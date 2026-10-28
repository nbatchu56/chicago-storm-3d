"""
LakeView live service.

    GET  /                 the 3-D viewer (live mode: loads api/scene.json, refreshes itself)
    GET  /sup              stand-up paddling wave forecast for the lakefront (loads api/sup.json)
    GET  /api/sup.json     SUP forecast: 8 spots x 48 h (refreshed every SUP_INTERVAL_MIN, default 30)
    GET  /api/lakefront.json  lakefront map (OSM), lake temperature/ice (NOAA GLSEA), station history (every 6 h)
    GET  /api/scene.json   latest scene (gzip when the client accepts it)
    GET  /world            LakeView: radar volume over the real lakefront, waves, GPX ride-along
    GET  /api/history?start=..&end=..   conditions for a past window (UTC ISO); 202 while it is built, then cached
    GET  /api/status       freshness, latest scan time, errors, next run
    GET  /healthz          liveness probe (always 200 while the process is up)
    GET  /readyz           200 once a scene exists, 503 before
    POST /api/refresh      run the pipeline now (needs header X-Refresh-Token = REFRESH_TOKEN)

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
Use ONE worker process: the pipeline runs in a background thread inside it.
"""

import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import threading

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .pipeline import Pipeline
from . import sup_waves as sup_waves_mod
from .sup_waves import SupForecaster
from .lakefront import LakefrontUpdater
from . import history as history_mod

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("klot.web")

STATIC = os.path.join(os.path.dirname(__file__), "static")
pipeline = Pipeline()
sup = SupForecaster(pipeline.cfg.data_dir)
lakefront = LakefrontUpdater(pipeline.cfg.data_dir)
SCHEDULER = os.environ.get("SCHEDULER", "internal")  # internal | external (cron calls /api/refresh)


@contextlib.asynccontextmanager
async def lifespan(_app):
    if SCHEDULER == "internal":
        log.info("Starting background pipeline: %s every %s min", pipeline.cfg.site, pipeline.cfg.interval_min)
        pipeline.start()
        if os.environ.get("SUP", "on") != "off":
            sup.start()
            lakefront.start()
    else:
        log.info("External scheduler: POST /api/refresh to update")
        if not pipeline.status["ready"]:
            threading.Thread(target=pipeline.run_once, daemon=True).start()
    yield
    pipeline.stop()
    sup.stop()
    lakefront.stop()


app = FastAPI(title="LakeView", lifespan=lifespan, docs_url=None, redoc_url=None)
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])


SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n')


def page(name):
    """Static pages are authored as fragments; wrap them in a proper HTML document."""
    with open(os.path.join(STATIC, name), encoding="utf-8") as fh:
        body = fh.read()
    html = body if body.lstrip().lower().startswith("<!doctype") else SKELETON + body + "\n</body>\n</html>\n"
    return Response(html, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
def index():
    return page("index.html")


@app.get("/api/scene.json")
def scene(request: Request):
    if not os.path.exists(pipeline.scene_path):
        raise HTTPException(503, "The first radar scans are still being processed. Try again in a minute.")
    etag = f'"{pipeline.status.get("updated") or os.path.getmtime(pipeline.scene_path)}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    headers = {"Cache-Control": "no-cache", "ETag": etag}
    if "gzip" in request.headers.get("accept-encoding", "") and os.path.exists(pipeline.scene_gz_path):
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
        return FileResponse(pipeline.scene_gz_path, media_type="application/json", headers=headers)
    return FileResponse(pipeline.scene_path, media_type="application/json", headers=headers)


@app.get("/sup", include_in_schema=False)
def sup_page():
    return page("sup.html")


@app.get("/api/sup.json")
def sup_json():
    if not os.path.exists(sup.path):
        raise HTTPException(503, sup.status.get("last_error") or "The first paddling forecast is still being built.")
    return FileResponse(sup.path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/lakefront.json")
def lakefront_json():
    if not os.path.exists(lakefront.path):
        raise HTTPException(503, lakefront.status.get("last_error") or "The lakefront map and lake temperatures are still loading.")
    return FileResponse(lakefront.path, media_type="application/json", headers={"Cache-Control": "no-cache"})


MODEL_FILES = {"bathy": "bathy.json", "swan": "swan.json", "funwave": "funwave.json"}


@app.get("/api/{name}.json", include_in_schema=False)
def model_file(name: str):
    """Static model products (lake bottom, SWAN library, FUNWAVE fields) placed in DATA_DIR by the workflows."""
    if name not in MODEL_FILES:
        raise HTTPException(404, "Not found")
    path = os.path.join(pipeline.cfg.data_dir, MODEL_FILES[name])
    if not os.path.exists(path):
        raise HTTPException(404, f"{MODEL_FILES[name]} isn't installed on this server (see README: wave models).")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/world", include_in_schema=False)
def world_page():
    return page("world.html")


HIST_DIR = os.path.join(pipeline.cfg.data_dir, "history")
HIST_JOBS = {}            # key -> {"state": "building"|"error", "error": str}
HIST_LOCK = threading.Lock()
HIST_MAX_MIN = 6 * 60     # longest window one request may build


def _hist_build(key, start, end, path):
    try:
        data = history_mod.build(start, end, pipeline.cfg.site, os.path.join(HIST_DIR, "radar_cache"))
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, path)
        with HIST_LOCK:
            HIST_JOBS.pop(key, None)
    except Exception as e:  # noqa: BLE001
        log.exception("history %s", key)
        with HIST_LOCK:
            HIST_JOBS[key] = {"state": "error", "error": str(e)}


@app.get("/api/history")
def history(start: str, end: str = ""):
    try:
        t0 = history_mod._parse_time(start)
        t1 = history_mod._parse_time(end) if end else t0 + dt.timedelta(minutes=30)
    except ValueError:
        raise HTTPException(400, "start and end must be ISO times, e.g. 2026-10-23T12:20Z")
    now = dt.datetime.now(dt.timezone.utc)
    if not (t0 < t1 <= now + dt.timedelta(minutes=5)) or (t1 - t0).total_seconds() > HIST_MAX_MIN * 60:
        raise HTTPException(400, f"Pick a past window shorter than {HIST_MAX_MIN // 60} hours.")
    if t0.year < 1995:
        raise HTTPException(400, "The radar archive starts in the 1990s.")
    t0, t1 = t0.replace(second=0, microsecond=0), t1.replace(second=0, microsecond=0)
    key = hashlib.sha1(f"{t0.isoformat()}|{t1.isoformat()}".encode()).hexdigest()[:16]
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"{key}.json")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json", headers={"Cache-Control": "public, max-age=86400"})
    with HIST_LOCK:
        job = HIST_JOBS.get(key)
        if job and job["state"] == "error":
            HIST_JOBS.pop(key)
            raise HTTPException(502, f"Couldn't build that period: {job['error']}")
        if not job:
            if sum(1 for j in HIST_JOBS.values() if j["state"] == "building") >= 2:
                raise HTTPException(429, "Two historical periods are already being built. Try again in a minute.")
            HIST_JOBS[key] = {"state": "building"}
            threading.Thread(target=_hist_build, args=(key, t0, t1, path), daemon=True).start()
    return JSONResponse({"state": "building", "retry_after_s": 15}, status_code=202, headers={"Retry-After": "15"})


@app.get("/api/status")
def status():
    return JSONResponse(pipeline.public_status() | {"sup": sup.status, "lakefront": lakefront.status}, headers={"Cache-Control": "no-store"})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


@app.get("/readyz", include_in_schema=False)
def readyz():
    if not pipeline.status["ready"]:
        raise HTTPException(503, "no scene yet")
    return {"ok": True, "latest_scan": pipeline.status.get("latest_scan")}


@app.post("/api/refresh")
def refresh(x_refresh_token: str = Header(default="")):
    token = os.environ.get("REFRESH_TOKEN", "")
    if not token or x_refresh_token != token:
        raise HTTPException(403, "Set REFRESH_TOKEN on the server and send it as X-Refresh-Token.")
    started = pipeline.run_once()
    sup_ok = sup.run_once()
    return {"ran": started, "status": pipeline.public_status(), "sup": sup_ok}
