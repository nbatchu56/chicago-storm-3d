"""
Chicago Storm 3D live service.

    GET  /                 the 3-D viewer (live mode: loads api/scene.json, refreshes itself)
    GET  /sup              stand-up paddling wave forecast for the lakefront (loads api/sup.json)
    GET  /api/sup.json     SUP forecast: 8 spots x 48 h (refreshed every SUP_INTERVAL_MIN, default 30)
    GET  /api/scene.json   latest scene (gzip when the client accepts it)
    GET  /api/status       freshness, latest scan time, errors, next run
    GET  /healthz          liveness probe (always 200 while the process is up)
    GET  /readyz           200 once a scene exists, 503 before
    POST /api/refresh      run the pipeline now (needs header X-Refresh-Token = REFRESH_TOKEN)

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
Use ONE worker process: the pipeline runs in a background thread inside it.
"""

import contextlib
import logging
import os
import threading

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .pipeline import Pipeline
from . import sup_waves as sup_waves_mod
from .sup_waves import SupForecaster

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("klot.web")

STATIC = os.path.join(os.path.dirname(__file__), "static")
pipeline = Pipeline()
sup = SupForecaster(pipeline.cfg.data_dir)
SCHEDULER = os.environ.get("SCHEDULER", "internal")  # internal | external (cron calls /api/refresh)


@contextlib.asynccontextmanager
async def lifespan(_app):
    if SCHEDULER == "internal":
        log.info("Starting background pipeline: %s every %s min", pipeline.cfg.site, pipeline.cfg.interval_min)
        pipeline.start()
        if os.environ.get("SUP", "on") != "off":
            sup.start()
    else:
        log.info("External scheduler: POST /api/refresh to update")
        if not pipeline.status["ready"]:
            threading.Thread(target=pipeline.run_once, daemon=True).start()
    yield
    pipeline.stop()
    sup.stop()


app = FastAPI(title="Chicago Storm 3D", lifespan=lifespan, docs_url=None, redoc_url=None)
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


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
    return FileResponse(os.path.join(STATIC, "sup.html"), media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/api/sup.json")
def sup_json():
    if not os.path.exists(sup.path):
        raise HTTPException(503, sup.status.get("last_error") or "The first paddling forecast is still being built.")
    return FileResponse(sup.path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    return JSONResponse(pipeline.public_status() | {"sup": sup.status}, headers={"Cache-Control": "no-store"})


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
