"""
Assemble the self-contained Lakefront Local page (app/static/world.html + embedded data).

    python tools/build_world.py --out world.html --scene scene.json --sup sup.json --lakefront lakefront.json \
        --bathy bathy.json --swan swan.json --funwave funwave_rough.json funwave_storm.json --validate validate.json \
        [--history history-*.json] [--track track.json]

Large model products are thinned to keep the page well under the 16 MB artifact limit: the lake-bottom file
keeps only the map grid, the SWAN library is block-averaged 2x, FUNWAVE surfaces are quantized to 1 byte.
"""

import argparse
import base64
import datetime as dt
import gzip
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "app", "static", "world.html")
TITLES = {"derecho-2026-08-11": "Aug 11 derecho", "paddle-2026-10-23": "Your paddle, Oct 23"}


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def put(html, tag, obj):
    i = html.index(tag) + len(tag)
    j = html.index("</script>", i)
    return html[:i] + json.dumps(obj, separators=(",", ":")).replace("</", "<\\/") + html[j:]


# Diversey Harbor and its entrance channel (m east/north of the page origin 41.8820, -87.6278): the 10 m map
# grid averages the ~25 m channel under Lake Shore Drive into dry land, so the page gets the 4 m grid here.
DETAIL_BOX = (-860.0, 4560.0, 0.0, 5760.0)   # x0, y0, x1, y1


def crop_detail(g, box=DETAIL_BOX):
    """Crop a bathy grid (z_cm int16 + src uint8, rows south->north) to box; None if it doesn't cover it."""
    if not g or "z_cm" not in g:
        return None
    nx, ny, dx = g["nx"], g["ny"], g["dx"]
    i0, j0 = int(np.floor((box[0] - g["x0"]) / dx)), int(np.floor((box[1] - g["y0"]) / dx))
    i1, j1 = int(np.ceil((box[2] - g["x0"]) / dx)) + 1, int(np.ceil((box[3] - g["y0"]) / dx)) + 1
    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, nx), min(j1, ny)
    if i1 - i0 < 8 or j1 - j0 < 8:
        return None
    z = np.frombuffer(gzip.decompress(base64.b64decode(g["z_cm"])), "<i2").reshape(ny, nx)[j0:j1, i0:i1]
    enc = lambda a: base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()  # noqa: E731
    out = {"x0": g["x0"] + i0 * dx, "y0": g["y0"] + j0 * dx, "dx": dx, "nx": i1 - i0, "ny": j1 - j0, "z_cm": enc(z.astype("<i2")),
           "order": g.get("order")}
    if g.get("src"):
        out["src"] = enc(np.frombuffer(gzip.decompress(base64.b64decode(g["src"])), np.uint8).reshape(ny, nx)[j0:j1, i0:i1])
    return out


def thin_bathy(b):
    if not b:
        return None
    out = {k: v for k, v in b.items() if k not in ("models", "probe")}
    cov = ((b.get("probe") or {}).get("world") or {}).get("coverage")
    out["probe"] = {"world": {"coverage": cov}} if cov else {}
    route = (b.get("models") or {}).get("route")
    if route and route.get("dx", 99) < b["world"]["dx"]:
        det = crop_detail(route)
        if det:
            out["detail"] = det
    return out


def coarsen_swan(d, k=2):
    if not d:
        return None
    for v in d.get("domains", {}).values():
        g = v["grid"]; nc = len(v["cases"])
        r = np.frombuffer(gzip.decompress(base64.b64decode(v["ratio"])), np.uint8).reshape(nc, g["ny"], g["nx"]).astype(np.float32)
        a = np.frombuffer(gzip.decompress(base64.b64decode(v["dir"])), np.uint8).reshape(nc, g["ny"], g["nx"]).astype(np.float32) * 2 * np.pi / 256
        ny, nx = g["ny"] // k * k, g["nx"] // k * k
        blk = lambda x: x[:, :ny, :nx].reshape(nc, ny // k, k, nx // k, k)  # noqa: E731
        wet = blk((r > 0).astype(np.float32)).sum(axis=(2, 4))
        rr = np.where(wet > 0, blk(r).sum(axis=(2, 4)) / np.maximum(wet, 1), 0)
        ss, cc = blk(np.sin(a) * (r > 0)).sum(axis=(2, 4)), blk(np.cos(a) * (r > 0)).sum(axis=(2, 4))
        dd = (np.degrees(np.arctan2(ss, cc)) % 360) * 256 / 360
        enc = lambda x: base64.b64encode(gzip.compress(np.ascontiguousarray(x).astype(np.uint8).tobytes(), 9)).decode()  # noqa: E731
        v["ratio"], v["dir"] = enc(np.round(rr)), enc(np.round(dd) % 256)
        v["grid"] = dict(g, dx=g["dx"] * k, nx=nx // k, ny=ny // k)
    return d


def quantize_funwave(runs, frames=32):
    out = []
    for d in runs:
        if not d:
            continue
        e = d["eta"]
        if e.get("format") != "i8":
            eta = np.frombuffer(gzip.decompress(base64.b64decode(e["data"])), "<f2").reshape(e["frames"], e["ny"], e["nx"]).astype(np.float32)[-frames:]
            sc = max(float(np.percentile(np.abs(eta), 99.9)) / 120, 0.005)
            q = np.clip(np.round(eta / sc), -127, 127).astype(np.int8)
            e.update(frames=int(q.shape[0]), data=base64.b64encode(gzip.compress(q.tobytes(), 9)).decode(), format="i8", scale=sc)
        out.append(d)
    return out


def summarize_validation(v):
    if not v:
        return None
    ph = (v.get("physics") or {}).get("6.0") or v.get("buoys") or []
    ph = [b for b in ph if b.get("stats")]
    if not ph:
        return None
    s0 = ph[0]["stats"]
    return {"buoys": [{"buoy": b["buoy"], "window": b["window"], "stats": b["stats"]} for b in ph],
            "peak_note": "%s measured %.1f ft, model %.1f ft" % (ph[0]["buoy"], s0["obs_peak"]["m"] * 3.281, s0["pred_peak"]["m"] * 3.281)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    for k in ("scene", "sup", "lakefront", "bathy", "swan", "validate", "track"):
        ap.add_argument("--" + k)
    ap.add_argument("--funwave", nargs="*", default=[])
    ap.add_argument("--imports", nargs="*", default=[], help="forecast tables read from screenshots (examples/*.json)")
    ap.add_argument("--history", nargs="*", default=[])
    ap.add_argument("--start", help='open on a history period: JSON {"label":..,"t":ISO,"view":[lat,lon,km,heading,pitch]}')
    a = ap.parse_args()
    html = open(TEMPLATE).read()
    blocks = {
        "scene-data": load(a.scene), "sup-data": load(a.sup), "lakefront-data": load(a.lakefront), "track-data": load(a.track),
        "bathy-data": thin_bathy(load(a.bathy)), "swan-data": coarsen_swan(load(a.swan)),
        "funwave-data": quantize_funwave([load(p) for p in a.funwave]) or None, "validate-data": summarize_validation(load(a.validate)),
        "import-data": [d for d in (load(p) for p in a.imports) if d] or None,
        "meta-data": {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **({"start": json.loads(a.start)} if a.start else {})},
    }
    hist = []
    for p in a.history:
        h = load(p)
        if not h:
            continue
        name = os.path.basename(p)
        h.setdefault("label", name[len("history-"):-len(".json")] if name.startswith("history-") else name)
        h.setdefault("title", TITLES.get(h["label"]))
        hist.append(h)
    hist.sort(key=lambda h: 0 if "derecho" in h["label"] else 1)
    if hist:
        blocks["history-data"] = hist
    for tag, obj in blocks.items():
        if obj is not None:
            html = put(html, f'<script id="{tag}" type="application/json">', obj)
    with open(a.out, "w") as fh:
        fh.write(html)
    print(f"built {a.out}: {len(html) // 1000} KB; " + ", ".join(k for k, v in blocks.items() if v is not None))


if __name__ == "__main__":
    main()
