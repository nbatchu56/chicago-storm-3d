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
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "app", "static", "world.html")
TITLES = {"derecho-2026-08-11": "Aug 11 derecho", "paddle-2026-10-23": "Your paddle, Oct 23"}


def paddle_title(label):
    """paddle-2026-10-24 -> "Your paddle, Oct 24" (any other label: no title)."""
    try:
        if label and label.startswith("paddle-"):
            d = dt.date.fromisoformat(label[7:17])
            return f"Your paddle, {d.strftime('%b')} {d.day}"
    except ValueError:
        pass
    return None


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def put(html, tag, obj):
    i = html.index(tag) + len(tag)
    j = html.index("</script>", i)
    return html[:i] + json.dumps(obj, separators=(",", ":")).replace("</", "<\\/") + html[j:]


# The page draws the shoreline and nearshore bottom from a 4 m grid: the corridor (Monroe Harbor to Montrose)
# when the bathy build has it, else the older route grid around Diversey Harbor. Only the band the lidar
# actually surveyed (plus a margin) is kept; cells beyond are NODATA and the page uses the 10 m grid there.
DETAIL_BOX = (-860.0, 4560.0, 0.0, 5760.0)   # x0, y0, x1, y1 (m east/north of 41.8820, -87.6278): route fallback
NODATA = -32768
BAND_M = 160.0


def crop_detail(g, box=None, band_m=BAND_M):
    """4 m grid for the page: crop to box (None: whole grid), NODATA beyond band_m of any lidar cell."""
    from scipy import ndimage
    if not g or "z_cm" not in g or not g.get("src"):
        return None
    nx, ny, dx = g["nx"], g["ny"], g["dx"]
    i0, j0, i1, j1 = 0, 0, nx, ny
    if box:
        i0, j0 = max(int(np.floor((box[0] - g["x0"]) / dx)), 0), max(int(np.floor((box[1] - g["y0"]) / dx)), 0)
        i1, j1 = min(int(np.ceil((box[2] - g["x0"]) / dx)) + 1, nx), min(int(np.ceil((box[3] - g["y0"]) / dx)) + 1, ny)
    if i1 - i0 < 8 or j1 - j0 < 8:
        return None
    z = np.frombuffer(gzip.decompress(base64.b64decode(g["z_cm"])), "<i2").reshape(ny, nx)[j0:j1, i0:i1].copy()
    src = np.frombuffer(gzip.decompress(base64.b64decode(g["src"])), np.uint8).reshape(ny, nx)[j0:j1, i0:i1].copy()
    keep = ndimage.binary_dilation(src == 1, iterations=max(1, int(round(band_m / dx))))
    z[~keep] = NODATA; src[~keep] = 0
    enc = lambda a: base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()  # noqa: E731
    return {"x0": g["x0"] + i0 * dx, "y0": g["y0"] + j0 * dx, "dx": dx, "nx": i1 - i0, "ny": j1 - j0, "z_cm": enc(z.astype("<i2")),
            "src": enc(src), "nodata": NODATA, "order": g.get("order")}


def thin_bathy(b):
    if not b:
        return None
    out = {k: v for k, v in b.items() if k not in ("models", "probe")}
    cov = ((b.get("probe") or {}).get("world") or {}).get("coverage")
    out["probe"] = {"world": {"coverage": cov}} if cov else {}
    models = b.get("models") or {}
    det = crop_detail(models["corridor"]) if "corridor" in models else crop_detail(models.get("route"), DETAIL_BOX)
    if det and det["dx"] < b["world"]["dx"]:
        out["detail"] = det
    return out


def coarsen_swan(d, k=2):
    if not d:
        return None
    for v in d.get("domains", {}).values():
        g = v["grid"]; nc = len(v["cases"])
        if g["dx"] >= 16:                       # already coarse (the offshore lane): keep as is
            continue
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


def summarize_learn(corr, card):
    import sys
    sys.path.insert(0, ROOT)
    from app.learn import summarize
    return summarize(corr, card)


# Chop trouble spots: how often each spot is Choppy or worse over the paddling-season wave climate (tools/chop_climate.json,
# from the buoys), with the SWAN library carrying each hour's offshore waves to shore (sheltering, wall reflection, breaking).
# The same energy scale as the page's chop bands: E = 613 Hs^2 J/m2, weighted by steepness clamp(4 / Tp, 0.6, 2).
CHOP_E = (70.0, 160.0)          # Choppy, Rough
SPOT_DX = 16.0


def chop_spots(swan, climate):
    if not swan or not climate or not swan.get("domains"):
        return None
    doms = []
    for v in swan["domains"].values():
        g = v["grid"]; nc = len(v["cases"])
        # the paddling-season cases only (the climate's waves): the storm cases would change the blend between them
        keep = [i for i, c in enumerate(v["cases"]) if c["hs"] in (0.6, 1.8) and c["tp"] <= 6.0]
        r = np.frombuffer(gzip.decompress(base64.b64decode(v["ratio"])), np.uint8).reshape(nc, g["ny"], g["nx"])[keep].astype(np.float32) / 100
        doms.append((g, [v["cases"][i] for i in keep], r))
    doms.sort(key=lambda d: -d[0]["dx"])          # coarse first; finer domains overwrite where they have water
    x0 = min(g["x0"] for g, _, _ in doms); y0 = min(g["y0"] for g, _, _ in doms)
    x1 = max(g["x0"] + (g["nx"] - 1) * g["dx"] for g, _, _ in doms); y1 = max(g["y0"] + (g["ny"] - 1) * g["dx"] for g, _, _ in doms)
    nx, ny = int((x1 - x0) / SPOT_DX) + 1, int((y1 - y0) / SPOT_DX) + 1
    out = np.zeros((2, ny, nx), np.float32); wet = np.zeros((ny, nx), bool)
    total = float(climate["hours"])
    bins = [(*map(float, k.split("|")), float(n)) for k, n in climate["bins"].items()]
    for di, (g, cases, r) in enumerate(doms):
        dirs = sorted({c["dir"] for c in cases}); tps = sorted({c["tp"] for c in cases}); hss = sorted({c["hs"] for c in cases})
        idx = {(c["dir"], c["tp"], c["hs"]): i for i, c in enumerate(cases)}
        cnt = np.zeros((2,) + r.shape[1:], np.float32)
        groups = {}
        for d, tp, hs, n in bins:
            if hs < 0.15:
                continue                                    # too small to be choppy anywhere (needs ~0.24 m locally)
            dn = d % 360
            if dirs[-1] + 15 < dn < 345:
                continue                                    # waves from the land side never reach the lakefront
            dn = 0 if dn >= 345 else dn
            dk = min(dirs, key=lambda x: abs(x - dn)); tk = min(tps, key=lambda x: abs(x - tp))
            groups.setdefault((dk, tk), []).append((tp, hs, n))
        for (dk, tk), items in groups.items():
            lo, hi = r[idx[(dk, tk, hss[0])]], r[idx[(dk, tk, hss[-1])]]
            for tp, hs, n in items:
                a = min(max((hs - hss[0]) / (hss[-1] - hss[0]), 0.0), 1.0)
                h = (lo + (hi - lo) * a) * hs
                e = 613.0 * h * h * min(max(4.0 / max(tp, 1.0), 0.6), 2.0)
                cnt[0] += (e >= CHOP_E[0]) * n; cnt[1] += (e >= CHOP_E[1]) * n
        w = (r.max(axis=0) > 0)
        # resample onto the common grid (nearest)
        xs = x0 + np.arange(nx) * SPOT_DX; ys = y0 + np.arange(ny) * SPOT_DX
        ii = np.round((xs - g["x0"]) / g["dx"]).astype(int); jj = np.round((ys - g["y0"]) / g["dx"]).astype(int)
        okx = (ii >= 0) & (ii < g["nx"]); oky = (jj >= 0) & (jj < g["ny"])
        J, I = np.meshgrid(np.clip(jj, 0, g["ny"] - 1), np.clip(ii, 0, g["nx"] - 1), indexing="ij")
        m = oky[:, None] & okx[None, :] & w[J, I]
        # a finer domain's offshore (east) edge carries its boundary condition, not the sea: it hands over to the coarser
        # one over 400-1400 m from that edge
        wt = np.ones(nx, np.float32) if not di else np.clip((g["x0"] + (g["nx"] - 1) * g["dx"] - 400 - xs) / 1000, 0, 1)
        wt = np.where(wet, wt[None, :], 1.0) * m
        out[:] = out * (1 - wt) + cnt[:, J, I] / total * wt; wet |= m
    enc = lambda x: base64.b64encode(gzip.compress(np.ascontiguousarray(x).astype(np.uint8).tobytes(), 9)).decode()  # noqa: E731
    q = np.round(np.clip(out, 0, 1) * 255)
    return {"grid": {"x0": x0, "y0": y0, "dx": SPOT_DX, "nx": nx, "ny": ny}, "layout": "2 channels (choppy+, rough+) x rows(south->north) x cols(west->east); byte = fraction of hours * 255",
            "choppy": enc(q[0]), "rough": enc(q[1]), "hours": climate["hours"], "seasons": climate.get("seasons"), "months": climate.get("months"),
            "local_hours": climate.get("local_hours"), "buoys": climate.get("buoys"), "bands_j_m2": CHOP_E,
            "max_choppy": round(float(out[0].max()), 3)}


SWAN_PART_BYTES = 8_000_000     # raw bytes of one array per SWAN part file (gzipped and base64'd, both arrays: ~6 MB)


def write_side(out, rel, obj):
    path = os.path.join(os.path.dirname(os.path.abspath(out)), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = json.dumps(obj, separators=(",", ":"))
    with open(path, "w") as fh:
        fh.write(txt)
    return len(txt) // 1000


SPLIT = ("bathy-data", "swan-data", "funwave-data", "history-data", "chopspots-data")   # funwave: one file per run


def merge_earth(lakefront, earth):
    """The fused satellite water temperature replaces GLSEA as the page's lake grid (it already holds GLSEA where the
    satellites saw nothing recent); the rest of earth.json goes to the page as is."""
    w = earth.get("water") or {}
    if lakefront is not None and w.get("temp") and any(v is not None for v in w["temp"].get("v", [])):
        lake = dict(lakefront.get("lake") or {})
        lake["temp"] = w["temp"]
        lakefront = dict(lakefront, lake=lake)
        earth = dict(earth, water={k: v for k, v in w.items() if k != "temp"} | {"temp": {"label": w["temp"].get("label"), "time": w["temp"].get("time")}})
    return lakefront, earth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    for k in ("scene", "sup", "lakefront", "bathy", "swan", "validate", "track", "levels", "earth", "wn3"):
        ap.add_argument("--" + k)
    ap.add_argument("--learn", help="correction.json from the learn branch (experimental learned wave correction)")
    ap.add_argument("--scorecard", help="scorecard.json from the learn branch (forecast vs buoy by lead time)")
    ap.add_argument("--funwave", nargs="*", default=[])
    ap.add_argument("--imports", nargs="*", default=[], help="forecast tables read from screenshots (examples/*.json)")
    ap.add_argument("--history", nargs="*", default=[])
    ap.add_argument("--chop-climate", default=os.path.join(ROOT, "tools", "chop_climate.json"),
                    help="paddling-season wave climate (tools/chop_climate.py) for the chop trouble-spot map")
    ap.add_argument("--sessions", nargs="*", default=[],
                    help="paddle sessions (sessions/*.json): what was observed on the water, attached to the history period with the same label")
    ap.add_argument("--split", action="store_true",
                    help="write the big, slow-changing blocks (" + ", ".join(SPLIT) + ") to data/*.json beside the page instead of inline")
    ap.add_argument("--start", help='open on a history period: JSON {"label":..,"t":ISO,"view":[lat,lon,km,heading,pitch]}')
    a = ap.parse_args()
    html = open(TEMPLATE).read()
    lakefront, earth = load(a.lakefront), load(a.earth)
    if earth:
        lakefront, earth = merge_earth(lakefront, earth)
    blocks = {
        "scene-data": load(a.scene), "sup-data": load(a.sup), "lakefront-data": lakefront, "earth-data": earth, "wn3-data": load(a.wn3), "track-data": load(a.track),
        "bathy-data": thin_bathy(load(a.bathy)), "swan-data": coarsen_swan(load(a.swan)),
        "funwave-data": quantize_funwave([load(p) for p in a.funwave]) or None, "validate-data": summarize_validation(load(a.validate)),
        "import-data": [d for d in (load(p) for p in a.imports) if d] or None,
        "learn-data": summarize_learn(load(a.learn), load(a.scorecard)),
        "chopspots-data": None, "levels-data": load(a.levels),
        "meta-data": {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **({"start": json.loads(a.start)} if a.start else {})},
    }
    blocks["chopspots-data"] = chop_spots(blocks["swan-data"], load(a.chop_climate))
    hist = []
    for p in a.history:
        h = load(p)
        if not h:
            continue
        name = os.path.basename(p)
        h.setdefault("label", name[len("history-"):-len(".json")] if name.startswith("history-") else name)
        h.setdefault("title", TITLES.get(h["label"]) or paddle_title(h["label"]))
        hist.append(h)
    for p in a.sessions:
        sess = load(p)
        h = next((h for h in hist if sess and h["label"] == sess.get("label")), None)
        if h:
            h["session"] = sess
            if sess.get("title"):
                h["title"] = sess["title"]
    hist.sort(key=lambda h: 0 if "derecho" in h["label"] else 1)
    if hist:
        blocks["history-data"] = hist
    sizes = {}
    for tag, obj in blocks.items():
        if obj is None:
            continue
        if a.split and tag == "funwave-data":   # one file per tile x condition, loaded when that tile is in view
            stubs = []
            for r in obj:
                rel = f"data/fw/{r.get('tile', 'diversey')}_{r.get('case', 'run')}.json"
                path = os.path.join(os.path.dirname(os.path.abspath(a.out)), rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as fh:
                    txt = json.dumps(r, separators=(",", ":")); fh.write(txt); sizes[rel] = len(txt) // 1000
                stubs.append({k: v for k, v in r.items() if k != "hsig"} | {"eta": {k: v for k, v in r["eta"].items() if k != "data"}, "src": rel})
            html = put(html, f'<script id="{tag}" type="application/json">', stubs)
        elif a.split and tag == "swan-data":     # case tables in data/swan.json; each domain's arrays in parts under data/swan/
            idx = {"version": obj.get("version"), "model": obj.get("model"), "domains": {}}
            for name, v in obj["domains"].items():
                g = v["grid"]; nc = len(v["cases"]); n = g["nx"] * g["ny"]
                arr = {k: np.frombuffer(gzip.decompress(base64.b64decode(v[k])), np.uint8).reshape(nc, n) for k in ("ratio", "dir")}
                per = max(1, SWAN_PART_BYTES // n)          # cases per part
                # parts carry the library's fingerprint: a page that gets a part from another build skips the domain
                lib = hashlib.sha1(json.dumps([g, v["cases"]], sort_keys=True).encode()).hexdigest()[:12]
                parts = []
                for k0 in range(0, nc, per):
                    rel = f"data/swan/{name}-{k0 // per}.json"
                    part = {"lib": lib, "cases": [k0, min(k0 + per, nc)]} | {k: base64.b64encode(gzip.compress(arr[k][k0:k0 + per].tobytes(), 9)).decode() for k in arr}
                    sizes[rel] = write_side(a.out, rel, part); parts.append({"src": rel, "cases": part["cases"]})
                idx["domains"][name] = {k: x for k, x in v.items() if k not in ("ratio", "dir")} | {"parts": parts, "lib": lib}
            sizes["data/swan.json"] = write_side(a.out, "data/swan.json", idx)
            html = html.replace(f'<script id="{tag}" type="application/json">', f'<script id="{tag}" type="application/json" data-src="data/swan.json">', 1)
        elif a.split and tag in SPLIT:
            rel = f"data/{tag[:-len('-data')]}.json"
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(a.out)), "data"), exist_ok=True)
            with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), rel), "w") as fh:
                txt = json.dumps(obj, separators=(",", ":")); fh.write(txt); sizes[rel] = len(txt) // 1000
            old = f'<script id="{tag}" type="application/json">'
            html = html.replace(old, f'<script id="{tag}" type="application/json" data-src="{rel}">', 1)
        else:
            html = put(html, f'<script id="{tag}" type="application/json">', obj)
    with open(a.out, "w") as fh:
        fh.write(html)
    print(f"built {a.out}: {len(html) // 1000} KB; " + ", ".join(k for k, v in blocks.items() if v is not None)
          + "".join(f"; {k} {v} KB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
