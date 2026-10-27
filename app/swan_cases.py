"""
SWAN wave-transformation library for the Lake Michigan Water Trail: Monroe Harbor to Montrose ("trail", 8 m)
inside a coarser lane out past the Harrison-Dever crib ("lane", 20 m); "route"/"montrose" are the older domains.

For each offshore wave condition (direction x a (peak period, height) case, CASES), SWAN (TU Delft, stationary, 2-D)
propagates a JONSWAP spectrum from the lake boundaries over the real bottom (app.bathymetry), with
refraction, shoaling, depth-induced breaking, bottom friction, triads, diffraction (phase-decoupled) and
obstacles: breakwaters/groynes/long piers from OpenStreetMap (opaque, partly reflecting) and the
seawall-type lakefront revetments (reflecting walls where deep water meets the shore).

    python -m app.swan_cases prepare --bathy bathy.json --lakefront lakefront.json --domain route --dirs 30,45 --out runs/
    (run swan in each case folder)
    python -m app.swan_cases collect --runs runs/ --out swan_route.json

Output per domain: Hs ratio (Hs / Hs_offshore, x100, uint8) and mean wave direction (nautical "from",
uint8 = deg * 256/360) for every case, on the SWAN grid, plus the case table.
"""

import argparse
import base64
import glob
import gzip
import json
import math
import os
import re

import numpy as np

from . import bathymetry as bm

DIRS = list(range(0, 180, 15))                 # N .. SSE: everything that can reach the Chicago shore from the lake
# (period s, offshore Hs m) cases run for every direction. The first six are the paddling-season grid; the rest are
# storm seas (Dec 29 2025 at Montrose: ~4 m at 7-9 s offshore), where breaking over the bars and the shoaling of long
# waves make the response height- and period-dependent. Storm heights aren't paired with short periods (too steep to
# exist); the page blends between the heights each period has.
CASES = [(2.5, 0.6), (2.5, 1.8), (4.0, 0.6), (4.0, 1.8), (6.0, 0.6), (6.0, 1.8),
         (6.0, 3.5), (8.5, 0.6), (8.5, 1.8), (8.5, 3.5), (8.5, 5.0)]
CASE_GROUPS = {"base": CASES[:6], "storm": CASES[6:]}
SWAN_DX = 10.0                                 # computational grid spacing (m) for the older route/montrose domains
# domains: bathy grid to take the bottom from, optional crop (S, W, N, E) and grid spacing (m)
DOMAINS = {
    "trail": {"grid": "corridor", "dx": 8.0},  # Monroe Harbor to Montrose nearshore, 4 m lidar -> 8 m
    # out past the Harrison-Dever crib, and north past Montrose to Kathy Osterman Beach, so waves from the north reach
    # Montrose over ~2 km of real nearshore instead of entering at a boundary 600 m from the beach
    "lane": {"grid": "world", "bbox": (41.866, -87.656, 41.988, -87.552), "dx": 20.0},
    "route": {"grid": "route", "dx": SWAN_DX},
    "montrose": {"grid": "montrose", "dx": SWAN_DX},
}
REFL = {"breakwater": 0.35, "groyne": 0.35, "pier": 0.5, "revetment": 0.45, "vertical": 0.85,
        # lakefront walls graded by the water depth 8 m out from the lidar waterline: in deep water the waves reach
        # the wall unbroken and come straight back (Theater on the Lake: 5 m deep 12 m out); over a shoal they
        # break first and little returns (Chess Pavilion, just south of the theater)
        "deep_wall": 0.8, "shoaled": 0.25}
TOE_DEEP, TOE_SHOAL = 3.0, 1.5             # m of water 8 m out from the waterline
# shore and structures known to be vertical walls rather than stepped revetment or rubble mound (S, W, N, E):
# Navy Pier's faces; Monroe Harbor's Grant Park seawall and the Easterly/Southerly Inner Breakwaters (crib
# breakwaters with vertical faces), all in 4-8 m of water by the lidar, so waves entering through the gap come
# back off the west wall instead of being half absorbed (a racer's GPS showed the slowdown it causes)
VERTICAL_ZONES = {"Navy Pier": (41.8885, -87.6150, 41.8945, -87.5955),
                  "Monroe Harbor": (41.8680, -87.6180, 41.8880, -87.6085)}


def case_id(d, tp, hs):
    return f"d{d:03d}_t{tp:.1f}_h{hs:.1f}".replace(".", "p")


def load_grid(bathy, domain):
    spec = DOMAINS.get(domain, {"grid": domain})
    name = spec["grid"]
    gr = bathy["models"][name] if name in bathy.get("models", {}) else bathy["world"]
    z, src = bm.unpack(gr)                     # bed elevation m IGLD85, rows south->north
    if spec.get("bbox"):
        s_, w, n, e = spec["bbox"]
        (xa, ya), (xb, yb) = bm.lf.xy(s_, w), bm.lf.xy(n, e)
        i0, j0 = max(int((xa - gr["x0"]) // gr["dx"]), 0), max(int((ya - gr["y0"]) // gr["dx"]), 0)
        i1, j1 = min(int((xb - gr["x0"]) // gr["dx"]) + 1, gr["nx"]), min(int((yb - gr["y0"]) // gr["dx"]) + 1, gr["ny"])
        z, src = z[j0:j1, i0:i1], src[j0:j1, i0:i1]
        gr = {"x0": gr["x0"] + i0 * gr["dx"], "y0": gr["y0"] + j0 * gr["dx"], "dx": gr["dx"], "nx": i1 - i0, "ny": j1 - j0}
    return gr, z, src


def resample(gr, z, dx):
    k = max(1, int(round(dx / gr["dx"])))
    zz = z[: (z.shape[0] // k) * k, : (z.shape[1] // k) * k]
    zz = zz.reshape(zz.shape[0] // k, k, zz.shape[1] // k, k).mean(axis=(1, 3))
    return {"x0": gr["x0"], "y0": gr["y0"], "dx": gr["dx"] * k, "nx": zz.shape[1], "ny": zz.shape[0]}, zz


def _area(pts):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))) / 2


def obstacles(lakefront, g, depth, fine=None):
    """Polylines (local metres) for SWAN OBSTACLE: structures and wall-like shoreline.

    Wall-like shore along the open lake is graded by the lidar depth 8 m out from the waterline: deep water at
    the wall (>= 3 m) reflects ~0.8, a shoal in front (< 1.5 m) ~0.25, otherwise ~0.45; Navy Pier's faces are
    vertical (~0.85); floating marina docks are left out. Long
    structures the 4 m lidar sees as open water are skipped (fine = (grid, depth, src))."""
    x0, y0, x1, y1 = g["x0"], g["y0"], g["x0"] + (g["nx"] - 1) * g["dx"], g["y0"] + (g["ny"] - 1) * g["dx"]
    inside = lambda p: x0 < p[0] < x1 and y0 < p[1] < y1
    out = []
    fg, fd = fine[:2] if fine else (g, depth)
    fs = fine[2] if fine and len(fine) > 2 else None

    def fine_at(px, py, lidar_only=False):
        i, j = int(round((px - fg["x0"]) / fg["dx"])), int(round((py - fg["y0"]) / fg["dx"]))
        if 0 <= j < fd.shape[0] and 0 <= i < fd.shape[1] and (not lidar_only or (fs is not None and fs[j, i] == 1)):
            return fd[j, i]
        return np.nan

    def submerged(pts):
        """A long mapped structure (>= 60 m) the lidar sees as open water (median > 2.5 m, < 15% above water):
        submerged or misplaced, not a wall. Short or narrow ones are left alone (a 4 m grid can't judge them)."""
        if sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) < 60:
            return False
        d = []
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(math.dist(a, b) / 3))
            d += [fine_at(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, True) for k in range(n + 1)]
        d = np.array(d, float); ok = d[np.isfinite(d)]
        return ok.size >= max(4, 0.7 * d.size) and float(np.median(ok)) > 2.5 and float((ok < 0).mean()) < 0.15
    zones = [(bm.lf.xy(s_, w), bm.lf.xy(n, e)) for s_, w, n, e in VERTICAL_ZONES.values()]

    def in_vertical_zone(pts):
        mx, my = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
        return any(a[0] <= mx <= b[0] and a[1] <= my <= b[1] for a, b in zones)
    for f in (lakefront or {}).get("osm", []):
        kind = f.get("kind")
        if kind not in ("breakwater", "groyne", "pier"):
            continue
        pts = [p for p in f["pts"] if inside(p)]
        if len(pts) < 2:
            continue
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        name = f.get("name") or ""
        if kind == "pier" and (length < 80 or re.search(r"\bdock\b", name, re.I)):   # floating marina docks, slips: transparent
            continue
        if f.get("closed"):                    # outline polygon: use its long axis as a wall
            pts = pts + [pts[0]]
        if submerged(pts):
            continue
        refl, trans = REFL[kind], 0.0
        if re.search(r"attenuator", name, re.I):          # floating wave attenuator: partly transmits
            refl, trans = 0.3, 0.5
        elif kind == "pier" and f.get("closed") and _area(f["pts"]) > 20000:   # Navy Pier: vertical faces
            refl = REFL["vertical"]
        elif kind in ("breakwater", "groyne") and in_vertical_zone(pts):
            refl = REFL["vertical"]
        out.append({"kind": kind, "name": name or None, "refl": refl, "trans": trans, "pts": [[round(p[0] - x0, 1), round(p[1] - y0, 1)] for p in pts]})
    # wall-like shore: coastline segments where the water is already > 2 m deep 15 m offshore, graded by toe depth
    def depth_at(px, py):
        i, j = int((px - x0) / g["dx"]), int((py - y0) / g["dx"])
        if 0 <= j < depth.shape[0] and 0 <= i < depth.shape[1]:
            return depth[j, i]
        return np.nan

    def toe_class(mx, my, nx, ny):
        """grade a wall segment by the depth 8 m lakeward of where the lidar puts the waterline"""
        sgn = 1.0 if np.nan_to_num(depth_at(mx + nx * 15, my + ny * 15), nan=-9) >= np.nan_to_num(depth_at(mx - nx * 15, my - ny * 15), nan=-9) else -1.0
        for t in np.arange(-20.0, 40.0, 1.0):     # the OSM line can sit metres off the real wall
            d = fine_at(mx + sgn * nx * t, my + sgn * ny * t)
            if np.isfinite(d) and d > 0.3:
                toe = fine_at(mx + sgn * nx * (t + 8), my + sgn * ny * (t + 8))
                if not np.isfinite(toe):
                    return "revetment"
                return "deep_wall" if toe >= TOE_DEEP else "shoaled" if toe < TOE_SHOAL else "revetment"
        return "revetment"

    def opening(a, b, nx, ny):
        """OSM closes the coastline across harbour mouths (Monroe Harbor's gap, 230 m and 8 m deep): a segment
        with lidar water along it and 10 m to both sides is an opening, not a wall"""
        L = math.dist(a, b)
        if L < 20:
            return False
        n = max(3, int(L / 4)); hits = 0
        for k in range(1, n):
            px, py = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
            ds = [fine_at(px + nx * t, py + ny * t) for t in (-10.0, 0.0, 10.0)]
            hits += all(np.isfinite(v) and v > 1.5 for v in ds)
        return hits >= 0.7 * (n - 1)

    def wall(run, kinds):
        mx, my = np.mean([p[0] for p in run]), np.mean([p[1] for p in run])
        k = "vertical" if any(a[0] <= mx <= b[0] and a[1] <= my <= b[1] for a, b in zones) else (kinds[0] if kinds else "revetment")
        return {"kind": "seawall", "wall": k, "refl": REFL[k], "trans": 0.0, "pts": [[round(p[0] - x0, 1), round(p[1] - y0, 1)] for p in run]}
    for f in (lakefront or {}).get("osm", []):
        if f.get("kind") != "coastline":
            continue
        run, kinds = [], []
        pts = f["pts"]
        # harbour mouths: runs of open segments, kept open only when wide enough for the grid (>= 60 m and 4 cells):
        # a gap of 1-2 cells is narrower than SWAN can resolve and its diffraction term goes unstable there
        segs = list(zip(pts, pts[1:]))
        op = []
        for a, b in segs:
            L = math.dist(a, b) or 1
            op.append(inside(a) and inside(b) and opening(a, b, (b[1] - a[1]) / L, -(b[0] - a[0]) / L))
        k = 0
        while k < len(op):
            if op[k]:
                e = k
                while e + 1 < len(op) and op[e + 1]:
                    e += 1
                if sum(math.dist(*segs[q]) for q in range(k, e + 1)) < max(60.0, 4 * g["dx"]):
                    for q in range(k, e + 1):
                        op[q] = False
                k = e + 1
            else:
                k += 1
        for (a, b), is_open in zip(segs, op):
            if not (inside(a) and inside(b)):
                continue
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            L = math.dist(a, b) or 1
            nx, ny = (b[1] - a[1]) / L, -(b[0] - a[0]) / L            # one side's normal; test both
            d = np.nanmax([depth_at(mx + nx * 15, my + ny * 15), depth_at(mx - nx * 15, my - ny * 15)])
            if np.isfinite(d) and d > 2.0 and not is_open:
                k = toe_class(mx, my, nx, ny)
                if run and kinds and k != kinds[0]:          # a new wall grade starts a new obstacle line
                    out.append(wall(run, kinds)); run, kinds = [], []
                run = run + [b] if run else [a, b]
                kinds.append(k)
            else:
                if len(run) >= 2:
                    out.append(wall(run, kinds))
                run, kinds = [], []
        if len(run) >= 2:
            out.append(wall(run, kinds))
    return out


def swn(g, level_above_lwd, d, tp, hs, obst):
    xl, yl = (g["nx"] - 1) * g["dx"], (g["ny"] - 1) * g["dx"]
    lines = [
        f"PROJECT 'chicago' 'c{d:03d}'",
        f"SET LEVEL={level_above_lwd:.3f} NAUTICAL",
        "MODE STATIONARY TWODIMENSIONAL",
        "COORDINATES CARTESIAN",
        # 0.06-1.2 Hz: the low end sits well below the peak of an 8.5 s storm sea (0.118 Hz)
        f"CGRID REGULAR 0. 0. 0. {xl:.1f} {yl:.1f} {g['nx'] - 1} {g['ny'] - 1} CIRCLE 36 0.06 1.2 28",
        f"INPGRID BOTTOM REGULAR 0. 0. 0. {g['nx'] - 1} {g['ny'] - 1} {g['dx']:.1f} {g['dx']:.1f} EXCEPTION -99.",
        "READINP BOTTOM 1. '../bottom.txt' 3 0 FREE",
        "BOUND SHAPESPEC JONSWAP 3.3 PEAK DSPR DEGREES",
    ]
    for side in ("E", "N", "S"):
        lines.append(f"BOUNDSPEC SIDE {side} CCW CONSTANT PAR {hs:.2f} {tp:.2f} {d:.1f} 25.")
    for o in obst:                             # long commands continue on the next line with "&"
        pairs = [f"{p[0]:.1f} {p[1]:.1f}" for p in o["pts"]]
        chunks = [" ".join(pairs[i:i + 5]) for i in range(0, len(pairs), 5)]
        lines.append(f"OBSTACLE TRANSM {o['trans']:.2f} REFL {o['refl']:.2f} LINE " + " &\n    ".join(chunks))
    lines += [
        "OFF QUADRUPL",
        "BREAKING CONSTANT 1.0 0.73",
        "FRICTION JONSWAP CONSTANT 0.038",
        "TRIAD",
        "DIFFRACTION 1 0.2",
        "NUMERIC STOPC 0.005 0.01 0.005 99.5 STAT 40",
        "BLOCK 'COMPGRID' NOHEAD 'out.mat' LAY 3 XP YP HSIGN DIR TM01 DEPTH",
        "COMPUTE",
        "STOP",
    ]
    return "\n".join(lines) + "\n"


def prepare(bathy_path, lakefront_path, domain, dirs, out_dir, periods=None, heights=None, cases=None):
    """cases: [(tp, hs), ...] (default CASES); periods/heights instead give the full grid of the two."""
    if cases is None:
        cases = [(tp, hs) for tp in periods for hs in heights] if periods and heights else CASES
    bathy = json.load(open(bathy_path))
    lakefront = json.load(open(lakefront_path)) if lakefront_path and os.path.exists(lakefront_path) else None
    gr, z, _ = load_grid(bathy, domain)
    g, zz = resample(gr, z, DOMAINS.get(domain, {}).get("dx", SWAN_DX))
    level = bathy["level"]["m_igld85"]
    depth_lwd = bm.LWD - zz                    # depth below LWD (SWAN adds LEVEL)
    fine = None
    if "corridor" in bathy.get("models", {}):
        fz, fsrc = bm.unpack(bathy["models"]["corridor"]); fine = (bathy["models"]["corridor"], level - fz, fsrc)
    obst = obstacles(lakefront, g, level - zz, fine or (gr, level - z))
    meta = {"domain": domain, "grid": g, "level_m_igld85": level, "obstacles": [{"kind": o["kind"], "n": len(o["pts"])} for o in obst]}
    os.makedirs(out_dir, exist_ok=True)
    bottom = "\n".join(" ".join(f"{v:.2f}" for v in row) for row in depth_lwd) + "\n"
    pairs, cases = cases, []
    os.makedirs(os.path.join(out_dir, domain), exist_ok=True)
    with open(os.path.join(out_dir, domain, "bottom.txt"), "w") as fh:
        fh.write(bottom)
    for d in dirs:
        for tp, hs in pairs:
            cid = case_id(d, tp, hs)
            cdir = os.path.join(out_dir, domain, cid)
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "INPUT"), "w") as fh:
                fh.write(swn(g, level - bm.LWD, d, tp, hs, obst))
            cases.append({"id": cid, "dir": d, "tp": tp, "hs": hs})
    meta["cases"] = cases
    with open(os.path.join(out_dir, domain, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    with open(os.path.join(out_dir, domain, "obstacles.json"), "w") as fh:
        json.dump(obst, fh)
    return meta


def read_case(cdir):
    from scipy.io import loadmat
    m = loadmat(os.path.join(cdir, "out.mat"))
    hs = m["Hsig"].astype(np.float32); di = m["Dir"].astype(np.float32); yp = m["Yp"]
    if yp[0, 0] > yp[-1, 0]:                    # make rows south -> north
        hs, di = hs[::-1], di[::-1]
    hs[(hs < 0) | ~np.isfinite(hs)] = 0
    di[~np.isfinite(di)] = 0
    return hs, di


def collect(runs_dir, out_path):
    out = {"version": 1, "model": "SWAN 41.51 stationary", "domains": {}}
    for meta_path in glob.glob(os.path.join(runs_dir, "*", "meta.json")):
        meta = json.load(open(meta_path)); dom = meta["domain"]; base = os.path.dirname(meta_path)
        got, ratio, dirs = [], [], []
        for c in meta["cases"]:
            cdir = os.path.join(base, c["id"])
            if not os.path.exists(os.path.join(cdir, "out.mat")):
                continue
            hs, di = read_case(cdir)
            if not (hs > 0).any():             # blew up (NaN read as zero): leave it out rather than show calm water
                print(f"::warning title=SWAN::{dom} {c['id']} has no waves anywhere; left out")
                continue
            ratio.append(np.clip(np.round(hs / c["hs"] * 100), 0, 255).astype(np.uint8))
            dirs.append((np.mod(di, 360) * 256 / 360).astype(np.uint8))
            got.append(c)
        if not got:
            continue
        enc = lambda arrs: base64.b64encode(gzip.compress(np.stack(arrs).tobytes(), 6)).decode()
        out["domains"][dom] = {"grid": meta["grid"], "level_m_igld85": meta["level_m_igld85"], "cases": got,
                               "obstacles": json.load(open(os.path.join(base, "obstacles.json"))),
                               "ratio": enc(ratio), "dir": enc(dirs),
                               "layout": "cases x rows(south->north) x cols(west->east); ratio = Hs/Hs0*100; dir = deg*256/360 (from)"}
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--bathy", required=True); p.add_argument("--lakefront"); p.add_argument("--domain", default="route")
    p.add_argument("--dirs", default=",".join(map(str, DIRS))); p.add_argument("--out", default="runs")
    p.add_argument("--group", choices=sorted(CASE_GROUPS), help="run only this group of (period, height) cases")
    c = sub.add_parser("collect"); c.add_argument("--runs", default="runs"); c.add_argument("--out", default="swan.json")
    a = ap.parse_args()
    if a.cmd == "prepare":
        m = prepare(a.bathy, a.lakefront, a.domain, [int(x) for x in a.dirs.split(",")], a.out,
                    cases=CASE_GROUPS[a.group] if a.group else None)
        print(f"{a.domain}: grid {m['grid']['nx']}x{m['grid']['ny']} at {m['grid']['dx']} m, {len(m['cases'])} cases, {len(m['obstacles'])} obstacles")
    else:
        o = collect(a.runs, a.out)
        print({k: len(v["cases"]) for k, v in o["domains"].items()})
