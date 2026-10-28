"""
FUNWAVE-TVD (fully nonlinear Boussinesq, phase-resolving) on the walls of the Lake Michigan Water Trail.

Resolves the individual waves, so it captures what a phase-averaged model only approximates: diffraction
around the harbour heads, reflection off the walls and revetments and the standing/crossing chop (clapotis)
that makes, breaking on the shoal. The walls are not idealized: the revetment steps and slopes come from the
4 m lidar bottom, so how much each one reflects falls out of the run. Tiles of ~1 km at 1.5 m (TILES).

    python -m app.funwave_case prepare --bathy bathy.json --tile diversey --case storm --out fw
    mpirun -np 4 funwave input.txt  (in fw)
    python -m app.funwave_case collect --run fw --out funwave_diversey_storm.json

The model x axis points west (onshore), so the internal wavemaker sits near the offshore (east) edge.
"""

import argparse
import base64
import glob
import gzip
import json
import math
import os

import numpy as np

from . import bathymetry as bm

# tiles in the shared local frame (m east/north of State & Madison); the east edge must be open lake
# (the wavemaker sits 75 m inside it). "diversey" is the original harbour-entrance domain, widened.
# x_wall: about the westmost lake wall in the tile; oblique seas need the wavemaker to reach up-wave of it
TILES = {
    "diversey": {"x_east": 200.0, "x_west": -650.0, "y_south": 5150.0, "y_north": 6000.0, "dx": 1.5, "x_wall": -400.0, "label": "Diversey Harbor"},
    "belmont": {"x_east": 150.0, "x_west": -950.0, "y_south": 5900.0, "y_north": 6850.0, "dx": 1.5, "x_wall": -650.0, "label": "Diversey to Belmont"},
    "northave": {"x_east": 800.0, "x_west": -150.0, "y_south": 3350.0, "y_north": 4300.0, "dx": 1.5, "x_wall": -150.0, "label": "North Ave to the Chess Pavilion"},
    "fullerton": {"x_east": 300.0, "x_west": -650.0, "y_south": 4200.0, "y_north": 5150.0, "dx": 1.5, "x_wall": -250.0, "label": "Chess Pavilion to Fullerton (Theater on the Lake)"},
    # the harbour gap between the Easterly and Southerly Inner Breakwaters (y -560..-790, 8 m deep) and the Grant
    # Park seawall it faces ~500 m west. The offshore band is measured opposite the gap: elsewhere the vertical
    # Easterly breakwater ~35 m behind the band would send its reflection back through it
    "monroe": {"x_east": 1850.0, "x_west": 880.0, "y_south": -1450.0, "y_north": -250.0, "dx": 1.5, "x_wall": 950.0, "label": "Monroe Harbor",
               "norm_y": (-800.0, -550.0), "fill_holes": True},
}
DOMAIN = TILES["diversey"]
CASES = {
    "rough": {"hs": 1.0, "tp": 5.0, "dir": 45.0, "label": "Typical rough NE day: 3.3 ft at 5 s"},
    "storm": {"hs": 2.4, "tp": 7.0, "dir": 43.0, "label": "Oct 16, 2026 peak at buoy 45198: 7.9 ft from NE"},
    "east": {"hs": 1.2, "tp": 5.5, "dir": 90.0, "label": "Moderate east sea: 3.9 ft at 5.5 s"},
    # Oct 3, 2026 race in Monroe Harbor: buoy 45198 0.7-0.8 m, waves from the E (mean dir 50-90), wind ESE 5-6 m/s
    "race": {"hs": 0.75, "tp": 4.0, "dir": 85.0, "label": "Oct 3, 2026 race: 2.5 ft at 4 s from the east"},
    # the glass sunrises of Oct 21 and Oct 28, 2026 (Diversey to North Ave): wind 1-4 m/s S-SSW (offshore), only the
    # leftover swell of the days before, 0.1-0.2 m at buoy 45174 from the ESE-SE (mean dir ~100-160; 45198 was out
    # for the season, no period reported at that height). 4 s keeps the run at the race case's resolution.
    "glass": {"hs": 0.15, "tp": 4.0, "dir": 120.0, "label": "Glass mornings, Oct 21 & 28, 2026: 0.5 ft leftover swell at 4 s from the ESE"},
}
TOTAL, STEADY, SNAP = 420.0, 240.0, 0.5         # s: run time, start of averaging, snapshot interval


def depth_grid(bathy, d=None):
    """Still-water depth on the model grid (model i -> west, j -> north), from the 4 m corridor (or route) grid."""
    d = d or DOMAIN
    gr = bathy["models"].get("corridor") or bathy["models"]["route"]
    z, src = bm.unpack(gr)
    level = bathy["level"]["m_igld85"]
    dep = level - z
    xs = d["x_east"] - np.arange(0, d["x_east"] - d["x_west"] + 1e-6, d["dx"])        # model i=0 at the east edge
    ys = d["y_south"] + np.arange(0, d["y_north"] - d["y_south"] + 1e-6, d["dx"])
    fx = (xs - gr["x0"]) / gr["dx"]; fy = (ys - gr["y0"]) / gr["dx"]
    i0 = np.clip(np.floor(fx).astype(int), 0, gr["nx"] - 2); j0 = np.clip(np.floor(fy).astype(int), 0, gr["ny"] - 2)
    tx = (fx - i0)[None, :]; ty = (fy - j0)[:, None]
    J0, I0 = np.meshgrid(j0, i0, indexing="ij")
    D = (dep[J0, I0] * (1 - tx) + dep[J0, I0 + 1] * tx) * (1 - ty) + (dep[J0 + 1, I0] * (1 - tx) + dep[J0 + 1, I0 + 1] * tx) * ty
    S = src[J0, I0]
    if d.get("fill_holes"):
        D = fill_holes(D, S)
    return xs, ys, D.astype(np.float32), S


def fill_holes(D, S, iters=4000):
    """Lidar holes enclosed by lidar-surveyed water (a mooring field the laser didn't see through: NCEI's
    interpolated 1.5-2 m there is a false shoal between 4-5 m lidar soundings): bridge them smoothly from
    the surrounding lidar depths (Laplace fill). Holes touching the tile edge or dry land are left alone."""
    from scipy import ndimage
    hole = (S != 1) & (D > 0)
    lab, n = ndimage.label(hole)
    D = D.copy()
    for k in range(1, n + 1):
        m = lab == k
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m.sum() < 50:
            continue
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        if (D[ring] <= 0.3).mean() > 0.1:      # touches the shore or a wall: not an enclosed hole
            continue
        j0, j1 = np.where(m.any(1))[0][[0, -1]]; i0, i1 = np.where(m.any(0))[0][[0, -1]]
        sl = (slice(max(j0 - 3, 0), j1 + 4), slice(max(i0 - 3, 0), i1 + 4))
        sub, mm = D[sl].copy(), m[sl]
        sub[mm] = float(D[ring].mean())
        for _ in range(iters):
            avg = 0.25 * (np.roll(sub, 1, 0) + np.roll(sub, -1, 0) + np.roll(sub, 1, 1) + np.roll(sub, -1, 1))
            sub[mm] = avg[mm]
        D[sl] = sub
    return D


def input_txt(m, n, case, dep_wk, d=None, tile="diversey", px=2, py=2):
    d = d or DOMAIN
    c = CASES[case]
    theta = c["dir"] - 90.0                   # nautical "from" -> direction of travel relative to model +x (west)
    return f"""TITLE = {tile}_{case}
PX = {px}
PY = {py}
DEPTH_TYPE = DATA
DEPTH_FILE = depth.txt
RESULT_FOLDER = output/
Mglob = {m}
Nglob = {n}
TOTAL_TIME = {TOTAL}
PLOT_START_TIME = {TOTAL - 20.0}
PLOT_INTV = {SNAP}
PLOT_INTV_STATION = 1.0
SCREEN_INTV = 20.0
DX = {d['dx']}
DY = {d['dx']}
WAVEMAKER = WK_IRR
DEP_WK = {dep_wk:.2f}
Xc_WK = 75.0
Yc_WK = {n * d['dx'] / 2:.1f}
Ywidth_WK = {n * d['dx'] * 0.9:.1f}
FreqPeak = {1.0 / c['tp']:.4f}
FreqMin = {0.6 / c['tp']:.4f}
FreqMax = {2.5 / c['tp']:.4f}
Hmo = {c['hs']}
GammaTMA = 3.3
ThetaPeak = {theta:.1f}
Sigma_Theta = 15.0
Delta_WK = 0.5
Time_ramp = 2.0
PERIODIC = F
DIFFUSION_SPONGE = F
FRICTION_SPONGE = T
DIRECT_SPONGE = T
Csp = 0.0
CDsponge = 1.0
Sponge_west_width = 45.0
Sponge_east_width = 0.0
Sponge_south_width = 45.0
Sponge_north_width = 45.0
Cd = 0.002
CFL = 0.4
FroudeCap = 1.0
MinDepth = 0.02
VISCOSITY_BREAKING = T
Cbrk1 = 0.65
Cbrk2 = 0.35
T_INTV_mean = {TOTAL - STEADY}
STEADY_TIME = {STEADY}
NumberStations = 0
FIELD_IO_TYPE = ASCII
DEPTH_OUT = F
U = F
V = F
ETA = T
MASK = F
Umean = F
Vmean = F
ETAmean = F
WaveHeight = T
"""


def run_domain(tile, case):
    """The tile, extended up-wave for an oblique sea: waves from the wavemaker travel at an angle, so without
    this the walls near the up-wave edge sit in a false shadow (half height at Theater on the Lake for NE seas)."""
    core = TILES[tile]; d = dict(core)
    ang = math.radians(abs(90.0 - CASES[case]["dir"]))
    ext = round(max(0.0, (core["x_east"] - 75.0 - core["x_wall"]) * math.tan(ang)) + 30.0, -1) if ang > 0.05 else 0.0
    if CASES[case]["dir"] < 90:
        d["y_north"] = core["y_north"] + ext       # from the NE: rays run to the SW
    else:
        d["y_south"] = core["y_south"] - ext
    return d, core


def prepare(bathy_path, case, out, tile="diversey", procs=4):
    bathy = json.load(open(bathy_path))
    d, core = run_domain(tile, case)
    xs, ys, D, S = depth_grid(bathy, d)
    os.makedirs(os.path.join(out, "output"), exist_ok=True)
    # wavemaker depth: median depth along the wavemaker line (model x = 75 m); it must be open water
    iw = int(75.0 / d["dx"])
    wet = D[:, iw] > 0.5
    if wet.mean() < 0.8:
        raise SystemExit(f"{tile}: wavemaker line only {wet.mean():.0%} wet; move the east edge offshore")
    dep_wk = float(np.median(D[:, iw][wet]))
    np.savetxt(os.path.join(out, "depth.txt"), np.where(D > -3, D, -3.0), fmt="%.3f")   # land: negative (dry)
    px, py = (2, procs // 2) if procs >= 2 else (1, 1)
    with open(os.path.join(out, "input.txt"), "w") as fh:
        fh.write(input_txt(D.shape[1], D.shape[0], case, dep_wk, d, tile, px, py))
    meta = {"case": case, "tile": tile, "tile_label": d["label"], **CASES[case], "domain": d, "core": core, "m": int(D.shape[1]), "n": int(D.shape[0]),
            "dep_wk": dep_wk, "lidar_frac": round(float((S == 1).mean()), 3)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    return meta


def _read(path, n, m):
    a = np.loadtxt(path, dtype=np.float32)
    return a.reshape(n, m)


def collect(run, out_path, step=2):
    meta = json.load(open(os.path.join(run, "meta.json")))
    n, m = meta["n"], meta["m"]
    hs_files = sorted(glob.glob(os.path.join(run, "output", "Hsig_*")))
    eta_files = sorted(glob.glob(os.path.join(run, "output", "eta_*")))
    if not hs_files:
        raise SystemExit("no Hsig output")
    hs = _read(hs_files[-1], n, m)
    # world orientation: columns west->east (model i runs east->west), rows south->north
    hs = hs[:, ::-1]
    # dry cells carry a placeholder (~12 m) in FUNWAVE's output: zero them, so they don't swamp the 1-byte
    # quantization of the surface on the page
    dep_path = os.path.join(run, "depth.txt")
    dry = (np.loadtxt(dep_path, dtype=np.float32).reshape(n, m)[:, ::-1] <= 0.0) if os.path.exists(dep_path) else np.zeros((n, m), bool)
    hs = np.where(dry, 0.0, hs)
    frames = [np.where(dry, 0.0, _read(f, n, m)[:, ::-1])[::step, ::step] for f in eta_files[-40:]]
    eta = np.stack(frames).astype(np.float16) if frames else np.zeros((0, 1, 1), np.float16)
    # the oblique internal wavemaker in a 600 m wide domain with absorbing sides under-delivers energy:
    # scale the run so the open-water band in front of the wavemaker carries the stated offshore height
    rows = slice(60, -60)
    nb = (meta.get("core") or {}).get("norm_y") or meta["domain"].get("norm_y")
    if nb:                                    # the tile names where the open-water band is clean
        y0 = meta["domain"]["y_south"]; dxm = meta["domain"]["dx"]
        rows = slice(max(0, int((nb[0] - y0) / dxm)), int((nb[1] - y0) / dxm))
    off = hs[rows, -130:-80]
    norm = float(meta["hs"] / max(off.mean(), 1e-3))
    hs = hs * norm; eta = (eta.astype(np.float32) * norm).astype(np.float16)
    enc = lambda a: base64.b64encode(gzip.compress(a.tobytes(), 6)).decode()
    d = meta["domain"]
    out = {"version": 3, "model": "FUNWAVE-TVD", "scaled_by": round(norm, 3), "tile": meta.get("tile", "diversey"), "tile_label": meta.get("tile_label", "Diversey Harbor"),
           **{k: meta[k] for k in ("case", "hs", "tp", "dir", "label", "lidar_frac")},
           "grid": {"x0": d["x_west"], "y0": d["y_south"], "dx": d["dx"], "nx": m, "ny": n},
           # the tile proper (the run domain may extend up-wave of it): the page shows a tile there
           "core": {k: (meta.get("core") or d)[k] for k in ("x_west", "x_east", "y_south", "y_north")},
           "hsig": enc(np.clip(np.round(hs * 100), 0, 65535).astype("<u2")), "hsig_units": "cm",
           "eta": {"dt": SNAP, "step": step, "nx": int(eta.shape[2]) if eta.size else 0, "ny": int(eta.shape[1]) if eta.size else 0,
                   "frames": int(eta.shape[0]), "data": enc(eta.astype("<f2")), "units": "m"}}
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--bathy", required=True); p.add_argument("--case", default="storm"); p.add_argument("--out", default="fw")
    p.add_argument("--tile", default="diversey", choices=sorted(TILES)); p.add_argument("--procs", type=int, default=4)
    c = sub.add_parser("collect"); c.add_argument("--run", default="fw"); c.add_argument("--out", default="funwave.json")
    a = ap.parse_args()
    if a.cmd == "prepare":
        print(prepare(a.bathy, a.case, a.out, a.tile, a.procs))
    else:
        o = collect(a.run, a.out)
        print({k: o[k] for k in ("case", "hs", "tp", "dir")}, o["grid"], o["eta"]["frames"], "frames")
