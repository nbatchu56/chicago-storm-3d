"""
FUNWAVE-TVD (fully nonlinear Boussinesq, phase-resolving) at the Diversey Harbor entrance.

Resolves the individual waves, so it captures what a phase-averaged model only approximates: diffraction
around the harbour heads, reflection off the vertical walls and the standing/crossing chop that makes,
breaking on the shoal. Domain: ~700 x 600 m over the entrance at 1.5 m, from the lidar/NCEI bottom.

    python -m app.funwave_case prepare --bathy bathy.json --case storm --out fw_storm
    mpirun -np 2 funwave  (in fw_storm)
    python -m app.funwave_case collect --run fw_storm --out funwave_storm.json

The model x axis points west (onshore), so the internal wavemaker sits near the offshore (east) edge.
"""

import argparse
import base64
import glob
import gzip
import json
import os

import numpy as np

from . import bathymetry as bm

# domain in the shared local frame (m east/north of State & Madison)
DOMAIN = {"x_east": 150.0, "x_west": -600.0, "y_south": 5300.0, "y_north": 5900.0, "dx": 1.5}
CASES = {
    "rough": {"hs": 1.0, "tp": 5.0, "dir": 45.0, "label": "Typical rough NE day: 3.3 ft at 5 s"},
    "storm": {"hs": 2.4, "tp": 7.0, "dir": 43.0, "label": "Oct 16, 2026 peak at buoy 45198: 7.9 ft from NE"},
}
TOTAL, STEADY, SNAP = 420.0, 240.0, 0.5         # s: run time, start of averaging, snapshot interval


def depth_grid(bathy):
    """Still-water depth on the model grid (model i -> west, j -> north), from the 4 m route grid."""
    gr = bathy["models"]["route"]
    z, src = bm.unpack(gr)
    level = bathy["level"]["m_igld85"]
    dep = level - z
    d = DOMAIN
    xs = d["x_east"] - np.arange(0, d["x_east"] - d["x_west"] + 1e-6, d["dx"])        # model i=0 at the east edge
    ys = d["y_south"] + np.arange(0, d["y_north"] - d["y_south"] + 1e-6, d["dx"])
    fx = (xs - gr["x0"]) / gr["dx"]; fy = (ys - gr["y0"]) / gr["dx"]
    i0 = np.clip(np.floor(fx).astype(int), 0, gr["nx"] - 2); j0 = np.clip(np.floor(fy).astype(int), 0, gr["ny"] - 2)
    tx = (fx - i0)[None, :]; ty = (fy - j0)[:, None]
    J0, I0 = np.meshgrid(j0, i0, indexing="ij")
    D = (dep[J0, I0] * (1 - tx) + dep[J0, I0 + 1] * tx) * (1 - ty) + (dep[J0 + 1, I0] * (1 - tx) + dep[J0 + 1, I0 + 1] * tx) * ty
    S = src[J0, I0]
    return xs, ys, D.astype(np.float32), S


def input_txt(m, n, case, dep_wk):
    c = CASES[case]
    theta = c["dir"] - 90.0                   # nautical "from" -> direction of travel relative to model +x (west)
    return f"""TITLE = diversey_{case}
PX = 2
PY = 1
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
DX = {DOMAIN['dx']}
DY = {DOMAIN['dx']}
WAVEMAKER = WK_IRR
DEP_WK = {dep_wk:.2f}
Xc_WK = 75.0
Yc_WK = {n * DOMAIN['dx'] / 2:.1f}
Ywidth_WK = {n * DOMAIN['dx'] * 0.9:.1f}
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


def prepare(bathy_path, case, out):
    bathy = json.load(open(bathy_path))
    xs, ys, D, S = depth_grid(bathy)
    os.makedirs(os.path.join(out, "output"), exist_ok=True)
    # wavemaker depth: median depth along the wavemaker line (model x = 75 m)
    iw = int(75.0 / DOMAIN["dx"])
    dep_wk = float(np.median(D[:, iw][D[:, iw] > 0.5])) if (D[:, iw] > 0.5).any() else 5.0
    np.savetxt(os.path.join(out, "depth.txt"), np.where(D > -3, D, -3.0), fmt="%.3f")   # land: negative (dry)
    with open(os.path.join(out, "input.txt"), "w") as fh:
        fh.write(input_txt(D.shape[1], D.shape[0], case, dep_wk))
    meta = {"case": case, **CASES[case], "domain": DOMAIN, "m": int(D.shape[1]), "n": int(D.shape[0]), "dep_wk": dep_wk,
            "lidar_frac": round(float((S == 1).mean()), 3)}
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
    frames = [_read(f, n, m)[:, ::-1][::step, ::step] for f in eta_files[-40:]]
    eta = np.stack(frames).astype(np.float16) if frames else np.zeros((0, 1, 1), np.float16)
    enc = lambda a: base64.b64encode(gzip.compress(a.tobytes(), 6)).decode()
    d = DOMAIN
    out = {"version": 1, "model": "FUNWAVE-TVD", **{k: meta[k] for k in ("case", "hs", "tp", "dir", "label", "lidar_frac")},
           "grid": {"x0": d["x_west"], "y0": d["y_south"], "dx": d["dx"], "nx": m, "ny": n},
           "hsig": enc(np.clip(np.round(hs * 100), 0, 65535).astype("<u2")), "hsig_units": "cm",
           "eta": {"dt": SNAP, "step": step, "nx": int(eta.shape[2]) if eta.size else 0, "ny": int(eta.shape[1]) if eta.size else 0,
                   "frames": int(eta.shape[0]), "data": enc(eta.astype("<f2")), "units": "m"}}
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--bathy", required=True); p.add_argument("--case", default="storm"); p.add_argument("--out", default="fw")
    c = sub.add_parser("collect"); c.add_argument("--run", default="fw"); c.add_argument("--out", default="funwave.json")
    a = ap.parse_args()
    if a.cmd == "prepare":
        print(prepare(a.bathy, a.case, a.out))
    else:
        o = collect(a.run, a.out)
        print({k: o[k] for k in ("case", "hs", "tp", "dir")}, o["grid"], o["eta"]["frames"], "frames")
