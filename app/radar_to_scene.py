#!/usr/bin/env python3
"""
radar_to_scene.py
=================
Turn NEXRAD radar volume scans into a 3-D weather scene for the Chicago Storm 3D viewer
(chicago_storm_3d.html): clouds, rain shafts and storm cores over Chicago and the lake,
for the observed scans plus a 0-2 h extrapolation forecast.

What it does
1. Reads KLOT (or other) Level II volume scans: downloads the latest ones from the public
   AWS archive, takes local files, or builds a synthetic demo storm (--demo).
2. Grids every tilt onto a 3-D box centred on downtown Chicago with Py-ART
   (2 km horizontal, 0.5 km vertical, 0-15 km), so storm tops, bright bands and anvils
   are kept, not just a flat rain map.
3. Tracks echo motion (same method as chicago_radar_nowcast.py) and advects every height
   level forward to make forecast frames every 10 min.
4. Infers a cloud field from the radar echoes: cloud above a cloud base wherever there is
   echo, a wider deck around stratiform rain, and anvils spreading from deep cores.
   Radar cannot see non-precipitating cloud, so this is an estimate, and it is labelled so.
5. Writes scene JSON (gzip-compressed volumes) and, with --html, a standalone copy of the
   viewer with the scene embedded so it opens with your storm.

Usage
    pip install arm_pyart boto3 opencv-python-headless scipy numpy
    python radar_to_scene.py --demo --html                     # synthetic storm
    python radar_to_scene.py --html                            # latest KLOT scans
    python radar_to_scene.py --time 2025-07-15T21:00 --html    # past event (UTC)
    python radar_to_scene.py --files KLOT2025...V06 ... --html # local Level II files
    python radar_to_scene.py --html --wind-speed 14 --wind-dir 30   # lake wind for waves
    python radar_to_scene.py --html --ndbc 45007               # lake wind from a buoy

Needs chicago_radar_nowcast.py in the same folder (motion tracking + advection) and
chicago_storm_3d.html as the viewer template for --html.
"""

import argparse
import base64
import datetime as dt
import gzip
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chicago_radar_nowcast as nc  # noqa: E402

UTC = dt.timezone.utc
DOWNTOWN = (41.8820, -87.6278)  # State & Madison: origin of Chicago's street numbering
DX = 2.0          # km
DZ = 0.5          # km
NZ = 30           # 0-15 km
EXT_HALF = 140.0  # km, working box (lets storms advect in from outside the view)
VIEW_HALF = 100.0  # km, box sent to the viewer


def xy_from_downtown(lat, lon):
    x = (lon - DOWNTOWN[1]) * 111.32 * math.cos(math.radians(DOWNTOWN[0]))
    y = (lat - DOWNTOWN[0]) * 110.57
    return x, y


def axes(half):
    n = int(round(2 * half / DX)) + 1
    return np.linspace(-half, half, n)


Z_LEVELS = (np.arange(NZ) + 0.5) * DZ  # level centres, km


# ---------------------------------------------------------------------------
# Real radar -> 3-D grid
# ---------------------------------------------------------------------------
RHOHV_MIN = 0.8      # rain, snow and cloud ice correlate > ~0.9; birds, insects and clutter ~0.3-0.8
RHOHV_KEEP_DBZ = 35  # strong cores (hail can read < 0.8) are kept whatever their correlation


def biological_mask(dbz, rhohv):
    """True where an echo is non-meteorological by dual-pol: weak and poorly correlated. Gates with no correlation
    reading (masked or NaN) are kept. On clear dawns in migration season a wide shallow layer of 5-15 dBZ birds and
    insects otherwise becomes a deck of cloud (Oct 21, 2026: a cloudless sunrise drawn with cumulus)."""
    dbz = np.ma.filled(np.ma.asarray(dbz, dtype=float), np.nan)
    rho = np.ma.filled(np.ma.asarray(rhohv, dtype=float), np.nan)
    with np.errstate(invalid="ignore"):
        return np.isfinite(rho) & (rho < RHOHV_MIN) & ~(dbz >= RHOHV_KEEP_DBZ)


def borrow_rhohv(radar, rho):
    """The lowest tilts are scanned twice (split cuts) and only one pass carries dual-pol: give each ray of a pass without a
    correlation reading the one from the nearest-azimuth ray of a pass at the same elevation that has it."""
    rho = np.ma.filled(np.ma.asarray(rho, dtype=float), np.nan)
    ang = np.asarray(radar.fixed_angle["data"], dtype=float)
    az = np.asarray(radar.azimuth["data"], dtype=float)
    sl = [(int(a), int(b) + 1) for a, b in zip(radar.sweep_start_ray_index["data"], radar.sweep_end_ray_index["data"])]
    has = [np.isfinite(rho[a:b]).mean() > 0.02 for a, b in sl]
    for i, (a, b) in enumerate(sl):
        if has[i]:
            continue
        j = min((k for k in range(len(sl)) if has[k] and abs(ang[k] - ang[i]) < 0.25), key=lambda k: abs(ang[k] - ang[i]), default=None)
        if j is None:
            continue
        pa, pb = sl[j]
        d = np.abs(((az[a:b, None] - az[None, pa:pb]) + 180) % 360 - 180)
        rho[a:b] = rho[pa + d.argmin(axis=1)]
    return rho


def smooth_rhohv(radar, rho, rays=5, gates=9):
    """Bird and insect returns are noisy gate to gate (a third of single gates read > 0.8 at a median of ~0.65), so
    each gate is judged on the mean correlation around it: ~5 rays x ~2 km, within its own sweep."""
    out = np.full_like(rho, np.nan)
    sl = [(int(a), int(b) + 1) for a, b in zip(radar.sweep_start_ray_index["data"], radar.sweep_end_ray_index["data"])]
    for a, b in sl:
        r = rho[a:b]; ok = np.isfinite(r)
        num = ndimage.uniform_filter(np.where(ok, r, 0.0), (rays, gates), mode=("wrap", "nearest"))
        den = ndimage.uniform_filter(ok.astype(float), (rays, gates), mode=("wrap", "nearest"))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(den > 0.2, num / den, np.nan)
    return out


def exclude_biological(radar, gf):
    f = radar.fields.get("cross_correlation_ratio")
    if f is None:
        print("dual-pol: no correlation coefficient in this volume", flush=True)
        return 0
    ref = radar.fields["reflectivity"]["data"]
    m = biological_mask(ref, smooth_rhohv(radar, borrow_rhohv(radar, f["data"])))
    gf.exclude_gates(m)
    echo = np.ma.filled(np.ma.asarray(ref, dtype=float), np.nan) >= 5
    rho = np.ma.filled(np.ma.asarray(f["data"], dtype=float), np.nan)[echo]
    print(f"dual-pol: dropped {int(m.sum())} of {int(echo.sum())} echo gates >= 5 dBZ; rho on echo: "
          f"{int(np.isfinite(rho).sum())} valid, median {np.nanmedian(rho) if np.isfinite(rho).any() else float('nan'):.2f}", flush=True)
    return int(m.sum())


def grid_volume(path):
    import pyart
    # reflectivity, plus the dual-pol correlation coefficient to drop birds and insects; skipping the other moments cuts
    # peak memory several-fold
    radar = pyart.io.read_nexrad_archive(path, include_fields=["reflectivity", "cross_correlation_ratio"])
    t = pyart.util.datetime_from_radar(radar)
    t = dt.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=UTC)
    rlat, rlon = float(radar.latitude["data"][0]), float(radar.longitude["data"][0])
    rx, ry = xy_from_downtown(rlat, rlon)  # radar position relative to downtown, km
    ax = axes(EXT_HALF)
    # drop gates beyond the farthest grid corner (NEXRAD reaches 460 km; we need ~240 km from KLOT)
    far_km = max(math.hypot(cx - rx, cy - ry) for cx in (ax[0], ax[-1]) for cy in (ax[0], ax[-1])) + 10
    keep = int(np.searchsorted(radar.range["data"], far_km * 1e3)) + 1
    if keep < radar.ngates:
        radar.range["data"] = radar.range["data"][:keep]
        radar.ngates = keep
        for f in radar.fields.values():
            f["data"] = f["data"][:, :keep]
        radar.init_gate_x_y_z(); radar.init_gate_altitude(); radar.init_gate_longitude_latitude()
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_invalid("reflectivity")
    gf.exclude_below("reflectivity", 0.0)   # no-echo gates add nothing but memory
    exclude_biological(radar, gf)
    radar.fields.pop("cross_correlation_ratio", None)   # not gridded
    # grid limits are relative to the radar
    lim_x = ((ax[0] - rx) * 1e3, (ax[-1] - rx) * 1e3)
    lim_y = ((ax[0] - ry) * 1e3, (ax[-1] - ry) * 1e3)
    g = pyart.map.grid_from_radars(
        (radar,), grid_shape=(NZ, ax.size, ax.size),
        grid_limits=((Z_LEVELS[0] * 1e3, Z_LEVELS[-1] * 1e3), lim_y, lim_x),
        fields=["reflectivity"], gatefilters=(gf,), weighting_function="Barnes2", roi_func="dist_beam",
        min_radius=1500.0)
    vol = np.ma.filled(g.fields["reflectivity"]["data"].astype(float), np.nan)  # (z, y, x)
    vol[vol < 5.0] = np.nan
    # remove isolated speckle in 3-D
    keep = ndimage.binary_opening(np.isfinite(vol), structure=np.ones((1, 2, 2)))
    vol[~keep] = np.nan
    return t, vol, (rx, ry), radar.metadata.get("instrument_name", "")


# ---------------------------------------------------------------------------
# Synthetic 3-D demo storm: squall line with trailing stratiform rain and an anvil
# ---------------------------------------------------------------------------
def synthetic_volume(t_min, X, Y, rng_state):
    rng, texture, cells = rng_state
    u, v = 0.85, -0.25                       # km/min, toward ESE (~53 km/h)
    ox, oy = u * t_min, v * t_min            # storm-relative offset
    Xs, Ys = X - ox, Y - oy                  # storm-relative coordinates
    Z = np.zeros((NZ,) + X.shape)
    # line axis: through (-55, 0) at t = 0, oriented NNE-SSW (perpendicular to motion)
    nx_, ny_ = 0.96, -0.28                   # unit vector along motion
    along = (Xs + 55) * nx_ + Ys * ny_       # + ahead of the line
    across = -(Xs + 55) * ny_ + Ys * nx_     # along-line coordinate
    z = Z_LEVELS[:, None, None]
    # convective cells along the line
    for (s0, d0, sig, top, grow) in cells:
        D = d0 + grow * t_min / 60.0
        T = top + 0.04 * (D - d0) * 10
        g = np.exp(-((along - 0) ** 2 + (across - s0) ** 2) / (2 * sig ** 2))
        prof = np.where(z <= T - 1.5, D - 1.4 * np.maximum(z - 4.0, 0),
                        np.maximum(D - 1.4 * (T - 5.5) - (z - (T - 1.5)) / 1.5 * 35, -10))
        Z += 10 ** (prof / 10.0) * g[None]
    # trailing stratiform region (behind = negative along), with a melting-layer bright band
    # sharp leading edge (ahead of the line), long trailing tail behind it
    strat_h = np.where(along > -8, np.exp(-((along + 8) / 3.0) ** 2),
                       np.exp(-((along + 30) / 22.0) ** 2) / np.exp(-((22 / 22.0) ** 2)) * 0.9)
    strat_h = np.minimum(strat_h, 0.9) * np.exp(-(across / 75) ** 4) * 0.8
    Ds = 33.0
    sprof = np.where(z < 4.0, Ds + 6 * np.exp(-((z - 3.6) / 0.35) ** 2), Ds - 5.5 * (z - 4.0))
    sprof = np.where(z > 7.5, -20, sprof)
    Z += 10 ** (sprof / 10.0) * strat_h[None]
    # forward anvil: weak echo aloft only
    anv_h = np.exp(-((along - 22) / 18) ** 2) * np.exp(-(across / 70) ** 4) * (along > -5)
    aprof = np.where((z > 8.0) & (z < 11.5), 18 - 4 * np.abs(z - 9.8), -20)
    Z += 10 ** (aprof / 10.0) * anv_h[None]
    # storm-relative texture so cells and gaps move with the storm
    rr = (Ys - Y.min()) / DX + 60
    cc = (Xs - X.min()) / DX + 60
    tex = ndimage.map_coordinates(texture, [rr, cc], order=1, mode="wrap")
    Z *= 10 ** (0.45 * tex)[None]
    dbz = 10 * np.log10(np.maximum(Z, 1e-6))
    dbz[dbz < 5] = np.nan
    return dbz


def synthetic_state(seed=7):
    rng = np.random.default_rng(seed)
    texture = ndimage.gaussian_filter(rng.standard_normal((400, 400)), 2.2)
    texture /= texture.std()
    cells = []
    for s in np.arange(-110, 111, 11.0):
        cells.append((s + rng.normal(0, 3), rng.uniform(46, 60), rng.uniform(3.5, 6.5),
                      rng.uniform(9.5, 13.5), rng.uniform(-6, 6)))
    return rng, texture, cells


# ---------------------------------------------------------------------------
# Forecast: advect every level with the low-level motion field
# ---------------------------------------------------------------------------
def forecast_volumes(times, vols, lead_min, step_min):
    comps = [np.nanmax(np.where(np.isfinite(v[:8]), v[:8], -30), axis=0) for v in vols]
    comps = [np.where(c > 5, c, np.nan) for c in comps]
    u, v = nc.estimate_motion(times, comps, tile=32, step=16, smooth_px=20, max_shift_px=12)
    n = lead_min // step_min
    last = vols[-1]
    zlin = np.where(np.isfinite(last), 10 ** (np.nan_to_num(last) / 10.0), 0.0)
    out = np.zeros((n,) + last.shape)
    for k in range(NZ):
        if zlin[k].max() <= 0:
            continue
        fr = nc.extrapolate(zlin[k], u, v, n, step_min)
        for j in range(n):
            out[j, k] = ndimage.gaussian_filter(fr[j], 0.08 * (j + 1))
    with np.errstate(divide="ignore"):
        dbz = 10 * np.log10(out)
    dbz[~np.isfinite(dbz) | (dbz < 5)] = np.nan
    mean_u = float(np.mean(u)) * DX  # km/min
    mean_v = float(np.mean(v)) * DX
    return [dbz[j] for j in range(n)], (mean_u, mean_v)


# ---------------------------------------------------------------------------
# Cloud field inferred from radar echoes
# ---------------------------------------------------------------------------
def infer_cloud(dbz, cloud_base_km=1.2):
    echo = np.isfinite(dbz)
    # ignore shallow, weak echo columns (tops < 2 km and < 15 dBZ): near the radar these are
    # almost always ground clutter, birds or insects, not cloud
    colmax = np.nanmax(np.where(echo, dbz, -99.0), axis=0)
    has = echo.any(axis=0)
    top_i = np.where(has, NZ - 1 - np.argmax(echo[::-1], axis=0), -1)
    nonmet = has & (Z_LEVELS[np.clip(top_i, 0, NZ - 1)] < 2.0) & (colmax < 15.0)
    echo = echo & ~nonmet[None]
    dbz = np.where(echo, dbz, np.nan)
    d = np.nan_to_num(dbz, nan=-10.0)
    z = Z_LEVELS[:, None, None]
    # 1) cloud wherever there is echo above cloud base, slightly grown
    core = ndimage.binary_dilation(echo, structure=np.ones((3, 3, 3)))
    dens = ndimage.gaussian_filter(core.astype(float), (0.8, 1.2, 1.2))
    dens *= np.clip((d + 10) / 40, 0.35, 1.0) if echo.any() else 1.0
    # 2) deck around rain reaching the lower levels (nimbostratus): base to echo top + 1 km.
    #    Columns with echo only aloft (anvil, virga) keep cloud just where the echo is.
    col = echo.any(axis=0)
    rain_col = echo[: int(3.0 / DZ)].any(axis=0)
    top_idx = np.where(col, NZ - 1 - np.argmax(echo[::-1], axis=0), 0)
    top_km = Z_LEVELS[top_idx] + 1.0
    deck_h = ndimage.gaussian_filter(rain_col.astype(float), 1.5)
    top_sm = ndimage.gaussian_filter(ndimage.grey_dilation(np.where(col, top_km, 0), size=(3, 3)), 1.5)
    deck = deck_h[None] * (z <= top_sm[None]) * 0.6
    # 3) anvil tops spreading a little beyond the deep, intense cores
    colmax = d.max(axis=0)
    deep = (colmax >= 45) & (top_km >= 9.0)
    anv = np.clip(ndimage.gaussian_filter(deep.astype(float), 3.0) * 2.5, 0, 1)
    anv_top = np.maximum(top_sm, 9.5)
    anvil = anv[None] * (np.abs(z - (anv_top[None] - 0.8)) < 1.0) * 0.7
    cloud = np.maximum.reduce([dens, deck, anvil])
    cloud[Z_LEVELS < cloud_base_km] = 0.0
    # soften the base over one level
    base_i = int(np.searchsorted(Z_LEVELS, cloud_base_km))
    if base_i < NZ:
        cloud[base_i] *= 0.6
    return np.clip(cloud, 0, 1)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode(cloud, dbz):
    """Bytes: cloud volume then dBZ volume, each (z, y, x) uint8, gzip + base64.
    dBZ byte: 0 = no echo, otherwise round((dBZ + 10) * 3) -> -10..75 dBZ."""
    c8 = np.round(cloud * 255).astype(np.uint8)
    d = np.where(np.isfinite(dbz), np.clip(np.round((dbz + 10) * 3), 1, 255), 0).astype(np.uint8)
    raw = np.ascontiguousarray(c8).tobytes() + np.ascontiguousarray(d).tobytes()
    return base64.b64encode(gzip.compress(raw, 9)).decode("ascii")


def crop(vol):
    ax = axes(EXT_HALF)
    i0 = int(np.argmin(np.abs(ax + VIEW_HALF)))
    i1 = int(np.argmin(np.abs(ax - VIEW_HALF))) + 1
    return vol[:, i0:i1, i0:i1]


def build_scene(times, vols, source, site, radar_xy, lead, step, wind, cloud_base):
    fc, motion = forecast_volumes(times, vols, lead, step)
    fc_times = [times[-1] + dt.timedelta(minutes=step * (k + 1)) for k in range(len(fc))]
    frames = []
    for t, v in zip(times, vols):
        cv = crop(v)
        frames.append({"t": t.isoformat(), "kind": "obs", "lead": 0,
                       "data": encode(infer_cloud(cv, cloud_base), cv)})
    for k, (t, v) in enumerate(zip(fc_times, fc)):
        cv = crop(v)
        frames.append({"t": t.isoformat(), "kind": "fc", "lead": step * (k + 1),
                       "data": encode(infer_cloud(cv, cloud_base), cv)})
    nxy = crop(vols[0]).shape[1]
    meta = {
        "version": 1, "source": source, "site": site,
        "origin": {"lat": DOWNTOWN[0], "lon": DOWNTOWN[1], "name": "State & Madison"},
        "grid": {"nx": nxy, "ny": nxy, "nz": NZ, "dx_km": DX, "dz_km": DZ,
                 "x0_km": -VIEW_HALF, "y0_km": -VIEW_HALF},
        "radar_xy_km": [round(radar_xy[0], 2), round(radar_xy[1], 2)],
        "motion_km_per_min": [round(motion[0], 4), round(motion[1], 4)],
        "wind": wind, "cloud_base_km": cloud_base,
        "created": dt.datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return {"meta": meta, "frames": frames}


def lake_wind(args):
    if args.ndbc:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from chicago_nearshore_waves import fetch_ndbc
            obs = fetch_ndbc(args.ndbc)
            return {"speed": round(obs["wspd10"], 1), "dir": obs["wdir"], "duration_h": args.duration,
                    "source": f"NDBC {args.ndbc.upper()} {obs['time']} UTC"}
        except Exception as e:  # noqa: BLE001
            print(f"NDBC {args.ndbc} unavailable ({e}); using --wind-speed/--wind-dir", file=sys.stderr)
    return {"speed": args.wind_speed, "dir": args.wind_dir, "duration_h": args.duration,
            "source": "scenario"}


def write_html(scene, template, out_html):
    with open(template, encoding="utf-8") as fh:
        html = fh.read()
    start_tag = '<script id="scene-data" type="application/json">'
    i = html.find(start_tag)
    j = html.find("</script>", i)
    if i < 0 or j < 0:
        raise RuntimeError(f"{template} has no scene-data block")
    html = html[:i + len(start_tag)] + json.dumps(scene, separators=(",", ":")) + html[j:]
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="KLOT")
    ap.add_argument("--time", help="end time UTC, e.g. 2025-07-15T21:00 (default: now)")
    ap.add_argument("--n-scans", type=int, default=6)
    ap.add_argument("--files", nargs="+", help="local Level II files")
    ap.add_argument("--cache", default="radar_cache")
    ap.add_argument("--demo", action="store_true", help="synthetic squall line, no download")
    ap.add_argument("--lead", type=int, default=120, help="forecast minutes (default 120)")
    ap.add_argument("--step", type=int, default=10, help="forecast frame step, min (default 10)")
    ap.add_argument("--cloud-base", type=float, default=1.2, help="cloud base height, km")
    ap.add_argument("--wind-speed", type=float, default=12.0, help="lake wind U10, m/s")
    ap.add_argument("--wind-dir", type=float, default=20.0, help="lake wind FROM, deg")
    ap.add_argument("--duration", type=float, default=8.0, help="hours the wind has blown")
    ap.add_argument("--ndbc", help="buoy for live lake wind, e.g. 45007")
    ap.add_argument("--out", default="chicago_storm_scene.json")
    ap.add_argument("--html", nargs="?", const="chicago_storm_3d_scene.html",
                    help="also write a standalone viewer with the scene embedded")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "chicago_storm_3d.html"))
    args = ap.parse_args()

    if args.demo:
        ax = axes(EXT_HALF)
        X, Y = np.meshgrid(ax, ax)
        state = synthetic_state()
        t_end = dt.datetime.now(UTC).replace(second=0, microsecond=0)
        times = [t_end + dt.timedelta(minutes=5 * k) for k in range(-args.n_scans + 1, 1)]
        vols = [synthetic_volume((t - t_end).total_seconds() / 60, X, Y, state) for t in times]
        source, site = "Demo storm (synthetic squall line)", "DEMO"
        rx, ry = xy_from_downtown(*nc.SITES["KLOT"])
    else:
        site = args.site.upper()
        if args.files:
            paths = args.files
        else:
            end = (dt.datetime.fromisoformat(args.time).replace(tzinfo=UTC) if args.time
                   else dt.datetime.now(UTC))
            scans = nc.list_scans(site, end, args.n_scans)
            print(f"Downloading {len(scans)} {site} volumes...")
            paths = [nc.download(b, k, args.cache) for _, b, k in scans]
        vols, times = [], []
        for p in paths:
            print(f"  gridding {os.path.basename(p)}")
            t, v, (rx, ry), name = grid_volume(p)
            times.append(t); vols.append(v)
        order = np.argsort(times)
        times = [times[i] for i in order]; vols = [vols[i] for i in order]
        source = f"NEXRAD {site} Level II"

    wind = lake_wind(args)
    scene = build_scene(times, vols, source, site, (rx, ry), args.lead, args.step, wind, args.cloud_base)
    with open(args.out, "w") as fh:
        json.dump(scene, fh, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1e6
    print(f"Scene: {len(scene['frames'])} frames "
          f"({sum(f['kind'] == 'obs' for f in scene['frames'])} observed), "
          f"motion {scene['meta']['motion_km_per_min']} km/min -> {args.out} ({size:.1f} MB)")
    if args.html:
        write_html(scene, args.template, args.html)
        print(f"Viewer with embedded scene -> {args.html}")


if __name__ == "__main__":
    main()
