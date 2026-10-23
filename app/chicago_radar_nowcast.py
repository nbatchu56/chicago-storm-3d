#!/usr/bin/env python3
"""
chicago_radar_nowcast.py
========================
Short-range (0-2 h) precipitation forecast for Chicago from NEXRAD radar returns.

How it works ("radar nowcasting" by extrapolation, as used by NWS/MRMS-style products)
1. Download the latest Level II volume scans from KLOT (Chicago/Romeoville) from the
   public NOAA/Unidata archive on AWS (no account needed).
2. Build a low-level composite reflectivity map (max of the lowest tilts) on a 2 km grid,
   remove speckle/clutter, and convert dBZ -> rain rate with Marshall-Palmer Z = 200 R^1.6.
3. Track how echoes move between scans with dense optical flow (OpenCV Farneback),
   smoothed and filled into rain-free areas.
4. Advect the latest rain field forward along that motion (semi-Lagrangian backtracking).
5. Run an ensemble with perturbed motion and growing multiplicative noise to get
   probabilities (chance of rain, chance of heavy rain) that widen with lead time.
6. Output an animated radar loop (observed + forecast), a point forecast for chosen
   locations (downtown, O'Hare, Midway, the lakefront), and a CSV.

Usage
-----
    pip install arm_pyart boto3 opencv-python-headless scipy matplotlib imageio
    python chicago_radar_nowcast.py                    # latest KLOT scans, 2 h forecast
    python chicago_radar_nowcast.py --demo             # synthetic storm (offline test)
    python chicago_radar_nowcast.py --time 2025-07-15T20:30 --lead 90   # past event (UTC)
    python chicago_radar_nowcast.py --files a_V06 b_V06 c_V06            # local Level II files
    python chicago_radar_nowcast.py --site KMKX        # another radar (e.g. Milwaukee)

Limits: extrapolation assumes storms keep moving the same way. Skill is good for
~30-60 min and drops after that; it cannot forecast new storms forming or rapid
growth/decay, and lake-effect bands that sit in place are handled poorly.
Radar sees precipitation, not cloud. The beam rises with distance (~3 km up at 200 km),
so low drizzle and lake-effect snow far from the radar can be missed.
"""

import argparse
import datetime as dt
import math
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

UTC = dt.timezone.utc

# Radar sites (lat, lon)
SITES = {"KLOT": (41.6045, -88.0847), "KMKX": (42.9678, -88.5506),
         "KIWX": (41.3586, -85.7000), "KILX": (40.1505, -89.3368)}

POINTS = {  # forecast locations
    "Downtown (Loop)": (41.8800, -87.6300),
    "O'Hare": (41.9786, -87.9048),
    "Midway": (41.7868, -87.7522),
    "Lake offshore (Navy Pier +5 km)": (41.8917, -87.5400),
}

# Approximate SW Lake Michigan shoreline, used as a map reference line
SHORELINE = [(42.80, -87.78), (42.50, -87.80), (42.25, -87.82), (42.05, -87.67),
             (41.88, -87.61), (41.72, -87.53), (41.66, -87.43), (41.61, -87.27),
             (41.66, -87.05), (41.72, -86.90), (41.82, -86.70), (42.10, -86.48),
             (42.50, -86.25), (42.80, -86.21)]

BUCKETS = ["unidata-nexrad-level2", "noaa-nexrad-level2"]  # new archive first

# NWS-style reflectivity colors
DBZ_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
DBZ_COLORS = ["#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
              "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
              "#f800fd", "#9854c6"]
DBZ_CMAP = ListedColormap(DBZ_COLORS)
DBZ_NORM = BoundaryNorm(DBZ_LEVELS, DBZ_CMAP.N)


# ---------------------------------------------------------------------------
# Grid / coordinate helpers
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, lat0, lon0, half_width_km=200.0, res_km=2.0):
        self.lat0, self.lon0, self.res = lat0, lon0, res_km
        self.x = np.arange(-half_width_km, half_width_km + res_km / 2, res_km)  # km east
        self.y = self.x.copy()                                                   # km north
        self.X, self.Y = np.meshgrid(self.x, self.y)  # row = y (south->north)

    def latlon_to_xy(self, lat, lon):
        x = (lon - self.lon0) * 111.32 * math.cos(math.radians(self.lat0))
        y = (lat - self.lat0) * 110.57
        return x, y

    def xy_to_rc(self, x, y):
        return (y - self.y[0]) / self.res, (x - self.x[0]) / self.res

    @property
    def shape(self):
        return self.X.shape


def dbz_to_rain(dbz):
    """Marshall-Palmer Z = 200 R^1.6 -> rain rate (mm/h). NaN/low echoes -> 0."""
    z = 10.0 ** (np.nan_to_num(dbz, nan=-30.0) / 10.0)
    r = (z / 200.0) ** (1 / 1.6)
    return np.where(np.nan_to_num(dbz, nan=-30.0) >= 10.0, r, 0.0)


def rain_to_dbz(r):
    with np.errstate(divide="ignore"):
        dbz = 10 * np.log10(200.0 * np.maximum(r, 1e-6) ** 1.6)
    return np.where(r > 0.05, dbz, np.nan)


# ---------------------------------------------------------------------------
# Radar data access (AWS) and reading
# ---------------------------------------------------------------------------
def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_scans(site, end_time, n_scans, lookback_min=90):
    """Return [(time, bucket, key)] of the n most recent volumes before end_time."""
    s3 = _s3()
    out = []
    days = {(end_time - dt.timedelta(minutes=lookback_min)).date(), end_time.date()}
    for bucket in BUCKETS:
        for day in sorted(days):
            prefix = f"{day:%Y/%m/%d}/{site}/"
            pager = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for page in pager:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    name = key.rsplit("/", 1)[-1]
                    if name.endswith("_MDM") or len(name) < 19:
                        continue
                    try:
                        t = dt.datetime.strptime(name[4:19], "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
                    except ValueError:
                        continue
                    if end_time - dt.timedelta(minutes=lookback_min) <= t <= end_time:
                        out.append((t, bucket, key))
        if out:
            break
    out.sort()
    if len(out) < 3:
        raise RuntimeError(f"Found only {len(out)} {site} scans before {end_time:%Y-%m-%d %H:%M} UTC")
    return out[-n_scans:]


def download(bucket, key, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        _s3().download_file(bucket, key, path)
    return path


def read_composite(path, grid, max_elev=2.5, min_dbz=10.0):
    """Max reflectivity of the lowest tilts, nearest-neighbour gridded. Returns (time, dBZ)."""
    import pyart
    radar = pyart.io.read_nexrad_archive(path)
    t0 = pyart.util.datetime_from_radar(radar)
    t0 = dt.datetime(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second, tzinfo=UTC)
    field = "reflectivity"
    xs, ys, vals = [], [], []
    rad_lat, rad_lon = radar.latitude["data"][0], radar.longitude["data"][0]
    off_x, off_y = grid.latlon_to_xy(rad_lat, rad_lon)
    for i in range(radar.nsweeps):
        if radar.fixed_angle["data"][i] > max_elev:
            continue
        sl = radar.get_slice(i)
        refl = radar.fields[field]["data"][sl]
        gx, gy, _ = radar.get_gate_x_y_z(i)
        good = ~np.ma.getmaskarray(refl)
        xs.append(gx[good] / 1000 + off_x)
        ys.append(gy[good] / 1000 + off_y)
        vals.append(np.asarray(refl)[good])
    xs, ys, vals = map(np.concatenate, (xs, ys, vals))
    comp = np.full(grid.shape, np.nan)
    tree = cKDTree(np.column_stack([grid.X.ravel(), grid.Y.ravel()]))
    dist, idx = tree.query(np.column_stack([xs, ys]), distance_upper_bound=grid.res)
    ok = np.isfinite(dist)
    flat = comp.ravel()
    order = np.argsort(vals[ok])  # later writes win -> keep max per cell
    flat[idx[ok][order]] = vals[ok][order]
    comp = flat.reshape(grid.shape)
    # fill small holes, remove speckle and very weak echoes
    comp = np.where(comp >= min_dbz, comp, np.nan)
    echo = ndimage.binary_opening(np.isfinite(comp), iterations=1)
    comp = np.where(echo, comp, np.nan)
    return t0, comp


# ---------------------------------------------------------------------------
# Synthetic storm for offline testing
# ---------------------------------------------------------------------------
def synthetic_frames(grid, n_past=6, n_future=24, step_min=5, seed=3):
    """Squall line + cells moving NE ~ 60 km/h, slowly veering and evolving."""
    rng = np.random.default_rng(seed)
    t_end = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    cells = []
    for k in range(18):  # line of cells
        s = -120 + 14 * k + rng.normal(0, 4)
        cells.append([s * 0.6 - 130, -s * 0.8 + 10, rng.uniform(8, 25), rng.uniform(5, 14), rng.uniform(3, 9)])
    for k in range(10):  # scattered stratiform/convective blobs
        cells.append([rng.uniform(-190, -40), rng.uniform(-150, 150), rng.uniform(2, 10),
                      rng.uniform(15, 35), rng.uniform(-0.3, 0.3)])
    # storm-relative small-scale texture (convective cores, gaps) that moves with the storm
    texture = ndimage.gaussian_filter(rng.standard_normal((400, 400)), 2.5)
    texture /= texture.std()
    frames, times = [], []
    for n in range(-n_past + 1, n_future + 1):
        t = n * step_min
        u = 0.75 + 0.002 * t                # km/min east (veering slightly)
        v = 0.55 - 0.002 * t                # km/min north
        R = np.zeros(grid.shape)
        for (x0, y0, amp, sig, growth) in cells:
            a = max(amp + growth * t / 60.0, 0)
            R += a * np.exp(-((grid.X - x0 - u * (t + 60)) ** 2 + (grid.Y - y0 - v * (t + 60)) ** 2) / (2 * sig**2))
        # sample the texture in storm-relative coordinates (grid cells)
        rr = (grid.Y - v * (t + 60) - grid.y[0]) / grid.res
        cc = (grid.X - u * (t + 60) - grid.x[0]) / grid.res
        tex = ndimage.map_coordinates(texture, [rr + 100, cc + 100], order=1, mode="wrap")
        R = R * np.exp(0.7 * tex - 0.25)
        frames.append(rain_to_dbz(R))
        times.append(t_end + dt.timedelta(minutes=t))
    return times[:n_past], frames[:n_past], times[n_past:], frames[n_past:]


# ---------------------------------------------------------------------------
# Motion estimation and extrapolation
# ---------------------------------------------------------------------------
def to_uint8(dbz):
    return (np.clip(np.nan_to_num(dbz, nan=0.0), 0, 70) / 70 * 255).astype(np.uint8)


def estimate_motion(times, frames, tile=64, step=32, smooth_px=35, max_shift_px=15):
    """Echo motion (grid cells per minute) by tile-wise phase correlation (TREC-style).

    Each tile with enough echo texture gives one displacement vector per scan pair;
    outliers are removed and vectors are spread to the whole grid with Gaussian weights,
    falling back to the domain-wide motion where there are no echoes."""
    import cv2
    ny, nx = frames[0].shape
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    win_all = cv2.createHanningWindow((nx, ny), cv2.CV_32F)
    vecs, glob = [], []  # (row, col, u, v, weight)
    for k in range(1, len(frames)):
        dt_min = (times[k] - times[k - 1]).total_seconds() / 60.0
        if dt_min <= 0.5:  # duplicate / out-of-order scan
            continue
        A = np.nan_to_num(frames[k - 1], nan=0.0).astype(np.float32)
        B = np.nan_to_num(frames[k], nan=0.0).astype(np.float32)
        # remove the mean first: otherwise the static window biases shifts toward zero
        (gx, gy), resp = cv2.phaseCorrelate(A - A.mean(), B - B.mean(), win_all)
        if resp > 0.05 and math.hypot(gx, gy) < max_shift_px:
            glob.append((gx / dt_min, gy / dt_min, resp))
        for r0 in range(0, ny - tile + 1, step):
            for c0 in range(0, nx - tile + 1, step):
                a, b = A[r0:r0 + tile, c0:c0 + tile], B[r0:r0 + tile, c0:c0 + tile]
                if (a > 0).mean() < 0.15 or a.std() < 3.0:
                    continue
                (dx, dy), resp = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), win)
                if resp > 0.15 and math.hypot(dx, dy) < max_shift_px:
                    vecs.append((r0 + tile / 2, c0 + tile / 2, dx / dt_min, dy / dt_min, resp))
    if glob:
        g = np.array(glob)
        mean_u, mean_v = np.average(g[:, 0], weights=g[:, 2]), np.average(g[:, 1], weights=g[:, 2])
    else:
        mean_u = mean_v = 0.0
    if len(vecs) < 3:
        return np.full((ny, nx), mean_u), np.full((ny, nx), mean_v)
    V = np.array(vecs)
    # reject outliers (> 3 robust std from the median vector)
    du, dv = V[:, 2] - np.median(V[:, 2]), V[:, 3] - np.median(V[:, 3])
    mad = 1.4826 * np.median(np.hypot(du, dv)) + 1e-6
    V = V[np.hypot(du, dv) < 3 * mad + 0.05]
    # Gaussian-weighted interpolation of the vectors onto the grid
    rr, cc = np.mgrid[0:ny:4, 0:nx:4]
    num_u = np.zeros(rr.shape); num_v = np.zeros(rr.shape); den = np.zeros(rr.shape)
    for r, c, uu, vv, w in V:
        k = w * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * smooth_px**2))
        num_u += k * uu; num_v += k * vv; den += k
    blend = np.clip(den / 0.5, 0, 1)
    u_c = blend * num_u / np.maximum(den, 1e-9) + (1 - blend) * mean_u
    v_c = blend * num_v / np.maximum(den, 1e-9) + (1 - blend) * mean_v
    zoom = (ny / u_c.shape[0], nx / u_c.shape[1])
    u = ndimage.zoom(u_c, zoom, order=1)[:ny, :nx]
    v = ndimage.zoom(v_c, zoom, order=1)[:ny, :nx]
    return u, v  # (d_col, d_row) per minute


def extrapolate(rain0, u, v, lead_steps, step_min, speed_fac=1.0, rot_deg=0.0):
    """Semi-Lagrangian backward advection of rain0 along (u=cols/min, v=rows/min)."""
    ny, nx = rain0.shape
    rows, cols = np.mgrid[0:ny, 0:nx].astype(float)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    uu = speed_fac * (c * u - s * v)
    vv = speed_fac * (s * u + c * v)
    pr, pc = rows.copy(), cols.copy()
    out = []
    sub = 4  # sub-steps per output step for curved trajectories
    for _ in range(lead_steps):
        for _ in range(sub):
            du = ndimage.map_coordinates(uu, [pr, pc], order=1, mode="nearest")
            dv = ndimage.map_coordinates(vv, [pr, pc], order=1, mode="nearest")
            pc -= du * step_min / sub
            pr -= dv * step_min / sub
        out.append(ndimage.map_coordinates(rain0, [pr, pc], order=1, mode="constant", cval=0.0))
    return out


def ensemble_nowcast(rain0, u, v, lead_steps, step_min, n_members=20, seed=0):
    rng = np.random.default_rng(seed)
    members = []
    for m in range(n_members):
        if m == 0:
            sf, rot = 1.0, 0.0  # control member
        else:
            sf, rot = rng.normal(1.0, 0.12), rng.normal(0.0, 8.0)
        fc = extrapolate(rain0, u, v, lead_steps, step_min, sf, rot)
        if m > 0:
            for k in range(lead_steps):
                lead = (k + 1) * step_min
                sigma = 0.25 * math.sqrt(lead / 30.0)          # log-space spread grows with lead
                noise = ndimage.gaussian_filter(rng.standard_normal(rain0.shape), 6)
                noise /= noise.std() + 1e-9
                fc[k] = fc[k] * np.exp(sigma * noise - 0.5 * sigma**2)
        members.append(fc)
    return np.array(members)  # (member, lead, ny, nx)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def local_str(t):
    try:
        from zoneinfo import ZoneInfo
        return t.astimezone(ZoneInfo("America/Chicago")).strftime("%I:%M %p %Z").lstrip("0")
    except Exception:
        return t.strftime("%H:%M UTC")


def draw_frame(ax, grid, dbz, title, prob=None):
    ext = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]
    ax.set_facecolor("#1b1f24")
    ax.imshow(np.ma.masked_invalid(dbz), origin="lower", extent=ext, cmap=DBZ_CMAP, norm=DBZ_NORM,
              interpolation="nearest")
    if prob is not None:
        cs = ax.contour(grid.X, grid.Y, prob, levels=[0.3, 0.7], colors=["w", "w"],
                        linestyles=["--", "-"], linewidths=0.8)
    sx, sy = zip(*[grid.latlon_to_xy(la, lo) for la, lo in SHORELINE])
    ax.plot(sx, sy, color="#9fb3c8", lw=1.0)
    for name, (la, lo) in POINTS.items():
        px, py = grid.latlon_to_xy(la, lo)
        ax.plot(px, py, "o", ms=3.5, color="white", mec="k")
    lx, ly = grid.latlon_to_xy(*POINTS["Downtown (Loop)"])
    ax.annotate("Chicago", (lx, ly), xytext=(6, 6), textcoords="offset points", color="w", fontsize=8)
    for r in (50, 100, 150, 200):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#555", lw=0.5))
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_xlabel("km east of radar"); ax.set_ylabel("km north of radar")
    ax.set_title(title, fontsize=10)


def make_animation(grid, obs_times, obs_frames, fc_times, fc_dbz, fc_prob, path, site):
    import imageio.v2 as imageio
    images = []
    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=90)
    sm = plt.cm.ScalarMappable(cmap=DBZ_CMAP, norm=DBZ_NORM)
    fig.colorbar(sm, ax=ax, label="reflectivity (dBZ)", shrink=0.8)
    seq = [(t, f, None, f"{site} observed  {local_str(t)}") for t, f in zip(obs_times, obs_frames)]
    seq += [(t, f, p, f"FORECAST +{int((t - obs_times[-1]).total_seconds() // 60)} min  "
             f"valid {local_str(t)}\n(white lines: 30% / 70% chance of rain)")
            for t, f, p in zip(fc_times, fc_dbz, fc_prob)]
    for t, f, p, title in seq:
        ax.clear()
        draw_frame(ax, grid, f, title, p)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    durations = [0.5] * len(obs_frames) + [0.35] * (len(fc_dbz) - 1) + [2.0]
    imageio.mimsave(path, images, duration=durations, loop=0)
    return path


def point_forecast(grid, members, fc_times, obs_rain, heavy=5.0):
    rows = []
    for name, (la, lo) in POINTS.items():
        x, y = grid.latlon_to_xy(la, lo)
        r, c = grid.xy_to_rc(x, y)
        r, c = int(round(r)), int(round(c))
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            continue
        # 3x3 neighbourhood to reduce pixel-level noise
        sl = (slice(max(r - 1, 0), r + 2), slice(max(c - 1, 0), c + 2))
        now = float(obs_rain[sl].mean())
        series = members[:, :, sl[0], sl[1]].mean(axis=(2, 3))  # (member, lead)
        rows.append((name, now, series))
    return rows


def plot_points(rows, fc_times, path, heavy=5.0):
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 2.3 * len(rows)), sharex=True)
    axs = np.atleast_1d(axs)
    lead = [(t - fc_times[0]).total_seconds() / 60 + (fc_times[1] - fc_times[0]).total_seconds() / 60
            for t in fc_times]
    for ax, (name, now, s) in zip(axs, rows):
        p10, p50, p90 = np.percentile(s, [10, 50, 90], axis=0)
        ax.fill_between(lead, p10, p90, color="#5aa0e0", alpha=0.35, label="10-90% range")
        ax.plot(lead, p50, color="#1f5fa0", lw=2, label="median")
        ax.set_ylabel("mm/h")
        ax2 = ax.twinx()
        ax2.plot(lead, 100 * (s > 0.5).mean(0), color="#e07b39", lw=1.5, label="chance of rain")
        ax2.plot(lead, 100 * (s > heavy).mean(0), color="#c0392b", lw=1.2, ls="--", label="chance of heavy")
        ax2.set_ylim(0, 105); ax2.set_ylabel("%")
        ax.set_title(f"{name}  (now: {now:.1f} mm/h)", fontsize=10, loc="left")
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[0].get_figure().axes[len(rows)].get_legend_handles_labels()
    axs[0].legend(h1 + h2, l1 + l2, fontsize=7, ncol=4, loc="upper right")
    axs[-1].set_xlabel("minutes from latest radar scan")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def verify(fc_rain, truth_dbz, rain0, step_min, thr=1.0):
    def csi(f, o):
        hits, miss, fa = (f & o).sum(), (~f & o).sum(), (f & ~o).sum()
        return hits / max(hits + miss + fa, 1)
    print(f"\n  Demo check vs synthetic truth (CSI, rain > {thr:g} mm/h; 1 = perfect):")
    print("    lead     nowcast   persistence (radar frozen)")
    for k in range(3, len(fc_rain) + 1, 3):
        o = dbz_to_rain(truth_dbz[k - 1]) > thr
        print(f"    +{k * step_min:3d} min   {csi(fc_rain[k - 1] > thr, o):.2f}      {csi(rain0 > thr, o):.2f}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="KLOT", help="NEXRAD site ID (default KLOT, Chicago)")
    ap.add_argument("--time", help="end time in UTC, e.g. 2025-07-15T20:30 (default: now)")
    ap.add_argument("--n-scans", type=int, default=6, help="past scans used for motion (default 6)")
    ap.add_argument("--lead", type=int, default=120, help="forecast length, minutes (default 120)")
    ap.add_argument("--step", type=int, default=5, help="forecast output step, minutes (default 5)")
    ap.add_argument("--members", type=int, default=20, help="ensemble size (default 20)")
    ap.add_argument("--files", nargs="+", help="local Level II files instead of downloading")
    ap.add_argument("--cache", default="radar_cache", help="download folder")
    ap.add_argument("--demo", action="store_true", help="synthetic storm, no download needed")
    ap.add_argument("--out", default="chicago_nowcast", help="output file prefix")
    args = ap.parse_args()

    lat0, lon0 = SITES.get(args.site.upper(), SITES["KLOT"])
    grid = Grid(lat0, lon0)
    n_lead = args.lead // args.step
    truth = None

    if args.demo:
        obs_times, obs_frames, _, truth = synthetic_frames(grid, args.n_scans, n_lead, args.step)
        source = "synthetic demo storm"
    else:
        if args.files:
            paths = args.files
        else:
            end = (dt.datetime.fromisoformat(args.time).replace(tzinfo=UTC) if args.time
                   else dt.datetime.now(UTC))
            scans = list_scans(args.site.upper(), end, args.n_scans)
            print(f"Downloading {len(scans)} {args.site} scans "
                  f"({scans[0][0]:%H:%M}-{scans[-1][0]:%H:%M} UTC)...")
            paths = [download(b, k, args.cache) for _, b, k in scans]
        obs = sorted(read_composite(p, grid) for p in paths)
        obs_times, obs_frames = [o[0] for o in obs], [o[1] for o in obs]
        source = f"NEXRAD {args.site.upper()} Level II"

    rain0 = dbz_to_rain(obs_frames[-1])
    u, v = estimate_motion(obs_times, obs_frames)
    wet = rain0 > 0.5
    if wet.any():
        spd = np.hypot(u[wet], v[wet]) * grid.res * 60  # km/h
        ang = (math.degrees(math.atan2(u[wet].mean(), v[wet].mean())) + 360) % 360
        print(f"\nSource: {source}; latest scan {local_str(obs_times[-1])}")
        print(f"Echo motion: {np.median(spd):.0f} km/h toward {ang:.0f}° "
              f"(rain covers {100 * wet.mean():.0f}% of the 400 x 400 km domain)")
    else:
        print("No significant precipitation on radar - forecast will be dry.")

    members = ensemble_nowcast(rain0, u, v, n_lead, args.step, args.members)
    fc_times = [obs_times[-1] + dt.timedelta(minutes=args.step * (k + 1)) for k in range(n_lead)]
    control = members[0]
    prob = (members > 0.5).mean(axis=0)
    # display field: control member, smoothed more at longer leads (lower predictability)
    fc_dbz = [rain_to_dbz(ndimage.gaussian_filter(control[k], 0.15 * k)) for k in range(n_lead)]

    gif = make_animation(grid, obs_times, obs_frames, fc_times, fc_dbz, prob, f"{args.out}_loop.gif",
                         args.site.upper())
    rows = point_forecast(grid, members, fc_times, rain0)
    plot_points(rows, fc_times, f"{args.out}_points.png")

    print("\n  Point forecasts (median mm/h | chance of rain):")
    hdr = "".join(f"  +{m:>3d}m" for m in (15, 30, 60, 90, 120) if m <= args.lead)
    print(f"  {'location':<32}{hdr}")
    with open(f"{args.out}_points.csv", "w") as fh:
        fh.write("location,valid_time_utc,lead_min,median_mm_h,p10,p90,prob_rain,prob_heavy\n")
        for name, now, s in rows:
            cells = []
            for m in (15, 30, 60, 90, 120):
                k = m // args.step - 1
                if 0 <= k < n_lead:
                    cells.append(f"  {np.median(s[:, k]):4.1f}|{100 * (s[:, k] > 0.5).mean():3.0f}%")
            print(f"  {name:<32}" + "".join(cells))
            for k, t in enumerate(fc_times):
                p10, p50, p90 = np.percentile(s[:, k], [10, 50, 90])
                fh.write(f"{name},{t:%Y-%m-%dT%H:%M},{(k + 1) * args.step},{p50:.2f},{p10:.2f},{p90:.2f},"
                         f"{(s[:, k] > 0.5).mean():.2f},{(s[:, k] > 5).mean():.2f}\n")
    if truth is not None:
        verify(control, truth, rain0, args.step)
    print(f"\n  Saved: {gif}, {args.out}_points.png, {args.out}_points.csv")


if __name__ == "__main__":
    main()
