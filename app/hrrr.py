"""
NOAA HRRR (3 km, hourly) 10 m wind and surface gust around Chicago, for the page's wind layer.

Downloads only the three records over the radar map's region through NOMADS' grib filter (no key):
the newest complete cycle (f00-f18, f00-f48 on the 00/06/12/18 runs), hours past f18 from the newest
00/06/12/18 run, and the 12 hours before the newest cycle from each of those hours' own analysis (f00).

HRRR's winds are relative to its Lambert grid: they are turned to true north here, then resampled onto a 3 km
grid in the page's frame (km east/north of State & Madison).

    python -m app.hrrr --out wind.json        (needs: pip install eccodes)
"""

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import json
import logging
import math
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from . import lakefront as lf
from . import sup_waves as sw

log = logging.getLogger("hrrr")
UTC = dt.timezone.utc
FILTER = os.environ.get("HRRR_FILTER", "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl")
BOX = (41.10, -88.50, 42.80, -86.70)      # south, west, north, east: the radar map's region and a margin
DX_KM = 3.0
SCALE = 0.25                              # m/s per step: u, v as int8 (+-31.75 m/s), gust as uint8 (to 63.75 m/s)
PAST_H = 12
LOV, LATIN = -97.5, 38.5                  # HRRR's Lambert conformal grid
MIN_INTERVAL = 1.25                       # s between requests (NOMADS blocks addresses that hit it > ~120 times a minute)


def max_fh(cyc_hour):
    return 48 if cyc_hour % 6 == 0 else 18


def url(cyc, fh, box=BOX):
    s, w, n, e = box
    q = {"dir": f"/hrrr.{cyc:%Y%m%d}/conus", "file": f"hrrr.t{cyc:%H}z.wrfsfcf{fh:02d}.grib2",
         "var_UGRD": "on", "var_VGRD": "on", "var_GUST": "on", "lev_10_m_above_ground": "on", "lev_surface": "on",
         "subregion": "", "toplat": n, "leftlon": w, "rightlon": e, "bottomlat": s}
    return FILTER + "?" + urllib.parse.urlencode(q)


class Throttle:
    def __init__(self, gap=MIN_INTERVAL):
        self.gap, self.last, self.lock = gap, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            d = self.last + self.gap - time.monotonic()
            if d > 0:
                time.sleep(d)
            self.last = time.monotonic()


def http(u, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": sw.UA}), timeout=timeout) as r:
        return r.read()


def is_grib(b):
    return bool(b) and b[:4] == b"GRIB"


def decode(blob):
    """grib2 bytes -> {'u','v','gust','lat','lon','rel'} (flat arrays; lon in -180..180)"""
    import eccodes
    out = {}
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(blob); tmp.flush()
        with open(tmp.name, "rb") as fh:
            while True:
                g = eccodes.codes_grib_new_from_file(fh)
                if g is None:
                    break
                try:
                    name = {"10u": "u", "10v": "v", "gust": "gust"}.get(eccodes.codes_get(g, "shortName"))
                    if not name:
                        continue
                    out[name] = np.asarray(eccodes.codes_get_values(g), float)
                    if name in ("u", "v"):
                        out["rel"] = bool(eccodes.codes_get(g, "uvRelativeToGrid"))
                    if "lat" not in out:
                        out["lat"] = np.asarray(eccodes.codes_get_array(g, "latitudes"), float)
                        lo = np.asarray(eccodes.codes_get_array(g, "longitudes"), float)
                        out["lon"] = np.where(lo > 180, lo - 360, lo)
                finally:
                    eccodes.codes_release(g)
    missing = {"u", "v", "gust", "lat"} - set(out)
    if missing:
        raise ValueError(f"grib is missing {sorted(missing)}")
    return out


def earth_relative(u, v, lon, lov=LOV, latin=LATIN):
    """Grid-relative Lambert winds -> true east/north components (the standard rotation by n·(lon − LoV))."""
    a = math.sin(math.radians(latin)) * np.radians(np.asarray(lon) - lov)
    c, s = np.cos(a), np.sin(a)
    return c * u + s * v, -s * u + c * v


def grid_spec(box=BOX, dx=DX_KM):
    s, w, n, e = box
    x0, y0 = (w - lf.REF[1]) * lf.KX / 1000, (s - lf.REF[0]) * lf.KY / 1000
    x1, y1 = (e - lf.REF[1]) * lf.KX / 1000, (n - lf.REF[0]) * lf.KY / 1000
    # a lon/lat box is wider at its south edge in km than the local frame's flat x: keep only what the data covers
    return {"x0": round(x0 + dx, 2), "y0": round(y0 + dx, 2), "dx": dx,
            "nx": int((x1 - x0) / dx) - 1, "ny": int((y1 - y0) / dx) - 1}


def resample(lat, lon, fields, G):
    """Scattered HRRR points -> the local grid (linear on the triangulation; nearest where it falls outside)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    px = (lon - lf.REF[1]) * lf.KX / 1000; py = (lat - lf.REF[0]) * lf.KY / 1000
    pts = np.column_stack([px, py])
    gx = G["x0"] + np.arange(G["nx"]) * G["dx"]; gy = G["y0"] + np.arange(G["ny"]) * G["dx"]
    X, Y = np.meshgrid(gx, gy)
    vals = np.column_stack(fields)
    lin = LinearNDInterpolator(pts, vals)(X, Y)
    bad = np.isnan(lin[..., 0])
    if bad.any():
        lin[bad] = NearestNDInterpolator(pts, vals)(X[bad], Y[bad])
    return [lin[..., k] for k in range(vals.shape[1])]   # rows south -> north


def pack(a, signed):
    q = np.round(np.asarray(a) / SCALE)
    q = np.clip(q, -127, 127).astype(np.int8) if signed else np.clip(q, 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


def unpack(s, signed, shape):
    return np.frombuffer(base64.b64decode(s), np.int8 if signed else np.uint8).astype(float).reshape(shape) * SCALE


def schedule(newest, synoptic, past=PAST_H):
    """[(valid time, cycle, forecast hour)] from `past` hours before the newest cycle to its last hour (or the synoptic run's)."""
    out = [(newest - dt.timedelta(hours=k), newest - dt.timedelta(hours=k), 0) for k in range(past, 0, -1)]
    last_new = newest + dt.timedelta(hours=max_fh(newest.hour))
    end = max(last_new, synoptic + dt.timedelta(hours=48)) if synoptic else last_new
    t = newest
    while t <= end:
        if t <= last_new:
            out.append((t, newest, int((t - newest).total_seconds() // 3600)))
        else:
            out.append((t, synoptic, int((t - synoptic).total_seconds() // 3600)))
        t += dt.timedelta(hours=1)
    return out


def find_cycles(now, get, throttle):
    """Newest cycle whose last hour is posted, and the newest 00/06/12/18 run whose f48 is posted (None if it's the same)."""
    newest = synoptic = None
    base = now.replace(minute=0, second=0, microsecond=0)
    for k in range(1, 9):
        c = base - dt.timedelta(hours=k)
        throttle.wait()
        try:
            ok = is_grib(get(url(c, max_fh(c.hour))))
        except Exception:
            ok = False
        if ok and newest is None:
            newest = c
            if c.hour % 6 == 0:
                return newest, None
        if ok and c.hour % 6 == 0:
            return newest, c
    if newest is None:
        raise RuntimeError("no complete HRRR cycle in the last 8 hours")
    return newest, synoptic


def fetch(out_path, now=None, get=http, gap=MIN_INTERVAL, box=BOX):
    now = now or dt.datetime.now(UTC)
    th = Throttle(gap)
    newest, syn = find_cycles(now, get, th)
    plan = schedule(newest, syn)
    G = grid_spec(box)
    log.info("hrrr: newest %s, synoptic %s, %d hours", newest, syn, len(plan))

    def one(item):
        t, c, f = item
        th.wait()
        try:
            b = get(url(c, f, box))
            if not is_grib(b):
                return t, c, f, None
            d = decode(b)
        except Exception as e:  # one missing hour shouldn't sink the layer
            log.info("hrrr %s f%02d: %s", c, f, e)
            return t, c, f, None
        u, v = (earth_relative(d["u"], d["v"], d["lon"]) if d.get("rel", True) else (d["u"], d["v"]))
        return t, c, f, resample(d["lat"], d["lon"], [u, v, d["gust"]], G)

    with cf.ThreadPoolExecutor(3) as ex:
        res = sorted((r for r in ex.map(one, plan) if r[3] is not None), key=lambda r: r[0])
    if not res:
        raise RuntimeError("no HRRR hours decoded")
    out = {"version": 1, "model": "NOAA HRRR 3 km", "source": FILTER, "cycle": newest.isoformat(),
           "synoptic": syn.isoformat() if syn else None, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
           "grid": G, "scale": SCALE, "t": [], "run": [], "u": [], "v": [], "g": []}
    for t, c, f, (u, v, g) in res:
        out["t"].append(t.isoformat()); out["run"].append(f"{c:%d%H}z f{f:02d}")
        out["u"].append(pack(u, True)); out["v"].append(pack(v, True)); out["g"].append(pack(np.maximum(g, np.hypot(u, v)), False))
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


def at(w, ti, x_km, y_km):
    """(u, v, gust) m/s at a grid time index and a local point (bilinear)."""
    G = w["grid"]; shape = (G["ny"], G["nx"])
    U, V, Gs = unpack(w["u"][ti], True, shape), unpack(w["v"][ti], True, shape), unpack(w["g"][ti], False, shape)
    fx = min(max((x_km - G["x0"]) / G["dx"], 0), G["nx"] - 1.001); fy = min(max((y_km - G["y0"]) / G["dx"], 0), G["ny"] - 1.001)
    i, j = int(fx), int(fy); a, b = fx - i, fy - j
    f = lambda A: (A[j, i] * (1 - a) + A[j, i + 1] * a) * (1 - b) + (A[j + 1, i] * (1 - a) + A[j + 1, i + 1] * a) * b
    return f(U), f(V), f(Gs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="wind.json")
    a = ap.parse_args()
    o = fetch(a.out)
    i = min(range(len(o["t"])), key=lambda k: abs(dt.datetime.fromisoformat(o["t"][k]) - dt.datetime.now(UTC)))
    u, v, g = at(o, i, 0.0, 0.0)
    print(f"{len(o['t'])} hours {o['t'][0]} .. {o['t'][-1]}; now at State & Madison {math.hypot(u, v) * 2.237:.0f} mph, gust {g * 2.237:.0f} mph",
          f"grid {o['grid']['nx']}x{o['grid']['ny']}, {os.path.getsize(a.out) // 1024} KB")
