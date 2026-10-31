"""
The jet stream: 250 hPa winds over North America for the page's Radar map card, where Chicago sits under it,
and what that has meant for gusts on the lakefront.

    python -m app.jet now --out jet.json [--climate jet_climate.json]     (GFS 1°, needs: pip install eccodes)
    python -m app.jet climate --out jet_climate.json [--first 2016]        (reanalysis + buoys; no key)

now:     the newest GFS run's 250 hPa u/v every 3 h to +48 h and the run 12 h before it to fill the past, over
         18-62 N, 135-55 W; per time: the speed over Chicago, the polar jet's axis on Chicago's meridian, the nearest
         jet streak (a local speed maximum ≥ 45 m/s within 1500 km) and Chicago's quadrant relative to it, and the
         subtropical jet's latitude (zonal-wind maximum 15-38 N, averaged 120-80 W).
climate: (1) NCEP/NCAR reanalysis 6-hourly 250 hPa winds (2.5°, PSL OPeNDAP) through the same metrics, matched with
         hourly gusts at the Chicago crib (CHII2) and the Wilmette and Chicago buoys (45174, 45198): the gust factor
         (gust / sustained) and the chance of a 30 mph gust in each quadrant against hours with no streak near,
         stratified by the sustained wind, with day-block bootstrap 90% intervals. The page changes a number only
         where the interval excludes "no change".
         (2) the subtropical jet's latitude by month from the monthly reanalysis, 1948+, and its winter (DJF)
         position by El Niño phase (ONI), for the El Niño tracker.
"""

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import json
import logging
import math
import os
import random
import re
import tempfile
import urllib.parse
import urllib.request

import numpy as np

from . import enso
from . import hrrr

log = logging.getLogger("jet")
UTC = dt.timezone.utc
UA = enso.UA
CHI = (41.88, -87.63)
GFS_FILTER = os.environ.get("GFS_FILTER", "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl")
BOX = (18.0, -135.0, 62.0, -55.0)            # south, west, north, east
STREAK_MIN = 45.0                            # m/s (~100 mph): a jet streak's core
RANGE_KM = 1500.0
CORE_KM = 300.0
STJ_LON = (-120.0, -80.0)
STJ_LAT = (15.0, 38.0)
PSL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
R1_BOX = (20.0, -130.0, 60.0, -60.0)         # the reanalysis box for the gust study (2.5°)
STATIONS = ["CHII2", "45174", "45198"]
GUST30 = 13.4                                # m/s
WIND_BINS = [(4.0, 7.0), (7.0, 10.0), (10.0, 40.0)]
QUADS = ["core", "left-exit", "right-entrance", "right-exit", "left-entrance"]
BOOT = 1000
MIN_EFFECT = 0.03                            # a change smaller than 3% is not worth showing, even when it is "significant"


def http(u, timeout=120, tries=5):
    """GET with a back-off on 429/5xx (PSL's THREDDS limits how fast one address may ask)."""
    import time
    import urllib.error
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 20 * 2 ** k
            log.info("%s: HTTP %s, waiting %d s", u[:80], e.code, wait)
            time.sleep(min(wait, 300))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:   # a dropped connection: try again
            if k == tries - 1:
                raise
            log.info("%s: %s, retrying", u[:80], e)
            time.sleep(10 * 2 ** k)


# ---------------------------------------------------------------------------- metrics (pure)
def local_km(lat, lon, ref=CHI):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lon - ref[1]) * 111.32 * np.cos(np.radians((lat + ref[0]) / 2)), (lat - ref[0]) * 110.57


def bilinear(lat, lon, A, la, lo):
    """A[nlat, nlon] on ascending lat/lon axes at one point."""
    fy = np.interp(la, lat, np.arange(len(lat))); fx = np.interp(lo, lon, np.arange(len(lon)))
    j, i = min(int(fy), len(lat) - 2), min(int(fx), len(lon) - 2); b, a = fy - j, fx - i
    return float((A[j, i] * (1 - a) + A[j, i + 1] * a) * (1 - b) + (A[j + 1, i] * (1 - a) + A[j + 1, i + 1] * a) * b)


def ascending(lat, lon, *arrs):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    lon = np.where(lon > 180, lon - 360, lon)
    arrs = [np.asarray(a, float) for a in arrs]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; arrs = [a[::-1] for a in arrs]
    o = np.argsort(lon); lon = lon[o]; arrs = [a[:, o] for a in arrs]
    return lat, lon, arrs


def stj(lat, lon, U):
    """Subtropical jet: latitude and speed of the 120-80 W mean zonal wind maximum between 15 and 38 N."""
    mlo = (lon >= STJ_LON[0]) & (lon <= STJ_LON[1]); mla = (lat >= STJ_LAT[0]) & (lat <= STJ_LAT[1])
    prof = U[np.ix_(mla, mlo)].mean(axis=1); la = lat[mla]
    k = int(np.argmax(prof))
    if 0 < k < len(prof) - 1:                     # parabolic peak between grid rows
        d = prof[k - 1] - 2 * prof[k] + prof[k + 1]
        off = 0.5 * (prof[k - 1] - prof[k + 1]) / d if d < 0 else 0.0
        return round(float(la[k] + off * (la[1] - la[0])), 2), round(float(prof[k] - 0.25 * (prof[k - 1] - prof[k + 1]) * off), 1)
    return round(float(la[k]), 2), round(float(prof[k]), 1)


def quadrant(cx, cy, sx, sy, su, sv):
    """Chicago (cx, cy km) relative to a streak centred at (sx, sy) blowing toward (su, sv): along > 0 is downstream (exit),
    cross > 0 is to the left looking downstream."""
    n = math.hypot(su, sv) or 1.0; ex, ey = su / n, sv / n
    dx, dy = cx - sx, cy - sy
    along, cross = dx * ex + dy * ey, -dx * ey + dy * ex
    return along, cross


def metrics(lat, lon, U, V):
    lat, lon, (U, V) = ascending(lat, lon, U, V)
    S = np.hypot(U, V)
    u0, v0 = bilinear(lat, lon, U, *CHI), bilinear(lat, lon, V, *CHI)
    spd = math.hypot(u0, v0)
    # the polar jet's axis on Chicago's meridian (the mean of the columns within 5° of it), 30-65 N
    cols = np.abs(lon - CHI[1]) <= 5.0; rows = (lat >= 30) & (lat <= 65)
    prof = S[np.ix_(rows, cols)].mean(axis=1); k = int(np.argmax(prof)); axis_lat = float(lat[rows][k])
    # the nearest jet streak: grid local maxima (3x3) of at least STREAK_MIN within RANGE_KM; the strongest wins
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); X, Y = local_km(LA, LO); D = np.hypot(X, Y)
    pad = np.pad(S, 1, mode="edge")
    nb = np.max(np.stack([pad[1 + dj:1 + dj + S.shape[0], 1 + di:1 + di + S.shape[1]] for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di]), axis=0)
    cand = (S >= STREAK_MIN) & (S >= nb) & (D <= RANGE_KM)
    out = {"spd": round(spd, 1), "dir": round((math.degrees(math.atan2(-u0, -v0)) + 360) % 360), "axis_lat": round(axis_lat, 1),
           "axis_spd": round(float(prof[k]), 1), "axis_km": round((axis_lat - CHI[0]) * 110.57), "streak": None, "quad": "none"}
    out["stj_lat"], out["stj_spd"] = stj(lat, lon, U)
    if cand.any():
        j, i = np.unravel_index(np.argmax(np.where(cand, S, -1)), S.shape)
        sx, sy, su, sv = float(X[j, i]), float(Y[j, i]), float(U[j, i]), float(V[j, i])
        along, cross = quadrant(0.0, 0.0, sx, sy, su, sv)
        out["streak"] = {"lat": float(lat[j]), "lon": float(lon[i]), "spd": round(float(S[j, i]), 1), "km": round(float(D[j, i])),
                         "along_km": round(along), "cross_km": round(cross)}
        if D[j, i] <= CORE_KM or spd >= 0.85 * S[j, i]:
            out["quad"] = "core"
        elif abs(cross) <= 900 and abs(along) <= RANGE_KM:
            out["quad"] = ("left" if cross > 0 else "right") + "-" + ("exit" if along > 0 else "entrance")
        else:
            out["quad"] = "near"
    return out


# ---------------------------------------------------------------------------- GFS now
def gfs_url(cyc, fh, box=BOX):
    s, w, n, e = box
    q = {"dir": f"/gfs.{cyc:%Y%m%d}/{cyc:%H}/atmos", "file": f"gfs.t{cyc:%H}z.pgrb2.1p00.f{fh:03d}",
         "var_UGRD": "on", "var_VGRD": "on", "lev_250_mb": "on", "subregion": "", "toplat": n, "leftlon": w, "rightlon": e, "bottomlat": s}
    return GFS_FILTER + "?" + urllib.parse.urlencode(q)


def decode_gfs(blob):
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
                    nm = eccodes.codes_get(g, "shortName")
                    if nm in ("u", "v"):
                        nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
                        out[nm] = np.asarray(eccodes.codes_get_values(g), float).reshape(nj, ni)
                        if "lat" not in out:
                            # the rows' own latitudes (the values' scan order), not the sorted distinct list
                            out["lat"] = np.asarray(eccodes.codes_get_array(g, "latitudes"), float).reshape(nj, ni)[:, 0]
                            out["lon"] = np.asarray(eccodes.codes_get_array(g, "longitudes"), float).reshape(nj, ni)[0]
                finally:
                    eccodes.codes_release(g)
    if not {"u", "v", "lat"} <= set(out):
        raise ValueError("grib is missing 250 hPa u/v")
    return out


def gfs_cycles(now, get, th):
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    for k in range(0, 5):
        c = base - dt.timedelta(hours=6 * k)
        th.wait()
        try:
            if hrrr.is_grib(get(gfs_url(c, 48))):
                return c
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError("no complete GFS cycle in the last day")


def gfs_schedule(cyc):
    prev = cyc - dt.timedelta(hours=12)
    return [(prev + dt.timedelta(hours=f), prev, f) for f in (0, 3, 6, 9)] + [(cyc + dt.timedelta(hours=f), cyc, f) for f in range(0, 49, 3)]


def pack_i8(a, scale=1.0):
    return base64.b64encode(np.clip(np.round(np.asarray(a) / scale), -127, 127).astype(np.int8).tobytes()).decode()


def build_now(climate=None, now=None, get=http, gap=hrrr.MIN_INTERVAL):
    now = now or dt.datetime.now(UTC)
    th = hrrr.Throttle(gap)
    cyc = gfs_cycles(now, get, th)
    plan = gfs_schedule(cyc)

    def one(item):
        t, c, f = item
        th.wait()
        try:
            b = get(gfs_url(c, f))
            return (t, c, f, decode_gfs(b)) if hrrr.is_grib(b) else (t, c, f, None)
        except Exception as e:  # noqa: BLE001
            log.info("gfs %s f%03d: %s", c, f, e)
            return t, c, f, None
    with cf.ThreadPoolExecutor(3) as ex:
        res = sorted((r for r in ex.map(one, plan) if r[3] is not None), key=lambda r: r[0])
    if not res:
        raise RuntimeError("no GFS hours decoded")
    lat, lon, _ = ascending(res[0][3]["lat"], res[0][3]["lon"], res[0][3]["u"])
    out = {"version": 1, "model": "NOAA GFS 1°, 250 hPa", "cycle": cyc.isoformat(), "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
           "grid": {"lat0": float(lat[0]), "lon0": float(lon[0]), "d": float(lat[1] - lat[0]), "nlat": len(lat), "nlon": len(lon)},
           "scale": 1.0, "t": [], "run": [], "u": [], "v": [], "chicago": []}
    for t, c, f, d in res:
        la, lo, (U, V) = ascending(d["lat"], d["lon"], d["u"], d["v"])
        out["t"].append(t.isoformat()); out["run"].append(f"{c:%d%H}z f{f:03d}")
        out["u"].append(pack_i8(U)); out["v"].append(pack_i8(V))
        out["chicago"].append(metrics(la, lo, U, V))
    ol = os.path.join(os.path.dirname(__file__), "static", "na_outline.json")   # tools/na_outline.py
    if os.path.exists(ol):
        out["outline"] = json.load(open(ol))
    if climate:
        out["climate"] = climate
        out["risk"] = risk_table(climate)
    return out


def risk_table(climate):
    """quadrant -> {gust: factor or None, p30: ratio or None} where the study found an effect: the 90% interval excludes 1
    and the change is at least MIN_EFFECT."""
    g = (climate or {}).get("gust") or {}
    out = {}
    for q, r in (g.get("quads") or {}).items():
        gf = r.get("gf_ratio"); p = r.get("p30_ratio")
        use = lambda x: x["value"] if x and x.get("sig") and abs(x["value"] - 1.0) >= MIN_EFFECT else None
        out[q] = {"gust": use(gf), "p30": use(p), "n": r.get("n")}
    return out


# ---------------------------------------------------------------------------- reanalysis (OPeNDAP ascii)
def parse_dods_ascii(txt, var):
    """The first `var.var[...]` grid block of an OPeNDAP .ascii response -> ndarray, plus the map vectors by name."""
    blocks = re.split(r"\n\s*\n", txt.split("-" * 20, 1)[-1].strip("-\n "))
    arr = None; maps = {}
    for b in blocks:
        lines = [l for l in b.strip().splitlines() if l.strip()]
        if not lines:
            continue
        m = re.match(r"^([\w.]+)\[(.+)\]$", lines[0].strip().replace("][", ","))
        if not m:
            continue
        name, dims = m.group(1), [int(x) for x in m.group(2).split(",")]
        if name == f"{var}.{var}" and arr is None:
            vals = []
            for l in lines[1:]:
                vals.extend(float(x) for x in l.split(",")[1:])
            arr = np.array(vals, float).reshape(dims)
        elif "." not in name and len(dims) == 1:
            maps[name] = np.array([float(x) for x in ",".join(lines[1:]).split(",") if x.strip()], float)
    if arr is None:
        raise ValueError(f"no {var} block")
    return arr, maps


def r1_index(lat0, lat1, lon0, lon1, d=2.5):
    """NCEP R1 grid indices (lat 90 -> -90, lon 0 -> 357.5) for a box (lon in -180..180)."""
    j0, j1 = int(round((90 - lat1) / d)), int(round((90 - lat0) / d))
    i0, i1 = int(round((lon0 % 360) / d)), int(round((lon1 % 360) / d))
    return j0, j1, i0, i1


def n_times(path, fetch=http):
    """Length of a dataset's time axis (OPeNDAP refuses a slice past the end)."""
    txt = fetch(f"{PSL}{path}.dds", timeout=120).decode("utf-8", "replace")
    return int(re.search(r"time\s*\[\s*time\s*=\s*(\d+)\s*\]", txt).group(1))


def r1_6h(var, year, box=R1_BOX, fetch=http):
    j0, j1, i0, i1 = r1_index(box[0], box[2], box[1], box[3])
    nt = n_times(f"pressure/{var}.{year}.nc", fetch) - 1
    u = f"{PSL}pressure/{var}.{year}.nc.ascii?{var}[0:1:{nt}][8][{j0}:1:{j1}][{i0}:1:{i1}],time[0:1:{nt}],lat[{j0}:1:{j1}],lon[{i0}:1:{i1}]"
    txt = fetch(u, timeout=600).decode("utf-8", "replace")
    a, m = parse_dods_ascii(txt, var)
    return m["time"][: a.shape[0]], m["lat"], m["lon"], a[:, 0]


def hours_to_dt(h):
    return dt.datetime(1800, 1, 1, tzinfo=UTC) + dt.timedelta(hours=float(h))


def r1_metrics_year(year, fetch=http):
    t, lat, lon, U = r1_6h("uwnd", year, fetch=fetch)
    t2, _, _, V = r1_6h("vwnd", year, fetch=fetch)
    n = min(len(t), len(t2))
    return [dict(metrics(lat, lon, U[k], V[k]), t=hours_to_dt(t[k]).isoformat()) for k in range(n)]


# ---------------------------------------------------------------------------- gusts at the lakefront by quadrant
def windows(jet_rows, obs_by_station, half=3):
    """One sample per reanalysis time: each station's median gust factor over ±half h (sustained ≥ 4 m/s, gust reported),
    pooled across stations; the max gust; the median sustained wind."""
    idx = {}
    for st, rows in obs_by_station.items():
        for r in rows:
            if r.get("wspd") is None or r.get("gst") is None or r["wspd"] < 4.0:
                continue
            h = r["t"].replace(minute=0, second=0, microsecond=0)
            idx.setdefault(h, []).append((st, r["wspd"], r["gst"]))
    out = []
    for j in jet_rows:
        t0 = dt.datetime.fromisoformat(j["t"])
        got = [x for k in range(-half, half + 1) for x in idx.get(t0 + dt.timedelta(hours=k), [])]
        if len(got) < 3:
            continue
        gfs, us, gs = {}, [], []
        for st, u, g in got:
            gfs.setdefault(st, []).append(max(g, u) / u); us.append(u); gs.append(g)
        out.append({"t": t0, "day": t0.date().isoformat(), "quad": j["quad"], "gf": float(np.mean([np.median(v) for v in gfs.values()])),
                    "u": float(np.median(us)), "gmax": float(max(gs)), "warm": 4 <= t0.month <= 10})
    return out


def _bin(u):
    for k, (a, b) in enumerate(WIND_BINS):
        if a <= u < b:
            return k
    return None


def strat_ratio(cat, base, key):
    """Wind-stratified ratio of means: sum_b w_b·mean_cat_b / sum_b w_b·mean_base_b with w_b = the category's count in bin b."""
    num = den = 0.0
    for b in range(len(WIND_BINS)):
        c = [s[key] for s in cat if _bin(s["u"]) == b]; z = [s[key] for s in base if _bin(s["u"]) == b]
        if len(c) < 3 or len(z) < 10:
            continue
        num += len(c) * float(np.mean(c)); den += len(c) * float(np.mean(z))
    return num / den if den > 0 else None


def boot_ratio(cat, base, key, n=BOOT, seed=7, alpha=0.10):
    """Day-block bootstrap of strat_ratio -> {value, lo, hi, sig} at 1 - alpha."""
    v = strat_ratio(cat, base, key)
    if v is None:
        return None
    rng = random.Random(seed)
    def by_day(xs):
        d = {}
        for s in xs:
            d.setdefault(s["day"], []).append(s)
        return list(d.values())
    cd, bd = by_day(cat), by_day(base)
    rs = []
    for _ in range(n):
        c = [s for g in (rng.choice(cd) for _ in cd) for s in g]; b = [s for g in (rng.choice(bd) for _ in bd) for s in g]
        r = strat_ratio(c, b, key)
        if r is not None:
            rs.append(r)
    if len(rs) < n // 2:
        return {"value": round(v, 3), "lo": None, "hi": None, "sig": False}
    lo, hi = np.percentile(rs, [50 * alpha, 100 - 50 * alpha])
    return {"value": round(v, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3), "sig": bool(lo > 1.0 or hi < 1.0)}


def gust_study(samples):
    for s in samples:
        s["p30"] = 1.0 if s["gmax"] >= GUST30 else 0.0
    base = [s for s in samples if s["quad"] == "none"]
    res = {"n": len(samples), "baseline": {"n": len(base), "gf": round(float(np.mean([s["gf"] for s in base])), 3) if base else None,
                                           "p30": round(float(np.mean([s["p30"] for s in base])), 3) if base else None},
           "quads": {}, "wind_bins_ms": WIND_BINS}
    tested = [q for q in QUADS + ["near"] if sum(1 for s in samples if s["quad"] == q) >= 20]
    alpha = 0.10 / max(len(tested), 1)          # Bonferroni: six quadrants are tested, so each interval is 1 - 0.10/6
    res["method"] = (f"wind-stratified ratio vs hours with no streak within 1500 km; day-block bootstrap, {100 * (1 - alpha):.1f}% intervals "
                     f"(90% across the {len(tested)} categories tested); a change under {MIN_EFFECT:.0%} is not applied")
    for q in QUADS + ["near"]:
        cat = [s for s in samples if s["quad"] == q]
        if len(cat) < 20:
            res["quads"][q] = {"n": len(cat)}
            continue
        res["quads"][q] = {"n": len(cat), "gf": round(float(np.mean([s["gf"] for s in cat])), 3), "p30": round(float(np.mean([s["p30"] for s in cat])), 3),
                           "gf_ratio": boot_ratio(cat, base, "gf", alpha=alpha), "p30_ratio": boot_ratio(cat, base, "p30", alpha=alpha)}
    return res


# ---------------------------------------------------------------------------- subtropical jet by month, and by El Niño phase
def r1_monthly_u(fetch=http):
    j0, j1, i0, i1 = r1_index(10.0, 45.0, STJ_LON[0], STJ_LON[1])
    nt = n_times("Monthlies/pressure/uwnd.mon.mean.nc", fetch) - 1
    u = f"{PSL}Monthlies/pressure/uwnd.mon.mean.nc.ascii?uwnd[0:1:{nt}][8][{j0}:1:{j1}][{i0}:1:{i1}],time[0:1:{nt}],lat[{j0}:1:{j1}],lon[{i0}:1:{i1}]"
    a, m = parse_dods_ascii(fetch(u, timeout=600).decode("utf-8", "replace"), "uwnd")
    return m["time"][: a.shape[0]], m["lat"], m["lon"], a[:, 0]


def stj_study(t, lat, lon, U, oni, base=(1991, 2020)):
    rows = []
    for k in range(len(t)):
        d = hours_to_dt(t[k]); la, lo, (u,) = ascending(lat, lon, U[k])
        s_lat, s_spd = stj(la, lo, u)
        rows.append({"ym": f"{d:%Y-%m}", "y": d.year, "m": d.month, "lat": s_lat, "spd": s_spd})
    clim = {}
    for m in range(1, 13):
        xs = [r for r in rows if r["m"] == m and base[0] <= r["y"] <= base[1]]
        if xs:
            clim[m] = {"lat": round(float(np.mean([r["lat"] for r in xs])), 2), "spd": round(float(np.mean([r["spd"] for r in xs])), 1)}
    for r in rows:
        c = clim.get(r["m"])
        if c:
            r["lat_anom"] = round(r["lat"] - c["lat"], 2); r["spd_anom"] = round(r["spd"] - c["spd"], 1)
    w_oni = enso.winter_oni(oni)
    winters = []
    for y in sorted(w_oni):
        ms = [r for r in rows if (r["y"] == y and r["m"] == 12) or (r["y"] == y + 1 and r["m"] in (1, 2))]
        if len(ms) == 3:
            winters.append({"y": y, "oni": w_oni[y], "class": enso.phase(w_oni[y]), "lat": round(float(np.mean([r["lat"] for r in ms])), 2),
                            "spd": round(float(np.mean([r["spd"] for r in ms])), 1)})
    comp = {}
    for cls in ["strong El Niño", "El Niño", "neutral", "La Niña", "all"]:
        xs = [w for w in winters if cls == "all" or w["class"] == cls or (cls == "El Niño" and w["class"] == "strong El Niño")]
        if xs:
            comp[cls] = {"n": len(xs), "lat": round(float(np.mean([w["lat"] for w in xs])), 2), "spd": round(float(np.mean([w["spd"] for w in xs])), 1)}
    r = None
    if len(winters) > 10:
        r = round(float(np.corrcoef([w["oni"] for w in winters], [w["lat"] for w in winters])[0, 1]), 2)
    return {"monthly": rows[-120:], "clim": clim, "winters": winters, "composite": comp, "r_oni_lat": r,
            "method": "latitude of the 250 hPa zonal-wind maximum 15-38 N, averaged 120-80 W (NCEP/NCAR reanalysis monthly means)"}


def build_climate(first=2016, last=None, fetch=http, stations=STATIONS, cache=None):
    """cache: a directory of per-year reanalysis metrics (jet_r1_YYYY.json), so a monthly rerun only reads what's new."""
    import time
    from . import learn
    last = last or dt.date.today().year - 1
    years = list(range(first, last + 1))
    jets = []
    for y in years:                                   # one year at a time: PSL answers 429 to parallel requests
        path = os.path.join(cache, f"jet_r1_{y}.json") if cache else None
        if path and os.path.exists(path):
            jets += json.load(open(path)); continue
        rows = r1_metrics_year(y, fetch)
        if path:
            os.makedirs(cache, exist_ok=True)
            json.dump(rows, open(path, "w"), separators=(",", ":"))
        jets += rows
        time.sleep(3)
    log.info("jet: %d reanalysis times, quadrants %s", len(jets), {q: sum(1 for j in jets if j["quad"] == q) for q in QUADS + ["none", "near"]})
    obs = {st: learn.station_rows(st, years) for st in stations}
    samples = windows(jets, obs)
    gust = gust_study(samples)
    gust.update({"years": [first, last], "stations": stations, "times": len(jets)})
    oni = enso.parse_oni(enso.get(enso.URLS["oni"]))
    s = stj_study(*r1_monthly_u(fetch), oni)
    return {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "gust": gust, "stj": s,
            "sources": ["NCEP/NCAR Reanalysis 1 (NOAA PSL)", "NDBC CHII2, 45174, 45198", "NOAA CPC ONI"]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("now"); n.add_argument("--out", required=True); n.add_argument("--climate")
    c = sub.add_parser("climate"); c.add_argument("--out", required=True); c.add_argument("--first", type=int, default=2016); c.add_argument("--last", type=int); c.add_argument("--cache")
    a = ap.parse_args()
    if a.cmd == "now":
        clim = json.load(open(a.climate)) if a.climate and os.path.exists(a.climate) else None
        d = build_now(clim)
        k = min(range(len(d["t"])), key=lambda i: abs(dt.datetime.fromisoformat(d["t"][i]) - dt.datetime.now(UTC)))
        print(len(d["t"]), "times", d["t"][0], "..", d["t"][-1], json.dumps(d["chicago"][k]))
    else:
        d = build_climate(a.first, a.last, cache=a.cache)
        print(json.dumps({"gust": d["gust"], "stj": {k: d["stj"][k] for k in ("composite", "r_oni_lat")}}, indent=1)[:5000])
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
