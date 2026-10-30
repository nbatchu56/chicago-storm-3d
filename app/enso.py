"""
El Niño tracker: the state of ENSO, how it has been moving, and what El Niño winters do to Lake Michigan off Chicago.

    python -m app.enso now --out enso.json [--climate enso_climate.json]      # every snapshot: indices + this season
    python -m app.enso climate --out enso_climate.json --era5 L/era5           # monthly: the 1979+ composites

Global metrics (NOAA CPC / PSL, no key):
  ONI      3-month running mean of Niño-3.4 SST anomaly (ERSSTv5), the official ENSO index (oni.ascii.txt)
  RONI     relative ONI: Niño-3.4 minus the tropical mean, CPC's strength measure since 2025 (RONI.ascii.txt)
  weekly   Niño 1+2, 3, 3.4, 4 SST anomalies from OISST, weekly (wksst9120.for)
  MEI v2   multivariate ENSO index, 2-month seasons (meiv2.data); SOI standardized Tahiti - Darwin pressure (soi)
  Pacific  the equatorial Pacific SST anomaly map, OISST v2.1 daily (CoastWatch ERDDAP), 1 degree
Trends: the weekly Niño-3.4 slope over the last 4 and 12 weeks, the ONI's run of seasons above +0.5, and the current
event lined up month by month with the strong events since 1979 (1982-83, 1997-98, 2015-16, 2023-24).

Lake Michigan off Chicago (winters Dec-Mar 1979 on, classed by their DJF ONI: strong El Niño >= +1.5, El Niño >= +0.5,
La Niña <= -0.5, else neutral):
  air            O'Hare daily mean temperature Dec-Feb, days with a low <= 0 F, seasonal snowfall (NCEI GHCN-D)
  ice            Lake Michigan annual maximum ice cover (GLERL AMIC), days with ice at the Chicago nearshore cell
                 (GLERL gridded daily ice), and the gridded Jan-Mar mean ice cover over the southern basin
  water          December lake surface temperature over the southern basin (GLERL GLSEA, 1995-2023)
  level          Calumet Harbor monthly mean, December to April change (NOAA CO-OPS)
  wind           ERA5 10 m wind at buoy 45198's cell (1991 on): mean speed, hours >= 10 m/s from the onshore sector
                 (N through ESE, the ones that put waves on the lakefront), and hours from the south (the southern storm track)
The gridded composites are differences from all winters (El Niño minus all), for the page's lake overlay.
"""

import argparse
import base64
import datetime as dt
import gzip
import io
import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request

import numpy as np

log = logging.getLogger("klot.enso")
UTC = dt.timezone.utc
UA = "chicago-storm-3d (github.com/nbatchu56/chicago-storm-3d)"
CPC = "https://www.cpc.ncep.noaa.gov/data/indices/"
URLS = {"oni": CPC + "oni.ascii.txt", "roni": CPC + "RONI.ascii.txt", "weekly": CPC + "wksst9120.for", "soi": CPC + "soi",
        "mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data"}
OISST = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"
GLERL = "https://apps.glerl.noaa.gov/erddap/griddap/"
AMIC = "https://www.glerl.noaa.gov/data/ice/glicd/AMIC.txt"
GHCN = "https://www.ncei.noaa.gov/access/services/data/v1"
COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
OHARE = "USW00094846"
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
STRONG, EL, LA = 1.5, 0.5, -0.5
ANALOGS = [1982, 1997, 2015, 2023]               # strong El Niño onset years (the event's winter starts in December)
LAKE_BOX = (41.62, -88.00, 42.70, -86.45)       # southern Lake Michigan: S, W, N, E
NEAR_CHI = (41.93, -87.60)                      # the Chicago nearshore cell for ice days (off Diversey / Fullerton)
BUOY_CELL = "45198"
ONSHORE = (345.0, 115.0)                        # wind-from sector that puts waves on the Chicago lakefront (N .. ESE)


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------------------- parsers (pure)
def parse_oni(txt):
    """ONI / RONI: rows 'SEAS YR [TOTAL] ANOM' -> [{season, year, anom}] in order."""
    out = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0] in SEASONS and p[1].isdigit():
            try:
                out.append({"season": p[0], "year": int(p[1]), "anom": float(p[-1])})
            except ValueError:
                pass
    return out


def parse_weekly(txt):
    """wksst9120.for: ' 07OCT2026     26.1 5.2 ...' (SST and anomaly glued with its sign) -> [{t, n12, n3, n34, n4}]."""
    out = []
    for line in txt.splitlines():
        m = re.match(r"\s*(\d{2}[A-Z]{3}\d{4})\s+(.*)$", line)
        if not m:
            continue
        nums = [float(x) for x in re.findall(r"-?\d+\.\d", m.group(2))]
        if len(nums) != 8:
            continue
        t = dt.datetime.strptime(m.group(1).title(), "%d%b%Y").date().isoformat()
        out.append({"t": t, "n12": nums[1], "n3": nums[3], "n34": nums[5], "n4": nums[7]})
    return out


def parse_mei(txt):
    """MEI v2: 'YEAR DJ JF ... ND' rows, -999 missing -> [{year, k (0=DJ), v}]."""
    out = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 13 and p[0].isdigit() and len(p[0]) == 4:
            for k, v in enumerate(p[1:]):
                f = float(v)
                if f > -900:
                    out.append({"year": int(p[0]), "k": k, "v": f})
    return out


def parse_soi(txt):
    """CPC soi: the second table (standardized) -> [{year, month, v}]."""
    blocks, cur = [], []
    for line in txt.splitlines():
        if line.strip().startswith("YEAR"):
            if cur:
                blocks.append(cur)
            cur = []
            continue
        m = re.match(r"(\d{4})(.*)$", line.strip())
        if m:
            vals = re.findall(r"-?\d+\.\d", m.group(2))
            cur.append((int(m.group(1)), [float(v) for v in vals]))
    if cur:
        blocks.append(cur)
    tab = blocks[-1] if blocks else []
    return [{"year": y, "month": i + 1, "v": v} for y, vals in tab for i, v in enumerate(vals) if v > -99]


def slope_per_week(vals):
    if len(vals) < 3:
        return None
    x = np.arange(len(vals), dtype=float)
    return round(float(np.polyfit(x, np.asarray(vals, float), 1)[0]), 3)


def phase(anom):
    if anom is None:
        return None
    return "strong El Niño" if anom >= STRONG else "El Niño" if anom >= EL else "La Niña" if anom <= LA else "neutral"


def winter_oni(oni):
    """DJF ONI by winter (the winter of Dec Y - Feb Y+1 is keyed Y): {Y: anom}."""
    return {r["year"] - 1: r["anom"] for r in oni if r["season"] == "DJF"}


def analog_tracks(oni, years=ANALOGS, current=None):
    """Each strong event's ONI from its MJJ (onset) to the next MAM, by season, and the current event's so far."""
    idx = {(r["year"], r["season"]): r["anom"] for r in oni}
    order = ["MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM"]
    def track(y0):
        return [idx.get((y0 if i < 7 else y0 + 1, s)) for i, s in enumerate(order)]
    out = {"seasons": order, "events": {f"{y}-{str(y + 1)[2:]}": track(y) for y in years}}
    if current:
        out["current"] = {"label": f"{current}-{str(current + 1)[2:]}", "oni": track(current)}
    return out


def run_length(oni, thr=EL):
    n = 0
    for r in reversed(oni):
        if r["anom"] >= thr:
            n += 1
        else:
            break
    return n


def summarize_indices(oni, roni, weekly, mei, soi):
    w34 = [w["n34"] for w in weekly]
    last = weekly[-1] if weekly else {}
    cur_oni = oni[-1] if oni else None
    cur_roni = roni[-1] if roni else None
    onset = None
    if cur_oni and cur_oni["anom"] >= EL:
        # the onset year: the calendar year of the first season of the current warm run
        k = len(oni) - run_length(oni)
        onset = oni[k]["year"]
    peak = max((w for w in weekly[-26:]), key=lambda w: w["n34"], default=None)
    return {
        "status": phase(cur_oni["anom"]) if cur_oni else None,
        "strength": None if not cur_roni else ("very strong" if cur_roni["anom"] >= 2.0 else "strong" if cur_roni["anom"] >= STRONG else
                                                 "moderate" if cur_roni["anom"] >= 1.0 else "weak" if cur_roni["anom"] >= EL else "below El Niño"),
        "oni": cur_oni, "roni": cur_roni,
        "weekly": {"t": last.get("t"), "n12": last.get("n12"), "n3": last.get("n3"), "n34": last.get("n34"), "n4": last.get("n4"),
                   "slope4": slope_per_week(w34[-4:]), "slope12": slope_per_week(w34[-12:]),
                   "peak26": peak and {"t": peak["t"], "n34": peak["n34"]}},
        "mei": mei[-1] if mei else None, "soi": soi[-1] if soi else None,
        "run_seasons": run_length(oni), "onset_year": onset,
        "series": {"weekly": [[w["t"], w["n34"], w["n12"], w["n4"]] for w in weekly[-104:]],
                   "oni": [[r["year"], r["season"], r["anom"]] for r in oni if r["year"] >= 1979],
                   "roni": [[r["year"], r["season"], r["anom"]] for r in roni if r["year"] >= 1979],
                   "mei": [[m["year"], m["k"], m["v"]] for m in mei[-36:]], "soi": [[s["year"], s["month"], s["v"]] for s in soi[-36:]]},
        "analogs": analog_tracks(oni, current=onset),
    }


# ---------------------------------------------------------------------------- the Pacific map
def parse_erddap_csv(txt, var):
    """ERDDAP CSV (header + units rows) -> (times, lats, lons, values) as lists, NaN kept."""
    rows = txt.splitlines()[2:]
    out = []
    for r in rows:
        p = r.split(",")
        if len(p) < 4:
            continue
        try:
            v = float(p[-1]) if p[-1] not in ("NaN", "") else float("nan")
            out.append((p[0], float(p[-3]), float(p[-2]), v))
        except ValueError:
            continue
    return out


def pacific_map(fetch=get):
    """Equatorial Pacific SST anomaly, 20S-20N, 140E-80W, 1 degree, newest OISST day -> byte grid (anomaly -4..+4 C)."""
    def q(lo0, lo1):
        return OISST + "?" + "anom" + urllib.parse.quote(f"[(last)][(0.0)][(-19.875):4:(19.875)][({lo0}):4:({lo1})]")
    a = parse_erddap_csv(fetch(q(140.125, 179.875)), "anom")   # the ERDDAP grid is -180..180: the dateline splits the box
    b = parse_erddap_csv(fetch(q(-179.875, -80.125)), "anom")
    pts = a + b
    if not pts:
        return None
    t = pts[0][0]
    lats = sorted({p[1] for p in pts}); lons = sorted({(p[2] + 360) % 360 for p in pts})
    li = {v: i for i, v in enumerate(lats)}; lj = {v: j for j, v in enumerate(lons)}
    g = np.full((len(lats), len(lons)), np.nan, np.float32)
    for _, la, lo, v in pts:
        g[li[la], lj[(lo + 360) % 360]] = v
    q8 = np.where(np.isfinite(g), np.clip(np.round((g + 4) / 8 * 253) + 1, 1, 254), 0).astype(np.uint8)
    return {"t": t, "lat0": lats[0], "lat1": lats[-1], "lon0": lons[0], "lon1": lons[-1], "nlat": len(lats), "nlon": len(lons),
            "lo": -4.0, "hi": 4.0, "b64": base64.b64encode(gzip.compress(q8.tobytes(), 9)).decode(),
            "regions": {"Niño 1+2": [-10, 0, 270, 280], "Niño 3": [-5, 5, 210, 270], "Niño 3.4": [-5, 5, 190, 240], "Niño 4": [-5, 5, 160, 210]}}


# ---------------------------------------------------------------------------- local records (Chicago / Lake Michigan)
def parse_amic(txt):
    out = {}
    for line in txt.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            try:
                out[int(p[0])] = float(p[2])     # columns: year sup mic hur eri ont bas
            except ValueError:
                pass
    return out


def ghcn_daily(start, end, fetch=get, station=OHARE):
    q = {"dataset": "daily-summaries", "stations": station, "startDate": start, "endDate": end, "dataTypes": "TMAX,TMIN,SNOW",
         "format": "json", "units": "standard"}
    rows = json.loads(fetch(GHCN + "?" + urllib.parse.urlencode(q), timeout=240) or "[]")
    out = []
    for r in rows:
        f = lambda k: float(r[k]) if r.get(k, "").strip() not in ("", None) else None  # noqa: E731
        out.append({"d": r["DATE"], "tmax": f("TMAX"), "tmin": f("TMIN"), "snow": f("SNOW")})
    return out


def winter_air(days, y):
    """DJF mean temperature (F), days with tmin <= 0 F (Dec-Mar), snowfall (in, Oct-May) for the winter keyed y."""
    djf = [((d["tmax"] + d["tmin"]) / 2) for d in days if d["tmax"] is not None and d["tmin"] is not None
           and ((d["d"][:7] >= f"{y}-12" and d["d"][:7] <= f"{y}-12") or (f"{y + 1}-01" <= d["d"][:7] <= f"{y + 1}-02"))]
    cold = sum(1 for d in days if d["tmin"] is not None and d["tmin"] <= 0 and f"{y}-12-01" <= d["d"] <= f"{y + 1}-03-31")
    snow = sum(d["snow"] or 0 for d in days if f"{y}-10-01" <= d["d"] <= f"{y + 1}-05-31")
    return {"djf_f": round(float(np.mean(djf)), 1) if len(djf) > 60 else None, "cold_days": cold, "snow_in": round(snow, 1)}


def coops_monthly(y0, y1, fetch=get):
    out = {}
    for a in range(y0, y1 + 1, 10):
        b = min(a + 9, y1)
        q = {"station": "9087044", "product": "monthly_mean", "datum": "IGLD", "begin_date": f"{a}0101", "end_date": f"{b}1231",
             "units": "metric", "time_zone": "gmt", "format": "json", "application": "chicago-storm-3d"}
        try:
            for r in json.loads(fetch(COOPS + "?" + urllib.parse.urlencode(q), timeout=120)).get("data", []):
                if r.get("MSL"):
                    out[(int(r["year"]), int(r["month"]))] = float(r["MSL"])
        except Exception as e:  # noqa: BLE001
            log.warning("CO-OPS %s-%s: %s", a, b, e)
    return out


def era5_winter(era5_dir, y, point=BUOY_CELL):
    """ERA5 10 m wind at a buoy cell, Dec-Mar of winter y: mean speed, hours >= 10 m/s onshore (N..ESE) and from the S half."""
    rows = []
    for yy in (y, y + 1):
        p = os.path.join(era5_dir, f"era5_{yy}.csv.gz")
        if not os.path.exists(p):
            return None
        with gzip.open(p, "rt") as fh:
            next(fh)
            for line in fh:
                t, pt, ws, wd, *_ = line.rstrip("\n").split(",")
                if pt == point and (f"{y}-12" <= t[:7] or t[:7] <= f"{y + 1}-03") and f"{y}-12-01" <= t[:10] <= f"{y + 1}-03-31" and ws and wd:
                    rows.append((float(ws), float(wd)))
    if len(rows) < 24 * 100:
        return None
    ws = np.array([r[0] for r in rows]); wd = np.array([r[1] for r in rows])
    on = (wd >= ONSHORE[0]) | (wd <= ONSHORE[1])
    south = (wd >= 135) & (wd <= 225)
    return {"wind_ms": round(float(ws.mean()), 2), "onshore10_h": int(((ws >= 10) & on).sum()), "south10_h": int(((ws >= 10) & south).sum())}


def erddap_grid(dataset, var, t0, t1, box=LAKE_BOX, stride=2, tstride=7, fetch=get):
    s, w, n, e = box
    q = f"{var}[({t0}T12:00:00Z):{tstride}:({t1}T12:00:00Z)][({s}):{stride}:({n})][({w}):{stride}:({e})]"
    return parse_erddap_csv(fetch(GLERL + dataset + ".csv?" + urllib.parse.quote(q, safe="=&"), timeout=300), var)


def grid_mean(pts):
    """[(t, lat, lon, v)] -> (lats, lons, mean over time with NaN where never valid, count of times)."""
    if not pts:
        return None
    lats = sorted({p[1] for p in pts}); lons = sorted({p[2] for p in pts}); times = sorted({p[0] for p in pts})
    li = {v: i for i, v in enumerate(lats)}; lj = {v: j for j, v in enumerate(lons)}; lt = {v: k for k, v in enumerate(times)}
    a = np.full((len(times), len(lats), len(lons)), np.nan, np.float32)
    for t, la, lo, v in pts:
        a[lt[t], li[la], lj[lo]] = v
    with np.errstate(invalid="ignore"):
        m = np.nanmean(a, axis=0) if np.isfinite(a).any() else np.full(a.shape[1:], np.nan)
    return lats, lons, m, a


def ice_days_at(a, lats, lons, lat, lon, thr=10.0, every=7):
    i = int(np.argmin(np.abs(np.asarray(lats) - lat))); j = int(np.argmin(np.abs(np.asarray(lons) - lon)))
    v = a[:, i, j]
    return int(np.nansum(v >= thr)) * every


def composite(by_winter, classes, key):
    """Mean of one metric by class over the winters that have it: {class: {mean, n, years}}."""
    out = {}
    for cls in ("strong El Niño", "El Niño", "neutral", "La Niña", "all"):
        ys = [y for y, v in by_winter.items() if v.get(key) is not None and (cls == "all" or classes.get(y) == cls or (cls == "El Niño" and classes.get(y) == "strong El Niño"))]
        if ys:
            out[cls] = {"mean": round(float(np.mean([by_winter[y][key] for y in ys])), 2), "n": len(ys)}
    return out


def byte_field(lats, lons, a, lo, hi):
    q8 = np.where(np.isfinite(a), np.clip(np.round((a - lo) / (hi - lo) * 253) + 1, 1, 254), 0).astype(np.uint8)
    return {"lat0": lats[0], "lat1": lats[-1], "lon0": lons[0], "lon1": lons[-1], "nlat": len(lats), "nlon": len(lons), "lo": lo, "hi": hi,
            "b64": base64.b64encode(gzip.compress(q8.tobytes(), 9)).decode()}


def build_climate(era5_dir=None, fetch=get, first=1979, last=None):
    """The composites. Winters keyed by their December's year."""
    oni = parse_oni(fetch(URLS["oni"]))
    w_oni = winter_oni(oni)
    last = last or max(w_oni)
    years = [y for y in range(first, last + 1) if y in w_oni]
    classes = {y: phase(w_oni[y]) for y in years}
    rec = {y: {"oni": w_oni[y], "class": classes[y]} for y in years}
    amic = parse_amic(fetch(AMIC))
    for y in years:
        rec[y]["max_ice_pct"] = amic.get(y + 1)
    try:
        days = []
        for a in range(first, last + 1, 10):
            days += ghcn_daily(f"{a}-10-01", f"{min(a + 9, last) + 1}-05-31", fetch=fetch)
        for y in years:
            rec[y].update(winter_air(days, y))
    except Exception as e:  # noqa: BLE001
        log.warning("GHCN: %s", e)
    lv = coops_monthly(first, last + 1, fetch=fetch)
    for y in years:
        a, b = lv.get((y, 12)), lv.get((y + 1, 4))
        rec[y]["level_dec_apr_cm"] = round((b - a) * 100, 1) if a is not None and b is not None else None
    if era5_dir:
        for y in years:
            e = era5_winter(era5_dir, y)
            if e:
                rec[y].update(e)
    # gridded: Jan-Mar ice cover and December water temperature, southern basin
    ice_all, ice_n, sst_all = [], {}, []
    for y in years:
        try:
            pts = erddap_grid("GL_Ice_Cover", "ice_cover", f"{y + 1}-01-03", f"{y + 1}-03-28", fetch=fetch)
            gm = grid_mean(pts)
            if gm:
                lats, lons, m, a = gm
                ice_all.append((y, m)); rec[y]["chicago_ice_days"] = ice_days_at(a, lats, lons, *NEAR_CHI)
                ice_n[y] = (lats, lons)
        except Exception as e:  # noqa: BLE001
            log.warning("ice %s: %s", y, e)
        if 1995 <= y <= 2023:
            try:
                gm = grid_mean(erddap_grid("GLSEA_GCS", "sst", f"{y}-12-01", f"{y}-12-29", fetch=fetch))
                if gm:
                    sst_all.append((y, gm[2])); rec[y]["dec_sst_c"] = round(float(np.nanmean(gm[2])), 2)
            except Exception as e:  # noqa: BLE001
                log.warning("sst %s: %s", y, e)
    grids = {}
    if ice_all:
        lats, lons = ice_n[ice_all[0][0]]
        allm = np.nanmean(np.stack([m for _, m in ice_all]), axis=0)
        en = [m for y, m in ice_all if classes[y] in ("El Niño", "strong El Niño")]
        sen = [m for y, m in ice_all if classes[y] == "strong El Niño"]
        if en:
            grids["ice_en_minus_all"] = byte_field(lats, lons, np.nanmean(np.stack(en), axis=0) - allm, -40.0, 40.0)
        if sen:
            grids["ice_strong_minus_all"] = byte_field(lats, lons, np.nanmean(np.stack(sen), axis=0) - allm, -40.0, 40.0)
        grids["ice_all"] = byte_field(lats, lons, allm, 0.0, 100.0)
    if sst_all:
        g0 = sst_all[0][1]
        same = [(y, m) for y, m in sst_all if m.shape == g0.shape]
        allm = np.nanmean(np.stack([m for _, m in same]), axis=0)
        en = [m for y, m in same if classes[y] in ("El Niño", "strong El Niño")]
        if en and ice_n:
            # GLSEA's grid matches the ice grid's spacing; the box is the same request
            lats, lons = next(iter(ice_n.values()))
            if allm.shape == (len(lats), len(lons)):
                grids["sst_dec_en_minus_all"] = byte_field(lats, lons, np.nanmean(np.stack(en), axis=0) - allm, -2.0, 2.0)
    metrics = ["djf_f", "cold_days", "snow_in", "max_ice_pct", "chicago_ice_days", "dec_sst_c", "level_dec_apr_cm", "wind_ms", "onshore10_h", "south10_h"]
    return {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "first": first, "last": last,
            "classes": {"strong El Niño": f">= +{STRONG}", "El Niño": f">= +{EL}", "La Niña": f"<= {LA}", "by": "DJF ONI"},
            "winters": {str(y): rec[y] for y in years}, "composite": {k: composite(rec, classes, k) for k in metrics}, "grids": grids,
            "sources": ["NOAA CPC ONI", "GLERL AMIC and gridded ice (ERDDAP GL_Ice_Cover)", "GLERL GLSEA (ERDDAP GLSEA_GCS)",
                        "NCEI GHCN-D O'Hare", "NOAA CO-OPS Calumet Harbor", "ERA5 (Copernicus) at buoy 45198"]}


# ---------------------------------------------------------------------------- this season so far
def season_so_far(today, fetch=get, climate=None):
    """Oct 1 to today at O'Hare (the cold season so far): mean temperature, snow, nights at or below 0 F."""
    y = today.year if today.month >= 8 else today.year - 1
    out = {"winter": y, "since": f"{y}-10-01"}
    if today < dt.date(y, 10, 2):
        return out
    try:
        days = ghcn_daily(f"{y}-10-01", today.isoformat(), fetch=fetch)
        t = [((d["tmax"] + d["tmin"]) / 2) for d in days if d["tmax"] is not None and d["tmin"] is not None]
        out["air"] = {"days": len(t), "mean_f": round(float(np.mean(t)), 1) if t else None, "snow_in": round(sum(d["snow"] or 0 for d in days), 1),
                      "cold_days": sum(1 for d in days if d["tmin"] is not None and d["tmin"] <= 0)}
    except Exception as e:  # noqa: BLE001
        out["air_error"] = str(e)[:200]
    return out


def flags(idx, season, climate, today):
    """What this season's weather shares with the El Niño winter pattern, when an El Niño is on: short notes for the page."""
    notes = []
    st = (idx or {}).get("status") or ""
    if "El Niño" not in st:
        return notes
    comp = (climate or {}).get("composite") or {}
    def c(key, cls):
        return ((comp.get(key) or {}).get(cls) or {}).get("mean")
    if c("max_ice_pct", "strong El Niño") is not None and c("max_ice_pct", "all") is not None:
        notes.append({"kind": "ice", "text": f"Strong El Niño winters peaked at {c('max_ice_pct', 'strong El Niño'):.0f}% Lake Michigan ice on average, against {c('max_ice_pct', 'all'):.0f}% for all winters"})
    if c("djf_f", "strong El Niño") is not None and c("djf_f", "all") is not None:
        notes.append({"kind": "air", "text": f"Dec-Feb at O'Hare averaged {c('djf_f', 'strong El Niño') - c('djf_f', 'all'):+.1f} °F against all winters in strong El Niños"})
    return notes


def build_now(climate=None, fetch=get, today=None):
    today = today or dt.date.today()
    oni = parse_oni(fetch(URLS["oni"])); roni = parse_oni(fetch(URLS["roni"])); weekly = parse_weekly(fetch(URLS["weekly"]))
    try:
        mei = parse_mei(fetch(URLS["mei"]))
    except Exception:  # noqa: BLE001
        mei = []
    try:
        soi = parse_soi(fetch(URLS["soi"]))
    except Exception:  # noqa: BLE001
        soi = []
    idx = summarize_indices(oni, roni, weekly, mei, soi)
    out = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "indices": idx}
    try:
        out["pacific"] = pacific_map(fetch)
    except Exception as e:  # noqa: BLE001
        out["pacific_error"] = str(e)[:200]
    out["season"] = season_so_far(today, fetch, climate)
    if climate:
        out["climate"] = {k: climate[k] for k in ("first", "last", "classes", "winters", "composite", "grids", "sources", "generated") if k in climate}
    out["flags"] = flags(idx, out["season"], climate, today)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("now"); n.add_argument("--out", required=True); n.add_argument("--climate")
    c = sub.add_parser("climate"); c.add_argument("--out", required=True); c.add_argument("--era5"); c.add_argument("--first", type=int, default=1979)
    a = ap.parse_args()
    if a.cmd == "now":
        clim = json.load(open(a.climate)) if a.climate and os.path.exists(a.climate) else None
        d = build_now(clim)
        i = d["indices"]
        print(json.dumps({k: i[k] for k in ("status", "strength", "oni", "roni", "weekly", "run_seasons", "onset_year")}, default=str))
    else:
        d = build_climate(a.era5, first=a.first)
        print(json.dumps(d["composite"], indent=1)[:4000])
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
