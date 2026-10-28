"""
Learned wave correction (experimental): learn how the open-lake wave model is usually wrong, from the buoys.

Three parts, all writing to the `learn` branch:

1. backfill  Rerun the wave model (same SeaState the paddling forecast uses) over years of NDBC archive at the
             Wilmette (45174) and Chicago (45198) buoys, driven by observed wind only, and keep every hour's
             model-vs-buoy pair. Because the model needs only wind, this gives years of training data at once.
             With ERA5 (app.era5) cached, hours with buoy waves but no measured wind get ERA5 wind (scaled to read
             like the buoy), and a 35-year ERA5-only hindcast shows how often big storms happen and how few of them
             the buoys saw (era5_climate.json).
2. fit       Learn a correction factor for wave height and period by wind direction (12 sectors) x wind speed
             (4 bins), plus one for swell-dominated hours (old waves after the wind drops or shifts), as the
             average log(buoy / model) with shrinkage toward broader averages where data is thin. Scored by
             leaving one year out at a time; the page shows it only as an experimental option.
3. log/score Every snapshot records what the forecast said at the two buoys for the next 48 h (raw model,
             learned, the live-buoy-nudged product, the NWS wave grid and NOAA GLWU when available); `score`
             lines those up with what the buoys measured and reports errors by lead time. It also keeps the
             next 6 h of each beach's forecast (scorecard/spots-*.jsonl) for photo spot checks (tools/spotcheck.py).

    python -m app.learn backfill --out learn/ [--years 2012-2026] [--era5 learn/era5]
    python -m app.learn fit --dir learn/
    python -m app.learn log --sup sup.json [--glwu glwu.json] --dir learn/
    python -m app.learn score --dir learn/
"""

import argparse
import csv
import datetime as dt
import glob
import gzip
import io
import json
import logging
import math
import os

import numpy as np

from . import sup_waves as sw

log = logging.getLogger("learn")
UTC = dt.timezone.utc

BUOYS = {"45174": (42.135, -87.655), "45198": (41.892, -87.563)}               # Wilmette, Chicago
WIND_ORDER = {"45174": ["45174", "CHII2", "45198"], "45198": ["45198", "CHII2", "45174"]}
SECTORS = 12
SPEED_EDGES = (4.0, 7.0, 10.0)               # m/s at 10 m, stability-adjusted: light / moderate / fresh / strong
MIN_HS = 0.15                                # below this the log ratio is mostly noise
SHRINK_K = 72.0                              # hours of data a cell needs before it counts as much as its parent
                                             # (hours are strongly correlated: ~72 h is a few storms' worth)
CLIP = (0.5, 2.0)
# Fit only on data that matches how each buoy measures today. 45174 read ~20% higher waves through 2024 than from
# 2025 on, for the same wind and model (45198 didn't change), so something changed at that buoy; the live nudge and
# the scorecard use today's readings, so the older 45174 years are kept in the backfill but left out of the fit.
FIT_FROM = {"45174": "2025-01-01"}
SWELL_FRAC = 0.5                             # share of wave energy in leftover/swell above this -> "swell" factor
LEAD_BUCKETS = [(0, 0), (1, 3), (4, 6), (7, 12), (13, 24), (25, 48)]
METHODS = ["raw", "learned", "product", "product_learned", "nws", "glwu", "gfs", "wn3"]
METHOD_LABELS = {"raw": "model", "learned": "model + learned", "product": "forecast (buoy-nudged)",
                 "product_learned": "forecast + learned", "nws": "NWS wave grid", "glwu": "NOAA GLWU",
                 "gfs": "model on GFS wind", "wn3": "model on WeatherNext 3 wind"}


# ---------------------------------------------------------------------------- data
def get_bytes(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": sw.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def station_rows(station, years, get=get_bytes, now=None):
    """All standard-met rows for the station in these years: yearly archive files, then this year's monthly
    files and the 45-day realtime file."""
    now = now or dt.datetime.now(UTC)
    st = station.lower()
    rows = {}

    def add(txt):
        for r in sw.parse_ndbc(txt, station):
            rows[r["t"]] = r

    for y in years:
        urls = []
        if y < now.year:
            urls.append(f"https://www.ndbc.noaa.gov/data/historical/stdmet/{st}h{y}.txt.gz")
        else:
            for m in range(1, now.month + 1):
                mon = dt.date(y, m, 1).strftime("%b")
                urls.append(f"https://www.ndbc.noaa.gov/data/stdmet/{mon}/{st}{m}{y}.txt.gz")
                urls.append(f"https://www.ndbc.noaa.gov/data/stdmet/{mon}/{st}.txt")     # newest month, plain text
        for url in urls:
            try:
                raw = get(url)
                txt = gzip.decompress(raw).decode("utf-8", "replace") if url.endswith(".gz") else raw.decode("utf-8", "replace")
                add(txt)
            except Exception as e:  # noqa: BLE001
                log.debug("%s: %s", url, e)
    if any(y >= now.year for y in years):
        try:
            add(get(f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt").decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            log.debug("realtime %s: %s", station, e)
    return [rows[t] for t in sorted(rows)]


def _hour(t):
    return t.replace(minute=0, second=0, microsecond=0)


def hourly_means(rows, fields=("wspd", "wvht", "dpd", "atmp", "wtmp")):
    acc = {}
    for r in rows:
        h = _hour(r["t"])
        a = acc.setdefault(h, {})
        for f in fields:
            if r.get(f) is not None:
                a.setdefault(f, []).append(r[f])
        if r.get("wdir") is not None:
            a.setdefault("wdir", []).append(math.radians(r["wdir"]))
        if r.get("mwd") is not None:
            a.setdefault("mwd", []).append(math.radians(r["mwd"]))
    out = {}
    for h, a in acc.items():
        o = {f: float(np.mean(v)) for f, v in a.items() if f not in ("wdir", "mwd")}
        for f in ("wdir", "mwd"):
            if f in a:
                o[f] = (math.degrees(math.atan2(np.mean(np.sin(a[f])), np.mean(np.cos(a[f])))) + 360) % 360
        out[h] = o
    return out


# ---------------------------------------------------------------------------- 1. backfill
def hindcast(buoy, hourly_by_station, gap_reset_h=12):
    """Model vs buoy, hour by hour, with the forecast's own wind handling (10 m height, stability)."""
    lat, lon = BUOYS[buoy]
    table = sw.fetch_table(lat, lon, 90)
    fetch = lambda d: sw.interp_dir(table, d) * 1000  # noqa: E731
    own = hourly_by_station.get(buoy, {})
    wave_hours = sorted(h for h, o in own.items() if "wvht" in o)
    if not wave_hours:
        return []
    out, sea, gap = [], sw.SeaState(), 0
    t, t1 = wave_hours[0] - dt.timedelta(hours=24), wave_hours[-1]
    era = era5_key(buoy)
    order = WIND_ORDER[buoy] + ([era] if era in hourly_by_station else [])
    while t <= t1:
        u = d = src = None
        for st in order:
            o = hourly_by_station.get(st, {}).get(t)
            if o and "wspd" in o and "wdir" in o:
                u, d, src = o["wspd"], o["wdir"], st
                break
        ob = own.get(t, {})
        air = ob.get("atmp", (hourly_by_station.get("CHII2", {}).get(t) or {}).get("atmp"))
        if air is None:
            air = (hourly_by_station.get(era, {}).get(t) or {}).get("atmp")
        if u is None:
            gap += 1
            if gap > gap_reset_h:
                sea = sw.SeaState()
        else:
            gap = 0
            u = sw.u10(u, src) * sw.stability(air, ob.get("wtmp"))
        sea.step(u, d, fetch)
        if "wvht" in ob and u is not None:
            comps = sea.components()
            e = sum(c["hs"] ** 2 for c in comps)
            dom = max(comps, key=lambda c: c["hs"], default=None)
            sw_e = sum(c["hs"] ** 2 for c in comps if c["kind"] == "swell" or c.get("left"))
            out.append({"t": t.isoformat(), "buoy": buoy, "obs_hs": round(ob["wvht"], 3), "obs_tp": _r(ob.get("dpd"), 2),
                        "obs_dir": _r(ob.get("mwd"), 0), "pred_hs": round(math.sqrt(e), 3), "pred_tp": _r(dom and dom["tp"], 2),
                        "pred_dir": _r(dom and dom["dir"], 0), "u": round(u, 2), "wdir": round(d), "swell_frac": round(sw_e / e, 2) if e > 0 else 0.0,
                        "wind_src": src})
        t += dt.timedelta(hours=1)
    return out


def _r(v, n):
    return None if v is None else round(float(v), n)


PAIR_FIELDS = ["t", "buoy", "obs_hs", "obs_tp", "obs_dir", "pred_hs", "pred_tp", "pred_dir", "u", "wdir", "swell_frac", "wind_src"]


def write_pairs(path, pairs):
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, PAIR_FIELDS)
        w.writeheader()
        w.writerows(pairs)


def read_pairs(paths):
    out = []
    for p in paths:
        with gzip.open(p, "rt") as fh:
            for r in csv.DictReader(fh):
                for k in ("obs_hs", "obs_tp", "obs_dir", "pred_hs", "pred_tp", "pred_dir", "u", "wdir", "swell_frac"):
                    r[k] = float(r[k]) if r.get(k) not in (None, "") else None
                out.append(r)
    return out


def backfill(out_dir, years, get=get_bytes, now=None, era5_dir=None):
    os.makedirs(os.path.join(out_dir, "backfill"), exist_ok=True)
    stations = sorted({s for v in WIND_ORDER.values() for s in v})
    by_year = {y: {st: hourly_means(station_rows(st, [y], get, now)) for st in stations} for y in years}
    era, calib = {}, {}
    if era5_dir and os.path.isdir(era5_dir):
        from . import era5
        era = era5.load(era5_dir)
        merged = {st: {h: o for y in years for h, o in by_year[y].get(st, {}).items()} for st in stations}
        calib = calibrate_era5(era, merged)
        log.info("ERA5 calibration: %s", json.dumps({b: {"ratio": c["overall"], "hours": c["hours"]} for b, c in calib.items()}))
    summary = {}
    for y in years:
        hourly = by_year[y]
        for b in BUOYS:
            if b in era and b in calib:
                hourly[era5_key(b)] = {h: o for h, o in calibrated(era[b], calib[b]).items() if h.year == y}
            pairs = hindcast(b, hourly)
            if not pairs:
                continue
            write_pairs(os.path.join(out_dir, "backfill", f"pairs_{b}_{y}.csv.gz"), pairs)
            summary.setdefault(b, {})[str(y)] = len(pairs)
            log.info("%s %s: %d hours (%d with ERA5 wind)", b, y, len(pairs), sum(1 for p in pairs if p["wind_src"].startswith("era5")))
    if era and calib:
        observed = {b: {h for y in years for h, o in by_year[y].get(b, {}).items() if "wvht" in o} for b in BUOYS}
        clim = era5_climate({b: calibrated(era[b], calib[b]) for b in BUOYS if b in era and b in calib}, observed)
        clim["calibration"] = calib
        with open(os.path.join(out_dir, "era5_climate.json"), "w") as fh:
            json.dump(clim, fh, indent=1)
        summary["era5"] = {b: v["totals"] for b, v in clim["buoys"].items()}
    return summary


# ---------------------------------------------------------------------------- ERA5 (filling wind, storm climate)
ERA5_MIN_U = 2.0
NORTH = (315.0, 45.0)        # wind from NW through NE: the longest fetch down the lake to Chicago


def era5_key(buoy):
    return f"era5:{buoy}"


def calibrated(series, cal):
    """ERA5 hours scaled to read like the buoy's 10 m wind (ratio by direction sector)."""
    out = {}
    for h, o in series.items():
        r = cal["sectors"][sector(o["wdir"])] if cal.get("sectors") else cal["overall"]
        out[h] = {"wspd": o["wspd"] * r, "wdir": o["wdir"], "atmp": o.get("atmp")}
    return out


def calibrate_era5(era, hourly_by_station):
    """Buoy 10 m wind / ERA5 wind over the hours both exist, overall and by direction sector (shrunk toward overall)."""
    out = {}
    for b in BUOYS:
        e, own = era.get(b) or {}, hourly_by_station.get(b) or {}
        num, den, n = [0.0] * SECTORS, [0.0] * SECTORS, [0] * SECTORS
        for h, o in own.items():
            q = e.get(h)
            if not q or "wspd" not in o or q["wspd"] < ERA5_MIN_U:
                continue
            k = sector(q["wdir"])
            num[k] += sw.u10(o["wspd"], b)
            den[k] += q["wspd"]
            n[k] += 1
        if sum(n) < 200:
            continue
        overall = sum(num) / sum(den)
        secs = [round((num[k] + SHRINK_K * overall * (den[k] / n[k] if n[k] else 1.0)) /
                      (den[k] + SHRINK_K * (den[k] / n[k] if n[k] else 1.0)), 3) for k in range(SECTORS)]
        out[b] = {"overall": round(overall, 3), "sectors": secs, "hours": sum(n), "hours_by_sector": n}
    return out


def _is_north(d):
    return d >= NORTH[0] or d <= NORTH[1]


def era5_climate(era_cal, observed, big=(1.0, 2.0), storm_gap_h=24):
    """Open-lake wave model run on ERA5 wind alone at each buoy, all years: hours of big waves by wind direction
    and how many of them the buoy was out to measure; the largest storms."""
    out = {"method": "CEM growth with memory on ERA5 10 m wind (scaled to the buoy), no stability term", "big_m": list(big), "buoys": {}}
    for b, ser in era_cal.items():
        lat, lon = BUOYS[b]
        table = sw.fetch_table(lat, lon, 90)
        fetch = lambda d: sw.interp_dir(table, d) * 1000  # noqa: E731
        sea, hours = sw.SeaState(), sorted(ser)
        tot = {f"{m:g}m": {"all": [0, 0], "north": [0, 0]} for m in big}
        storms, cur, last_big = [], None, None
        monthly = {m: [] for m in range(1, 13)}
        by_sector = [[0, 0] for _ in range(8)]
        for h in hours:
            o = ser[h]
            sea.step(o["wspd"], o["wdir"], fetch)
            hs, obs = sea.total_hs(), h in observed.get(b, ())
            monthly[h.month].append(hs)
            for m in big:
                if hs >= m:
                    k = f"{m:g}m"
                    tot[k]["all"][0] += 1; tot[k]["all"][1] += obs
                    if _is_north(o["wdir"]):
                        tot[k]["north"][0] += 1; tot[k]["north"][1] += obs
            if hs >= big[0]:
                s8 = int(((o["wdir"] + 22.5) % 360) // 45)
                by_sector[s8][0] += 1; by_sector[s8][1] += obs
                if cur is None or (h - last_big).total_seconds() > storm_gap_h * 3600:
                    if cur:
                        storms.append(cur)
                    cur = {"start": h.isoformat(), "peak_hs": 0.0, "hours": 0, "observed_hours": 0}
                cur["hours"] += 1; cur["observed_hours"] += obs
                if hs > cur["peak_hs"]:
                    cur.update(peak_hs=round(hs, 2), peak_t=h.isoformat(), peak_wind=round(o["wspd"], 1), peak_dir=round(o["wdir"]))
                last_big = h
        if cur:
            storms.append(cur)
        storms.sort(key=lambda s: -s["peak_hs"])
        years = sorted({h.year for h in hours})
        out["buoys"][b] = {
            "years": [years[0], years[-1]] if years else None,
            "totals": {k: {"model_hours": v["all"][0], "buoy_saw": v["all"][1], "north_model_hours": v["north"][0], "north_buoy_saw": v["north"][1]}
                       for k, v in tot.items()},
            "big_hours_by_dir": {d: {"model_hours": a, "buoy_saw": c} for d, (a, c) in zip(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], by_sector)},
            "top_storms": storms[:25],
            "monthly_hs_m": {str(m): {"p50": round(float(np.percentile(v, 50)), 2), "p90": round(float(np.percentile(v, 90)), 2),
                                      "p99": round(float(np.percentile(v, 99)), 2)} for m, v in monthly.items() if v}}
    return out


# ---------------------------------------------------------------------------- 2. fit
def sector(wdir):
    return int(((wdir + 180.0 / SECTORS) % 360) // (360.0 / SECTORS))


def speed_bin(u):
    return sum(u >= e for e in SPEED_EDGES)


def _usable(p, key_obs, key_pred):
    return p.get(key_obs) is not None and p.get(key_pred) is not None and p[key_obs] >= (MIN_HS if key_obs == "obs_hs" else 1.0) \
        and p[key_pred] >= (MIN_HS if key_pred == "pred_hs" else 1.0) and p.get("u") is not None and p.get("wdir") is not None


def _shrunk_table(pairs, key_obs, key_pred):
    """Least-squares factor f = sum(obs*model)/sum(model^2) (the scale that minimizes the squared error, so a noisy
    model is not blown up), lake-wide -> swell / per sector -> sector x speed, each shrunk toward its parent by
    adding SHRINK_K typical hours that agree with the parent."""
    xs = [p for p in pairs if _usable(p, key_obs, key_pred)]
    if not xs:
        return None
    op = lambda ps: sum(p[key_obs] * p[key_pred] for p in ps)  # noqa: E731
    pp = lambda ps: sum(p[key_pred] ** 2 for p in ps)  # noqa: E731
    w = pp(xs) / len(xs)                                        # one typical hour's weight
    g = op(xs) / pp(xs)
    shrink = lambda ps, parent: (op(ps) + SHRINK_K * w * parent) / (pp(ps) + SHRINK_K * w)  # noqa: E731
    swell = [p for p in xs if (p.get("swell_frac") or 0) > SWELL_FRAC]
    wind = [p for p in xs if (p.get("swell_frac") or 0) <= SWELL_FRAC]
    sec_ps = [[] for _ in range(SECTORS)]
    cell_ps = [[[] for _ in range(len(SPEED_EDGES) + 1)] for _ in range(SECTORS)]
    for p in wind:
        sec_ps[sector(p["wdir"])].append(p)
        cell_ps[sector(p["wdir"])][speed_bin(p["u"])].append(p)
    sec = [shrink(v, g) for v in sec_ps]
    table = [[shrink(cell_ps[s][b], sec[s]) for b in range(len(SPEED_EDGES) + 1)] for s in range(SECTORS)]
    c = lambda x: round(min(max(x, CLIP[0]), CLIP[1]), 3)  # noqa: E731
    return {"global": c(g), "swell": c(shrink(swell, g)), "n_swell": len(swell), "table": [[c(x) for x in row] for row in table],
            "n": [[len(v) for v in row] for row in cell_ps]}


# Wave height: a small regression (model output statistics), buoy Hs ~ these terms. The model's height alone,
# scaled by direction, isn't enough: the wind speed itself carries what the growth curves miss (they grow seas
# too slowly here, which is also why the model's peaks come late), and light-wind hours have a background sea.
FEATURES = ["1", "m", "m cos d", "m sin d", "cos d", "sin d", "u", "u^2/10", "m swell"]
RIDGE = 1e-3
RATIO_CLIP = (0.33, 3.0)     # applied as a factor on the model's sea (so sheltered spots stay sheltered)
MIN_M = 0.05                 # below this the model has no sea to scale: left as is


def features(m, u, wdir, swell_frac):
    d = math.radians(wdir)
    return [1.0, m, m * math.cos(d), m * math.sin(d), math.cos(d), math.sin(d), u, u * u / 10, m * (swell_frac or 0.0)]


def _regression(pairs):
    xs = [p for p in pairs if p.get("obs_hs") is not None and p.get("pred_hs") is not None and p.get("u") is not None and p.get("wdir") is not None]
    if len(xs) < 50:
        return None
    X = np.array([features(p["pred_hs"], p["u"], p["wdir"], p.get("swell_frac")) for p in xs])
    y = np.array([p["obs_hs"] for p in xs])
    A = X.T @ X + RIDGE * len(xs) * np.diag([0.0] + [1.0] * (X.shape[1] - 1))
    coef = np.linalg.solve(A, X.T @ y)
    return {"kind": "regression", "features": FEATURES, "coef": [round(float(c), 5) for c in coef], "ratio_clip": list(RATIO_CLIP), "min_m": MIN_M}


def fit_pairs(pairs):
    hs, tp = _regression(pairs), _shrunk_table(pairs, "obs_tp", "pred_tp")
    return {"version": 2, "sectors": SECTORS, "speed_edges": list(SPEED_EDGES), "swell_frac": SWELL_FRAC, "clip": list(CLIP),
            "hs": hs, "tp": tp, "hours": len(pairs)}


def factor(corr, wdir, u, swell_frac=0.0, what="hs", m=None):
    """Correction factor for this hour's model sea (1.0 when there is no correction). Height needs the model's
    height m at the place (the regression gives the buoy-like height, applied as a ratio)."""
    t = (corr or {}).get(what)
    if not t or wdir is None or u is None:
        return 1.0
    if t.get("kind") == "regression":
        if m is None or m < t.get("min_m", MIN_M):
            return 1.0
        est = sum(c * f for c, f in zip(t["coef"], features(m, u, wdir, swell_frac)))
        lo, hi = t.get("ratio_clip", RATIO_CLIP)
        return min(max(est / m, lo), hi)
    if (swell_frac or 0) > corr.get("swell_frac", SWELL_FRAC):
        return t["swell"]
    return t["table"][sector(wdir)][speed_bin(u)]


def apply_pairs(corr, pairs):
    return [p["pred_hs"] * factor(corr, p["wdir"], p["u"], p.get("swell_frac"), m=p["pred_hs"]) for p in pairs]


def skill(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    if len(o) < 3:
        return None
    big = o >= 1.0
    return {"hours": int(len(o)), "bias_m": round(float(np.mean(p - o)), 3), "rmse_m": round(float(np.sqrt(np.mean((p - o) ** 2))), 3),
            "mae_m": round(float(np.mean(np.abs(p - o))), 3), "r": round(float(np.corrcoef(o, p)[0, 1]), 3) if np.std(p) > 0 and np.std(o) > 0 else None,
            "big_hours": int(big.sum()), "big_bias_m": round(float(np.mean(p[big] - o[big])), 3) if big.any() else None,
            "big_rmse_m": round(float(np.sqrt(np.mean((p[big] - o[big]) ** 2))), 3) if big.any() else None}


def best_lag(pairs, max_lag=12):
    """Hours the model runs early (<0) or late (>0) against the buoy: the shift that best lines the two up."""
    by = {}
    for p in pairs:
        by.setdefault(p["buoy"], {})[dt.datetime.fromisoformat(p["t"])] = (p["obs_hs"], p["pred_hs"])
    best = None
    for lag in range(-max_lag, max_lag + 1):
        o, m = [], []
        for d in by.values():
            for t, (ob, _) in d.items():
                q = d.get(t + dt.timedelta(hours=lag))
                if q is not None:
                    o.append(ob)
                    m.append(q[1])
        if len(o) > 50:
            r = float(np.corrcoef(o, m)[0, 1])
            if best is None or r > best[1]:
                best = (lag, r)
    return None if best is None else {"lag_h": best[0], "r": round(best[1], 3)}


def cross_validate(pairs):
    years = sorted({p["t"][:4] for p in pairs})
    rows, all_o, all_raw, all_l, by_buoy = [], [], [], [], {}
    for y in years:
        train = [p for p in pairs if p["t"][:4] != y]
        test = [p for p in pairs if p["t"][:4] == y]
        if len(train) < 200 or len(test) < 50:
            continue
        c = fit_pairs(train)
        lp = apply_pairs(c, test)
        o = [p["obs_hs"] for p in test]
        r = [p["pred_hs"] for p in test]
        rows.append({"year": y, "raw": skill(o, r), "learned": skill(o, lp)})
        all_o += o
        all_raw += r
        all_l += lp
        for p, q in zip(test, lp):
            b = by_buoy.setdefault(p["buoy"], ([], [], []))
            b[0].append(p["obs_hs"])
            b[1].append(p["pred_hs"])
            b[2].append(q)
    if not rows:
        return {"years": [], "note": "not enough years to hold one out"}
    better = sum(1 for x in rows if x["learned"]["rmse_m"] < x["raw"]["rmse_m"])
    tot_raw, tot_l = skill(all_o, all_raw), skill(all_o, all_l)
    held = [p for r in rows for p in pairs if p["t"][:4] == r["year"]]
    sel = [i for i, p in enumerate(held) if p["obs_hs"] >= 1.0 and p.get("wdir") is not None and _is_north(p["wdir"])]
    north_big = {"raw": skill([all_o[i] for i in sel], [all_raw[i] for i in sel]), "learned": skill([all_o[i] for i in sel], [all_l[i] for i in sel])}
    return {"years": rows, "overall": {"raw": tot_raw, "learned": tot_l}, "north_big": north_big,
            "by_buoy": {b: {"raw": skill(v[0], v[1]), "learned": skill(v[0], v[2])} for b, v in by_buoy.items()},
            "years_better": better, "years_scored": len(rows),
            "improves": bool(tot_l["rmse_m"] < 0.97 * tot_raw["rmse_m"] and better * 2 > len(rows))}


def by_year(pairs):
    """Mean buoy / mean model per buoy and year: shows instrument changes and drifts."""
    acc = {}
    for p in pairs:
        a = acc.setdefault(p["buoy"], {}).setdefault(p["t"][:4], [0, 0.0, 0.0])
        a[0] += 1
        a[1] += p["obs_hs"]
        a[2] += p["pred_hs"]
    return {b: {y: {"hours": n, "obs_m": round(o / n, 3), "model_m": round(m / n, 3), "ratio": round(o / m, 3) if m else None}
                for y, (n, o, m) in sorted(ys.items())} for b, ys in acc.items()}


def fit_dir(d):
    paths = sorted(glob.glob(os.path.join(d, "backfill", "pairs_*.csv.gz")))
    every = read_pairs(paths)
    if not every:
        raise SystemExit("no backfill pairs under " + d)
    pairs = [p for p in every if p["t"] >= FIT_FROM.get(p["buoy"], "")]
    corr = fit_pairs(pairs)
    corr["generated"] = dt.datetime.now(UTC).isoformat(timespec="seconds")
    corr["method"] = ("height: regression of buoy Hs on the model's Hs (and its variation with wind direction), wind direction, 10 m wind "
                      "speed and the swell share, applied as a factor on the model's sea (%.2f-%.1f); period: least-squares factor by wind "
                      "direction x speed, shrunk toward broader averages" % RATIO_CLIP)
    corr["data"] = {b: sorted({p["t"][:4] for p in pairs if p["buoy"] == b}) for b in BUOYS}
    corr["hours_by_buoy"] = {b: sum(1 for p in pairs if p["buoy"] == b) for b in BUOYS}
    corr["fit_from"] = FIT_FROM
    corr["left_out_hours"] = len(every) - len(pairs)
    corr["by_year"] = by_year(every)
    corr["cv"] = cross_validate(pairs)
    corr["lag"] = best_lag(pairs)
    corr["era5_filled_hours"] = sum(1 for p in pairs if str(p.get("wind_src", "")).startswith("era5"))
    cp = os.path.join(d, "era5_climate.json")
    if os.path.exists(cp):
        clim = json.load(open(cp))
        corr["era5"] = {b: {"years": v["years"], "totals": v["totals"], "top_storms": v["top_storms"][:5], "monthly_hs_m": v.get("monthly_hs_m")}
                        for b, v in clim["buoys"].items()}
    corr["in_sample"] = {"raw": skill([p["obs_hs"] for p in pairs], [p["pred_hs"] for p in pairs]),
                         "learned": skill([p["obs_hs"] for p in pairs], apply_pairs(corr, pairs))}
    with open(os.path.join(d, "correction.json"), "w") as fh:
        json.dump(corr, fh, indent=1)
    return corr


def load_correction(path=None):
    path = path or os.environ.get("LEARN_PATH")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            c = json.load(fh)
        return c if c.get("hs") else None
    except Exception as e:  # noqa: BLE001
        log.warning("learned correction %s unreadable: %s", path, e)
        return None


# ---------------------------------------------------------------------------- 3. scorecard
def wind_driven(buoy, series, t0, n):
    """The open-lake wave model run on another model's 10 m wind at a buoy: spun up from the first hour of the
    series (the run's start, usually 6-12 h before t0), then n hours from t0. series: {"t", "wspd", "wdir"}."""
    if not series or not series.get("t"):
        return None
    lat, lon = BUOYS[buoy]
    table = sw.fetch_table(lat, lon, 90)
    fetch = lambda d: sw.interp_dir(table, d) * 1000  # noqa: E731
    by = {dt.datetime.fromisoformat(t): (u, d) for t, u, d in zip(series["t"], series["wspd"], series["wdir"]) if u is not None and d is not None}
    if not by:
        return None
    t, t1 = min(by), t0 + dt.timedelta(hours=n - 1)
    if t > t0:
        return None                                      # the run starts after the forecast: no spin-up
    sea, out = sw.SeaState(), []
    while t <= t1:
        u, d = by.get(t, (None, None))
        sea.step(u, d, fetch)
        if t >= t0:
            out.append(round(sea.total_hs(), 2) if t in by else None)
        t += dt.timedelta(hours=1)
    return out if any(v is not None for v in out) else None


def log_forecast(sup, out_dir, glwu=None, winds=None):
    """Append what this run forecast at the buoys to scorecard/forecasts-YYYY-MM.jsonl. winds: other models' wind at
    the buoys ({"gfs": {buoy: series}, "wn3": ...}), run through the same wave model."""
    bf = (sup or {}).get("buoy_fc") or {}
    if not bf:
        return None
    entry = {"issued": sup["start"], "generated": sup.get("generated"), "learned": bool(sup.get("learned")), "buoys": {}}
    for b, d in bf.items():
        e = {k: d.get(k) for k in ("t0", "raw", "learned", "product", "product_learned", "nws", "tp_raw", "tp_learned") if d.get(k) is not None}
        g = ((glwu or {}).get("points") or {}).get(b)
        if g and g.get("hs"):
            t0 = dt.datetime.fromisoformat(d["t0"])
            gt = {dt.datetime.fromisoformat(t): v for t, v in zip(g["t"], g["hs"])}
            e["glwu"] = [None if gt.get(t0 + dt.timedelta(hours=k)) is None else round(gt[t0 + dt.timedelta(hours=k)], 2)
                         for k in range(len(d.get("raw") or []))]
            e["glwu_cycle"] = glwu.get("cycle")
        for m, pts in (winds or {}).items():
            try:
                v = wind_driven(b, (pts or {}).get(b), dt.datetime.fromisoformat(d["t0"]), len(d.get("raw") or []))
            except Exception as ex:  # noqa: BLE001
                log.warning("%s at %s: %s", m, b, ex)
                v = None
            if v:
                e[m] = v
        entry["buoys"][b] = e
    os.makedirs(os.path.join(out_dir, "scorecard"), exist_ok=True)
    path = os.path.join(out_dir, "scorecard", "forecasts-%s.jsonl" % sup["start"][:7])
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return path


SPOT_FIELDS = ("wind", "gust", "dir", "offshore", "hs_offshore", "hs_paddle", "tp", "rating", "sky")


def log_spots(sup, out_dir, hours=6):
    """Append this run's beach forecasts (first `hours` hours) to scorecard/spots-YYYY-MM.jsonl, so a photo or
    paddle report from any later day can be checked against what the page said beforehand (tools/spotcheck.py)."""
    spots = (sup or {}).get("spots") or []
    if not spots or not sup.get("start"):
        return None
    entry = {"issued": sup["start"], "generated": sup.get("generated"), "spots": {}}
    for s in spots:
        rows = []
        for h in (s.get("hourly") or [])[:hours]:
            r = {"t": h["t"]}
            r.update({k: h[k] for k in SPOT_FIELDS if h.get(k) is not None})
            lr = h.get("learned") or {}
            if lr.get("rating"):
                r["learned_rating"] = lr["rating"]
                r["learned_hs_paddle"] = lr.get("hs_paddle")
            rows.append(r)
        if rows:
            entry["spots"][s["id"]] = {"lat": s.get("lat"), "lon": s.get("lon"), "hourly": rows}
    if not entry["spots"]:
        return None
    os.makedirs(os.path.join(out_dir, "scorecard"), exist_ok=True)
    path = os.path.join(out_dir, "scorecard", "spots-%s.jsonl" % sup["start"][:7])
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return path


def score(out_dir, obs_hourly=None, days=45, now=None):
    """Line up logged forecasts with what the buoys measured; errors by method and lead time."""
    now = now or dt.datetime.now(UTC)
    if obs_hourly is None:
        obs_hourly = {}
        for b in BUOYS:
            rows = sw.fetch_ndbc_rows(b)
            obs_hourly[b] = {h: o["wvht"] for h, o in hourly_means(rows, ("wvht",)).items() if "wvht" in o}
    entries = []
    for p in sorted(glob.glob(os.path.join(out_dir, "scorecard", "forecasts-*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    cut = now - dt.timedelta(days=days)
    acc = {m: {i: ([], []) for i in range(len(LEAD_BUCKETS))} for m in METHODS}
    acc_b = {b: {m: ([], []) for m in METHODS} for b in BUOYS}
    issued = 0
    for e in entries:
        if dt.datetime.fromisoformat(e["issued"]) < cut:
            continue
        issued += 1
        for b, d in e["buoys"].items():
            t0 = dt.datetime.fromisoformat(d["t0"])
            ob = obs_hourly.get(b, {})
            for m in METHODS:
                vals = d.get(m) or []
                for k, v in enumerate(vals):
                    t = t0 + dt.timedelta(hours=k)
                    if v is None or t not in ob or t > now:
                        continue
                    bi = next((i for i, (lo, hi) in enumerate(LEAD_BUCKETS) if lo <= k <= hi), None)
                    if bi is None:
                        continue
                    acc[m][bi][0].append(ob[t])
                    acc[m][bi][1].append(v)
                    acc_b[b][m][0].append(ob[t])
                    acc_b[b][m][1].append(v)
    stats = {m: [skill(*acc[m][i]) for i in range(len(LEAD_BUCKETS))] for m in METHODS}
    return {"version": 1, "generated": now.isoformat(timespec="seconds"), "window_days": days, "forecasts": issued,
            "buckets": ["%d h" % lo if lo == hi else "%d-%d h" % (lo, hi) for lo, hi in LEAD_BUCKETS],
            "methods": [m for m in METHODS if any(stats[m])], "labels": METHOD_LABELS,
            "stats": {m: v for m, v in stats.items() if any(v)},
            "by_buoy": {b: {m: skill(*v) for m, v in d.items() if skill(*v)} for b, d in acc_b.items()}}


def summarize(corr, card):
    """What the page needs: the correction's skill and the live scorecard (small)."""
    if not corr and not card:
        return None
    out = {}
    if corr:
        cv = corr.get("cv") or {}
        out["correction"] = {k: corr.get(k) for k in ("generated", "method", "hours", "data", "hours_by_buoy", "lag")}
        out["correction"].update(cv_overall=cv.get("overall"), years_better=cv.get("years_better"), years_scored=cv.get("years_scored"),
                                 improves=cv.get("improves"), by_buoy=cv.get("by_buoy"), fit_from=corr.get("fit_from"),
                                 north_big=cv.get("north_big"), era5=corr.get("era5"), era5_filled_hours=corr.get("era5_filled_hours"))
    if card:
        out["scorecard"] = card
    return out


def parse_years(s, now=None):
    now = now or dt.datetime.now(UTC)
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if s.startswith("last"):
        n = int(s[4:])
        return list(range(now.year - n + 1, now.year + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("backfill"); p.add_argument("--out", required=True); p.add_argument("--years", default="2012-%d" % dt.datetime.now(UTC).year)
    p.add_argument("--era5", help="cached ERA5 (app.era5) directory: fills missing wind, 35-year storm climate")
    p = sub.add_parser("fit"); p.add_argument("--dir", required=True)
    p = sub.add_parser("log"); p.add_argument("--sup", required=True); p.add_argument("--glwu"); p.add_argument("--dir", required=True)
    p.add_argument("--earth", help="earth.json (GFS wind at the buoys)"); p.add_argument("--wn3", help="wn3.json (WeatherNext 3 wind at the buoys)")
    p = sub.add_parser("score"); p.add_argument("--dir", required=True); p.add_argument("--days", type=int, default=45)
    a = ap.parse_args()
    if a.cmd == "backfill":
        s = backfill(a.out, parse_years(a.years), era5_dir=a.era5)
        print(json.dumps(s))
    elif a.cmd == "fit":
        c = fit_dir(a.dir)
        cv = c["cv"]
        print("hours", c["hours"], json.dumps(c["hours_by_buoy"]), "left out", c["left_out_hours"], json.dumps(c["fit_from"]), "lag", json.dumps(c["lag"]))
        for b, ys in c["by_year"].items():
            print("  buoy/model by year", b, " ".join("%s:%.2f" % (y, v["ratio"]) for y, v in ys.items()))
        print("held-out years:", json.dumps(cv.get("overall")))
        for r in cv.get("years", []):
            print(" ", r["year"], "raw rmse", r["raw"]["rmse_m"], "learned", r["learned"]["rmse_m"], "big-wave bias", r["raw"]["big_bias_m"], "->", r["learned"]["big_bias_m"])
        print("improves:", cv.get("improves"), "by buoy", json.dumps(cv.get("by_buoy")))
        print("north-wind hours with 1 m+ waves (held out):", json.dumps(cv.get("north_big")))
        print("hours with ERA5-filled wind in the fit:", c.get("era5_filled_hours"))
        for b, v in (c.get("era5") or {}).items():
            t1 = v["totals"].get("1m", {})
            print("  ERA5 %s %s: 1 m+ model hours %d, buoy out for %d (%.0f%%); north-wind 1 m+ %d, buoy saw %d (%.0f%%)" % (
                b, v["years"], t1.get("model_hours", 0), t1.get("buoy_saw", 0), 100 * t1.get("buoy_saw", 0) / max(t1.get("model_hours", 0), 1),
                t1.get("north_model_hours", 0), t1.get("north_buoy_saw", 0), 100 * t1.get("north_buoy_saw", 0) / max(t1.get("north_model_hours", 0), 1)))
            for s_ in v["top_storms"][:3]:
                print("    storm", s_["peak_t"][:13], "peak %.1f m" % s_["peak_hs"], "wind %s m/s from %s" % (s_.get("peak_wind"), s_.get("peak_dir")), "buoy saw %d/%d h" % (s_["observed_hours"], s_["hours"]))
        print("height regression:", " ".join("%s %+.4f" % fc for fc in zip(FEATURES, c["hs"]["coef"])))
        for u_ in (3, 6, 9, 12):
            print("  factor at %2d m/s (model 0.5 m)" % u_, " ".join("%d:%.2f" % (d_, factor(c, d_, u_, 0.0, m=0.5)) for d_ in range(0, 360, 45)))
    elif a.cmd == "log":
        sup = json.load(open(a.sup))
        rd = lambda p: json.load(open(p)) if p and os.path.exists(p) and os.path.getsize(p) else None  # noqa: E731
        g, ea, wn = rd(a.glwu), rd(a.earth), rd(a.wn3)
        winds = {}
        if ea and (ea.get("gfs") or {}).get("points"):
            winds["gfs"] = ea["gfs"]["points"]
        if wn and wn.get("points") and wn.get("init"):
            winds["wn3"] = {b: {"t": p["t"], "wspd": p.get("wspd"), "wdir": p.get("wdir")} for b, p in wn["points"].items() if p.get("wspd")}
        print(log_forecast(sup, a.dir, g, winds) or "no buoy forecasts in sup.json")
        print(log_spots(sup, a.dir) or "no beach forecasts in sup.json")
    elif a.cmd == "score":
        card = score(a.dir, days=a.days)
        with open(os.path.join(a.dir, "scorecard.json"), "w") as fh:
            json.dump(card, fh, indent=1)
        print(json.dumps({m: [s and s["rmse_m"] for s in v] for m, v in card["stats"].items()}))
