"""
Learned wave correction (experimental): learn how the open-lake wave model is usually wrong, from the buoys.

Three parts, all writing to the `learn` branch:

1. backfill  Rerun the wave model (same SeaState the paddling forecast uses) over years of NDBC archive at the
             Wilmette (45174) and Chicago (45198) buoys, driven by observed wind only, and keep every hour's
             model-vs-buoy pair. Because the model needs only wind, this gives years of training data at once.
2. fit       Learn a correction factor for wave height and period by wind direction (12 sectors) x wind speed
             (4 bins), plus one for swell-dominated hours (old waves after the wind drops or shifts), as the
             average log(buoy / model) with shrinkage toward broader averages where data is thin. Scored by
             leaving one year out at a time; the page shows it only as an experimental option.
3. log/score Every snapshot records what the forecast said at the two buoys for the next 48 h (raw model,
             learned, the live-buoy-nudged product, the NWS wave grid and NOAA GLWU when available); `score`
             lines those up with what the buoys measured and reports errors by lead time.

    python -m app.learn backfill --out learn/ [--years 2012-2026]
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
SWELL_FRAC = 0.5                             # share of wave energy in leftover/swell above this -> "swell" factor
LEAD_BUCKETS = [(0, 0), (1, 3), (4, 6), (7, 12), (13, 24), (25, 48)]
METHODS = ["raw", "learned", "product", "product_learned", "nws", "glwu"]
METHOD_LABELS = {"raw": "model", "learned": "model + learned", "product": "forecast (buoy-nudged)",
                 "product_learned": "forecast + learned", "nws": "NWS wave grid", "glwu": "NOAA GLWU"}


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
    while t <= t1:
        u = d = src = None
        for st in WIND_ORDER[buoy]:
            o = hourly_by_station.get(st, {}).get(t)
            if o and "wspd" in o and "wdir" in o:
                u, d, src = o["wspd"], o["wdir"], st
                break
        ob = own.get(t, {})
        air = ob.get("atmp", (hourly_by_station.get("CHII2", {}).get(t) or {}).get("atmp"))
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


def backfill(out_dir, years, get=get_bytes, now=None):
    os.makedirs(os.path.join(out_dir, "backfill"), exist_ok=True)
    stations = sorted({s for v in WIND_ORDER.values() for s in v})
    summary = {}
    for y in years:
        hourly = {st: hourly_means(station_rows(st, [y], get, now)) for st in stations}
        for b in BUOYS:
            pairs = hindcast(b, hourly)
            if not pairs:
                continue
            write_pairs(os.path.join(out_dir, "backfill", f"pairs_{b}_{y}.csv.gz"), pairs)
            summary.setdefault(b, {})[str(y)] = len(pairs)
            log.info("%s %s: %d hours", b, y, len(pairs))
    return summary


# ---------------------------------------------------------------------------- 2. fit
def sector(wdir):
    return int(((wdir + 180.0 / SECTORS) % 360) // (360.0 / SECTORS))


def speed_bin(u):
    return sum(u >= e for e in SPEED_EDGES)


def _usable(p, key_obs, key_pred):
    return p.get(key_obs) is not None and p.get(key_pred) is not None and p[key_obs] >= (MIN_HS if key_obs == "obs_hs" else 1.0) \
        and p[key_pred] >= (MIN_HS if key_pred == "pred_hs" else 1.0) and p.get("u") is not None and p.get("wdir") is not None


def _shrunk_table(pairs, key_obs, key_pred):
    """log(obs/pred): global -> swell / per-sector -> sector x speed, each shrunk toward its parent."""
    ys = [(p, math.log(p[key_obs] / p[key_pred])) for p in pairs if _usable(p, key_obs, key_pred)]
    if not ys:
        return None
    g = float(np.mean([y for _, y in ys]))
    swell = [y for p, y in ys if (p.get("swell_frac") or 0) > SWELL_FRAC]
    wind = [(p, y) for p, y in ys if (p.get("swell_frac") or 0) <= SWELL_FRAC]
    shrink = lambda vals, parent: (sum(vals) + SHRINK_K * parent) / (len(vals) + SHRINK_K)  # noqa: E731
    sw_f = shrink(swell, g)
    sec_vals = [[] for _ in range(SECTORS)]
    cell_vals = [[[] for _ in range(len(SPEED_EDGES) + 1)] for _ in range(SECTORS)]
    for p, y in wind:
        s, b = sector(p["wdir"]), speed_bin(p["u"])
        sec_vals[s].append(y)
        cell_vals[s][b].append(y)
    sec = [shrink(v, g) for v in sec_vals]
    table = [[shrink(cell_vals[s][b], sec[s]) for b in range(len(SPEED_EDGES) + 1)] for s in range(SECTORS)]
    lo, hi = math.log(CLIP[0]), math.log(CLIP[1])
    c = lambda x: round(math.exp(min(max(x, lo), hi)), 3)  # noqa: E731
    return {"global": c(g), "swell": c(sw_f), "n_swell": len(swell), "table": [[c(x) for x in row] for row in table],
            "n": [[len(v) for v in row] for row in cell_vals]}


def fit_pairs(pairs):
    hs, tp = _shrunk_table(pairs, "obs_hs", "pred_hs"), _shrunk_table(pairs, "obs_tp", "pred_tp")
    return {"version": 1, "sectors": SECTORS, "speed_edges": list(SPEED_EDGES), "swell_frac": SWELL_FRAC, "clip": list(CLIP),
            "hs": hs, "tp": tp, "hours": len(pairs)}


def factor(corr, wdir, u, swell_frac=0.0, what="hs"):
    """Correction factor for this hour (1.0 when there is no correction)."""
    t = (corr or {}).get(what)
    if not t or wdir is None or u is None:
        return 1.0
    if (swell_frac or 0) > corr.get("swell_frac", SWELL_FRAC):
        return t["swell"]
    return t["table"][sector(wdir)][speed_bin(u)]


def apply_pairs(corr, pairs):
    out = []
    for p in pairs:
        f = factor(corr, p["wdir"], p["u"], p.get("swell_frac"))
        out.append(p["pred_hs"] * f)
    return out


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
    return {"years": rows, "overall": {"raw": tot_raw, "learned": tot_l},
            "by_buoy": {b: {"raw": skill(v[0], v[1]), "learned": skill(v[0], v[2])} for b, v in by_buoy.items()},
            "years_better": better, "years_scored": len(rows),
            "improves": bool(tot_l["rmse_m"] < 0.97 * tot_raw["rmse_m"] and better * 2 > len(rows))}


def fit_dir(d):
    paths = sorted(glob.glob(os.path.join(d, "backfill", "pairs_*.csv.gz")))
    pairs = read_pairs(paths)
    if not pairs:
        raise SystemExit("no backfill pairs under " + d)
    corr = fit_pairs(pairs)
    corr["generated"] = dt.datetime.now(UTC).isoformat(timespec="seconds")
    corr["method"] = ("log(buoy/model) by wind direction (12 sectors) x 10 m wind speed (<4, 4-7, 7-10, >10 m/s), and for "
                      "swell-dominated hours, shrunk toward the sector and lake-wide averages (k = %d h); clipped %.1f-%.1f" % (SHRINK_K, *CLIP))
    corr["data"] = {b: sorted({p["t"][:4] for p in pairs if p["buoy"] == b}) for b in BUOYS}
    corr["hours_by_buoy"] = {b: sum(1 for p in pairs if p["buoy"] == b) for b in BUOYS}
    corr["cv"] = cross_validate(pairs)
    corr["lag"] = best_lag(pairs)
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
def log_forecast(sup, out_dir, glwu=None):
    """Append what this run forecast at the buoys to scorecard/forecasts-YYYY-MM.jsonl."""
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
        entry["buoys"][b] = e
    os.makedirs(os.path.join(out_dir, "scorecard"), exist_ok=True)
    path = os.path.join(out_dir, "scorecard", "forecasts-%s.jsonl" % sup["start"][:7])
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
                                 improves=cv.get("improves"), swell=corr["hs"]["swell"], global_factor=corr["hs"]["global"],
                                 table=corr["hs"]["table"], speed_edges=corr.get("speed_edges"))
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
    p = sub.add_parser("fit"); p.add_argument("--dir", required=True)
    p = sub.add_parser("log"); p.add_argument("--sup", required=True); p.add_argument("--glwu"); p.add_argument("--dir", required=True)
    p = sub.add_parser("score"); p.add_argument("--dir", required=True); p.add_argument("--days", type=int, default=45)
    a = ap.parse_args()
    if a.cmd == "backfill":
        s = backfill(a.out, parse_years(a.years))
        print(json.dumps(s))
    elif a.cmd == "fit":
        c = fit_dir(a.dir)
        cv = c["cv"]
        print("hours", c["hours"], json.dumps(c["hours_by_buoy"]), "lag", json.dumps(c["lag"]))
        print("held-out years:", json.dumps(cv.get("overall")))
        for r in cv.get("years", []):
            print(" ", r["year"], "raw rmse", r["raw"]["rmse_m"], "learned", r["learned"]["rmse_m"], "big-wave bias", r["raw"]["big_bias_m"], "->", r["learned"]["big_bias_m"])
        print("improves:", cv.get("improves"), "global factor", c["hs"]["global"], "swell", c["hs"]["swell"])
        for s_, row in enumerate(c["hs"]["table"]):
            print("  from %3d deg" % (s_ * 30), row, c["hs"]["n"][s_])
    elif a.cmd == "log":
        sup = json.load(open(a.sup))
        g = json.load(open(a.glwu)) if a.glwu and os.path.exists(a.glwu) else None
        print(log_forecast(sup, a.dir, g) or "no buoy forecasts in sup.json")
    elif a.cmd == "score":
        card = score(a.dir, days=a.days)
        with open(os.path.join(a.dir, "scorecard.json"), "w") as fh:
            json.dump(card, fh, indent=1)
        print(json.dumps({m: [s and s["rmse_m"] for s in v] for m, v in card["stats"].items()}))
