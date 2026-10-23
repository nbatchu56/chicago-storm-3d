"""
Check the wave model against the Chicago buoy.

Drives the same wave-memory model the paddling forecast uses (CEM growth over the real fetch, decay, swell
on wind shifts) with observed wind only, hour by hour, and compares the predicted significant wave height
at buoy 45198 (1.3 mi off Navy Pier, ~9 m deep) with what the buoy measured. No buoy waves are fed in.

    python -m app.validate --days 21 --out validate.json
"""

import argparse
import datetime as dt
import json
import math

import numpy as np

from . import sup_waves as sw

UTC = dt.timezone.utc
BUOYS = [("45198", 41.892, -87.563), ("45174", 42.135, -87.655)]   # Chicago, Wilmette
WIND = ["45198", "CHII2", "45174", "45007"]       # first with data wins, hour by hour


def hourly(rows, field):
    out = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        h = r["t"].replace(minute=0, second=0, microsecond=0)
        out.setdefault(h, []).append(v)
    return {h: float(np.mean(v)) for h, v in out.items()}


def hourly_dir(rows, field="wdir"):
    out = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        h = r["t"].replace(minute=0, second=0, microsecond=0)
        out.setdefault(h, []).append(math.radians(v))
    return {h: (math.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) + 360) % 360 for h, a in out.items()}


def rows_for(st, t0, t1, get):
    from .history import _ndbc_rows_for
    return _ndbc_rows_for(st, t0, t1, lambda url, timeout=30: get(url, "text/plain", timeout))


BUOY_WIND = {"45198", "45174", "45177", "45007"}      # anemometers ~3-4 m above the water


def run_buoy(BUOY, days, get, cache, fb=1.0, fc=1.0, series=True):
    now = dt.datetime.now(UTC)
    for st in [BUOY[0]] + WIND:
        if st not in cache:
            cache[st] = rows_for(st, now - dt.timedelta(days=days + 1), now, get)
    rows = cache
    obs_hs = hourly(rows[BUOY[0]], "wvht")
    if not obs_hs:
        return {"buoy": BUOY[0], "error": "no wave data in the realtime file or the monthly archive"}
    t1 = max(obs_hs); t0 = max(min(obs_hs), t1 - dt.timedelta(days=days))
    winds = {st: (hourly(rows[st], "wspd"), hourly_dir(rows[st])) for st in WIND}
    table = sw.fetch_table(BUOY[1], BUOY[2], 90)
    fetch = lambda d: sw.interp_dir(table, d) * 1000  # noqa: E731
    sea = sw.SeaState()
    t, out, used = t0 - dt.timedelta(hours=12), [], {}
    while t <= t1:
        u = d = None
        for st in WIND:
            if t in winds[st][0] and t in winds[st][1]:
                u, d = winds[st][0][t], winds[st][1][t]
                u *= fb if st in BUOY_WIND else fc              # to 10 m wind
                used[st] = used.get(st, 0) + 1
                break
        sea.step(u, d, fetch)
        if t >= t0:
            comps = sea.components()
            pred = math.sqrt(sum(c["hs"] ** 2 for c in comps))
            out.append({"t": t.isoformat(), "obs": obs_hs.get(t), "pred": round(pred, 3), "wind": u, "dir": d})
        t += dt.timedelta(hours=1)
    pairs = np.array([(r["obs"], r["pred"]) for r in out if r["obs"] is not None])
    o, p = pairs[:, 0], pairs[:, 1]
    big = o >= 1.0
    stats = {"hours": int(len(o)), "bias_m": round(float(np.mean(p - o)), 3), "rmse_m": round(float(np.sqrt(np.mean((p - o) ** 2))), 3),
             "r": round(float(np.corrcoef(o, p)[0, 1]), 3) if len(o) > 2 else None,
             "scatter_index": round(float(np.sqrt(np.mean((p - o - np.mean(p - o)) ** 2)) / max(np.mean(o), 1e-3)), 3),
             "obs_peak": {"m": float(o.max()), "t": next(r["t"] for r in out if r["obs"] == o.max())},
             "pred_peak": {"m": float(p.max()), "t": next(r["t"] for r in out if r["pred"] == p.max())},
             "big_waves_hours": int(big.sum()), "big_waves_bias_m": round(float(np.mean(p[big] - o[big])), 3) if big.any() else None,
             "wind_hours_by_station": used}
    return {"buoy": BUOY[0], "window": {"start": t0.isoformat(), "end": t1.isoformat()}, "stats": stats, **({"series": out} if series else {})}


def tune(days, get, cache):
    """Grid search: wind height factors (buoy, crib) and leftover-wave decay time, scored on both buoys
    by RMSE plus the bias on big-wave hours (what matters on a rough day)."""
    rows = []
    for fb in (1.0, 1.1, 1.2, 1.3):
        for fc in (0.9, 1.0, 1.1):
            for dh in (3.0, 4.0, 6.0, 8.0):
                sw.DECAY_H = dh
                res = [run_buoy(b, days, get, cache, fb, fc, series=False) for b in BUOYS]
                st = [r["stats"] for r in res if "stats" in r]
                if not st:
                    continue
                score = sum(x["rmse_m"] + 0.5 * abs(x["big_waves_bias_m"] or 0) for x in st) / len(st)
                rows.append({"fb": fb, "fc": fc, "decay_h": dh, "score": round(score, 4),
                             **{r["buoy"]: {k: r["stats"][k] for k in ("rmse_m", "bias_m", "r", "big_waves_bias_m")} for r in res if "stats" in r}})
    sw.DECAY_H = 4.0
    rows.sort(key=lambda r: r["score"])
    return rows


def run(days=21, get=sw.http_get, do_tune=True):
    cache = {}
    base = [run_buoy(b, days, get, cache) for b in BUOYS]
    grid = tune(days, get, cache) if do_tune else []
    best = grid[0] if grid else None
    tuned = []
    if best:
        sw.DECAY_H = best["decay_h"]
        tuned = [run_buoy(b, days, get, cache, best["fb"], best["fc"]) for b in BUOYS]
        sw.DECAY_H = 4.0
    return {"version": 3, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "buoys": base, "tuned": tuned, "best": best, "grid_top": grid[:8],
            "model": "sup_waves.SeaState (CEM growth, decay %.1f h, swell on shifts) driven by observed wind" % sw.DECAY_H}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--out", default="validate.json")
    a = ap.parse_args()
    d = run(a.days)
    json.dump(d, open(a.out, "w"), separators=(",", ":"))
    for b in d["buoys"]:
        print("as is", b["buoy"], json.dumps(b.get("stats") or b.get("error")))
    print("best", json.dumps(d["best"]))
    for b in d["tuned"]:
        print("tuned", b["buoy"], json.dumps(b.get("stats")))
