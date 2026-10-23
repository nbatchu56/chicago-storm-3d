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


def run_buoy(BUOY, days, get, cache):
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
                if st == "CHII2":
                    u *= 1.0                              # crib anemometer is near 10 m already
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
    return {"buoy": BUOY[0], "window": {"start": t0.isoformat(), "end": t1.isoformat()}, "stats": stats, "series": out}


def run(days=21, get=sw.http_get):
    cache = {}
    res = [run_buoy(b, days, get, cache) for b in BUOYS]
    return {"version": 2, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "buoys": res,
            "model": "sup_waves.SeaState (CEM growth, decay %.1f h, swell on shifts) driven by observed wind" % sw.DECAY_H}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--out", default="validate.json")
    a = ap.parse_args()
    d = run(a.days)
    json.dump(d, open(a.out, "w"), separators=(",", ":"))
    for b in d["buoys"]:
        print(b["buoy"], json.dumps(b.get("stats") or b.get("error")))
