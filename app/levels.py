"""
Hourly Lake Michigan water level at Chicago (NOAA CO-OPS Calumet Harbor 9087044, IGLD85), for the page to raise or
lower the lake for the time on screen: storm surge, seiche and the seasonal level, as measured.

    python -m app.levels --days 800 --out levels.json

Output: {"station", "datum": "IGLD85", "t0": first hour (Unix s, UTC), "step_s": 3600, "cm": [level - 17600 cm, or
null]} - one value per hour, small enough to inline in the page.
"""

import argparse
import datetime as dt
import json
import logging
import urllib.parse

from . import bathymetry as bm
from . import lakefront as lf

log = logging.getLogger("klot.levels")
UTC = dt.timezone.utc
BASE_CM = 17600


def _fetch(station, product, start, end, get, interval=None):
    q = {"station": station, "product": product, "datum": "IGLD", "units": "metric", "time_zone": "gmt",
         "format": "json", "application": "chicago-storm-3d",
         "begin_date": start.strftime("%Y%m%d %H:%M"), "end_date": end.strftime("%Y%m%d %H:%M")}
    if interval:
        q["interval"] = interval
    js = json.loads(get(bm.COOPS + "?" + urllib.parse.urlencode(q), timeout=90))
    if "error" in js:
        raise RuntimeError(js["error"].get("message", "error"))
    out = {}
    for r in js.get("data", []):
        try:
            t = dt.datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            out[t] = float(r["v"])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def hourly(station, start, end, get=lf.http_get):
    """Hourly levels: hourly water_level in year-long requests (31-day ones if refused), then 6-minute data for the
    last few days (the hourly series can lag), averaged to the hour."""
    vals, t = {}, start
    while t < end:
        t1 = min(t + dt.timedelta(days=364), end)
        try:
            vals.update(_fetch(station, "water_level", t, t1, get, interval="h"))
        except Exception as e:  # noqa: BLE001
            log.info("%s hourly %s..%s: %s; trying monthly", station, t.date(), t1.date(), e)
            s = t
            while s < t1:
                s1 = min(s + dt.timedelta(days=30), t1)
                for prod, iv in (("water_level", "h"), ("hourly_height", None)):
                    try:
                        vals.update(_fetch(station, prod, s, s1, get, interval=iv))
                        break
                    except Exception as e2:  # noqa: BLE001
                        log.info("%s %s %s..%s: %s", station, prod, s.date(), s1.date(), e2)
                s = s1
        t = t1
    try:
        six = _fetch(station, "water_level", end - dt.timedelta(days=4), end, get)
        acc = {}
        for k, v in six.items():
            acc.setdefault(k.replace(minute=0), []).append(v)
        for k, vs in acc.items():
            vals[k] = sum(vs) / len(vs)
    except Exception as e:  # noqa: BLE001
        log.info("%s 6-min: %s", station, e)
    return vals


def build(days=800, get=lf.http_get):
    end = dt.datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=days)
    for st in bm.LEVEL_STATIONS:
        vals = hourly(st, start, end, get)
        if len(vals) > 24:
            n = int((end - start).total_seconds() // 3600) + 1
            cm = [None] * n
            for k, v in vals.items():
                i = int((k - start).total_seconds() // 3600)
                if 0 <= i < n:
                    cm[i] = round(v * 100 - BASE_CM)
            while cm and cm[-1] is None:
                cm.pop()
            return {"station": st, "datum": "IGLD85", "base_cm": BASE_CM, "t0": int(start.timestamp()), "step_s": 3600,
                    "cm": cm, "n_obs": len(vals)}
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--out", default="levels.json")
    a = ap.parse_args()
    d = build(a.days)
    if d:
        with open(a.out, "w") as fh:
            json.dump(d, fh, separators=(",", ":"))
        print(f"wrote {a.out}: {d['station']}, {d['n_obs']} hours, last {d['cm'][-1] if d['cm'] else None} cm over {d['base_cm']}")
    else:
        print("no water levels")
