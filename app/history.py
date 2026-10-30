"""
Conditions for any past time on the Chicago lakefront, from public archives.

  radar   NEXRAD Level II from the NOAA/Unidata archive on AWS (KLOT back to the 1990s), gridded in 3-D
  wind/waves/water temperature
          NDBC observations: the realtime files cover the last ~45 days; older times use the
          monthly/yearly historical archive files
  lake    NOAA GLERL GLSEA lake-surface temperature for that day (2006-present) and ice concentration
          (1995-present, ice season)

    python -m app.history --start 2026-10-23T12:20 --end 2026-10-23T12:51 --out history.json

Output: {"window": .., "scene": <radar scene for the window>, "stations": [...], "lake": {...}, "notes": [...]}.
"""

import argparse
import datetime as dt
import gzip
import io
import json
import logging
import os

import numpy as np

from . import chicago_radar_nowcast as nc
from . import lakefront as lf
from . import radar_to_scene as r2s
from .sup_waves import parse_ndbc

log = logging.getLogger("klot.history")
UTC = dt.timezone.utc
STATIONS = ["45174", "45198", "45177", "45186", "45007", "CHII2", "CNII2", "OKSI2", "FSTI2", "CMTI2"]


def _parse_time(s):
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def radar_scene(start, end, site="KLOT", cache="radar_cache", cloud_base=1.2, max_scans=12, wind=None):
    """Every volume from ~20 min before start to end (at least 3), gridded, no forecast frames beyond 10 min."""
    span = (end - start).total_seconds() / 60 + 25
    scans = nc.list_scans(site, end + dt.timedelta(minutes=3), max_scans, lookback_min=max(40, int(span)))
    scans = [s for s in scans if s[0] >= start - dt.timedelta(minutes=25)] or scans
    times, vols, radar_xy = [], [], (0.0, 0.0)
    for _, bucket, key in scans:
        path = nc.download(bucket, key, cache)
        t, v, radar_xy, _ = r2s.grid_volume(path)
        times.append(t); vols.append(v)
        try:
            os.remove(path)
        except OSError:
            pass
    order = np.argsort(times)
    times = [times[i] for i in order]; vols = [vols[i] for i in order]
    wind = wind or {"speed": 0, "dir": 0, "duration_h": 8, "source": "history"}
    scene = r2s.build_scene(times, vols, f"NEXRAD {site} Level II (archive)", site, radar_xy, 10, 10, wind, cloud_base)
    scene["meta"]["historical"] = True
    return scene


def _ndbc_rows_for(station, start, end, get):
    """Observations for [start, end] from the realtime file, or the historical stdmet archive."""
    now = dt.datetime.now(UTC)
    rows = []
    if now - start < dt.timedelta(days=44):
        try:
            rows = parse_ndbc(get(f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt", timeout=30), station)
        except Exception as e:  # noqa: BLE001
            log.info("realtime %s: %s", station, e)
    if not rows:
        st = station.lower()
        cands = []
        if start.year == now.year:   # current year: monthly files
            mon = start.strftime("%b")
            cands.append(f"https://www.ndbc.noaa.gov/data/stdmet/{mon}/{st}{start.month}{start.year}.txt.gz")
        cands.append(f"https://www.ndbc.noaa.gov/data/historical/stdmet/{st}h{start.year}.txt.gz")
        for url in cands:
            try:
                raw = get_bytes(url)
                txt = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "replace")
                rows = parse_ndbc(txt, station)
                if rows:
                    break
            except Exception as e:  # noqa: BLE001
                log.info("historical %s: %s", url, e)
    lo, hi = start - dt.timedelta(hours=12), end + dt.timedelta(hours=3)
    out = [r for r in rows if lo <= r["t"] <= hi]
    return sorted(out, key=lambda r: r["t"])


def get_bytes(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": lf.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def station_window(start, end, get=lf.http_get):
    meta = {}
    try:
        meta = {s["id"]: s for s in lf.fetch_stations(get=get)}
    except Exception as e:  # noqa: BLE001
        log.info("station list: %s", e)
    out = []
    for st in STATIONS:
        rows = _ndbc_rows_for(st, start, end, get)
        if not rows:
            continue
        m = meta.get(st, {"id": st, "name": st, "lat": None, "lon": None})
        s = dict(m)
        if s.get("lat") is not None:
            s["x"], s["y"] = lf.xy(s["lat"], s["lon"])
        s["history"] = [{"t": r["t"].isoformat(), "wspd": r["wspd"], "wdir": r["wdir"], "gst": r["gst"], "wvht": r["wvht"],
                         "dpd": r["dpd"], "mwd": r["mwd"], "wtmp": r["wtmp"], "atmp": r["atmp"]} for r in rows]
        last_t = next((h for h in reversed(s["history"]) if h["wtmp"] is not None), None)
        s["wtmp_now"] = last_t and {"c": last_t["wtmp"], "t": last_t["t"]}
        out.append(s)
    return out


def lake_for_date(t, get=lf.http_get):
    """GLSEA temperature (and ice, if any) for the day of t."""
    out, probe = {}, {}
    host = lf.ERDDAP_HOSTS[0]
    # the analysis for a day appears about a day later: try that day, the day before, then the latest
    days = [t.strftime("%Y-%m-%dT12:00:00Z"), (t - dt.timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z"), "last"]
    for ds, var, key, day in [(a, b, c, d) for a, b, c in (("GLSEA_ACSPO_GCS", "sst", "temp"), ("GLSEA_GCS", "sst", "temp"),
                                                             ("GL_Ice_Concentration_GCS", "ice_concentration", "ice")) for d in days]:
        if key in out:
            continue
        if day == "last" and (dt.datetime.now(UTC) - t).days > 3:
            continue                      # "latest" only stands in for very recent times
        s, w, n, e = lf.TEMP_BBOX
        url = f"{host}/griddap/{ds}.json?{var}%5B({day})%5D%5B({s}):1:({n})%5D%5B({w}):1:({e})%5D"
        try:
            js = json.loads(get(url, timeout=120))
            cols = js["table"]["columnNames"]
            rows = js["table"]["rows"]
            pts = [(r[cols.index("latitude")], r[cols.index("longitude")], r[cols.index(var)]) for r in rows]
            g = lf.grid_from_points(pts)
            vals = [v for v in g["v"] if v is not None]
            if not vals:
                continue
            if key == "ice" and max(vals) > 1.5:
                g["v"] = [None if v is None else round(v / 100.0, 3) for v in g["v"]]
            out[key] = g | {"time": rows[0][cols.index("time")], "source": f"{ds} ({host})", "var": var}
        except Exception as e:  # noqa: BLE001
            probe[f"{ds} {day}"] = str(e)
    if "ice" in out and (abs((dt.datetime.fromisoformat(out["ice"]["time"].replace("Z", "+00:00")) - t).days) > 3):
        out["ice_status"] = f"no ice analysis near that date (nearest {out['ice']['time'][:10]})"
        del out["ice"]
    return out, probe


def build(start, end, site="KLOT", cache="radar_cache", get=lf.http_get, max_scans=12, lake_date=None):
    notes = []
    data = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
            "window": {"start": start.isoformat(), "end": end.isoformat()}, "site": site}
    data["stations"] = station_window(start, end, get)
    wind = None
    w = next((s for s in data["stations"] if any(h["wspd"] is not None for h in s["history"])), None)
    if w:
        mid = start + (end - start) / 2
        h = min((h for h in w["history"] if h["wspd"] is not None), key=lambda h: abs(_parse_time(h["t"]) - mid))
        wind = {"speed": h["wspd"], "dir": h["wdir"], "duration_h": 8, "source": f"NDBC {w['id']} {h['t']}"}
    try:
        data["scene"] = radar_scene(start, end, site, cache, max_scans=max_scans, wind=wind)
    except Exception as e:  # noqa: BLE001
        notes.append(f"radar unavailable: {e}")
    data["lake"], probe = lake_for_date(lake_date or start, get)
    if lake_date:   # e.g. the ice peak a few days after the cold snap: say so
        notes.append(f"lake temperature and ice from {lake_date.strftime('%b %-d, %Y')}, when the nearshore ice was greatest")
        data["lake"]["date_note"] = lake_date.strftime("%b %-d, %Y")
    if "temp" not in data["lake"]:
        notes.append("satellite lake temperature unavailable for that day")
    data["notes"] = notes
    data["probe"] = probe
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="UTC, e.g. 2026-10-23T12:20")
    ap.add_argument("--end", help="UTC (default start + 30 min)")
    ap.add_argument("--site", default="KLOT")
    ap.add_argument("--cache", default="radar_cache")
    ap.add_argument("--out", default="history.json")
    ap.add_argument("--max-scans", type=int, default=12, help="most radar volumes gridded for the window")
    ap.add_argument("--lake-date", help="take the lake temperature and ice analysis from this day instead (UTC date, e.g. 2026-01-31)")
    a = ap.parse_args()
    t0 = _parse_time(a.start)
    t1 = _parse_time(a.end) if a.end else t0 + dt.timedelta(minutes=30)
    d = build(t0, t1, a.site, a.cache, max_scans=a.max_scans, lake_date=_parse_time(a.lake_date + "T12:00") if a.lake_date else None)
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    sc = d.get("scene")
    print(f"wrote {a.out}: radar frames {len(sc['frames']) if sc else 0}, stations {[s['id'] for s in d['stations']]}, "
          f"lake {list(d['lake'])}, notes {d['notes']}")
