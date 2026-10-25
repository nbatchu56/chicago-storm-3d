"""
NOAA Great Lakes Wave model (GLWU, WAVEWATCH III on an unstructured lake grid, output on a 2.5 km grid) at the
wave buoys, as a second opinion for the scorecard (app.learn).

Reads the newest cycle on NOMADS, downloads only the significant-height, peak-period and peak-direction records
for the first 48 h (byte ranges from the .idx file) and samples the nearest water point to each buoy.

    python -m app.glwu --out glwu.json        (needs: pip install eccodes)
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import math
import os
import re
import tempfile
import urllib.request

from . import learn
from . import sup_waves as sw

log = logging.getLogger("glwu")
UTC = dt.timezone.utc
BASE = os.environ.get("GLWU_BASE", "https://nomads.ncep.noaa.gov/pub/data/nccf/com/glwu/prod/")
VARS = {"HTSGW": "hs", "PERPW": "tp", "DIRPW": "dir"}
MAX_LEAD = 48


def http(url, rng=None, timeout=60):
    h = {"User-Agent": sw.UA}
    if rng:
        h["Range"] = "bytes=%d-%s" % (rng[0], "" if rng[1] is None else rng[1])
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def pick_file(listing):
    """Newest Great Lakes 2.5 km grib2 file in a day's directory listing -> (name, cycle hour)."""
    names = sorted(set(re.findall(r'href="([^"]+\.grib2)"', listing)))
    best = []
    for n in names:
        m = re.search(r"\.t(\d\d)z\.", n)
        if not m:
            continue
        rank = 0 if "grlc_2p5km." in n else 1 if "grlc" in n and "2p5km" in n else 2 if "grlc" in n else 9
        best.append((rank, -int(m.group(1)), n))
    best.sort()
    return (best[0][2], -best[0][1]) if best and best[0][0] < 9 else (None, None)


def parse_idx(text, size=None):
    """wgrib2 inventory -> [(offset, end or None, var, lead_h)]."""
    rows = []
    for line in text.splitlines():
        p = line.split(":")
        if len(p) < 6:
            continue
        lead = 0 if p[5].strip() == "anl" else None
        m = re.match(r"(\d+) hour fcst", p[5].strip())
        if m:
            lead = int(m.group(1))
        rows.append([int(p[1]), None, p[3], lead])
    for i in range(len(rows) - 1):
        rows[i][1] = rows[i + 1][0] - 1
    if rows and size:
        rows[-1][1] = size - 1
    return [tuple(r) for r in rows]


def wanted_ranges(idx, max_lead=MAX_LEAD):
    """Records we need, merged into as few byte ranges as possible (kept in file order)."""
    recs = [r for r in idx if r[2] in VARS and r[3] is not None and r[3] <= max_lead]
    merged = []
    for off, end, var, lead in recs:
        if merged and merged[-1][1] is not None and merged[-1][1] + 1 == off:
            merged[-1][1] = end
            merged[-1][2].append((var, lead))
        else:
            merged.append([off, end, [(var, lead)]])
    return recs, merged


def nearest(lats, lons, vals, missing, lat, lon, max_km=8.0):
    """Index of the nearest grid point with a value (water) to (lat, lon), and its distance in km."""
    import numpy as np
    la, lo, v = np.asarray(lats, float), np.asarray(lons, float), np.asarray(vals, float)
    lo = np.where(lo > 180, lo - 360, lo)
    d = np.hypot((la - lat) * 110.57, (lo - lon) * 111.32 * math.cos(math.radians(lat)))
    d[(v == missing) | ~np.isfinite(v)] = np.inf
    i = int(np.argmin(d))
    bd = float(d[i])
    return (i, bd) if bd <= max_km else (None, None if not np.isfinite(bd) else bd)


def fetch(out_path, get=http):
    top = get(BASE).decode("utf-8", "replace")
    days = sorted(set(re.findall(r"glwu\.(\d{8})/", top)), reverse=True)[:2]
    if not days:
        raise RuntimeError("no glwu.YYYYMMDD directories at " + BASE)
    url = cyc = None
    for d in days:
        listing = get(f"{BASE}glwu.{d}/").decode("utf-8", "replace")
        name, hh = pick_file(listing)
        if name and f'"{name}.idx"' in listing:
            url, cyc = f"{BASE}glwu.{d}/{name}", f"{d}{hh:02d}"
            break
        log.info("glwu.%s: no usable file (grib2 files: %s)", d, re.findall(r'href="([^"]+\.grib2)"', listing)[:12])
    if not url:
        raise RuntimeError("no GLWU grib2 file with an index found")
    log.info("GLWU %s", url)
    idx = parse_idx(get(url + ".idx").decode())
    recs, merged = wanted_ranges(idx)
    if not recs:
        raise RuntimeError("no HTSGW/PERPW/DIRPW records in the index")
    with cf.ThreadPoolExecutor(6) as ex:
        chunks = list(ex.map(lambda m: get(url, (m[0], m[1])), merged))
    import eccodes
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        for c in chunks:
            tmp.write(c)
        tmp.flush()
        pts = {b: None for b in learn.BUOYS}
        series = {b: {} for b in learn.BUOYS}
        lats = lons = None
        with open(tmp.name, "rb") as fh:
            k = 0
            while True:
                gid = eccodes.codes_grib_new_from_file(fh)
                if gid is None:
                    break
                try:
                    var, lead = recs[k][2], recs[k][3]
                    k += 1
                    vals = eccodes.codes_get_values(gid)
                    miss = eccodes.codes_get(gid, "missingValue")
                    if lats is None:
                        lats = eccodes.codes_get_array(gid, "latitudes")
                        lons = eccodes.codes_get_array(gid, "longitudes")
                    for b, (la, lo) in learn.BUOYS.items():
                        if pts[b] is None and var == "HTSGW":
                            i, d = nearest(lats, lons, vals, miss, la, lo)
                            pts[b] = {"i": i, "dist_km": None if d is None else round(d, 2)}
                        i = (pts[b] or {}).get("i")
                        if i is None:
                            continue
                        v = float(vals[i])
                        t = dt.datetime.strptime(cyc, "%Y%m%d%H").replace(tzinfo=UTC) + dt.timedelta(hours=lead)
                        series[b].setdefault(t, {})[VARS[var]] = None if v == miss else round(v, 3)
                finally:
                    eccodes.codes_release(gid)
    out = {"version": 1, "model": "NOAA GLWU (WAVEWATCH III)", "file": url, "cycle": cyc,
           "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "points": {}}
    for b, ser in series.items():
        if not ser or not pts[b] or pts[b]["i"] is None:
            continue
        ts = sorted(ser)
        i = pts[b]["i"]
        out["points"][b] = {"lat": round(float(lats[i]), 4), "lon": round(float(lons[i] - 360 if lons[i] > 180 else lons[i]), 4),
                            "dist_km": pts[b]["dist_km"], "t": [t.isoformat() for t in ts],
                            **{f: [ser[t].get(f) for t in ts] for f in VARS.values()}}
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="glwu.json")
    a = ap.parse_args()
    o = fetch(a.out)
    print(json.dumps({b: {"dist_km": p["dist_km"], "hours": len(p["t"]), "hs0": p["hs"][0] if p["hs"] else None} for b, p in o["points"].items()}), o["cycle"])
