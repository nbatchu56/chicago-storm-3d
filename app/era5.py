"""
ECMWF ERA5 hourly reanalysis (via Google Earth Engine) at the two wave buoys, for the learned correction (app.learn).

ERA5 is a consistent hourly record of 10 m wind and 2 m air temperature back to 1940 on a 0.25 deg grid. We take the
open-lake cell next to each buoy and cache one file per year (era5/era5_YYYY.csv.gz on the `learn` branch). The
backfill uses it two ways:

  * filling wind: hours the buoy reported waves but no station had wind get ERA5 wind, scaled to read like the buoy
    (ratio by direction sector from the hours both exist), so those hours count in the fit too;
  * a 35-year wave hindcast from ERA5 wind alone, to see how often big storms (especially north winds, which have the
    longest fetch to Chicago) happen and how few of them the buoys were out for, i.e. where the fit is extrapolating.

    python -m app.era5 --dir learn/era5 --years 1991-2026        (needs: pip install earthengine-api; GEE_SERVICE_ACCOUNT_KEY)
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import glob
import gzip
import logging
import math
import os

log = logging.getLogger("era5")
UTC = dt.timezone.utc
COLLECTION = "ECMWF/ERA5/HOURLY"
BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m", "temperature_2m"]
# ERA5 cell centres (0.25 deg) in open water next to each buoy; the cells holding the buoys themselves reach the shore
POINTS = {"45174": (42.25, -87.50), "45198": (42.00, -87.50)}
SCALE_M = 27830
FIELDS = ["t", "point", "wspd", "wdir", "atmp"]


def met_dir(u, v):
    return (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0


def rows_from_region(region, point):
    """ee getRegion result ([header, rows...]) -> [{t, point, wspd, wdir, atmp}] sorted by time."""
    if not region:
        return []
    h = region[0]
    it, iu, iv, ia = h.index("time"), h.index(BANDS[0]), h.index(BANDS[1]), h.index(BANDS[2])
    out = {}
    for r in region[1:]:
        u, v, a = r[iu], r[iv], r[ia]
        if u is None or v is None or r[it] is None:
            continue
        t = dt.datetime.fromtimestamp(r[it] / 1000.0, UTC).replace(minute=0, second=0, microsecond=0)
        out[t] = {"t": t.isoformat(), "point": point, "wspd": round(math.hypot(u, v), 2), "wdir": round(met_dir(u, v)),
                  "atmp": None if a is None else round(a - 273.15, 2)}
    return [out[t] for t in sorted(out)]


def fetch_chunk(ee, point, start, end):
    lat, lon = POINTS[point]
    col = ee.ImageCollection(COLLECTION).filterDate(start.isoformat(), end.isoformat()).select(BANDS)
    return rows_from_region(col.getRegion(ee.Geometry.Point([lon, lat]), SCALE_M).getInfo(), point)


def year_path(d, y):
    return os.path.join(d, f"era5_{y}.csv.gz")


def fetch_year(ee, y, now):
    """Both points, one year, in quarter-year requests (getRegion has a size limit)."""
    rows = []
    for p in POINTS:
        for q in range(4):
            a = dt.datetime(y, 3 * q + 1, 1, tzinfo=UTC)
            b = dt.datetime(y + (q == 3), (3 * q + 3) % 12 + 1, 1, tzinfo=UTC)
            if a > now:
                break
            rows += fetch_chunk(ee, p, a, min(b, now))
    return rows


def write_year(path, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def update(d, years, now=None, workers=6):
    """Fetch the years not cached yet; always refresh the current year (ERA5 runs ~6 days behind) and the last one
    in January. Returns {year: rows fetched}."""
    from .weathernext import init_ee
    now = now or dt.datetime.now(UTC)
    os.makedirs(d, exist_ok=True)
    todo = [y for y in years if y <= now.year and (not os.path.exists(year_path(d, y)) or y == now.year or (y == now.year - 1 and now.month == 1))]
    if not todo:
        return {}
    ee = init_ee()
    done = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_year, ee, y, now): y for y in todo}
        for f in cf.as_completed(futs):
            y = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("ERA5 %s: %s", y, e)
                continue
            if rows:
                write_year(year_path(d, y), rows)
            done[y] = len(rows)
            log.info("ERA5 %s: %d rows", y, len(rows))
    return done


def load(d, years=None):
    """Cached ERA5 -> {point: {datetime: {"wspd", "wdir", "atmp"}}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "era5_*.csv.gz"))):
        y = int(os.path.basename(path)[5:9])
        if years and y not in years:
            continue
        with gzip.open(path, "rt") as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["point"], {})[dt.datetime.fromisoformat(r["t"])] = {
                    "wspd": float(r["wspd"]), "wdir": float(r["wdir"]), "atmp": float(r["atmp"]) if r["atmp"] not in ("", None) else None}
    return out


if __name__ == "__main__":
    from .learn import parse_years
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--years", default="1991-%d" % dt.datetime.now(UTC).year)
    a = ap.parse_args()
    print(update(a.dir, parse_years(a.years)))
