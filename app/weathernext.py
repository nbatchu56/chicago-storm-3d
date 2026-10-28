"""
Google DeepMind WeatherNext 3 (experimental ensemble) at a few lakefront points, from Earth Engine.

  0.1 deg  projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg   10 m wind (+ spread), cloud, rain, MSLP
  0.05 deg projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p05deg  station-tuned 2 m temperature, dewpoint

Each collection holds ensemble statistics (mean, p10..p90) per forecast hour; we take the newest init that is
fully ingested for the first 48 h and sample the points. Earth Engine gets a run ~7.5 h after its init time.

Two ways in, same output. Earth Engine: a service-account key (JSON) registered for Earth Engine with WeatherNext
access, in the GEE_SERVICE_ACCOUNT_KEY environment variable (or a path to it in GEE_SERVICE_ACCOUNT_FILE);
GEE_PROJECT overrides the key's project. BigQuery (used when WN3_BQ_TABLE is set): the WeatherNext 3 Analytics
Hub listing linked into your project, e.g. WN3_BQ_TABLE=my-project.weathernext_3.weathernext_3_0_0_0p1deg (~100 MB
a run; WN3_BQ_STATION=1 adds the 0.05 station table from the same dataset, which a free-tier project can't query); the same key must be able to run query jobs in that project
(BigQuery Job User) and read the linked dataset. Either way WeatherNext access is granted on request by Google
(developers.google.com/weathernext). Data is experimental (DeepMind terms), not for real-world decisions.

    python -m app.weathernext --out wn3.json        (needs: pip install earthengine-api, or google-cloud-bigquery)
"""

import argparse
import datetime as dt
import json
import logging
import math
import os

log = logging.getLogger("weathernext")
UTC = dt.timezone.utc
COL_10 = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
COL_05 = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p05deg"
HOURS = 48
POINTS = {                                   # id -> (lat, lon)
    "45174": (42.135, -87.655),              # Wilmette buoy
    "45198": (41.892, -87.563),              # Chicago buoy
    "northave": (41.915, -87.622),           # North Ave Beach
}
BANDS_10 = ["wind_speed_10m_mean", "wind_speed_10m_p10", "wind_speed_10m_p50", "wind_speed_10m_p90",
            "u_component_of_wind_10m_mean", "v_component_of_wind_10m_mean",
            "total_cloud_cover_mean", "total_precipitation_1hr_mean", "total_precipitation_1hr_p90",
            "temperature_2m_mean", "mean_sea_level_pressure_mean"]
BANDS_05 = ["station_head_temperature_2m_mean", "station_head_temperature_2m_p10", "station_head_temperature_2m_p90",
            "station_head_dewpoint_temperature_2m_mean"]
SCALE = {COL_10: 11132, COL_05: 5566}
BANDS_10_BQ = BANDS_10 + ["dewpoint_temperature_2m_mean"]   # the BigQuery table also has dewpoint at 0.1 deg


def parse_time(s):
    """start_time / end_time property (ISO string, or epoch ms) -> aware datetime."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return dt.datetime.fromtimestamp(s / 1000.0, UTC)
    s = str(s).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def pick_init(counts, hours=HOURS):
    """{start_time string: number of images with forecast_hour 1..hours} -> newest init with all of them, else None."""
    full = [(parse_time(k), k) for k, n in counts.items() if n >= hours]
    return max(full)[1] if full else None


def met_dir(u, v):
    """Wind components (towards) -> meteorological direction the wind comes FROM, degrees."""
    if u is None or v is None:
        return None
    return round((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0, 1)


def k2c(k):
    return None if k is None else round(k - 273.15, 2)


def to_series(features, init):
    """reduceRegions features (properties: id, forecast_hour, bands) -> {id: {t: [...], field: [...]}}."""
    t0 = parse_time(init)
    rows = {}
    for f in features:
        p = f.get("properties", f)
        pid, fh = p.get("id"), p.get("forecast_hour")
        if pid is None or fh is None:
            continue
        rows.setdefault(pid, {})[int(fh)] = p
    out = {}
    for pid, by_h in rows.items():
        hs = sorted(by_h)
        g = lambda b, f=None: [(f(by_h[h].get(b)) if f else _round(by_h[h].get(b), 2)) for h in hs]
        s = {"t": [(t0 + dt.timedelta(hours=h)).isoformat() for h in hs]}
        if any("wind_speed_10m_mean" in by_h[h] for h in hs):
            s.update(wspd=g("wind_speed_10m_mean"), wspd_p10=g("wind_speed_10m_p10"), wspd_p50=g("wind_speed_10m_p50"),
                     wspd_p90=g("wind_speed_10m_p90"),
                     wdir=[met_dir(by_h[h].get("u_component_of_wind_10m_mean"), by_h[h].get("v_component_of_wind_10m_mean")) for h in hs],
                     cloud=g("total_cloud_cover_mean"),
                     precip_mm=g("total_precipitation_1hr_mean", lambda v: _round(None if v is None else v * 1000.0, 2)),
                     precip_mm_p90=g("total_precipitation_1hr_p90", lambda v: _round(None if v is None else v * 1000.0, 2)),
                     t2m_c=g("temperature_2m_mean", k2c),
                     mslp_hpa=g("mean_sea_level_pressure_mean", lambda v: _round(None if v is None else v / 100.0, 1)))
            if any("dewpoint_temperature_2m_mean" in by_h[h] for h in hs):
                s["td2m_c"] = g("dewpoint_temperature_2m_mean", k2c)
        if any("station_head_temperature_2m_mean" in by_h[h] for h in hs):
            s.update(st_t2m_c=g("station_head_temperature_2m_mean", k2c), st_t2m_c_p10=g("station_head_temperature_2m_p10", k2c),
                     st_t2m_c_p90=g("station_head_temperature_2m_p90", k2c), st_td2m_c=g("station_head_dewpoint_temperature_2m_mean", k2c))
        out[pid] = s
    return out


def _round(v, n):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


# One init per query, by equality on init_time (the partition key): BigQuery's up-front estimate for this clustered
# view ignores the clustering, so a 30 h range is "2.7 TB" and a free-tier project refuses it, while one init is
# ~180 GB estimated and a few MB read. Runs land in BigQuery ~7.5 h after their init time (6-hourly cycles ~8 h 10).
BQ_LAG_H, BQ_TRIES = 7.0, 6


def bq_inits(now, lag_h=BQ_LAG_H, tries=BQ_TRIES):
    """Init times to try, newest first: whole hours from now - lag back tries hours."""
    t0 = (now - dt.timedelta(hours=lag_h)).replace(minute=0, second=0, microsecond=0)
    return [t0 - dt.timedelta(hours=k) for k in range(tries)]


def bq_query(table, bands, init, hours=HOURS):
    """GoogleSQL for the point samples of one init from a WeatherNext 3 BigQuery table (one row per point x lead hour).
    The cells are picked with a constant MULTIPOINT so the table's geography clustering prunes the scan (a join
    predicate alone doesn't); then each point takes its own cell (two points can share one: North Ave and 45198)."""
    pts = ", ".join(f"STRUCT('{k}' AS id, ST_GEOGPOINT({lo}, {la}) AS g)" for k, (la, lo) in POINTS.items())
    multi = ", ".join(f"({lo} {la})" for la, lo in POINTS.values())
    cols = ", ".join(f"f.{b}" for b in bands)
    return (f"SELECT p.id, t.init_time, f.hours AS forecast_hour, {cols}\n"
            f"FROM `{table}` AS t, UNNEST(t.forecast) AS f, UNNEST([{pts}]) AS p\n"
            f"WHERE t.init_time = TIMESTAMP('{init.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC')\n"
            f"  AND ST_INTERSECTS(t.geography_polygon, ST_GEOGFROMTEXT('MULTIPOINT({multi})'))\n"
            f"  AND ST_INTERSECTS(t.geography_polygon, p.g) AND f.hours BETWEEN 1 AND {hours}")


def bq_series(rows, hours=HOURS):
    """Query rows (dicts with id, init_time, forecast_hour, bands) -> (init ISO string, series) for the newest init
    that has every lead hour 1..hours at every point; None, {} when no init is complete yet."""
    by_init = {}
    for r in rows:
        t = r["init_time"]
        t = t if isinstance(t, dt.datetime) else parse_time(t)
        by_init.setdefault(t.astimezone(UTC), []).append(r)
    for init in sorted(by_init, reverse=True):
        rs = by_init[init]
        if all(len({r["forecast_hour"] for r in rs if r["id"] == k}) >= hours for k in POINTS):
            key = init.isoformat()
            return key, to_series(rs, key)
    return None, {}


def init_bq():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    info = json.loads(_key())
    creds = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    project = os.environ.get("WN3_BQ_PROJECT") or os.environ.get("GEE_PROJECT") or info.get("project_id")
    return bigquery.Client(project=project, credentials=creds)


def sample_bq(client, table, bands, now, hours=HOURS):
    from google.cloud import bigquery
    billed, seen = 0, []
    for t in bq_inits(now):
        job = client.query(bq_query(table, bands, t, hours), job_config=bigquery.QueryJobConfig(labels={"app": "lakeview"}))
        rows = [dict(r) for r in job.result()]
        billed += job.total_bytes_billed or 0
        seen.append(f"{t:%H}Z:{len(rows)}")
        init, ser = bq_series(rows, hours)
        if init:
            log.info("%s init %s: %d rows, %.0f MB billed (%s)", table.rsplit(".", 1)[-1], init, len(rows), billed / 1e6, " ".join(seen))
            return init, ser
    raise RuntimeError(f"{table}: no init with all {hours} hours at every point in the last {BQ_TRIES} tried ({' '.join(seen)})")


def _key():
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if not key and os.environ.get("GEE_SERVICE_ACCOUNT_FILE"):
        key = open(os.environ["GEE_SERVICE_ACCOUNT_FILE"]).read()
    if not key:
        raise RuntimeError("no GEE_SERVICE_ACCOUNT_KEY / GEE_SERVICE_ACCOUNT_FILE")
    return key


def init_ee():
    import ee
    key = _key()
    info = json.loads(key)
    creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key)
    ee.Initialize(creds, project=os.environ.get("GEE_PROJECT") or info.get("project_id"))
    return ee


def sample(ee, col_id, bands, now, hours=HOURS):
    """Newest complete init of a collection, sampled at POINTS -> (init string, series by point)."""
    col = ee.ImageCollection(col_id).filterDate(ee.Date((now - dt.timedelta(days=2)).isoformat()),
                                                ee.Date((now + dt.timedelta(days=3)).isoformat()))
    first = col.filter(ee.Filter.rangeContains("forecast_hour", 1, hours))
    counts = first.aggregate_histogram("start_time").getInfo() or {}
    init = pick_init(counts, hours)
    if not init:
        raise RuntimeError(f"{col_id}: no init with all {hours} hours ({len(counts)} partial)")
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"id": k}) for k, (la, lo) in POINTS.items()])
    imgs = first.filter(ee.Filter.eq("start_time", init)).select(bands)

    def per_image(img):
        fh = img.get("forecast_hour")
        return img.reduceRegions(pts, ee.Reducer.first(), SCALE[col_id]).map(lambda f: f.set("forecast_hour", fh))

    feats = imgs.map(per_image).flatten().getInfo()["features"]
    log.info("%s init %s: %d samples", col_id.rsplit("/", 1)[-1], init, len(feats))
    return init, to_series(feats, init)


def fetch(out_path, now=None):
    now = now or dt.datetime.now(UTC)
    out = {"version": 1, "model": "Google DeepMind WeatherNext 3 (experimental)", "generated": now.isoformat(timespec="seconds"),
           "terms": "Experimental data, (c) DeepMind Technologies Limited; not intended, validated, or approved for real world use.",
           "points": {k: {"lat": la, "lon": lo} for k, (la, lo) in POINTS.items()}, "notes": []}
    bq_table = os.environ.get("WN3_BQ_TABLE")
    if bq_table:                                     # BigQuery: the linked listing; the station table sits beside the 0.1 one
        client = init_bq()
        base = bq_table.rsplit(".", 1)[0]
        products = (("init", bq_table, BANDS_10_BQ),)
        # the 0.05 station table has no geography clustering worth the name for a free-tier (sandbox) project: its
        # worst-case estimate (~1.3 TB a run) is refused there, so it's opt-in for a billed project
        if os.environ.get("WN3_BQ_STATION") == "1":
            products += (("init_station", f"{base}.{COL_05.rsplit('/', 1)[-1]}", BANDS_05),)
        out["source"] = "bigquery"
    else:
        ee = init_ee()
        products = (("init", COL_10, BANDS_10), ("init_station", COL_05, BANDS_05))
        out["source"] = "earthengine"
    for key, col, bands in products:
        try:
            init, ser = sample_bq(client, col, bands, now) if bq_table else sample(ee, col, bands, now)
        except Exception as e:                       # one product failing keeps the other
            log.warning("%s: %s", col, e)
            out["notes"].append(f"{col.rsplit('/', 1)[-1]}: {e}")
            continue
        out[key] = parse_time(init).isoformat()
        for pid, s in ser.items():
            p = out["points"].setdefault(pid, {})
            if key == "init_station":                # hourly times can differ between the two inits
                p["t_station"] = s.pop("t")
            p.update(s)
    if "init" not in out and "init_station" not in out:
        raise RuntimeError("; ".join(out["notes"]) or "no WeatherNext data")
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="wn3.json")
    a = ap.parse_args()
    o = fetch(a.out)
    p = o["points"].get("northave", {})
    print(o.get("init"), o.get("init_station"), "wspd0", (p.get("wspd") or [None])[0], "t2m0", (p.get("st_t2m_c") or [None])[0], o["notes"])
