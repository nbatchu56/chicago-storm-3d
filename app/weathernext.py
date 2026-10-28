"""
Google DeepMind WeatherNext 3 (experimental ensemble) at a few lakefront points, from Earth Engine.

  0.1 deg  projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg   10 m wind (+ spread), cloud, rain, MSLP
  0.05 deg projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p05deg  station-tuned 2 m temperature, dewpoint

Each collection holds ensemble statistics (mean, p10..p90) per forecast hour; we take the newest init that is
fully ingested for the first 48 h and sample the points. Earth Engine gets a run ~7.5 h after its init time.

Credentials: a service-account key (JSON) registered for Earth Engine with WeatherNext access, in the
GEE_SERVICE_ACCOUNT_KEY environment variable (or a path to it in GEE_SERVICE_ACCOUNT_FILE). GEE_PROJECT
overrides the key's project. Data is experimental (DeepMind terms), not for real-world decisions.

    python -m app.weathernext --out wn3.json        (needs: pip install earthengine-api)
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
        if any("station_head_temperature_2m_mean" in by_h[h] for h in hs):
            s.update(st_t2m_c=g("station_head_temperature_2m_mean", k2c), st_t2m_c_p10=g("station_head_temperature_2m_p10", k2c),
                     st_t2m_c_p90=g("station_head_temperature_2m_p90", k2c), st_td2m_c=g("station_head_dewpoint_temperature_2m_mean", k2c))
        out[pid] = s
    return out


def _round(v, n):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


def init_ee():
    import ee
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if not key and os.environ.get("GEE_SERVICE_ACCOUNT_FILE"):
        key = open(os.environ["GEE_SERVICE_ACCOUNT_FILE"]).read()
    if not key:
        raise RuntimeError("no GEE_SERVICE_ACCOUNT_KEY / GEE_SERVICE_ACCOUNT_FILE")
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
    ee = init_ee()
    out = {"version": 1, "model": "Google DeepMind WeatherNext 3 (experimental)", "generated": now.isoformat(timespec="seconds"),
           "terms": "Experimental data, (c) DeepMind Technologies Limited; not intended, validated, or approved for real world use.",
           "points": {k: {"lat": la, "lon": lo} for k, (la, lo) in POINTS.items()}, "notes": []}
    for key, col, bands in (("init", COL_10, BANDS_10), ("init_station", COL_05, BANDS_05)):
        try:
            init, ser = sample(ee, col, bands, now)
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
