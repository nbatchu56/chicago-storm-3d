"""
Which Earth Engine datasets can our service account read near the Chicago lakefront, and how fresh are they?
Prints one line per dataset: newest image time, age, and a sample value at a point off North Ave.

    GEE_SERVICE_ACCOUNT_KEY=... python tools/ee_probe.py      (needs: pip install earthengine-api)
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.weathernext import init_ee  # noqa: E402

LAKE = (-87.58, 41.92)        # lon, lat ~3 km off North Ave (open water)
DAYS = 40
PROBES = [   # id, band, scale m, lookback days
    ("MODIS/061/MOD11A1", "LST_Day_1km", 1000, DAYS),
    ("MODIS/061/MYD11A1", "LST_Day_1km", 1000, DAYS),
    ("LANDSAT/LC09/C02/T1_L2", "ST_B10", 30, DAYS),
    ("LANDSAT/LC08/C02/T1_L2", "ST_B10", 30, DAYS),
    ("COPERNICUS/S1_GRD", "VV", 10, DAYS),
    ("COPERNICUS/S2_SR_HARMONIZED", "B4", 10, DAYS),
    ("ECMWF/ERA5_LAND/HOURLY", "lake_mix_layer_temperature", 11132, DAYS),
    ("ECMWF/ERA5/HOURLY", "u_component_of_wind_10m", 27830, 4000),
    ("NOAA/GOES/19/MCMIPC", "CMI_C13", 2000, 2),
    ("NOAA/GOES/16/MCMIPC", "CMI_C13", 2000, 2),
    ("NOAA/GFS0P25", "u_component_of_wind_10m_above_ground", 27830, 3),
    ("projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg", "wind_speed_10m_mean", 11132, 3),
]


def main():
    ee = init_ee()
    now = dt.datetime.now(dt.timezone.utc)
    pt = ee.Geometry.Point(list(LAKE))
    for cid, band, scale, days in PROBES:
        try:
            col = ee.ImageCollection(cid).filterBounds(pt).filterDate(ee.Date((now - dt.timedelta(days=days)).isoformat()),
                                                                      ee.Date((now + dt.timedelta(days=1)).isoformat()))
            n = col.size().getInfo()
            if not n:
                print(f"{cid:72s} readable, no images in the last {days} d")
                continue
            img = col.sort("system:time_start", False).first()
            t = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd'T'HH:mm'Z'", "UTC").getInfo()
            v = img.select(band).reduceRegion(ee.Reducer.first(), pt, scale).get(band).getInfo()
            age = now - dt.datetime.strptime(t, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
            print(f"{cid:72s} OK  {n:5d} imgs  newest {t}  age {age.total_seconds() / 3600:7.1f} h  {band}={v}")
        except Exception as e:
            print(f"{cid:72s} FAIL {str(e)[:200]}")


if __name__ == "__main__":
    main()
