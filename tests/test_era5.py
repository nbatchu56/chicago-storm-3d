"""ERA5 in the learned correction (synthetic data only, no Earth Engine)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import era5, learn  # noqa: E402
from app import sup_waves as sw  # noqa: E402

UTC = dt.timezone.utc
T0 = dt.datetime(2024, 6, 1, tzinfo=UTC)


def test_rows_from_region_converts_and_sorts():
    h = ["id", "longitude", "latitude", "time", *era5.BANDS]
    ms = lambda k: (T0 + dt.timedelta(hours=k)).timestamp() * 1000  # noqa: E731
    region = [h, ["b", -87.5, 42.0, ms(1), 0.0, -5.0, 283.15], ["a", -87.5, 42.0, ms(0), -3.0, 0.0, 273.15], ["c", -87.5, 42.0, ms(2), None, 1.0, 280.0]]
    rows = era5.rows_from_region(region, "45198")
    assert [r["t"][11:13] for r in rows] == ["00", "01"]
    assert rows[0]["wdir"] == 90 and rows[0]["wspd"] == 3.0 and rows[0]["atmp"] == 0.0     # blowing west = from the east
    assert rows[1]["wdir"] == 0 and rows[1]["atmp"] == 10.0


def test_cache_roundtrip(tmp_path):
    rows = [{"t": (T0 + dt.timedelta(hours=k)).isoformat(), "point": "45174", "wspd": 5.0 + k, "wdir": 180, "atmp": None} for k in range(3)]
    era5.write_year(era5.year_path(str(tmp_path), 2024), rows)
    d = era5.load(str(tmp_path))
    assert d["45174"][T0 + dt.timedelta(hours=2)]["wspd"] == 7.0 and d["45174"][T0]["atmp"] is None
    assert era5.load(str(tmp_path), years=[2023]) == {}


def _overlap(ratio_s, ratio_n, hours=600):
    """ERA5 reads 1/ratio of the buoy's 10 m wind: ratio_s in south winds, ratio_n in north winds."""
    era, own = {}, {}
    for k in range(hours):
        t, d = T0 + dt.timedelta(hours=k), 180 if k % 2 else 0
        u10 = 8.0
        era[t] = {"wspd": u10 / (ratio_s if d == 180 else ratio_n), "wdir": d, "atmp": 15.0}
        own[t] = {"wspd": u10 / sw.u10(1.0, "45198"), "wdir": d, "wvht": 0.5}      # anemometer height: reads below 10 m
    return era, own


def test_calibration_by_sector_and_apply():
    era, own = _overlap(1.2, 0.8)
    c = learn.calibrate_era5({"45198": era}, {"45198": own})["45198"]
    assert c["hours"] == 600 and abs(c["overall"] - 1.0) < 0.1
    assert abs(c["sectors"][learn.sector(180)] - 1.2) < 0.06 and abs(c["sectors"][learn.sector(0)] - 0.8) < 0.06
    cal = learn.calibrated(era, c)
    t = T0 + dt.timedelta(hours=1)
    assert abs(cal[t]["wspd"] - 8.0) < 0.5
    assert learn.calibrate_era5({"45198": dict(list(era.items())[:50])}, {"45198": own}) == {}     # too little overlap


def test_hindcast_fills_missing_wind_from_era5():
    era, own = _overlap(1.0, 1.0, hours=120)
    for k in range(48, 72):                     # buoy anemometer out for a day, waves still reported
        own[T0 + dt.timedelta(hours=k)].pop("wspd")
    hourly = {"45198": own, learn.era5_key("45198"): era}
    pairs = learn.hindcast("45198", hourly)
    src = {p["t"]: p["wind_src"] for p in pairs}
    assert src[(T0 + dt.timedelta(hours=60)).isoformat()] == "era5:45198"
    assert src[(T0 + dt.timedelta(hours=30)).isoformat()] == "45198"
    assert len(pairs) == 120


def test_climate_counts_storms_and_buoy_coverage():
    ser = {}
    for k in range(24 * 20):
        t = T0 + dt.timedelta(hours=k)
        storm = 100 <= k < 130 or 300 <= k < 320           # two north blows, the buoy saw only the second
        ser[t] = {"wspd": 16.0 if storm else 3.0, "wdir": 0.0 if storm else 200.0, "atmp": 10.0}
    observed = {"45198": {T0 + dt.timedelta(hours=k) for k in range(290, 340)}}
    c = learn.era5_climate({"45198": ser}, observed)["buoys"]["45198"]
    t1 = c["totals"]["1m"]
    assert t1["model_hours"] > 30 and 0 < t1["buoy_saw"] < t1["model_hours"] and 0.6 * t1["model_hours"] < t1["north_model_hours"] < t1["model_hours"]   # leftover waves after the wind turns S
    assert len(c["top_storms"]) == 2 and c["top_storms"][0]["peak_hs"] >= 1.0
    assert sum(s["observed_hours"] for s in c["top_storms"]) == t1["buoy_saw"]
    assert c["big_hours_by_dir"]["N"]["model_hours"] == t1["north_model_hours"]
    assert sum(v["model_hours"] for v in c["big_hours_by_dir"].values()) == t1["model_hours"]


def test_cv_reports_north_big():
    from test_learn import synth_pairs
    ps = synth_pairs()
    for p in ps:
        p["obs_hs"] = max(p["obs_hs"], 1.2) if p["wdir"] in (30, 300) else p["obs_hs"]
    nb = learn.cross_validate(ps)["north_big"]
    assert nb["raw"]["hours"] > 100 and nb["learned"]["hours"] == nb["raw"]["hours"]


def test_wind_driven_spins_up_and_aligns():
    t0 = T0 + dt.timedelta(hours=9)
    ser = {"t": [(T0 + dt.timedelta(hours=k)).isoformat() for k in range(55)], "wspd": [12.0] * 55, "wdir": [0.0] * 55}
    v = learn.wind_driven("45198", ser, t0, 49)
    assert len(v) == 49 and v[0] > 0.5 and v[-1] is None and v[45] >= v[0]       # the series ends 46 h after t0
    assert learn.wind_driven("45198", {"t": [(t0 + dt.timedelta(hours=2)).isoformat()], "wspd": [5], "wdir": [0]}, t0, 5) is None
    assert learn.wind_driven("45198", None, t0, 5) is None


def test_log_forecast_with_gfs(tmp_path):
    t0 = T0 + dt.timedelta(hours=6)
    sup = {"start": t0.isoformat(), "buoy_fc": {"45198": {"t0": t0.isoformat(), "raw": [0.5] * 10}}}
    ser = {"t": [(T0 + dt.timedelta(hours=k)).isoformat() for k in range(20)], "wspd": [8.0] * 20, "wdir": [200.0] * 20}
    import json
    path = learn.log_forecast(sup, str(tmp_path), winds={"gfs": {"45198": ser}, "wn3": {}})
    e = json.loads(open(path).read().strip())
    assert len(e["buoys"]["45198"]["gfs"]) == 10 and "wn3" not in e["buoys"]["45198"]
