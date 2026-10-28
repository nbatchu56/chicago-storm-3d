from app import weathernext as wn


def test_pick_init_newest_complete():
    counts = {"2026-10-27T18:00:00Z": 48, "2026-10-27T19:00:00Z": 48, "2026-10-27T20:00:00Z": 12}
    assert wn.pick_init(counts) == "2026-10-27T19:00:00Z"
    assert wn.pick_init({"2026-10-27T20:00:00Z": 3}) is None
    assert wn.pick_init({}) is None


def test_parse_time_forms():
    a = wn.parse_time("2026-10-27T18:00:00Z")
    assert a == wn.parse_time("2026-10-27 18:00:00") == wn.parse_time(a.timestamp() * 1000)
    assert a.utcoffset().total_seconds() == 0


def test_met_dir():
    assert wn.met_dir(0.0, -5.0) == 0.0          # blowing south = from the north
    assert wn.met_dir(-5.0, 0.0) == 90.0         # blowing west = from the east
    assert wn.met_dir(None, 1.0) is None


def test_to_series_units_and_order():
    feats = [{"properties": {"id": "northave", "forecast_hour": h, "wind_speed_10m_mean": 5.0 + h,
                             "wind_speed_10m_p10": 3.0, "wind_speed_10m_p50": 5.0, "wind_speed_10m_p90": 8.0,
                             "u_component_of_wind_10m_mean": 0.0, "v_component_of_wind_10m_mean": -5.0,
                             "total_cloud_cover_mean": 0.5, "total_precipitation_1hr_mean": 0.0012,
                             "total_precipitation_1hr_p90": 0.003, "temperature_2m_mean": 283.15,
                             "mean_sea_level_pressure_mean": 101325.0}} for h in (2, 1)]
    feats.append({"properties": {"id": "45198", "forecast_hour": 1, "station_head_temperature_2m_mean": 273.15}})
    s = wn.to_series(feats, "2026-10-27T18:00:00Z")
    p = s["northave"]
    assert p["t"] == ["2026-10-27T19:00:00+00:00", "2026-10-27T20:00:00+00:00"]
    assert p["wspd"] == [6.0, 7.0] and p["wdir"] == [0.0, 0.0]
    assert p["precip_mm"] == [1.2, 1.2] and p["t2m_c"] == [10.0, 10.0] and p["mslp_hpa"] == [1013.2, 1013.2]
    assert "st_t2m_c" not in p and s["45198"]["st_t2m_c"] == [0.0] and "wspd" not in s["45198"]
