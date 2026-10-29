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


def test_bq_query_one_init():
    import datetime as dt
    now = dt.datetime(2026, 10, 28, 20, 25, tzinfo=dt.timezone.utc)
    inits = wn.bq_inits(now)
    assert inits[0] == dt.datetime(2026, 10, 28, 13, tzinfo=dt.timezone.utc) and len(inits) == wn.BQ_TRIES and inits[1] == inits[0] - dt.timedelta(hours=1)
    q = wn.bq_query("p.d.weathernext_3_0_0_0p1deg", wn.BANDS_10, inits[0])
    assert "FROM `p.d.weathernext_3_0_0_0p1deg` AS t, UNNEST(t.forecast) AS f" in q
    assert "ST_GEOGPOINT(-87.622, 41.915)" in q and "'northave' AS id" in q      # lon first
    assert "t.init_time = TIMESTAMP('2026-10-28 13:00:00 UTC')" in q and "f.hours BETWEEN 1 AND 360" in q   # equality: one partition; 15 days
    assert "ST_GEOGFROMTEXT('MULTIPOINT((-87.655 42.135), (-87.563 41.892), (-87.622 41.915))')" in q   # constant: prunes
    assert all(f"f.{b}" in q for b in wn.BANDS_10)


def test_bq_series_newest_complete_init():
    import datetime as dt
    t0, t1 = dt.datetime(2026, 10, 28, 6, tzinfo=dt.timezone.utc), dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc)
    rows = [{"id": k, "init_time": t0, "forecast_hour": h, "wind_speed_10m_mean": 4.0} for k in wn.POINTS for h in range(1, 49)]
    rows += [{"id": k, "init_time": t1, "forecast_hour": h, "wind_speed_10m_mean": 6.0} for k in wn.POINTS for h in range(1, 20)]
    init, ser = wn.bq_series(rows)                  # the 12Z init is newer but only 19 h in: the 06Z one is used
    assert init == "2026-10-28T06:00:00+00:00"
    assert ser["northave"]["wspd"][0] == 4.0 and ser["northave"]["t"][0] == "2026-10-28T07:00:00+00:00"
    assert wn.bq_series(rows[-5:]) == (None, {})


def test_to_series_dewpoint_from_bq():
    feats = [{"id": "northave", "forecast_hour": 1, "wind_speed_10m_mean": 3.0, "dewpoint_temperature_2m_mean": 278.15}]
    assert wn.to_series(feats, "2026-10-28T13:00:00Z")["northave"]["td2m_c"] == [5.0]


def test_bq_series_keeps_leads_past_48h_but_needs_the_first_48():
    import datetime as dt
    t0, t1 = dt.datetime(2026, 10, 28, 6, tzinfo=dt.timezone.utc), dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc)
    rows = [{"id": k, "init_time": t0, "forecast_hour": h, "wind_speed_10m_mean": 4.0} for k in wn.POINTS for h in range(1, 361)]
    # the 12Z init has 300 leads but hour 5 is missing: a count alone would pass it, the first 48 must all be there
    rows += [{"id": k, "init_time": t1, "forecast_hour": h, "wind_speed_10m_mean": 6.0} for k in wn.POINTS for h in range(1, 302) if h != 5]
    init, ser = wn.bq_series(rows)
    assert init == "2026-10-28T06:00:00+00:00"
    p = ser["northave"]
    assert len(p["wspd"]) == 360 and p["t"][-1] == "2026-11-12T06:00:00+00:00"
    assert wn._last_lead(ser, init) == 360 and wn._last_lead({}, init) == 0
