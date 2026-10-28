"""Earth Engine layers: the pure parts (no Earth Engine)."""
import datetime as dt
import math

import numpy as np

from app import earth

UTC = dt.timezone.utc


def _glsea(val=15.0, time="1970-01-01T00:00:00Z"):
    return {"lat0": 41.6, "lat1": 42.4, "lon0": -87.95, "lon1": -87.1, "nlat": 9, "nlon": 9, "v": [val] * 81, "time": time}


def test_sample_grid_bilinear_and_gaps():
    g = {"lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0, "nlat": 2, "nlon": 2, "v": [0.0, 1.0, 2.0, None]}
    assert earth.sample_grid(g, 0.0, 0.5) == 0.5
    assert abs(earth.sample_grid(g, 0.5, 0.5) - 1.0) < 1e-9          # the missing corner is left out, not zero
    assert earth.sample_grid(g, 5.0, 5.0) is None and earth.sample_grid(None, 0, 0) is None


def test_fusion_prefers_newest_recent_satellite():
    lats, lons = np.array([41.9, 41.905]), np.array([-87.6, -87.595, -87.59])
    now_h = 500000.0
    modis_t = np.array([[10.0, 10.0, np.nan], [10.0, np.nan, np.nan]])
    modis_h = np.array([[now_h - 5, now_h - 80, np.nan], [now_h - 5, np.nan, np.nan]])     # one pixel too old
    ls_t = np.array([[12.0, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    ls_h = np.array([[now_h - 30, np.nan, np.nan], [np.nan, np.nan, np.nan]])           # older than MODIS there
    out, src, counts = earth.fuse_temperature(lats, lons, {"landsat": (ls_t, ls_h), "modis": (modis_t, modis_h)}, _glsea(15.0), now_h)
    assert out[0, 0] == 10.0 and src[0, 0] == 3          # newest wins
    assert abs(out[0, 1] - 15.0) < 1e-9 and src[0, 1] == 1          # 60 h old MODIS -> GLSEA
    assert counts == {"glsea": 4, "landsat": 0, "modis": 2}
    # a satellite pass older than the GLSEA analysis day is already in GLSEA
    gt = dt.datetime.fromtimestamp((now_h - 2) * 3600, UTC).isoformat()
    out2, src2, _ = earth.fuse_temperature(lats, lons, {"landsat": (ls_t, ls_h), "modis": (modis_t, modis_h)}, _glsea(15.0, gt), now_h)
    assert (src2 == 1).all()


def test_upwelling_detects_cold_nearshore():
    lats, lons = earth.grid_axes(earth.TEMP_BBOX, earth.TEMP_D)
    t = np.full((len(lats), len(lons)), 15.0)
    shore = int((-87.62 - lons[0]) / earth.TEMP_D)
    t[:, :shore] = np.nan                                     # land
    t[:, shore - 20:shore - 17] = 9.0                         # a lagoon inland: ignored
    t[:, shore:shore + 5] = 9.0                               # cold strip ~2 km wide
    u = earth.upwelling(lats, lons, t)
    assert u["upwelling"] and abs(u["delta_c"] - 6.0) < 0.6 and u["rows"] > 30
    t[:, shore:shore + 5] = 14.5
    assert not earth.upwelling(lats, lons, t)["upwelling"]


def test_byte_grid_roundtrip():
    lats, lons = np.arange(3) * 0.1, np.arange(4) * 0.1
    a = np.array([[-30.0, -15.0, 0.0, np.nan], [-20, -10, -5, -1], [-29, -28, -2, -3]])
    g = earth.byte_grid(lats, lons, a, -30.0, 0.0)
    b = earth.decode_byte_grid(g)
    assert np.isnan(b[0, 3]) and g["valid"] == 11
    assert np.nanmax(np.abs(b - a)) < 0.2
    f = earth.byte_grid(lats, lons, np.array([[0.5, 5, 50, np.nan]] * 3), 0.0, 60.0, log_scale=True)
    assert abs(earth.decode_byte_grid(f)[1, 2] - 50) < 2


def test_nechad_and_cloud_fraction():
    assert abs(float(earth.nechad_fnu(0.01)) - 2.43) < 0.05 and np.isnan(earth.nechad_fnu(0.2))
    bt = np.array([290.0, 279.0, 260.0, np.nan])
    assert earth.cloud_fraction(bt, None, 288.0, False) == 2 / 3
    assert earth.cloud_fraction(bt, np.array([0.5, 0.1, 0.1, 0.1]), 288.0, True) == 1.0


def test_sun_elevation_sane():
    noon = dt.datetime(2026, 6, 21, 17, 50, tzinfo=UTC)       # ~solar noon in Chicago
    assert 68 < earth.sun_elevation(41.9, -87.6, noon) < 73
    assert earth.sun_elevation(41.9, -87.6, dt.datetime(2026, 6, 21, 5, 0, tzinfo=UTC)) < 0


def test_era5_normals_window():
    ser = {}
    t0 = dt.datetime(1995, 1, 1, tzinfo=UTC)
    for k in range(24 * 365 * 3):
        t = t0 + dt.timedelta(hours=k)
        ser[t] = {"wspd": 10.0 if t.month == 10 else 2.0, "wdir": 180.0, "atmp": 5.0}
    n = earth.era5_normals(ser, dt.datetime(2026, 10, 15, tzinfo=UTC))
    assert n["wind_ms"]["p50"] == 10.0 and n["air_c"]["p50"] == 5.0
    cal = {"sectors": [1.0] * 12}
    cal["sectors"][6] = 1.5                                   # south sector
    assert earth.era5_normals(ser, dt.datetime(2026, 10, 15, tzinfo=UTC), calib=cal)["wind_ms"]["p50"] == 15.0
    assert earth.era5_normals({}, dt.datetime(2026, 10, 15, tzinfo=UTC)) is None


def test_newest_complete_and_gfs_series():
    assert earth.newest_complete({"1000": 49, "2000": 30}, 49) == "1000"
    feats = [{"properties": {"id": "northave", "fh": h, "u_component_of_wind_10m_above_ground": 0.0, "v_component_of_wind_10m_above_ground": -4.0,
                             "temperature_2m_above_ground": 12.34, "total_cloud_cover_entire_atmosphere": 55.5}} for h in (1, 0)]
    s = earth.gfs_series(feats, 1761609600000)["northave"]
    assert s["wspd"] == [4.0, 4.0] and s["wdir"] == [0.0, 0.0] and s["t"][0].startswith("2025-10-28T00:00") and s["cloud"] == [56, 56]


def test_log_passes_dedupes(tmp_path):
    e = {"s1": {"time": "2026-10-23T23:39Z", "orbit": "ASCENDING", "median_db": -15.1, "launches": {"Montrose": -20.0}},
         "s2": {"time": "2026-10-26T16:51", "median_fnu": 3.0, "launches": {"North Ave": 0.7}}}
    assert sorted(earth.log_passes(e, str(tmp_path))) == ["s1", "s2"]
    assert earth.log_passes(e, str(tmp_path)) == []
    assert open(tmp_path / "sat" / "s1_passes.jsonl").read().count("\n") == 1


def test_cloud_top():
    bt = np.array([288.0 - 26.0, 288.0 - 26.0, 288.0])
    m = earth.cloud_mask(bt, None, 288.0, False)
    assert list(m) == [1.0, 1.0, 0.0] and earth.cloud_top_km(bt, m, 288.0) == 4.0
    assert earth.cloud_top_km(bt, np.zeros(3), 288.0) is None


def test_fuse_four_satellites_counts_and_labels():
    import numpy as np
    lats, lons = np.array([41.9, 41.91]), np.array([-87.6, -87.59])
    now_h = 500000.0
    nan = np.full((2, 2), np.nan)
    eco_t, eco_h = np.array([[12.0, np.nan], [np.nan, np.nan]]), np.array([[now_h - 2, np.nan], [np.nan, np.nan]])
    vi_t, vi_h = np.array([[11.0, 11.5], [np.nan, np.nan]]), np.array([[now_h - 20, now_h - 20], [np.nan, np.nan]])
    sat = {"landsat": (nan, nan), "modis": (nan, nan), "viirs": (vi_t, vi_h), "ecostress": (eco_t, eco_h)}
    out, src, counts = earth.fuse_temperature(lats, lons, sat, None, now_h)
    assert out[0, 0] == 12.0 and out[0, 1] == 11.5 and np.isnan(out[1, 0])      # newest pass wins per cell
    assert counts == {"glsea": 0, "landsat": 0, "modis": 0, "viirs": 1, "ecostress": 1}
    assert earth.SAT_LABEL["ecostress"] == "ECOSTRESS" and earth.SAT_LABEL["viirs"] == "VIIRS"
