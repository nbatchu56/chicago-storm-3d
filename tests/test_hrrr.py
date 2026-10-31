"""HRRR wind layer: rotation, the hour plan, cycle discovery, packing and resampling (no network)."""
import datetime as dt
import math
import urllib.parse as up

import numpy as np

from app import hrrr

UTC = dt.timezone.utc


def test_rotation_to_true_north():
    u, v = hrrr.earth_relative(np.array([0.0]), np.array([10.0]), np.array([hrrr.LOV]))
    assert abs(u[0]) < 1e-9 and abs(v[0] - 10) < 1e-9                  # on the central meridian the grid is true north
    u, v = hrrr.earth_relative(np.array([0.0]), np.array([10.0]), np.array([-87.6]))
    a = math.sin(math.radians(38.5)) * math.radians(-87.6 + 97.5)       # ~6.2° at Chicago
    assert abs(math.degrees(a) - 6.16) < 0.05
    assert abs(u[0] - 10 * math.sin(a)) < 1e-9 and abs(v[0] - 10 * math.cos(a)) < 1e-9
    assert abs(math.hypot(u[0], v[0]) - 10) < 1e-9


def test_schedule_hourly_and_synoptic():
    new, syn = dt.datetime(2026, 10, 30, 22, tzinfo=UTC), dt.datetime(2026, 10, 30, 18, tzinfo=UTC)
    p = hrrr.schedule(new, syn)
    ts = [t for t, _, _ in p]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert ts[0] == new - dt.timedelta(hours=12) and p[0][2] == 0             # past hours: their own analysis
    assert p[12] == (new, new, 0)
    assert (new + dt.timedelta(hours=18), new, 18) in p
    assert (new + dt.timedelta(hours=19), syn, 23) in p                       # beyond f18: the 18z run
    assert ts[-1] == syn + dt.timedelta(hours=48)
    only = hrrr.schedule(syn, None)
    assert only[-1] == (syn + dt.timedelta(hours=48), syn, 48)


def test_find_cycles_skips_incomplete():
    now = dt.datetime(2026, 10, 30, 23, 20, tzinfo=UTC)
    posted = {("2026103021", 18), ("2026103018", 48), ("2026103020", 18)}

    def get(u):
        q = dict(up.parse_qsl(up.urlparse(u).query))
        c = q["dir"][6:14] + q["file"][6:8]; f = int(q["file"][-8:-6])
        return b"GRIB...." if (c, f) in posted else b"<html>no file</html>"
    new, syn = hrrr.find_cycles(now, get, hrrr.Throttle(0))
    assert new == dt.datetime(2026, 10, 30, 21, tzinfo=UTC) and syn == dt.datetime(2026, 10, 30, 18, tzinfo=UTC)


def test_url_asks_for_three_records_in_the_box():
    q = dict(up.parse_qsl(up.urlparse(hrrr.url(dt.datetime(2026, 10, 30, 6, tzinfo=UTC), 7)).query))
    assert q["file"] == "hrrr.t06z.wrfsfcf07.grib2" and q["dir"] == "/hrrr.20261030/conus"
    assert q["var_UGRD"] == q["var_VGRD"] == q["var_GUST"] == "on" and q["toplat"] == str(hrrr.BOX[2])


def test_pack_roundtrip_and_clip():
    a = np.array([[-40.0, -1.13, 0.0], [0.26, 5.0, 40.0]])
    s = hrrr.unpack(hrrr.pack(a, True), True, a.shape)
    assert s[0, 0] == -31.75 and s[1, 2] == 31.75 and abs(s[0, 1] + 1.13) <= hrrr.SCALE / 2 and s[1, 1] == 5.0
    g = hrrr.unpack(hrrr.pack(np.array([-3.0, 70.0, 12.3]), False), False, (3,))
    assert g[0] == 0 and g[1] == 63.75 and abs(g[2] - 12.3) <= hrrr.SCALE / 2


def test_resample_and_sample():
    G = hrrr.grid_spec()
    lat, lon = np.meshgrid(np.linspace(41.0, 42.9, 40), np.linspace(-88.6, -86.6, 40), indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    x_km = (lon - (-87.6278)) * 111320 * math.cos(math.radians(41.882)) / 1000
    u, v, g = hrrr.resample(lat, lon, [x_km * 0.1, np.full_like(x_km, -3.0), np.full_like(x_km, 9.0)], G)
    assert u.shape == (G["ny"], G["nx"]) and not np.isnan(u).any()
    w = {"grid": G, "u": [hrrr.pack(u, True)], "v": [hrrr.pack(v, True)], "g": [hrrr.pack(g, False)]}
    uu, vv, gg = hrrr.at(w, 0, 20.0, 5.0)
    assert abs(uu - 2.0) < 0.2 and vv == -3.0 and gg == 9.0                  # linear in x: 0.1 m/s per km
