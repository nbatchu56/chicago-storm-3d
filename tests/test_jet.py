"""Jet stream metrics, quadrants, the OPeNDAP parser and the gust study's statistics (no network)."""
import datetime as dt
import urllib.parse as up

import numpy as np

from app import jet

UTC = dt.timezone.utc


def field(streak_lat, streak_lon, peak=60.0, base=15.0, width=6.0, length=14.0):
    """A westerly jet along streak_lat with a streak (an along-flow maximum) centred at streak_lon."""
    lat = np.arange(18.0, 63.0, 1.0); lon = np.arange(-135.0, -54.0, 1.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    U = base + (peak - base) * np.exp(-((LA - streak_lat) / width) ** 2) * np.exp(-((LO - streak_lon) / length) ** 2)
    U += 20.0 * np.exp(-((LA - 28.0) / 4.0) ** 2)           # a subtropical jet near 28 N
    return lat, lon, U, np.zeros_like(U)


def test_quadrant_geometry():
    # a westerly streak: downstream is east, left is north
    assert jet.quadrant(500, 300, 0, 0, 50, 0) == (500, 300)        # NE of the centre: exit, left
    a, c = jet.quadrant(-500, -300, 0, 0, 50, 0)
    assert a < 0 and c < 0                                          # SW: entrance, right


def test_metrics_quadrants():
    # streak centred NW of Chicago (upstream, to the left... Chicago is downstream and to the right): right-exit
    lat, lon, U, V = field(47.0, -100.0)
    m = jet.metrics(lat, lon, U, V)
    assert m["streak"] and m["quad"] == "right-exit" and m["streak"]["spd"] >= 45
    # streak to the north-east: Chicago upstream and to the right -> right-entrance
    m = jet.metrics(*field(47.0, -78.0))
    assert m["quad"] == "right-entrance"
    # streak to the south-east: Chicago upstream and to the left -> left-entrance; to the south-west: left-exit
    assert jet.metrics(*field(36.0, -78.0))["quad"] == "left-entrance"
    assert jet.metrics(*field(36.0, -97.0))["quad"] == "left-exit"
    # overhead
    m = jet.metrics(*field(41.9, -88.0))
    assert m["quad"] == "core" and abs(m["axis_lat"] - 42.0) <= 1.0
    # a weak jet: no streak
    m = jet.metrics(*field(47.0, -90.0, peak=35.0))
    assert m["streak"] is None and m["quad"] == "none"
    assert abs(m["stj_lat"] - 28.0) < 1.5


def test_metrics_accepts_north_to_south_and_0_360():
    lat, lon, U, V = field(47.0, -100.0)
    m1 = jet.metrics(lat, lon, U, V)
    m2 = jet.metrics(lat[::-1], lon % 360, U[::-1], V[::-1])
    assert m1 == m2


def test_gfs_url_and_schedule():
    c = dt.datetime(2026, 10, 30, 18, tzinfo=UTC)
    q = dict(up.parse_qsl(up.urlparse(jet.gfs_url(c, 6)).query))
    assert q["file"] == "gfs.t18z.pgrb2.1p00.f006" and q["dir"] == "/gfs.20261030/18/atmos" and q["lev_250_mb"] == "on"
    s = jet.gfs_schedule(c)
    ts = [t for t, _, _ in s]
    assert ts == sorted(ts) and ts[0] == c - dt.timedelta(hours=12) and ts[-1] == c + dt.timedelta(hours=48) and len(s) == 21


DODS = """Dataset {
    Float32 lat[lat = 2];
} x;
---------------------------------------------
lat[2]
65.0, 62.5

lon[3]
225.0, 227.5, 230.0

time[2]
1963536.0, 1963542.0

uwnd.uwnd[2][1][2][3]
[0][0][0], 13.0, 14.3, 15.1
[0][0][1], 12.5, 14.1, 15.8
[1][0][0], 13.2, 13.6, 14.5
[1][0][1], 14.0, 13.7, 14.6

uwnd.time[2]
1963536.0, 1963542.0
"""


def test_parse_dods_ascii():
    a, m = jet.parse_dods_ascii(DODS, "uwnd")
    assert a.shape == (2, 1, 2, 3) and a[1, 0, 1, 2] == 14.6 and list(m["lat"]) == [65.0, 62.5] and len(m["time"]) == 2
    assert jet.hours_to_dt(1963536.0) == dt.datetime(2024, 1, 1, tzinfo=UTC)
    assert jet.r1_index(20, 60, -130, -60) == (12, 28, 92, 120)


def test_windows_and_study():
    t0 = dt.datetime(2024, 6, 1, tzinfo=UTC)
    jets, obs = [], {"CHII2": [], "45198": []}
    for d in range(120):
        t = t0 + dt.timedelta(hours=6 * d)
        q = "left-exit" if d % 3 == 0 else "none"
        jets.append({"t": t.isoformat(), "quad": q})
        for k in range(-3, 4):
            u = 5.0 + (d % 4) * 1.5
            gf = 1.6 if q == "left-exit" else 1.3
            for st in obs:
                obs[st].append({"t": t + dt.timedelta(hours=k), "wspd": u, "gst": u * gf})
    S = jet.windows(jets, obs)
    assert len(S) == 120 and abs(S[0]["gf"] - 1.6) < 1e-9 and abs(S[1]["gf"] - 1.3) < 1e-9
    r = jet.gust_study(S)
    le = r["quads"]["left-exit"]
    assert le["n"] == 40 and abs(le["gf_ratio"]["value"] - 1.6 / 1.3) < 1e-3 and le["gf_ratio"]["sig"]
    assert r["quads"]["core"] == {"n": 0}
    t = jet.risk_table({"gust": r})
    assert abs(t["left-exit"]["gust"] - 1.6 / 1.3) < 1e-3 and t["core"]["gust"] is None


def test_no_effect_is_not_significant():
    import random
    for seed, sig in ((0, False), (3, True)):        # seed 3: a 2.5% difference that the interval calls real
        rng = random.Random(seed)
        S = []
        for d in range(400):
            u = rng.uniform(4, 14)
            S.append({"t": None, "day": f"d{d // 4}", "quad": "core" if d % 5 == 0 else "none", "gf": 1.35 + rng.gauss(0, 0.1), "u": u, "gmax": u * 1.35})
        r = jet.gust_study(S)
        assert r["quads"]["core"]["gf_ratio"]["sig"] == sig
        assert jet.risk_table({"gust": r})["core"]["gust"] is None      # too small to change a number either way


def test_stj_study_composites():
    oni = [{"season": "DJF", "year": y + 1, "anom": a} for y, a in [(2000, 2.0), (2001, -1.0), (2002, 0.0), (2003, 1.0)]]
    lat = np.arange(45.0, 9.0, -2.5); lon = np.arange(240.0, 282.5, 2.5)
    t, U = [], []
    for y in range(2000, 2005):
        for m in range(1, 13):
            t.append((dt.datetime(y, m, 1, tzinfo=UTC) - dt.datetime(1800, 1, 1, tzinfo=UTC)).total_seconds() / 3600)
            w = (y if m == 12 else y - 1)
            jl = {2000: 25.0, 2001: 32.0, 2002: 29.0, 2003: 27.0}.get(w, 29.0)
            U.append(np.repeat(40 * np.exp(-((lat - jl) / 4) ** 2)[:, None], len(lon), axis=1))
    r = jet.stj_study(np.array(t), lat, lon, np.array(U), oni, base=(2000, 2004))
    by = {w["y"]: w for w in r["winters"]}
    assert abs(by[2000]["lat"] - 25.0) < 0.6 and by[2000]["class"] == "strong El Niño" and abs(by[2001]["lat"] - 32.0) < 0.6
    assert r["composite"]["strong El Niño"]["lat"] < r["composite"]["La Niña"]["lat"]
