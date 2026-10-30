"""El Niño tracker: parsers, trends and composites on fixed text (no network)."""
import base64
import datetime as dt
import gzip
import os

import numpy as np

from app import enso

ONI = """ SEAS  YR   TOTAL   ANOM
  DJF 1997  25.21  -0.47
  NDJ 1997  28.80   2.37
  DJF 1998  28.71   2.22
  DJF 1999  25.66  -1.55
  JJA 2026  29.09   1.80
  JAS 2026  29.12   2.16
"""
WEEKLY = """ Weekly SST data starts week centered on 2Sept1981

                Nino1+2      Nino3        Nino34        Nino4
 Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA
 30SEP2026     26.0 5.3     29.0 4.1     29.9 3.2     29.9 1.3
 07OCT2026     26.1 5.2     29.0 4.1     30.0 3.3     29.9 1.3
 14OCT2026     26.2 5.2     29.2 4.2     30.0 3.3     29.7 1.1
 21OCT2026     26.3 5.2     29.4 4.3     30.1 3.4     29.8 1.1
 28OCT2026     20.1-0.6     24.7-0.2     26.5-0.2     28.4-0.2
"""
SOI = """(STAND TAHITI - STAND DARWIN)  SEA LEVEL PRESS
YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
2026   1.8   2.4   2.0  -1.1  -1.5  -2.4  -4.0  -1.8  -3.3-999.9-999.9-999.9
 STANDARDIZED    DATA
YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
2026   1.1   1.4   1.2  -0.6  -0.9  -1.4  -2.4  -1.1  -2.0-999.9-999.9-999.9
"""
MEI = """1979     2026
2026    -0.76    -0.95    -1.03    -0.64     0.27     1.52     2.41     2.54     2.37  -999.00  -999.00  -999.00
"""


def test_parsers():
    oni = enso.parse_oni(ONI)
    assert oni[0] == {"season": "DJF", "year": 1997, "anom": -0.47} and oni[-1]["anom"] == 2.16
    w = enso.parse_weekly(WEEKLY)
    assert w[0] == {"t": "2026-09-30", "n12": 5.3, "n3": 4.1, "n34": 3.2, "n4": 1.3} and w[-1]["n12"] == -0.6 and w[-1]["n34"] == -0.2
    s = enso.parse_soi(SOI)
    assert s[0] == {"year": 2026, "month": 1, "v": 1.1} and len(s) == 9                 # the standardized table, missing months out
    m = enso.parse_mei(MEI)
    assert len(m) == 9 and m[-1] == {"year": 2026, "k": 8, "v": 2.37}


def test_phase_trend_and_winters():
    assert [enso.phase(x) for x in (2.2, 0.9, 0.0, -0.8, None)] == ["strong El Niño", "El Niño", "neutral", "La Niña", None]
    assert enso.slope_per_week([3.2, 3.3, 3.3, 3.4]) > 0.05 and enso.slope_per_week([1.0]) is None
    oni = enso.parse_oni(ONI)
    assert enso.winter_oni(oni) == {1996: -0.47, 1997: 2.22, 1998: -1.55}             # DJF 1998 is the winter of Dec 1997
    assert enso.run_length(oni) == 2
    tr = enso.analog_tracks(oni, years=[1997], current=2026)
    assert tr["events"]["1997-98"][tr["seasons"].index("NDJ")] == 2.37 and tr["events"]["1997-98"][tr["seasons"].index("DJF")] == 2.22
    assert tr["current"]["oni"][:3] == [None, 1.80, 2.16]


def test_composite_and_fields():
    rec = {1997: {"x": 10.0}, 2015: {"x": 20.0}, 2000: {"x": 0.0}, 2010: {"x": -10.0}, 2005: {"x": None}}
    cls = {1997: "strong El Niño", 2015: "El Niño", 2000: "neutral", 2010: "La Niña", 2005: "neutral"}
    c = enso.composite(rec, cls, "x")
    assert c["strong El Niño"] == {"mean": 10.0, "n": 1} and c["El Niño"] == {"mean": 15.0, "n": 2} and c["all"] == {"mean": 5.0, "n": 4}
    f = enso.byte_field([41.0, 42.0], [-88.0, -87.0, -86.0], np.array([[-40.0, 0.0, 40.0], [np.nan, 20.0, -20.0]]), -40.0, 40.0)
    q = np.frombuffer(gzip.decompress(base64.b64decode(f["b64"])), np.uint8).reshape(2, 3)
    assert q[0].tolist() == [1, 127, 254] and q[1, 0] == 0


def test_grid_mean_and_ice_days():
    pts = [("t1", 41.0, -87.0, 50.0), ("t2", 41.0, -87.0, 0.0), ("t1", 41.0, -86.0, float("nan")), ("t2", 41.0, -86.0, 30.0)]
    lats, lons, m, a = enso.grid_mean(pts)
    assert m.tolist() == [[25.0, 30.0]] and enso.ice_days_at(a, lats, lons, 41.0, -87.0) == 7


def test_era5_winter(tmp_path):
    rows = ["t,point,wspd,wdir,atmp"]
    t = dt.datetime(2015, 12, 1)
    while t < dt.datetime(2016, 4, 1):
        rows.append(f"{t.isoformat()}+00:00,45198,{12.0 if t.day == 1 else 5.0},{0 if t.month == 12 else 180},0")
        t += dt.timedelta(hours=1)
    for y in (2015, 2016):
        with gzip.open(tmp_path / f"era5_{y}.csv.gz", "wt") as fh:
            fh.write("\n".join([rows[0]] + [r for r in rows[1:] if r.startswith(str(y))]) + "\n")
    e = enso.era5_winter(str(tmp_path), 2015)
    assert e["onshore10_h"] == 24 and e["south10_h"] == 72 and 5.0 < e["wind_ms"] < 5.6     # Dec 1 from the N; Jan-Mar 1sts from the S
    assert enso.era5_winter(str(tmp_path), 2016) is None


def test_build_now_offline():
    def fetch(url, timeout=120):
        if "oni" in url:
            return ONI
        if "RONI" in url:
            return ONI.replace("2.16", "1.69")
        if "wksst" in url:
            return WEEKLY
        if "soi" in url:
            return SOI
        if "mei" in url:
            return MEI
        if "Oisst" in url:
            return "time,zlev,latitude,longitude,anom\nUTC,m,degrees_north,degrees_east,degree_C\n2026-10-15T12:00:00Z,0.0,-0.125,-150.125,3.1\n"
        return "[]"
    d = enso.build_now(climate={"composite": {"max_ice_pct": {"strong El Niño": {"mean": 22.0}, "all": {"mean": 40.0}}}}, fetch=fetch, today=dt.date(2026, 10, 30))
    i = d["indices"]
    assert i["status"] == "strong El Niño" and i["strength"] == "strong" and i["onset_year"] == 2026 and i["weekly"]["n34"] == -0.2
    assert d["pacific"]["nlat"] == 1 and d["flags"][0]["kind"] == "ice" and "22%" in d["flags"][0]["text"]
