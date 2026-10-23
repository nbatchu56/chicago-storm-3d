"""Offline tests for the SUP wave forecast, with synthetic (made-up) buoy and NWS inputs."""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import sup_waves as sw  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 23, 15, 0, tzinfo=UTC)


def ndbc_text(rows):
    head = "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE\n#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft\n"
    body = ""
    for t, wd, ws, gs, wv, dp in rows:
        body += f"{t:%Y %m %d %H %M} {wd:>4} {ws:>4} {gs:>4} {wv:>5} {dp:>5}   MM  MM 1015.0  12.0  13.5   MM   MM   MM    MM\n"
    return head + body


def make_get(wind_ms=8.0, wind_dir=20, buoy_hs="1.1"):
    rows = [(NOW - dt.timedelta(minutes=10 * k), wind_dir, wind_ms, wind_ms * 1.3, buoy_hs if k % 3 == 0 else "MM", "4" if k % 3 == 0 else "MM") for k in range(0, 24 * 6)]
    grid = "https://api.weather.gov/gridpoints/LOT/75,73"
    vals = lambda v: {"values": [{"validTime": f"{(NOW - dt.timedelta(hours=1)):%Y-%m-%dT%H:00:00+00:00}/PT60H", "value": v}]}
    raw = {"properties": {"windSpeed": vals(wind_ms * 3.6), "windGust": vals(wind_ms * 1.3 * 3.6), "windDirection": vals(wind_dir),
                          "waveHeight": vals(1.2), "temperature": vals(11), "skyCover": vals(60), "probabilityOfPrecipitation": vals(10)}}

    def get(url, accept="", timeout=20):
        if "realtime2/45174" in url:
            return ndbc_text(rows)
        if "ndbc" in url:
            raise OSError("not reporting")
        if "/points/" in url:
            return json.dumps({"properties": {"forecastGridData": grid}})
        if "/gridpoints/" in url:
            return json.dumps(raw)
        if "/zones/forecast/" in url:
            return json.dumps({"properties": {"updated": NOW.isoformat(), "periods": [{"name": "This Afternoon", "detailedForecast": "North winds 15 to 20 kt. Waves 3 to 5 ft."}]}})
        raise OSError(url)
    return get


def by_id(data):
    return {s["id"]: s for s in data["spots"]}


def test_fetch_geometry():
    # Rainbow Beach looking NNE sees most of the lake; Ohio Street looking SE is blocked by geometry or shelter
    r = sw.fetch_table(41.7595, -87.5505, 50)
    assert r[20] > 250
    assert sw.fetch_table(41.9145, -87.6235, 70)[270] < 1.5    # west wind: only the strip of water off the beach


def test_nne_wind_builds_waves_south_side():
    data = sw.build(now=NOW, hours=12, get=make_get(8.0, 20))
    s = by_id(data)
    first = {k: v["hourly"][0] for k, v in s.items()}
    for k, v in first.items():
        print(f"{k:9s} paddle {v['hs_paddle']:.2f} m  offshore {v['hs_offshore']:.2f} m  Tp {v['tp']} s  fetch {v['fetch_km']} km  {v['rating']}")
    assert data["sources"]["wind"] == "45174" and data["buoy_now"]["hs_m"] == 1.1
    assert first["montrose"]["hs_offshore"] < first["63rd"]["hs_offshore"]      # hook shelters north waves
    assert all(v["rating"] in ("choppy", "rough", "no-go") for v in first.values())
    assert data["marine"]["LMZ741"]["periods"][0]["name"] == "This Afternoon"
    assert len(s["rainbow"]["hourly"]) == 13


def test_navy_pier_shelters_ohio_street_from_ese():
    data = sw.build(now=NOW, hours=3, get=make_get(8.0, 110, "MM"))
    s = by_id(data)
    assert s["ohio-st"]["hourly"][0]["hs_offshore"] < 0.5 * s["rainbow"]["hourly"][0]["hs_offshore"]


def test_offshore_wind_is_flagged():
    data = sw.build(now=NOW, hours=3, get=make_get(6.0, 270, "0.2"))
    v = by_id(data)["north-ave"]["hourly"][0]
    assert v["offshore"] and v["hs_paddle"] < 0.2
    assert v["rating"] in ("offshore", "no-go") and "offshore" in v["reasons"][0]


def test_light_wind_glassy():
    data = sw.build(now=NOW, hours=3, get=make_get(1.5, 90, "0.1"))
    assert by_id(data)["ohio-st"]["hourly"][0]["rating"] in ("glassy", "good")


NSH = """FZUS53 KLOT 231445
NSHLOT

Nearshore Marine Forecast
National Weather Service Chicago IL
945 AM CDT Fri Oct 23 2026

LMZ740-232100-
Winthrop Harbor to Wilmette Harbor IL-
945 AM CDT Fri Oct 23 2026

.TODAY...East winds 5 to 10 kt. Waves 1 foot or less.
.TONIGHT...Southeast winds 10 kt. Waves 1 foot or less.
$$

LMZ741-232100-
Wilmette Harbor to Northerly Island IL-
945 AM CDT Fri Oct 23 2026

.TODAY...East winds 5 to 10 kt. Waves 1 foot or less.
.TONIGHT...Southeast winds 10 to 15 kt.
Waves 1 to 2 ft.
.SATURDAY...South winds 10 kt.
$$
"""


def test_parse_nearshore_product():
    m = sw.parse_nsh(NSH, "LMZ741")
    assert m["title"].startswith("Wilmette Harbor") and [p["name"] for p in m["periods"]] == ["Today", "Tonight", "Saturday"]
    assert m["periods"][1]["text"] == "Southeast winds 10 to 15 kt. Waves 1 to 2 ft."
    assert sw.parse_nsh(NSH, "LMZ745") is None
