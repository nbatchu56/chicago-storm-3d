import json

from app import wx_now

SQ = lambda w, s, e, n: {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}

SPC = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": SQ(-90, 40, -85, 44),
     "properties": {"DN": 2, "VALID": "202610271300", "EXPIRE": "202610281200", "ISSUE": "202610271246",
                    "LABEL": "TSTM", "LABEL2": "General Thunderstorms Risk", "fill": "#C1E9C1"}},
    {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[-88, 41.5], [-87, 41.5], [-87, 42.5], [-88, 42.5], [-88, 41.5]]]]},
     "properties": {"DN": 4, "VALID": "202610271300", "EXPIRE": "202610281200", "ISSUE": "202610271246",
                    "LABEL": "SLGT", "LABEL2": "Slight Risk", "fill": "#F6F67F"}},
    {"type": "Feature", "geometry": SQ(-80, 30, -79, 31), "properties": {"LABEL": "HIGH", "LABEL2": "High Risk"}},
]}


def test_spc_highest_category_at_point():
    s = wx_now.parse_spc(json.dumps(SPC))
    assert s["cat"] == "SLGT" and s["level"] == 2 and s["label"] == "Slight Risk"
    assert s["valid"].startswith("2026-10-27T13:00") and s["expire"].startswith("2026-10-28T12:00")


def test_spc_none_and_holes():
    hole = {"type": "Polygon", "coordinates": [[[-89, 41], [-86, 41], [-86, 43], [-89, 43], [-89, 41]],
                                               [[-88, 41.8], [-87, 41.8], [-87, 42.1], [-88, 42.1], [-88, 41.8]]]}
    fc = {"features": [{"geometry": hole, "properties": {"LABEL": "MRGL", "VALID": "202610271300"}}]}
    s = wx_now.parse_spc(json.dumps(fc))
    assert s["cat"] is None and s["level"] == 0 and s["valid"]
    assert wx_now.parse_spc(json.dumps({"features": []}))["cat"] is None


def test_alerts_dedup_and_sort():
    feats = [{"properties": {"event": "Small Craft Advisory", "severity": "Minor", "status": "Actual", "messageType": "Alert",
                             "headline": "SCA", "onset": "2026-10-27T18:00:00-05:00", "ends": "2026-10-28T06:00:00-05:00"}},
             {"properties": {"event": "Small Craft Advisory", "severity": "Minor", "status": "Actual", "headline": "SCA dup"}},
             {"properties": {"event": "Gale Warning", "severity": "Moderate", "status": "Actual", "headline": "Gale"}},
             {"properties": {"event": "Test Message", "status": "Test"}}]
    a = wx_now.parse_alerts(json.dumps({"features": feats}))
    assert [x["event"] for x in a] == ["Gale Warning", "Small Craft Advisory"]


def test_metar_latest_per_station():
    rows = [{"icaoId": "KMDW", "obsTime": 1793140000, "temp": 12.2, "dewp": 3.9, "wdir": 230, "wspd": 14, "wgst": 24,
             "visib": "10+", "rawOb": "KMDW 272253Z 23014G24KT 10SM FEW250 12/04 A2990", "clouds": [{"cover": "FEW", "base": 25000}], "fltCat": "VFR"},
            {"icaoId": "KMDW", "obsTime": 1793136400, "temp": 13, "wdir": 220, "wspd": 10},
            {"icaoId": "KORD", "obsTime": 1793140000, "temp": 11.1, "wdir": "VRB", "wspd": 3}]
    m = wx_now.parse_metar(json.dumps(rows))
    assert [x["id"] for x in m] == ["KMDW", "KORD"]
    assert m[0]["wspd_kt"] == 14 and m[0]["wgst_kt"] == 24 and m[0]["cover"] == "FEW" and m[0]["t"].endswith("+00:00")
    assert m[1]["wdir"] is None and m[1]["vrb"]


def test_build_records_failures():
    def get(url, **kw):
        if "spc" in url:
            raise OSError("403")
        if "aviationweather" in url:
            return "[]"
        return json.dumps({"features": []})
    d = wx_now.build(get)
    assert d["alerts"] == [] and d["metar"] == [] and "spc" not in d
    assert any("SPC" in n for n in d["notes"])
