"""The AIS reader: folding aisstream messages, the wake classes, the rolling log (no network)."""
import datetime as dt

from app import ais


def pos(mmsi, lat, lon, sog, cog, name="", kind="PositionReport"):
    return {"MessageType": kind, "MetaData": {"MMSI": mmsi, "ShipName": name, "latitude": lat, "longitude": lon},
            "Message": {kind: {"UserID": mmsi, "Latitude": lat, "Longitude": lon, "Sog": sog, "Cog": cog}}}


def test_classes():
    assert ais.wake_class(60, 30) == "tour" and ais.wake_class(69, 18) == "taxi"
    assert ais.wake_class(52, 20) == "tug" and ais.wake_class(55, 13) == "police" and ais.wake_class(37, 8) == "power"
    assert ais.wake_class(0, None, "LINNEA") == "tour" and ais.wake_class(0, None, "SUNLINER") == "taxi"
    assert ais.wake_class(0, None, "CPD MARINE 1") == "police" and ais.wake_class(0, None, "") == "other"


def test_absorb_and_summarize():
    st = {}
    assert ais.absorb(st, pos(1, 41.8885, -87.6250, 5.8, 270.0, "LINNEA"), now=100) == 1
    assert ais.absorb(st, pos(1, 41.8880, -87.6290, 6.2, 268.0), now=110) == 1
    assert ais.absorb(st, pos(2, 42.5, -87.6, 3.0, 0.0), now=100) is None          # outside the box
    assert ais.absorb(st, pos(3, 41.89, -87.62, 102.3, 360.0), now=100) == 3       # SOG / COG not available
    assert st[3]["track"][0][3] is None and st[3]["track"][0][4] is None
    ais.absorb(st, {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 1},
                    "Message": {"ShipStaticData": {"Name": "LINNEA  ", "Type": 60, "Dimension": {"A": 20, "B": 8, "C": 4, "D": 4}}}})
    assert st[1]["len_m"] == 28 and st[1]["beam_m"] == 8 and st[1]["type"] == 60
    t0 = dt.datetime(2026, 7, 4, 15, tzinfo=dt.timezone.utc)
    d = ais.summarize(st, t0, t0 + dt.timedelta(minutes=4))
    v = next(v for v in d["vessels"] if v["mmsi"] == 1)
    assert v["cls"] == "tour" and len(v["track"]) == 2 and d["fleet"]["1"]["len_m"] == 28
    row = next(r for r in d["seen"] if r["mmsi"] == 1)
    assert row["sog_kn"] == 6.0 and row["cls"] == "tour"
    # the next listen hears only a position: the registry fills in the name and the length
    t15 = t0 + dt.timedelta(days=15)
    st2 = {}; ais.absorb(st2, pos(1, 41.8885, -87.6300, 6.0, 270.0), now=t15.timestamp())
    d2 = ais.summarize(st2, t15, t15, d)
    assert d2["vessels"][0]["len_m"] == 28 and d2["vessels"][0]["cls"] == "tour"
    assert [r["t"] for r in d2["seen"]] == [round(t15.timestamp())]   # the 14-day log drops old rows


def test_usual_speeds():
    seen = [{"cls": "tour", "sog_kn": s} for s in (5.0, 6.0, 6.5, 7.0, 5.5)] + [{"cls": "tug", "sog_kn": 3.0}]
    assert ais.usual_speeds(seen) == {"tour": 6.0}


def test_page_block_drops_the_registry():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("build_world", os.path.join(os.path.dirname(__file__), "..", "tools", "build_world.py"))
    bw = importlib.util.module_from_spec(spec); spec.loader.exec_module(bw)
    assert bw.ais_block(None) is None
    assert bw.ais_block({"vessels": [], "seen": [], "fleet": {"1": {}}, "usual_kn": {}}) == {"vessels": [], "seen": [], "usual_kn": {}}
