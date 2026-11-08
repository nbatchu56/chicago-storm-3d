"""
Boats on the Chicago River from AIS (aisstream.io), for the page's boat-wake model.

    AISSTREAM_KEY=... python -m app.ais --out ais.json [--seconds 240] [--prev old_ais.json]

The snapshot listens to aisstream's websocket for a few minutes over the river and the harbor and writes ais.json:
  window    when it listened (ISO, UTC)
  vessels   every vessel heard inside the box: MMSI, name, AIS type, length / beam (from the static reports), the page's
            wake class (tour, taxi, tug, police, power, other) and its track [[unix s, lat, lon, SOG kn, COG deg], ...]
  fleet     what each MMSI's static reports said (name, type, length, beam), carried forward so a short listen knows them
  seen      a rolling 14-day log, one row per vessel per snapshot (carried forward with --prev): the class, the length,
            the median moving speed on the river. The page takes the boats' usual speeds from it once it has enough rows.

Who carries AIS: the tour boats and water taxis (commercial passenger vessels), the tugs, the police and Coast Guard
boats; most private boats do not, so the page keeps its schedule model for those. Without the key the step does nothing.
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import statistics
import time

log = logging.getLogger("ais")
UTC = dt.timezone.utc
URL = "wss://stream.aisstream.io/v0/stream"
BOX = (41.845, -87.705, 41.945, -87.595)          # S, W, N, E: the river from Belmont to Ping Tom, the lock, the harbor mouth
RIVER_W = -87.6090                                 # west of the lock's lake gate: on the river
KEEP_DAYS = 14
TOUR_MIN_M = 24.0                                  # passenger vessels this long or longer are tour boats; shorter ones water taxis
# the river's passenger fleets by name, for boats heard before their static report (class A sends it every 6 min, so a
# few minutes' listen often has a position but no type): Wendella (tours and the water taxi), Shoreline Sightseeing,
# Chicago's First Lady / Chicago Architecture Center, Mercury. Names only: the lengths come from AIS or the class default.
FLEET = {
    "tour": ("LINNEA", "LUCIA", "LENA", "WENDELLA", "OUILMETTE", "BRIGHT STAR", "MARQUETTE", "STAR OF CHICAGO", "CAP STREETER",
             "SKYLINE QUEEN", "MERCURY", "FIRST LADY", "LEADING LADY", "LITTLE LADY", "CLASSIC LADY", "FAIR LADY", "ODYSSEY", "SPIRIT OF CHICAGO"),
    "taxi": ("SUNLINER", "WATER TAXI", "TAXI"),
}


def wake_class(ais_type, length_m, name=""):
    """The page's wake class from the AIS ship type (ITU-R M.1371 table 53) and the length."""
    t = int(ais_type or 0)
    if 60 <= t <= 69:
        return "tour" if (length_m or 0) >= TOUR_MIN_M else "taxi"
    if t in (31, 32, 52):
        return "tug"
    if t in (35, 51, 55):
        return "police"
    if t in (36, 37):
        return "power"
    n = (name or "").upper()
    if any(w in n for w in ("POLICE", "CPD", "FIRE", "CFD", "USCG", "COAST GUARD")):
        return "police"
    for cls in ("taxi", "tour"):
        if any(w in n for w in FLEET[cls]):
            return cls
    return "other"


def in_box(lat, lon, box=BOX):
    s, w, n, e = box
    return s <= lat <= n and w <= lon <= e


def absorb(state, msg, now=None):
    """Fold one aisstream message into the state {mmsi: vessel}. Returns the MMSI it touched, or None."""
    md = msg.get("MetaData") or {}
    kind = msg.get("MessageType")
    body = (msg.get("Message") or {}).get(kind) or {}
    mmsi = md.get("MMSI") or body.get("UserID")
    if not mmsi:
        return None
    v = state.setdefault(int(mmsi), {"mmsi": int(mmsi), "name": "", "type": 0, "len_m": None, "beam_m": None, "track": []})
    name = (md.get("ShipName") or "").strip()
    if name:
        v["name"] = name
    if kind in ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"):
        lat, lon = body.get("Latitude", md.get("latitude")), body.get("Longitude", md.get("longitude"))
        if lat is None or lon is None or not in_box(lat, lon):
            return None
        sog, cog = body.get("Sog"), body.get("Cog")
        sog = None if sog is None or sog >= 102.2 else round(float(sog), 1)       # 102.3: not available
        cog = None if cog is None or cog >= 360 else round(float(cog), 1)
        t = round(now if now is not None else time.time(), 1)
        v["track"].append([t, round(float(lat), 6), round(float(lon), 6), sog, cog])
        if kind == "ExtendedClassBPositionReport":
            _dims(v, body.get("Dimension"), body.get("Type"))
        return v["mmsi"]
    if kind == "ShipStaticData":
        _dims(v, body.get("Dimension"), body.get("Type"))
        if body.get("Name"):
            v["name"] = body["Name"].strip()
        return v["mmsi"]
    if kind == "StaticDataReport":
        b = body.get("ReportB") or {}
        if b.get("Valid"):
            _dims(v, b.get("Dimension"), b.get("ShipType"))
        a = body.get("ReportA") or {}
        if a.get("Valid") and a.get("Name"):
            v["name"] = a["Name"].strip()
        return v["mmsi"]
    return None


def _dims(v, dim, typ):
    if dim:
        L = (dim.get("A") or 0) + (dim.get("B") or 0)
        B = (dim.get("C") or 0) + (dim.get("D") or 0)
        if 3 <= L <= 400:
            v["len_m"] = float(L)
        if 1 <= B <= 60:
            v["beam_m"] = float(B)
    if typ:
        v["type"] = int(typ)


async def _listen(key, seconds, state):
    import websockets
    s, w, n, e = BOX
    async with websockets.connect(URL, open_timeout=30, close_timeout=5) as ws:
        await ws.send(json.dumps({"APIKey": key, "BoundingBoxes": [[[s, w], [n, e]]]}))
        end = time.time() + seconds
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, end - time.time()))
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if "error" in msg:
                raise RuntimeError(f"aisstream: {msg['error']}")
            absorb(state, msg)


def summarize(state, t0, t1, prev=None):
    vessels, fleet = [], dict((prev or {}).get("fleet") or {})
    for v in state.values():                     # the static data heard before fills in what this listen missed
        f = fleet.get(str(v["mmsi"]))
        if f:
            v["name"] = v["name"] or f.get("name", "")
            v["type"] = v["type"] or f.get("type", 0)
            v["len_m"] = v["len_m"] or f.get("len_m")
            v["beam_m"] = v["beam_m"] or f.get("beam_m")
        if v["name"] or v["type"] or v["len_m"]:
            fleet[str(v["mmsi"])] = {"name": v["name"], "type": v["type"], "len_m": v["len_m"], "beam_m": v["beam_m"]}
    for v in state.values():
        if not v["track"]:
            continue
        v = dict(v)
        v["track"].sort(key=lambda r: r[0])
        v["cls"] = wake_class(v["type"], v["len_m"], v["name"])
        vessels.append(v)
    vessels.sort(key=lambda v: (v["cls"], v["name"]))
    seen = list((prev or {}).get("seen") or [])
    cut = t1.timestamp() - KEEP_DAYS * 86400
    seen = [r for r in seen if r.get("t", 0) >= cut]
    for v in vessels:
        on_river = [r for r in v["track"] if r[2] <= RIVER_W]
        moving = [r[3] for r in on_river if r[3] is not None and r[3] >= 1.0]
        if not on_river:
            continue
        seen.append({"t": round(on_river[-1][0]), "mmsi": v["mmsi"], "name": v["name"], "cls": v["cls"], "len_m": v["len_m"],
                     "sog_kn": round(statistics.median(moving), 1) if moving else None, "n": len(on_river)})
    return {"version": 1, "generated": t1.isoformat(timespec="seconds"),
            "window": {"t0": t0.isoformat(timespec="seconds"), "t1": t1.isoformat(timespec="seconds")},
            "box": BOX, "source": "aisstream.io", "vessels": vessels, "seen": seen, "fleet": fleet}


def usual_speeds(seen, min_rows=5):
    """The median moving speed (kn) on the river by wake class, from the rolling log (classes with enough rows only)."""
    by = {}
    for r in seen or []:
        if r.get("sog_kn") is not None and r.get("cls") in ("tour", "taxi", "tug", "police"):
            by.setdefault(r["cls"], []).append(r["sog_kn"])
    return {k: round(statistics.median(v), 1) for k, v in by.items() if len(v) >= min_rows}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--prev", help="the previous snapshot's ais.json (carries the 14-day log)")
    a = ap.parse_args()
    key = os.environ.get("AISSTREAM_KEY", "").strip()
    if not key:
        print("no AISSTREAM_KEY; skipping")
        return
    prev = None
    if a.prev and os.path.exists(a.prev):
        try:
            prev = json.load(open(a.prev))
        except ValueError:
            prev = None
    state, t0 = {}, dt.datetime.now(UTC)
    try:
        asyncio.run(_listen(key, a.seconds, state))
    except Exception as e:                       # a dropped socket keeps what it heard
        log.warning("listen ended: %s", e)
    d = summarize(state, t0, dt.datetime.now(UTC), prev)
    d["usual_kn"] = usual_speeds(d["seen"])
    with open(a.out, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    print(json.dumps({"vessels": len(d["vessels"]), "classes": {c: sum(1 for v in d["vessels"] if v["cls"] == c) for c in {v["cls"] for v in d["vessels"]}},
                      "seen": len(d["seen"]), "usual_kn": d["usual_kn"]}))


if __name__ == "__main__":
    main()
