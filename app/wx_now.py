"""
Weather headlines for the conditions box: NWS alerts for the lakefront, today's SPC severe outlook at
the lakefront, and the latest Midway / O'Hare METARs.

  alerts  api.weather.gov active alerts for LMZ741/LMZ742 (nearshore), ILZ104 (Central Cook) and
          ILC031 (Cook County), which carries county-based watches such as Severe Thunderstorm Watch
  spc     SPC Day 1 categorical outlook (GeoJSON): the highest risk whose polygon contains the lakefront
  metar   aviationweather.gov data API, decoded JSON for KMDW and KORD

Each part is optional; build() returns what it got plus "notes" for what failed. Standard library only.
"""

import datetime as dt
import json

UTC = dt.timezone.utc
ALERT_ZONES = ("LMZ741", "LMZ742", "ILZ104", "ILC031")
SPC_URL = "https://www.spc.noaa.gov/products/outlook/day1otlk_cat.nolyr.geojson"
METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMDW,KORD&format=json&hours=2"
POINT = (41.9203, -87.6195)          # North Avenue Beach (lat, lon)
SPC_ORDER = ["TSTM", "MRGL", "SLGT", "ENH", "MDT", "HIGH"]
SPC_NAMES = {"TSTM": "General thunderstorms", "MRGL": "Marginal risk", "SLGT": "Slight risk", "ENH": "Enhanced risk",
             "MDT": "Moderate risk", "HIGH": "High risk"}
SEVERITY = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1}


def _in_ring(lon, lat, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_geometry(lon, lat, g):
    """Point in a GeoJSON Polygon / MultiPolygon (holes respected)."""
    if not g:
        return False
    polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
    for p in polys:
        if p and _in_ring(lon, lat, [tuple(c[:2]) for c in p[0]]) and not any(_in_ring(lon, lat, [tuple(c[:2]) for c in h]) for h in p[1:]):
            return True
    return False


def _spc_time(s):
    try:
        return dt.datetime.strptime(str(s), "%Y%m%d%H%M").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return None


def parse_spc(text, point=POINT):
    fc = json.loads(text)
    best, props = -1, {}
    any_props = {}
    for f in fc.get("features", []):
        p = f.get("properties") or {}
        any_props = any_props or p
        lab = str(p.get("LABEL", "")).upper()
        if lab in SPC_ORDER and in_geometry(point[1], point[0], f.get("geometry")):
            k = SPC_ORDER.index(lab)
            if k > best:
                best, props = k, p
    src = props or any_props
    out = {"valid": _spc_time(src.get("VALID")), "expire": _spc_time(src.get("EXPIRE")), "issued": _spc_time(src.get("ISSUE"))}
    if best < 0:
        return out | {"cat": None, "label": "No thunderstorms expected", "level": 0}
    lab = SPC_ORDER[best]
    return out | {"cat": lab, "label": props.get("LABEL2") or SPC_NAMES[lab], "level": best, "color": props.get("fill")}


def parse_alerts(text):
    out = []
    for f in json.loads(text).get("features", []):
        p = f.get("properties") or {}
        if p.get("status") not in (None, "Actual") or p.get("messageType") == "Cancel":
            continue
        out.append({"event": p.get("event"), "headline": p.get("headline"), "severity": p.get("severity"),
                    "onset": p.get("onset") or p.get("effective"), "ends": p.get("ends") or p.get("expires"),
                    "sent": p.get("sent"), "area": p.get("areaDesc")})
    seen, uniq = set(), []
    for a in sorted(out, key=lambda a: (-SEVERITY.get(a["severity"], 0), a["onset"] or "")):
        if a["event"] not in seen:          # one line per event type (zones repeat the same alert)
            seen.add(a["event"]); uniq.append(a)
    return uniq


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_metar(text):
    rows = json.loads(text)
    latest = {}
    for r in rows if isinstance(rows, list) else []:
        sid = r.get("icaoId")
        t = r.get("obsTime")
        if not sid or t is None:
            continue
        if sid not in latest or t > latest[sid]["_t"]:
            latest[sid] = r | {"_t": t}
    out = []
    for sid in ("KMDW", "KORD"):
        r = latest.get(sid)
        if not r:
            continue
        wdir = r.get("wdir")
        out.append({"id": sid, "name": {"KMDW": "Midway", "KORD": "O'Hare"}[sid],
                    "t": dt.datetime.fromtimestamp(int(r["_t"]), UTC).isoformat(),
                    "temp_c": _num(r.get("temp")), "dewp_c": _num(r.get("dewp")),
                    "wdir": None if wdir in (None, "VRB") else _num(wdir), "vrb": wdir == "VRB",
                    "wspd_kt": _num(r.get("wspd")), "wgst_kt": _num(r.get("wgst")),
                    "visib": r.get("visib"), "wx": r.get("wxString"), "cover": (r.get("clouds") or [{}])[0].get("cover") if r.get("clouds") else None,
                    "flt": r.get("fltCat"), "raw": r.get("rawOb")})
    return out


def build(get):
    data, notes = {"generated": dt.datetime.now(UTC).isoformat(timespec="seconds")}, []
    try:
        data["alerts"] = parse_alerts(get("https://api.weather.gov/alerts/active?zone=" + ",".join(ALERT_ZONES),
                                          timeout=30, accept="application/geo+json"))
    except Exception as e:  # noqa: BLE001
        notes.append(f"NWS alerts unavailable: {e}")
    try:
        data["spc"] = parse_spc(get(SPC_URL, timeout=30))
    except Exception as e:  # noqa: BLE001
        notes.append(f"SPC outlook unavailable: {e}")
    try:
        data["metar"] = parse_metar(get(METAR_URL, timeout=30, accept="application/json"))
    except Exception as e:  # noqa: BLE001
        notes.append(f"METAR unavailable: {e}")
    data["notes"] = notes
    return data
