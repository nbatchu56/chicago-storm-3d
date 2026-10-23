"""
Lakefront layers for the SUP route view.

  geometry   OpenStreetMap via the Overpass API: lake coastline, harbors and lagoons, breakwaters,
             piers, beaches, parks, Lake Shore Drive, tall buildings (data (c) OpenStreetMap contributors, ODbL)
  water temp NOAA GLERL GLSEA daily lake-surface temperature (satellite, ~1.4 km) and ice
             concentration when present, via ERDDAP; plus every NDBC station in the area reporting WTMP
  history    last 48 h of wind, waves and water temperature from those stations

    python -m app.lakefront --out lakefront.json [--probe probe.json]

Every source is optional; what failed is recorded in "notes" (and in --probe for debugging).
"""

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request

log = logging.getLogger("klot.lakefront")
UTC = dt.timezone.utc
UA = os.environ.get("NWS_USER_AGENT", "chicago-storm-3d (github.com/nbatchu56/chicago-storm-3d)")

# Map area: Oak Street to Montrose, with room for the lake (S, W, N, E)
MAP_BBOX = (41.890, -87.660, 41.975, -87.585)
# Temperature area: the Chicago nearshore and a slice of open lake
TEMP_BBOX = (41.60, -87.95, 42.40, -87.10)
REF = (41.8820, -87.6278)          # State & Madison, same origin as the radar scene
KX = 111320.0 * math.cos(math.radians(REF[0]))
KY = 110570.0

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
ERDDAP_HOSTS = ["https://apps.glerl.noaa.gov/erddap", "https://coastwatch.glerl.noaa.gov/erddap"]
GLSEA_CANDIDATES = ["GLSEA_ACSPO_GCS", "GLSEA_GCS"]          # ACSPO GLSEA is current; the older GLSEA_GCS ends in 2023
ICE_CANDIDATES = ["GL_Ice_Concentration_GCS"]
MAX_AGE_DAYS = 5


def xy(lat, lon):
    """metres east/north of State & Madison"""
    return round((lon - REF[1]) * KX, 1), round((lat - REF[0]) * KY, 1)


def http_get(url, data=None, timeout=90, accept="*/*"):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------------------- OpenStreetMap
def overpass_query(b):
    s, w, n, e = b
    bb = f"{s},{w},{n},{e}"
    tall = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:90];
(
  way["natural"="coastline"]({bb});
  way["natural"="water"]({bb});
  relation["natural"="water"]({bb});
  way["leisure"="marina"]({bb});
  way["harbour"]({bb});
  way["man_made"~"^(breakwater|pier|groyne)$"]({bb});
  way["natural"="beach"]({bb});
  way["leisure"="park"]({bb});
  way["landuse"="grass"]({bb});
  way["highway"~"^(motorway|trunk|primary)$"]({bb});
  way["building"]["building:levels"~"^([2-9][0-9]|[1-9][0-9][0-9])$"]({tall});
  way["building"]["height"~"^([6-9][0-9]|[1-9][0-9][0-9])"]({tall});
);
out geom tags;"""


def classify(tags):
    t = tags or {}
    if t.get("natural") == "coastline":
        return "coastline"
    if t.get("man_made") in ("breakwater", "pier", "groyne"):
        return t["man_made"]
    if t.get("natural") == "water" or "harbour" in t or t.get("leisure") == "marina":
        return "water"
    if t.get("natural") == "beach":
        return "beach"
    if t.get("leisure") == "park" or t.get("landuse") == "grass":
        return "park"
    if "highway" in t:
        return "road"
    if "building" in t:
        return "building"
    return None


def building_height(t):
    h = t.get("height", "")
    m = re.match(r"^\s*([0-9.]+)", h)
    if m:
        v = float(m.group(1))
        return round(v * 0.3048, 1) if "ft" in h or "'" in h else v
    try:
        return round(float(t.get("building:levels")) * 3.7, 1)
    except (TypeError, ValueError):
        return None


def fetch_osm(bbox=MAP_BBOX, get=http_get):
    q = overpass_query(bbox).encode()
    last = None
    for url in OVERPASS:
        try:
            js = json.loads(get(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=120))
            break
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("Overpass %s failed: %s", url, e)
    else:
        raise RuntimeError(f"Overpass unavailable: {last}")
    feats = []
    for el in js.get("elements", []):
        kind = classify(el.get("tags"))
        if not kind:
            continue
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings.append(el["geometry"])
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") in ("outer", "") and "geometry" in m:
                    rings.append(m["geometry"])
        for g in rings:
            pts = [xy(p["lat"], p["lon"]) for p in g]
            if len(pts) < 2:
                continue
            f = {"kind": kind, "pts": pts, "closed": pts[0] == pts[-1]}
            name = (el.get("tags") or {}).get("name")
            if name:
                f["name"] = name
            if kind == "building":
                f["h"] = building_height(el.get("tags") or {})
            feats.append(f)
    return feats


# ---------------------------------------------------------------------------- water temperature
def _erddap_grid(host, ds, var, b, get):
    s, w, n, e = b
    url = (f"{host}/griddap/{ds}.json?{var}%5B(last)%5D%5B({s}):1:({n})%5D%5B({w}):1:({e})%5D")
    js = json.loads(get(url, timeout=120))
    tab = js["table"]
    cols = tab["columnNames"]
    it, ila, ilo, iv = cols.index("time"), cols.index("latitude"), cols.index("longitude"), cols.index(var)
    rows = tab["rows"]
    return rows[0][it] if rows else None, [(r[ila], r[ilo], r[iv]) for r in rows]


def _erddap_info(host, ds, get):
    js = json.loads(get(f"{host}/info/{ds}/index.json", timeout=60))
    variables = [r[1] for r in js["table"]["rows"] if r[0] == "variable"]
    return variables


def _erddap_search(host, words, get):
    js = json.loads(get(f"{host}/search/index.json?page=1&itemsPerPage=50&searchFor={urllib.parse.quote(words)}", timeout=60))
    cols = js["table"]["columnNames"]
    idc, tc = cols.index("Dataset ID"), cols.index("Title")
    return [(r[idc], r[tc]) for r in js["table"]["rows"]]


def grid_from_points(points):
    """[(lat, lon, value)] on a regular grid -> compact dict with lat/lon axes and a row-major value list."""
    lats = sorted({round(p[0], 5) for p in points})
    lons = sorted({round(p[1], 5) for p in points})
    li = {v: i for i, v in enumerate(lats)}
    lo = {v: i for i, v in enumerate(lons)}
    vals = [None] * (len(lats) * len(lons))
    for la, ln, v in points:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals[li[round(la, 5)] * len(lons) + lo[round(ln, 5)]] = round(float(v), 2)
    return {"lat0": lats[0], "lat1": lats[-1], "lon0": lons[0], "lon1": lons[-1], "nlat": len(lats), "nlon": len(lons), "v": vals}


def _fresh(t):
    try:
        return (dt.datetime.now(UTC) - dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))).days <= MAX_AGE_DAYS
    except ValueError:
        return False


def fetch_glsea(probe, get=http_get):
    """Lake-surface temperature (deg C) and ice concentration (0-1) grids from GLERL's ERDDAP."""
    out = {}
    for host in ERDDAP_HOSTS:
        cands = list(GLSEA_CANDIDATES) + list(ICE_CANDIDATES)
        try:
            found = _erddap_search(host, "GLSEA", get)
            probe.setdefault("erddap_search", {})[host] = found[:30]
            cands += [d for d, t in found if d not in cands and d.endswith("_GCS")]
        except Exception as e:  # noqa: BLE001
            probe.setdefault("erddap_errors", []).append(f"search {host}: {e}")
        for ds in cands:
            try:
                variables = _erddap_info(host, ds, get)
            except Exception as e:  # noqa: BLE001
                probe.setdefault("erddap_errors", []).append(f"info {host} {ds}: {e}")
                continue
            probe.setdefault("erddap_vars", {})[f"{host}/{ds}"] = variables
            tvar = next((v for v in variables if v.lower() in ("sst", "temp", "lst", "temperature", "surface_temperature")), None)
            ivar = next((v for v in variables if "ice" in v.lower()), None)
            try:
                if tvar and "temp" not in out:
                    t, pts = _erddap_grid(host, ds, tvar, TEMP_BBOX, get)
                    probe.setdefault("erddap_times", {})[f"{ds}.{tvar}"] = t
                    if pts and _fresh(t):
                        out["temp"] = grid_from_points(pts) | {"time": t, "source": f"{ds} ({host})", "var": tvar}
                if ivar and "ice" not in out:
                    t, pts = _erddap_grid(host, ds, ivar, TEMP_BBOX, get)
                    probe.setdefault("erddap_times", {})[f"{ds}.{ivar}"] = t
                    if pts and not _fresh(t):
                        out["ice_status"] = f"no current ice analysis (latest {str(t)[:10]}; NOAA analyzes ice roughly December to May)"
                    if pts and _fresh(t):
                        g = grid_from_points(pts)
                        mx = max([v for v in g["v"] if v is not None] or [0])
                        if mx > 1.5:      # percent -> fraction
                            g["v"] = [None if v is None else round(v / 100.0, 3) for v in g["v"]]
                        out["ice"] = g | {"time": t, "source": f"{ds} ({host})", "var": ivar}
            except Exception as e:  # noqa: BLE001
                probe.setdefault("erddap_errors", []).append(f"grid {host} {ds}: {e}")
            if "temp" in out and "ice" in out:
                return out
        if "temp" in out:
            return out
    return out


def fetch_stations(bbox=TEMP_BBOX, get=http_get):
    """All active NDBC stations in the area."""
    xml = get("https://www.ndbc.noaa.gov/activestations.xml", timeout=60)
    s, w, n, e = bbox
    out = []
    for m in re.finditer(r"<station\s+([^>]*?)/?>", xml):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        try:
            la, lo = float(attrs["lat"]), float(attrs["lon"])
        except (KeyError, ValueError):
            continue
        if s <= la <= n and w <= lo <= e:
            out.append({"id": attrs.get("id", "").upper(), "name": attrs.get("name"), "lat": la, "lon": lo,
                        "owner": attrs.get("owner"), "type": attrs.get("type")})
    return out


def station_history(st, hours=48, get=http_get):
    from .sup_waves import parse_ndbc
    rows = parse_ndbc(get(f"https://www.ndbc.noaa.gov/data/realtime2/{st}.txt", timeout=30), st)
    cut = dt.datetime.now(UTC) - dt.timedelta(hours=hours)
    out = []
    for r in rows:
        if r["t"] < cut:
            break
        out.append({"t": r["t"].isoformat(), "wspd": r["wspd"], "wdir": r["wdir"], "gst": r["gst"],
                    "wvht": r["wvht"], "dpd": r["dpd"], "mwd": r["mwd"], "wtmp": r["wtmp"], "atmp": r["atmp"]})
    return out[::-1]


def probe_osm(probe, bbox=MAP_BBOX, get=http_get):
    s_, w, n, e = bbox
    qs = {
        "coastline_ways": f'[out:json][timeout:60];way["natural"="coastline"]({s_},{w},{n},{e});out tags 5;',
        "lake_relation": '[out:json][timeout:60];relation["name"="Lake Michigan"];out tags;',
        "diversey": f'[out:json][timeout:60];nwr["name"~"Diversey"]({s_},{w},{n},{e});out tags center;',
        "water_rel_in_bbox": f'[out:json][timeout:60];relation["natural"="water"]({s_},{w},{n},{e});out tags;',
        "is_in_lake": '[out:json][timeout:60];is_in(41.93,-87.62);out tags;',
        "water_ways_big": f'[out:json][timeout:60];way["natural"="water"]({s_},{w},{n},{e});out tags 40;',
        "any_coast_nearby": '[out:json][timeout:60];way["natural"="coastline"](41.80,-87.70,42.05,-87.50);out tags 3;',
    }
    import time
    for k, q in qs.items():
        res = None
        for attempt in range(4):
            url = OVERPASS[attempt % len(OVERPASS)]
            try:
                js = json.loads(get(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=90))
                res = [{"type": el["type"], "id": el["id"], "tags": el.get("tags"), "center": el.get("center")} for el in js.get("elements", [])][:12]
                break
            except Exception as ex:  # noqa: BLE001
                res = f"error: {ex} ({url})"
                time.sleep(8)
        probe.setdefault("osm_probe", {})[k] = res
        time.sleep(4)


def build(get=http_get, probe=None):
    probe = probe if probe is not None else {}
    notes = []
    data = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
            "ref": {"lat": REF[0], "lon": REF[1], "units": "m east/north"}, "map_bbox": MAP_BBOX}
    if os.environ.get("OSM_PROBE"):
        probe_osm(probe, get=get)
    try:
        data["osm"] = fetch_osm(get=get)
        data["osm_attribution"] = "© OpenStreetMap contributors (ODbL)"
        probe["osm_counts"] = {k: sum(1 for f in data["osm"] if f["kind"] == k) for k in sorted({f["kind"] for f in data["osm"]})}
    except Exception as e:  # noqa: BLE001
        notes.append(f"map geometry unavailable: {e}")
    try:
        data["lake"] = fetch_glsea(probe, get)
        if "temp" not in data["lake"]:
            notes.append("satellite lake temperature unavailable")
    except Exception as e:  # noqa: BLE001
        notes.append(f"satellite lake temperature unavailable: {e}")
    stations = []
    try:
        for st in fetch_stations(get=get):
            try:
                hist = station_history(st["id"], get=get)
            except Exception as e:  # noqa: BLE001
                probe.setdefault("station_errors", []).append(f"{st['id']}: {e}")
                continue
            if hist:
                st["x"], st["y"] = xy(st["lat"], st["lon"])
                st["history"] = hist
                last_t = next((h for h in reversed(hist) if h["wtmp"] is not None), None)
                st["wtmp_now"] = last_t and {"c": last_t["wtmp"], "t": last_t["t"]}
                stations.append(st)
    except Exception as e:  # noqa: BLE001
        notes.append(f"station list unavailable: {e}")
    data["stations"] = stations
    data["notes"] = notes
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="lakefront.json")
    ap.add_argument("--probe", help="write discovery/debug info here")
    a = ap.parse_args()
    probe = {}
    d = build(probe=probe)
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    if a.probe:
        with open(a.probe, "w") as fh:
            json.dump(probe, fh, indent=1, default=str)
    print(f"wrote {a.out}: {len(d.get('osm', []))} map features, lake {list((d.get('lake') or {}).keys())}, "
          f"{len(d['stations'])} stations; notes {d['notes']}")
