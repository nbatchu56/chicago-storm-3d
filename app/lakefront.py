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
MAP_BBOX = (41.855, -87.665, 41.975, -87.560)    # Museum Campus / Loop to Montrose, out past the Harrison-Dever crib
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
LAKE_MICHIGAN_REL = 1205149   # OSM multipolygon "Lake Michigan" (natural=water, water=lake)


def overpass_queries(b):
    """Small queries (one big one times out). The lake shore comes from the member ways of the
    Lake Michigan multipolygon inside the map box; they are used as the shoreline."""
    s, w, n, e = b
    bb = f"{s},{w},{n},{e}"
    head = "[out:json][timeout:120];"
    return {
        "shore": head + f"rel({LAKE_MICHIGAN_REL});way(r)({bb});out geom;",
        "water": head + f'(way["natural"="water"]({bb});way["leisure"="marina"]({bb});way["harbour"]({bb}););out geom tags;',
        # water drawn as multipolygons (e.g. Diversey Harbor), minus the lake itself
        "water_rel": head + f'(rel["natural"="water"]({bb}); - rel({LAKE_MICHIGAN_REL}););out geom tags;',
        "structures": head + f'way["man_made"~"^(breakwater|pier|groyne)$"]({bb});out geom tags;',
        "land": head + f'(way["natural"="beach"]({bb});way["leisure"="park"]({bb});way["highway"~"^(motorway|trunk|primary)$"]({bb}););out geom tags;',
        "buildings": head + f'(way["building"]["building:levels"~"^([2-9][0-9]|[1-9][0-9][0-9])$"]({bb});'
                            f'way["building"]["height"~"^([6-9][0-9]|[1-9][0-9][0-9])"]({bb}););out geom tags;',
    }


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


def is_harbour(t):
    """Water connected to the lake (harbours, marinas, lagoons, basins), which shares the lake temperature."""
    name = (t.get("name") or "").lower()
    return (t.get("water") in ("harbour", "lagoon", "basin", "reservoir_harbour") or "harbour" in t or t.get("leisure") == "marina"
            or any(w in name for w in ("harbor", "harbour", "marina", "lagoon", "slip")))


def join_rings(parts, tol=0.5):
    """Join multipolygon member ways end to end into closed rings."""
    parts = [list(p) for p in parts if len(p) >= 2]
    rings = []
    while parts:
        ring = parts.pop(0)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for i, p in enumerate(parts):
                close = lambda a, b: abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol  # noqa: E731
                if close(ring[-1], p[0]):
                    ring += p[1:]
                elif close(ring[-1], p[-1]):
                    ring += p[::-1][1:]
                elif close(ring[0], p[-1]):
                    ring = p[:-1] + ring
                elif close(ring[0], p[0]):
                    ring = p[::-1][:-1] + ring
                else:
                    continue
                parts.pop(i)
                changed = True
                break
        if len(ring) >= 4:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings.append(ring)
    return rings


def _overpass(q, get, probe=None, tag=""):
    import time
    last = None
    for attempt in range(5):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            return json.loads(get(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=150))
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("Overpass %s %s failed (%s), retrying", tag, url, e)
            time.sleep(6 * (attempt + 1))
    raise RuntimeError(f"Overpass {tag} unavailable: {last}")


def fetch_osm(bbox=MAP_BBOX, get=http_get, probe=None):
    import time
    feats, errors = [], []
    for tag, q in overpass_queries(bbox).items():
        try:
            js = _overpass(q, get, probe, tag)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
            continue
        for el in js.get("elements", []):
            if el.get("type") == "relation":
                tags = el.get("tags") or {}
                parts = [[xy(p["lat"], p["lon"]) for p in m["geometry"]] for m in el.get("members", [])
                         if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry")]
                for ring in join_rings(parts):
                    f = {"kind": "water", "pts": ring, "closed": True, "rel": el["id"]}
                    if tags.get("name"):
                        f["name"] = tags["name"]
                    if is_harbour(tags):
                        f["harbour"] = True
                    feats.append(f)
                continue
            if el.get("type") != "way" or "geometry" not in el:
                continue
            kind = "coastline" if tag == "shore" else classify(el.get("tags"))
            if not kind:
                continue
            pts = [xy(p["lat"], p["lon"]) for p in el["geometry"]]
            if len(pts) < 2:
                continue
            f = {"kind": kind, "pts": pts, "closed": pts[0] == pts[-1]}
            name = (el.get("tags") or {}).get("name")
            if name:
                f["name"] = name
            if kind == "building":
                f["h"] = building_height(el.get("tags") or {})
            if kind == "water" and is_harbour(el.get("tags") or {}):
                f["harbour"] = True
            feats.append(f)
        time.sleep(2)
    if not any(f["kind"] == "coastline" for f in feats):
        errors.append("lake shoreline missing from OpenStreetMap response")
    return feats, errors


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


class _SkipOSM(Exception):
    pass


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


def build(get=http_get, probe=None, skip_osm=False):
    probe = probe if probe is not None else {}
    notes = []
    data = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
            "ref": {"lat": REF[0], "lon": REF[1], "units": "m east/north"}, "map_bbox": MAP_BBOX}
    if os.environ.get("OSM_PROBE"):
        probe_osm(probe, get=get)
    try:
        if skip_osm:
            raise _SkipOSM()
        data["osm"], errs = fetch_osm(get=get, probe=probe)
        data["osm_time"] = data["generated"]
        notes += [f"map: {e}" for e in errs]
        data["osm_attribution"] = "© OpenStreetMap contributors (ODbL)"
        probe["osm_counts"] = {k: sum(1 for f in data["osm"] if f["kind"] == k) for k in sorted({f["kind"] for f in data["osm"]})}
    except _SkipOSM:
        pass
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


class LakefrontUpdater:
    """Keeps lakefront.json fresh inside the web service: temperatures and stations every
    LAKEFRONT_INTERVAL_H (default 6 h), map geometry reused for up to 7 days."""

    def __init__(self, data_dir):
        import threading
        self.path = os.path.join(data_dir, "lakefront.json")
        self.interval_h = float(os.environ.get("LAKEFRONT_INTERVAL_H", 6))
        self.status = {"updated": None, "last_error": None, "runs": 0}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        os.makedirs(data_dir, exist_ok=True)

    def run_once(self):
        if not self._lock.acquire(blocking=False):
            return False
        try:
            old = None
            if os.path.exists(self.path):
                with open(self.path) as fh:
                    old = json.load(fh)
            reuse_map = bool(old and old.get("osm") and old.get("osm_time") and
                             (dt.datetime.now(UTC) - dt.datetime.fromisoformat(old["osm_time"])).days < 7)
            data = build_fn(skip_osm=reuse_map)
            if reuse_map:
                data["osm"], data["osm_time"] = old["osm"], old["osm_time"]
                data["osm_attribution"] = old.get("osm_attribution")
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, separators=(",", ":"))
            os.replace(tmp, self.path)
            self.status.update(updated=data["generated"], last_error=None, notes=data.get("notes"))
            return True
        except Exception as e:  # noqa: BLE001
            self.status["last_error"] = f"{type(e).__name__}: {e}"
            log.error("Lakefront update failed: %s", e)
            return False
        finally:
            self.status["runs"] += 1
            self._lock.release()

    def loop(self):
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(max(1800.0, self.interval_h * 3600))

    def start(self):
        import threading
        threading.Thread(target=self.loop, name="lakefront", daemon=True).start()

    def stop(self):
        self._stop.set()


def build_fn(skip_osm=False):
    return build(skip_osm=skip_osm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="lakefront.json")
    ap.add_argument("--probe", help="write discovery/debug info here")
    ap.add_argument("--reuse", help="earlier lakefront.json: keep its map if it covers the same area and is under 7 days old")
    a = ap.parse_args()
    probe = {}
    prev = None
    if a.reuse and os.path.exists(a.reuse):
        try:
            prev = json.load(open(a.reuse))
            age = dt.datetime.now(UTC) - dt.datetime.fromisoformat(prev.get("osm_time") or prev["generated"])
            if not prev.get("osm") or list(prev.get("map_bbox", [])) != list(MAP_BBOX) or age > dt.timedelta(days=7):
                prev = None
        except Exception as e:  # noqa: BLE001
            log.info("can't reuse %s: %s", a.reuse, e); prev = None
    d = build(probe=probe, skip_osm=prev is not None)
    if prev is not None:
        for k in ("osm", "osm_time", "osm_attribution", "osm_errors"):
            if k in prev:
                d[k] = prev[k]
        d.setdefault("notes", []).append("map reused from the previous snapshot")
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    if a.probe:
        with open(a.probe, "w") as fh:
            json.dump(probe, fh, indent=1, default=str)
    print(f"wrote {a.out}: {len(d.get('osm', []))} map features, lake {list((d.get('lake') or {}).keys())}, "
          f"{len(d['stations'])} stations; notes {d['notes']}")
