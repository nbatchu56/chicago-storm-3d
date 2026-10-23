"""
Stand-up paddling (SUP) wave forecast for the Chicago lakefront.

For each launch spot, every hour for the next ~48 h:
  wind (NWS hourly grid forecast, recent hours from buoys/stations)
  -> fetch over the lake from that spot (ray-cast on the shoreline) and how long the wind has blown
  -> offshore wave height/period (USACE CEM fetch/duration-limited growth)
  -> calibrated to the nearest wave buoy now, fading back to the model over ~6 h
  -> shoaling, refraction and breaking over a beach profile to the paddle zone (~2.5 m deep)
  -> plus local chop from the last few hundred metres of water
  -> a SUP rating with reasons (wave height, wind, gusts, offshore wind, cold water).

Data (all public, no keys): NDBC realtime observations, NWS api.weather.gov gridpoint and
nearshore marine zone forecasts.

    python -m app.sup_waves --out sup.json
"""

import datetime as dt
import functools
import json
import logging
import math
import os
import re
import urllib.request

import numpy as np

from .chicago_nearshore_waves import cem_wave_growth, dean_profile, transform

log = logging.getLogger("klot.sup")
UTC = dt.timezone.utc
UA = os.environ.get("NWS_USER_AGENT", "chicago-storm-3d (github.com/nbatchu56/chicago-storm-3d)")

# Lake Michigan shoreline (lat, lon), west shore north->south, round the south end, up the east shore
SHORE_LL = [
    (45.0, -87.42), (44.40, -87.50), (43.75, -87.70), (43.04, -87.88), (42.73, -87.78), (42.58, -87.81), (42.49, -87.80),
    (42.36, -87.82), (42.25, -87.80), (42.16, -87.76), (42.07, -87.68), (42.02, -87.665), (41.97, -87.640), (41.92, -87.626),
    (41.900, -87.619), (41.892, -87.598), (41.886, -87.611), (41.870, -87.611), (41.866, -87.608), (41.850, -87.609),
    (41.810, -87.580), (41.790, -87.572), (41.760, -87.550), (41.735, -87.530), (41.710, -87.521), (41.680, -87.500),
    (41.660, -87.450), (41.620, -87.380), (41.610, -87.260), (41.630, -87.100), (41.680, -87.000), (41.720, -86.900),
    (41.770, -86.800), (41.850, -86.650), (41.950, -86.570), (42.100, -86.480), (42.400, -86.280), (42.780, -86.210),
    (43.230, -86.340), (43.950, -86.450), (44.630, -86.240), (45.300, -85.600), (46.2, -85.4), (46.2, -87.3)]
REF = (41.8820, -87.6278)
KX = 111.32 * math.cos(math.radians(REF[0]))
KY = 110.57


def xy(lat, lon):
    return (lon - REF[1]) * KX, (lat - REF[0]) * KY


LAKE = [xy(a, b) for a, b in SHORE_LL]


def in_lake(x, y):
    c = False
    j = len(LAKE) - 1
    for i in range(len(LAKE)):
        ax, ay = LAKE[i]; bx, by = LAKE[j]
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            c = not c
        j = i
    return c


# Launch spots. normal = direction (deg) waves come FROM when they hit the beach head-on.
# shelter = [(from_deg, to_deg, factor)] for breakwaters / piers that block part of the lake.
SPOTS = [
    {"id": "montrose", "name": "Montrose Beach", "lat": 41.9655, "lon": -87.6345, "normal": 110,
     "shelter": [(330, 40, 0.5)], "zone": "LMZ741", "note": "Beach sits inside the Montrose hook; north waves wrap in weaker."},
    {"id": "north-ave", "name": "North Avenue Beach", "lat": 41.9145, "lon": -87.6235, "normal": 70,
     "shelter": [], "zone": "LMZ741", "note": "Open beach; revetments north and south reflect waves into choppy, confused water."},
    {"id": "oak-st", "name": "Oak Street Beach", "lat": 41.9025, "lon": -87.6230, "normal": 50,
     "shelter": [], "zone": "LMZ741", "note": "Short beach at the curve; concrete revetment on both sides reflects waves."},
    {"id": "ohio-st", "name": "Ohio Street Beach", "lat": 41.8935, "lon": -87.6125, "normal": 45,
     "shelter": [(80, 200, 0.25)], "zone": "LMZ741", "note": "Navy Pier blocks east and southeast waves; often the calmest downtown water."},
    {"id": "31st", "name": "31st Street Beach", "lat": 41.8385, "lon": -87.6055, "normal": 90,
     "shelter": [(20, 160, 0.45)], "zone": "LMZ742", "note": "The 31st Street Harbor breakwater shelters the beach from open-lake waves."},
    {"id": "63rd", "name": "63rd Street Beach", "lat": 41.7815, "lon": -87.5735, "normal": 65,
     "shelter": [], "zone": "LMZ742", "note": "Open to the northeast; long fetch down the lake on north winds."},
    {"id": "rainbow", "name": "Rainbow Beach", "lat": 41.7595, "lon": -87.5505, "normal": 50,
     "shelter": [], "zone": "LMZ742", "note": "Faces northeast; among the biggest waves in the city on north and northeast winds."},
    {"id": "calumet", "name": "Calumet Beach", "lat": 41.7285, "lon": -87.5255, "normal": 45,
     "shelter": [(100, 200, 0.5)], "zone": "LMZ742", "note": "Calumet Harbor breakwater blocks some southeast waves."},
]

# Buoys and shore stations, tried in order. Seasonal buoys are usually pulled for winter (Oct-Nov).
WAVE_BUOYS = [("45174", 42.135, -87.655), ("45198", 41.892, -87.563), ("45177", 41.794, -87.533),
              ("45186", 42.368, -87.786), ("45187", 42.491, -87.779), ("45007", 42.674, -87.026)]
WIND_STATIONS = ["45174", "45198", "45177", "CHII2", "CNII2", "45007"]

PADDLE_DEPTH_M = 2.5     # where you'd normally paddle, ~100-200 m off a Chicago beach
SUP = {                    # rules of thumb for flat-water SUP (m, m/s)
    "glassy": {"hs": 0.15, "wind": 3.0, "gust": 5.0},
    "good":   {"hs": 0.30, "wind": 5.0, "gust": 7.0},
    "choppy": {"hs": 0.60, "wind": 7.0, "gust": 9.5},
    "rough":  {"hs": 1.00, "wind": 9.0, "gust": 12.0},
}
COLD_WATER_C = 15.5      # below ~60 F: dress for immersion


# ---------------------------------------------------------------------------- data access
def http_get(url, accept="application/json", timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_ndbc(text, station):
    """NDBC realtime2 standard met file -> list of dicts, newest first."""
    lines = text.splitlines()
    head = lines[0].lstrip("#").split()
    idx = {k: i for i, k in enumerate(head)}
    out = []
    for line in lines[2:]:
        row = line.split()
        if len(row) < 5:
            continue
        try:
            t = dt.datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), tzinfo=UTC)
        except ValueError:
            continue

        def v(k):
            try:
                x = row[idx[k]]
                return None if x == "MM" else float(x)
            except (KeyError, IndexError, ValueError):
                return None
        out.append({"t": t, "station": station, "wspd": v("WSPD"), "wdir": v("WDIR"), "gst": v("GST"),
                    "wvht": v("WVHT"), "dpd": v("DPD"), "apd": v("APD"), "mwd": v("MWD"),
                    "wtmp": v("WTMP"), "atmp": v("ATMP")})
    return out


def fetch_ndbc_rows(station, get=http_get):
    try:
        return parse_ndbc(get(f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt", "text/plain"), station)
    except Exception as e:  # noqa: BLE001
        log.info("NDBC %s unavailable: %s", station, e)
        return []


_DUR = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?")


def expand_series(prop, conv=lambda x: x):
    """NWS gridpoint property {'values':[{'validTime':'2026-10-23T18:00:00+00:00/PT2H','value':x}]} -> {hour: value}."""
    out = {}
    for item in (prop or {}).get("values", []):
        if item.get("value") is None:
            continue
        start, dur = item["validTime"].split("/")
        t0 = dt.datetime.fromisoformat(start).astimezone(UTC)
        m = _DUR.fullmatch(dur)
        hours = (int(m.group(1) or 0) * 24 + int(m.group(2) or 0)) if m else 1
        for h in range(max(hours, 1)):
            out[t0 + dt.timedelta(hours=h)] = conv(item["value"])
    return out


def fetch_nws(lat=41.90, lon=-87.60, get=http_get):
    """Hourly wind/gust/wave/temp/sky from the NWS gridpoint covering the lakefront."""
    pts = json.loads(get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", "application/geo+json"))
    grid = pts["properties"]["forecastGridData"]
    raw = json.loads(get(grid, "application/geo+json"))["properties"]
    kmh = lambda v: v / 3.6  # noqa: E731
    series = {
        "wind": expand_series(raw.get("windSpeed"), kmh),
        "gust": expand_series(raw.get("windGust"), kmh),
        "dir": expand_series(raw.get("windDirection")),
        "wave": expand_series(raw.get("waveHeight")),
        "temp": expand_series(raw.get("temperature")),
        "sky": expand_series(raw.get("skyCover")),
        "pop": expand_series(raw.get("probabilityOfPrecipitation")),
    }
    return {"grid": grid.rsplit("/gridpoints/", 1)[-1], "series": series}


def parse_nsh(text, zone):
    """Pull one zone's periods out of an NWS Nearshore Marine Forecast (NSH) product."""
    lines = text.replace("\r", "").split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(zone)), None)
    if start is None:
        return None
    block = []
    for l in lines[start + 1:]:
        if l.strip() == "$$":
            break
        block.append(l)
    body = "\n".join(block)
    periods = []
    for m in re.finditer(r"^\.([A-Z][A-Z0-9 ]+?)\.\.\.(.*?)(?=^\.[A-Z][A-Z0-9 ]+?\.\.\.|\Z)", body, re.S | re.M):
        txt = " ".join(m.group(2).split())
        txt = re.sub(r"(^|\.\s+)([a-z])", lambda g: g.group(1) + g.group(2).upper(), txt.lower())   # sentence case
        periods.append({"name": m.group(1).title(), "text": txt})
    title = next((l.strip() for l in block if l.strip() and not l.startswith(".")), "")
    return {"zone": zone, "title": title, "periods": periods[:4]} if periods else None


def fetch_marine_text(zone, get=http_get):
    for kind in ("marine", "forecast"):
        try:
            js = json.loads(get(f"https://api.weather.gov/zones/{kind}/{zone}/forecast", "application/geo+json"))
            p = js["properties"]
            periods = [{"name": q["name"], "text": q["detailedForecast"]} for q in p.get("periods", [])[:4]]
            if periods:
                return {"zone": zone, "updated": p.get("updated"), "periods": periods}
        except Exception as e:  # noqa: BLE001
            log.info("Marine zone %s (%s) unavailable: %s", zone, kind, e)
    try:  # latest Nearshore Marine Forecast from NWS Chicago
        lst = json.loads(get("https://api.weather.gov/products/types/NSH/locations/LOT", "application/ld+json"))
        items = lst.get("@graph") or lst.get("products") or []
        if items:
            prod = json.loads(get(f"https://api.weather.gov/products/{items[0]['id']}", "application/ld+json"))
            out = parse_nsh(prod.get("productText", ""), zone)
            if out:
                out["updated"] = prod.get("issuanceTime")
                return out
    except Exception as e:  # noqa: BLE001
        log.info("NSH product unavailable: %s", e)
    return None


# ---------------------------------------------------------------------------- physics
def ray_fetch_km(lat, lon, from_deg, normal, step=0.5, max_km=500):
    """Open-water distance upwind from a spot (starting just offshore)."""
    x0, y0 = xy(lat, lon)
    a = math.radians(normal)
    for off in np.arange(0.3, 3.01, 0.1):                      # walk off the beach to open water
        if in_lake(x0 + math.sin(a) * off, y0 + math.cos(a) * off):
            x0 += math.sin(a) * off; y0 += math.cos(a) * off
            break
    else:
        return 0.0
    b = math.radians(from_deg)
    dx, dy = math.sin(b), math.cos(b)
    d = 0.0
    while d < max_km and in_lake(x0 + dx * d, y0 + dy * d):
        d += step if d < 5 else 1.0      # fine near the beach, coarser out in the lake
    return d


def fetch_table(lat, lon, normal):
    """Effective fetch by direction (cos^2-weighted over +-30 deg), every 10 deg."""
    raw = {a: ray_fetch_km(lat, lon, a, normal) for a in range(0, 360, 5)}
    out = {}
    for d in range(0, 360, 10):
        num = den = 0.0
        for off in range(-30, 31, 5):
            w = math.cos(math.radians(off)) ** 2
            num += w * raw[(d + off) % 360]; den += w
        out[d] = num / den
    return out


def interp_dir(table, deg):
    deg %= 360
    a = int(deg // 10) * 10; b = (a + 10) % 360
    f = (deg - a) / 10
    return table[a] * (1 - f) + table[b] * f


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def shelter_factor(spot, from_deg):
    f = 1.0
    for lo, hi, k in spot["shelter"]:
        inside = (lo <= from_deg <= hi) if lo <= hi else (from_deg >= lo or from_deg <= hi)
        if inside:
            f = min(f, k)
    return f


_PROFILE = dean_profile(depth_offshore=10.0, A=0.10, dx=2.0)


def nearshore(hs_off, tp, theta0):
    """Offshore waves (10 m depth) -> Hs at the paddle zone, breaker depth and distance from shore,
    and the wave angle to the shore normal at the paddle zone."""
    if hs_off < 0.03 or abs(theta0) >= 85 or tp < 0.8:
        return 0.0, None, None, None
    return _nearshore_cached(round(hs_off, 2), round(tp, 1), round(theta0 / 2) * 2)


@functools.lru_cache(maxsize=20000)
def _nearshore_cached(hs_off, tp, theta0):
    x, h = _PROFILE
    res = transform(x, h, hs_off, tp, theta0)
    i = int(np.argmin(np.abs(res["h_still"] - PADDLE_DEPTH_M)))
    ang = float(res["theta"][i])
    brk = np.where(res["qb"] > 0.05)[0]
    if brk.size:
        ib = int(brk.max())
        return float(res["hs"][i]), round(float(res["h_still"][ib]), 1), int(res["x"][ib]), ang
    return float(res["hs"][i]), None, None, ang


def wind_duration(series, i, max_h=18):
    """Hours the wind has blown from about the same direction, and the mean speed over them."""
    u0, d0 = series[i]["wind"], series[i]["dir"]
    if u0 is None or d0 is None:
        return 0, 0.0
    speeds = [u0]
    for k in range(i - 1, max(-1, i - max_h - 1), -1):
        u, d = series[k]["wind"], series[k]["dir"]
        if u is None or d is None or abs(angdiff(d, d0)) > 35 or u < 0.45 * u0:
            break
        speeds.append(u)
    return len(speeds), float(np.mean(speeds))


# ---------------------------------------------------------------------------- sea state with memory
G = 9.81
DECAY_H = float(os.environ.get("SUP_DECAY_H", 4.0))   # e-folding time of leftover waves on the lake (hours)
SHIFT_DEG = 45.0                                        # a wind shift beyond this turns the old sea into swell


def _ustar(u):
    return u * math.sqrt(0.001 * (1.1 + 0.035 * u))


def sea_limits(u, fetch_m):
    """Largest Hs/Tp this wind can hold over this fetch (fetch-limited or fully developed), CEM II-2."""
    if u < 0.5 or fetch_m <= 0:
        return 0.0, 0.0
    us = _ustar(u)
    x = G * fetch_m / us ** 2
    hs, tp = 4.13e-2 * us ** 2 / G * math.sqrt(x), 0.651 * us / G * x ** (1 / 3)
    hs_fd, tp_fd = 211.5 * us ** 2 / G, 239.8 * us / G
    return (hs_fd, tp_fd) if hs >= hs_fd else (hs, tp)


def grow(hs_prev, u, fetch_m, dt_h=1.0):
    """One step of the CEM equivalent-duration method: the sea already present is treated as if the
    current wind had been blowing long enough to make it, then the wind blows for dt_h more."""
    hs_max, tp_max = sea_limits(u, fetch_m)
    if hs_max <= 0:
        return 0.0, 0.0
    us = _ustar(u)
    x_eq = (max(hs_prev, 0.0) * G / (4.13e-2 * us ** 2)) ** 2
    t_eq = 77.23 * us / G * x_eq ** 0.67 / 3600.0 if x_eq > 0 else 0.0
    x = min((G * (t_eq + dt_h) * 3600.0 / (77.23 * us)) ** (1 / 0.67), G * fetch_m / us ** 2)
    hs, tp = 4.13e-2 * us ** 2 / G * math.sqrt(x), 0.651 * us / G * x ** (1 / 3)
    return (hs_max, tp_max) if hs >= hs_max else (hs, tp)


class SeaState:
    """Wind sea plus leftover swell at one place, stepped hour by hour."""

    def __init__(self):
        self.ws = {"hs": 0.0, "tp": 0.0, "dir": None, "left": False}
        self.swell = []

    def step(self, u, wdir, fetch_fn, dt_h=1.0):
        decay = math.exp(-dt_h / DECAY_H)
        if u is None or wdir is None:                       # no wind data: everything just decays
            self.ws["hs"] *= decay
            for sw in self.swell:
                sw["hs"] *= decay
            return
        ws = self.ws
        if ws["dir"] is not None and ws["hs"] > 0.03 and abs(angdiff(wdir, ws["dir"])) > SHIFT_DEG:
            self.swell.append(dict(ws))                      # wind shifted: old sea keeps running as swell
            ws.update(hs=0.0, tp=0.0)
        ws["dir"] = wdir
        hs_max, tp_max = sea_limits(u, fetch_fn(wdir))
        if ws["hs"] > hs_max * 1.05:                         # wind eased: the extra height dies away slowly
            ws["hs"] = max(hs_max, ws["hs"] * decay)
            ws["tp"] = max(ws["tp"], tp_max)
            ws["left"] = True
        else:
            ws["hs"], ws["tp"] = grow(ws["hs"], u, fetch_fn(wdir), dt_h)
            ws["left"] = False
        for sw in self.swell:                                # opposing wind knocks swell down about twice as fast
            opposed = u > 3.0 and abs(angdiff(wdir, sw["dir"])) > 120
            sw["hs"] *= math.exp(-dt_h / (DECAY_H * (0.5 if opposed else 1.0)))
        self.swell = sorted([sw for sw in self.swell if sw["hs"] >= 0.03], key=lambda c: -c["hs"])[:3]

    def components(self):
        out = [dict(self.ws, kind="wind sea")] if self.ws["hs"] >= 0.02 and self.ws["dir"] is not None else []
        return out + [dict(sw, kind="swell") for sw in self.swell]

    def total_hs(self):
        return math.sqrt(sum(c["hs"] ** 2 for c in self.components()))


def rate(hs, wind, gust, offshore, onshore_angle):
    """SUP rating + plain-language reasons."""
    gust = gust if gust is not None else wind * 1.35
    reasons = []
    level = "no-go"
    for name in ("glassy", "good", "choppy", "rough"):
        t = SUP[name]
        if hs <= t["hs"] and wind <= t["wind"] and gust <= t["gust"]:
            level = name
            break
    if hs > SUP["good"]["hs"]:
        reasons.append(f"{hs * 3.281:.1f} ft waves in the paddle zone")
    if wind > SUP["good"]["wind"]:
        reasons.append(f"wind {wind * 2.237:.0f} mph")
    if gust > SUP["good"]["gust"]:
        reasons.append(f"gusts {gust * 2.237:.0f} mph")
    if offshore and wind > 3.5:
        # offshore wind flattens the water near shore but pushes you out into the lake
        order = ["glassy", "good", "choppy", "offshore", "rough", "no-go"]
        worst = "offshore" if wind <= 6.5 else "no-go"
        if order.index(level) < order.index(worst):
            level = worst
        reasons.insert(0, f"offshore wind {wind * 2.237:.0f} mph pushes boards away from shore")
    elif abs(onshore_angle) > 60 and wind > 4:
        reasons.append("cross-shore wind: drift along the beach")
    if not reasons:
        reasons.append("light wind and small waves")
    return level, reasons


# ---------------------------------------------------------------------------- build
def build(now=None, hours=48, get=http_get):
    now = (now or dt.datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    sources = {"buoys": [], "wind": None, "nws_grid": None, "errors": []}

    # observations (recent wind history + current waves + water temperature)
    obs = {st: fetch_ndbc_rows(st, get) for st in dict.fromkeys(WIND_STATIONS + [b[0] for b in WAVE_BUOYS])}
    recent = lambda rows: [r for r in rows if now - r["t"] < dt.timedelta(hours=3)]  # noqa: E731
    wind_obs = []
    for st in WIND_STATIONS:
        rows = [r for r in obs.get(st, []) if r["wspd"] is not None and r["wdir"] is not None]
        if recent(rows):
            wind_obs = rows
            sources["wind"] = st
            break
    wave_now = None
    for st, la, lo in WAVE_BUOYS:
        rows = [r for r in recent(obs.get(st, [])) if r["wvht"] is not None]
        if rows:
            wave_now = dict(rows[0], lat=la, lon=lo)
            sources["buoys"].append(st)
            break
    wtmp = None
    for st in WIND_STATIONS + [b[0] for b in WAVE_BUOYS]:
        rows = [r for r in recent(obs.get(st, [])) if r["wtmp"] is not None]
        if rows:
            wtmp = {"c": rows[0]["wtmp"], "station": st, "t": rows[0]["t"].isoformat()}
            break

    try:
        nws = fetch_nws(get=get)
        sources["nws_grid"] = nws["grid"]
    except Exception as e:  # noqa: BLE001
        nws = {"series": {k: {} for k in ("wind", "gust", "dir", "wave", "temp", "sky", "pop")}}
        sources["errors"].append(f"NWS forecast unavailable: {e}")
        log.warning("NWS forecast unavailable: %s", e)

    # hourly series: 18 h of observed history, then forecast
    times = [now + dt.timedelta(hours=k) for k in range(-18, hours + 1)]
    series = []
    for t in times:
        ser = nws["series"]
        w, d, g = ser["wind"].get(t), ser["dir"].get(t), ser["gust"].get(t)
        if t <= now and wind_obs:
            near = min(wind_obs, key=lambda r: abs((r["t"] - t).total_seconds()))
            if abs((near["t"] - t).total_seconds()) <= 5400:
                w, d, g = near["wspd"], near["wdir"], near["gst"] if near["gst"] is not None else g
        series.append({"t": t, "wind": w, "dir": d, "gust": g, "wave_nws": ser["wave"].get(t),
                       "temp": ser["temp"].get(t), "sky": ser["sky"].get(t), "pop": ser["pop"].get(t)})
    i_now = 18
    if series[i_now]["wind"] is None:
        raise RuntimeError("No wind observation or forecast for the current hour")

    # calibration against the wave buoy, if one is reporting: hindcast the sea at the buoy the same way
    calib = 1.0
    if wave_now and wave_now["wvht"]:
        bt = fetch_table(wave_now["lat"], wave_now["lon"], 90)
        sea = SeaState()
        for k in range(0, i_now + 1):
            sea.step(series[k]["wind"], series[k]["dir"], lambda d: interp_dir(bt, d) * 1000)
        hs_model = sea.total_hs()
        if hs_model > 0.1:
            calib = float(np.clip(wave_now["wvht"] / hs_model, 0.5, 2.0))
    hist_hours = sum(1 for k in range(0, i_now) if series[k]["wind"] is not None)
    if hist_hours < 6:
        sources["errors"].append(f"only {hist_hours} h of wind history: waves already on the lake may be underestimated")

    spots_out = []
    for spot in SPOTS:
        table = fetch_table(spot["lat"], spot["lon"], spot["normal"])
        fetch_m = lambda d, t=table: interp_dir(t, d) * 1000  # noqa: E731
        sea = SeaState()
        hourly = []
        for k in range(0, len(series)):
            s_ = series[k]
            sea.step(s_["wind"], s_["dir"], fetch_m)
            if k < i_now or s_["wind"] is None or s_["dir"] is None:
                continue
            lead = k - i_now
            c_t = 1 + (calib - 1) * math.exp(-lead / 6.0)          # buoy correction fades over ~6 h
            comps, pz = [], []
            for c in sea.components():
                hs_off = c["hs"] * c_t * shelter_factor(spot, c["dir"])
                tp = c["tp"] * c_t ** (2 / 3)                        # growth laws: Tp ~ Hs^(2/3)
                theta0 = angdiff(c["dir"], spot["normal"])
                hs_pz, brk_depth, brk_dist, ang = nearshore(hs_off, tp, theta0)
                comps.append({"kind": c["kind"], "hs_off": hs_off, "tp": tp, "dir": c["dir"], "hs": hs_pz,
                              "ang": ang, "brk": (brk_depth, brk_dist), "left": c["kind"] == "swell" or c.get("left", False)})
            chop, chop_tp, _ = cem_wave_growth(s_["wind"], 300.0, 1)      # last ~300 m of water
            hs_paddle = math.sqrt(sum(c["hs"] ** 2 for c in comps) + chop ** 2)
            hs_off_tot = math.sqrt(sum(c["hs_off"] ** 2 for c in comps))
            swell_pz = math.sqrt(sum(c["hs"] ** 2 for c in comps if c["kind"] == "swell"))
            left_pz = math.sqrt(sum(c["hs"] ** 2 for c in comps if c["left"]))
            dom = max(comps, key=lambda c: c["hs"], default=None)
            theta_w = angdiff(s_["dir"], spot["normal"])
            offshore = abs(theta_w) > 100
            n_dur, _ = wind_duration(series, k)
            level, reasons = rate(hs_paddle, s_["wind"], s_["gust"], offshore, theta_w)
            if left_pz > 0.1 and left_pz > 0.6 * hs_paddle:
                reasons.append(f"leftover waves from earlier wind ({left_pz * 3.281:.1f} ft), dying down slowly")
            brk = max((c for c in comps if c["brk"][0] is not None), key=lambda c: c["hs"], default=None)
            use_dom = dom is not None and dom["hs"] > chop
            hourly.append({
                "t": s_["t"].isoformat(), "wind": round(s_["wind"], 1), "gust": None if s_["gust"] is None else round(s_["gust"], 1),
                "dir": round(s_["dir"]), "offshore": offshore, "fetch_km": round(interp_dir(table, s_["dir"]), 1), "duration_h": n_dur,
                "hs_offshore": round(hs_off_tot, 2), "tp": round(dom["tp"] if use_dom else chop_tp, 1),
                "hs_paddle": round(hs_paddle, 2), "chop": round(chop, 2), "swell": round(swell_pz, 2),
                "break_depth_m": brk["brk"][0] if brk else None, "break_dist_m": brk["brk"][1] if brk else None,
                "wave_dir": round(dom["dir"] if use_dom else s_["dir"]), "rating": level, "reasons": reasons,
                "components": [{"kind": c["kind"], "hs": round(c["hs"], 3), "tp": round(c["tp"], 1), "dir": round(c["dir"]),
                                "ang": None if c["ang"] is None else round(c["ang"], 1)} for c in comps if c["hs"] >= 0.01],
                "nws_wave_m": s_["wave_nws"], "air_c": s_["temp"], "sky": s_["sky"], "pop": s_["pop"],
            })
        spots_out.append({k: spot[k] for k in ("id", "name", "lat", "lon", "normal", "zone", "note")} | {"hourly": hourly})

    marine = {}
    for zone in sorted({s["zone"] for s in SPOTS}):
        m = fetch_marine_text(zone, get)
        if m:
            marine[zone] = m

    return {
        "version": 2, "decay_h": DECAY_H, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "start": now.isoformat(),
        "sources": sources, "calibration": round(calib, 2),
        "buoy_now": None if not wave_now else {
            "station": wave_now["station"], "t": wave_now["t"].isoformat(), "hs_m": wave_now["wvht"],
            "dpd_s": wave_now["dpd"], "apd_s": wave_now["apd"], "mwd": wave_now["mwd"]},
        "wind_now": {"speed": series[i_now]["wind"], "dir": series[i_now]["dir"], "gust": series[i_now]["gust"],
                     "source": sources["wind"] or "NWS forecast"},
        "water_temp": wtmp, "cold_water_c": COLD_WATER_C, "thresholds": SUP, "paddle_depth_m": PADDLE_DEPTH_M,
        "marine": marine, "spots": spots_out,
    }


class SupForecaster:
    """Rebuilds sup.json on a schedule inside the web service."""

    def __init__(self, data_dir, interval_min=None):
        import threading
        self.path = os.path.join(data_dir, "sup.json")
        self.interval = float(interval_min or os.environ.get("SUP_INTERVAL_MIN", 30))
        self.status = {"updated": None, "last_error": None, "runs": 0}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        os.makedirs(data_dir, exist_ok=True)

    def run_once(self):
        if not self._lock.acquire(blocking=False):
            return False
        try:
            data = build()
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, separators=(",", ":"))
            os.replace(tmp, self.path)
            self.status.update(updated=data["generated"], last_error=None, errors=data["sources"]["errors"])
            log.info("SUP forecast updated (%d spots)", len(data["spots"]))
            return True
        except Exception as e:  # noqa: BLE001 - keep serving the last forecast
            self.status["last_error"] = f"{type(e).__name__}: {e}"
            log.error("SUP forecast failed: %s", e)
            return False
        finally:
            self.status["runs"] += 1
            self._lock.release()

    def loop(self):
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(max(300.0, self.interval * 60))

    def start(self):
        import threading
        threading.Thread(target=self.loop, name="sup-forecast", daemon=True).start()

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Chicago lakefront SUP wave forecast")
    ap.add_argument("--out", default="sup.json")
    a = ap.parse_args()
    data = build()
    with open(a.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    s = data["spots"][0]["hourly"][0]
    print(f"wrote {a.out}: {len(data['spots'])} spots x {len(data['spots'][0]['hourly'])} h; "
          f"wind {data['wind_now']} buoy {data['buoy_now']} water {data['water_temp']}; "
          f"{data['spots'][0]['name']} now {s['hs_paddle']} m {s['rating']}; errors {data['sources']['errors']}")
