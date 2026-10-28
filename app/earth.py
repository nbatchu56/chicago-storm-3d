"""
Satellite and model layers from Google Earth Engine for the Lakefront Local page (earth.json).

  water     nearshore lake-surface temperature: the newest clear-sky MODIS (Terra + Aqua, day + night, 1 km) or
            Landsat 8/9 thermal (100 m) pixel from the last 48 h, NOAA GLSEA (from lakefront.json) elsewhere,
            on a 0.005 deg grid; plus an upwelling check (nearshore vs 8-15 km out along the Chicago shore)
  s1        Sentinel-1 radar backscatter (VV, dB) over the map: rough water is bright, wind-sheltered water dark
  s2        Sentinel-2 water turbidity (Nechad 2009, red band, FNU) from the newest clear pass over the map
  goes      GOES-19 cloud cover now (10.3 um brightness vs the GFS surface temperature; visible reflectance by day)
  gfs       NOAA GFS 0.25 deg: 10 m wind, 2 m air and cloud cover for 48 h at the lakefront (a second opinion)
  normals   ERA5 (cached by app.era5) wind and air for this time of year, ERA5-Land lake temperature normal

Every part is optional; failures go to "notes". Credentials as in app.weathernext (GEE_SERVICE_ACCOUNT_KEY).

    python -m app.earth --out earth.json --lakefront lakefront.json [--era5 learn/era5] [--prev old_earth.json]
"""

import argparse
import base64
import datetime as dt
import gzip
import json
import logging
import math
import os

import numpy as np

log = logging.getLogger("earth")
UTC = dt.timezone.utc
TEMP_BBOX = (41.60, -87.95, 42.40, -87.10)      # S, W, N, E (as app.lakefront)
MAP_BBOX = (41.855, -87.665, 41.990, -87.560)
TEMP_D = 0.005
MAP_D = 0.0005
SAT_MAX_AGE_H = 72.0              # and newer than the GLSEA analysis it would replace
NORTH_AVE = (41.915, -87.610)                   # ~1 km off North Ave Beach
POINTS = {"northave": (41.915, -87.610), "45198": (41.892, -87.563), "45174": (42.135, -87.655)}
LAUNCHES = {"Montrose": (41.96741, -87.63436), "North Ave": (41.91426, -87.62252), "Diversey": (41.92649, -87.63272)}
UPWELL_ROWS = (41.85, 42.06)
UPWELL_FLAG_C = 2.5


# ---------------------------------------------------------------------------- grids
def grid_axes(bbox, d):
    s, w, n, e = bbox
    nx, ny = int(round((e - w) / d)), int(round((n - s) / d))
    lats = s + d * (np.arange(ny) + 0.5)
    lons = w + d * (np.arange(nx) + 0.5)
    return lats, lons


def compute_grid(ee, image, bbox, d):
    """Image -> {band: array (ny, nx), row 0 = south} on a regular lat/lon grid (cell centres)."""
    s, w, n, e = bbox
    nx, ny = int(round((e - w) / d)), int(round((n - s) / d))
    arr = ee.data.computePixels({
        "expression": image, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": nx, "height": ny}, "crsCode": "EPSG:4326",
                 "affineTransform": {"scaleX": d, "shearX": 0, "translateX": w, "shearY": 0, "scaleY": -d, "translateY": s + ny * d}}})
    return {k: np.flipud(np.asarray(arr[k], float)) for k in arr.dtype.names}


def sample_grid(g, lat, lon):
    """Bilinear value of a page-format grid (lat0..lat1 centres, row-major from south, None = no data)."""
    if not g or not g.get("v"):
        return None
    nlat, nlon = g["nlat"], g["nlon"]
    fy = (lat - g["lat0"]) / ((g["lat1"] - g["lat0"]) or 1) * (nlat - 1)
    fx = (lon - g["lon0"]) / ((g["lon1"] - g["lon0"]) or 1) * (nlon - 1)
    if fx < -0.5 or fy < -0.5 or fx > nlon - 0.5 or fy > nlat - 0.5:
        return None
    i0, j0 = min(max(int(math.floor(fx)), 0), nlon - 1), min(max(int(math.floor(fy)), 0), nlat - 1)
    acc = wsum = 0.0
    for j, i in ((j0, i0), (j0, i0 + 1), (j0 + 1, i0), (j0 + 1, i0 + 1)):
        if 0 <= i < nlon and 0 <= j < nlat:
            v = g["v"][j * nlon + i]
            if v is not None:
                wgt = (1 - abs(fx - i)) * (1 - abs(fy - j))
                if wgt > 0:
                    acc += v * wgt
                    wsum += wgt
    return acc / wsum if wsum > 0 else None


def page_grid(lats, lons, a, nd=2):
    v = [None if not np.isfinite(x) else round(float(x), nd) for x in a.ravel()]
    return {"lat0": round(float(lats[0]), 5), "lat1": round(float(lats[-1]), 5), "lon0": round(float(lons[0]), 5),
            "lon1": round(float(lons[-1]), 5), "nlat": len(lats), "nlon": len(lons), "v": v}


def byte_grid(lats, lons, a, lo, hi, log_scale=False):
    """Array -> compact uint8 grid (0 = no data, 1..255 over lo..hi) for page textures."""
    x = np.asarray(a, float)
    ok = np.isfinite(x)
    y = np.where(ok, x, lo)
    if log_scale:
        y, lo_, hi_ = np.log10(np.clip(y, lo, hi) + 1.0), math.log10(lo + 1.0), math.log10(hi + 1.0)
    else:
        lo_, hi_ = lo, hi
    q = np.where(ok, 1 + np.round(np.clip((y - lo_) / (hi_ - lo_), 0, 1) * 254), 0).astype(np.uint8)
    return {"lat0": round(float(lats[0]), 6), "lat1": round(float(lats[-1]), 6), "lon0": round(float(lons[0]), 6),
            "lon1": round(float(lons[-1]), 6), "nlat": len(lats), "nlon": len(lons), "lo": lo, "hi": hi, "log": log_scale,
            "b64": base64.b64encode(gzip.compress(q.tobytes(), 9)).decode(), "valid": int(ok.sum())}


def decode_byte_grid(g):
    q = np.frombuffer(gzip.decompress(base64.b64decode(g["b64"])), np.uint8).reshape(g["nlat"], g["nlon"]).astype(float)
    f = (q - 1) / 254
    if g.get("log"):
        lo_, hi_ = math.log10(g["lo"] + 1), math.log10(g["hi"] + 1)
        v = 10 ** (lo_ + f * (hi_ - lo_)) - 1
    else:
        v = g["lo"] + f * (g["hi"] - g["lo"])
    v[q == 0] = np.nan
    return v


# ---------------------------------------------------------------------------- water temperature
def fuse_temperature(lats, lons, sat, glsea, now_h, max_age_h=SAT_MAX_AGE_H):
    """Per cell: the newest satellite observation no older than max_age_h and newer than the GLSEA analysis day
    (GLSEA already holds the older passes), else GLSEA; land stays empty.
    sat: {name: (temp_c array, obs_hours array)} on the grid; glsea: page-format grid or None; water: bool array."""
    ny, nx = len(lats), len(lons)
    out = np.full((ny, nx), np.nan)
    src = np.zeros((ny, nx), np.int8)                  # 0 none, 1 GLSEA, 2.. satellites in order
    best_h = np.full((ny, nx), -np.inf)
    floor_h = now_h - max_age_h
    try:
        gt = dt.datetime.fromisoformat(str((glsea or {}).get("time")).replace("Z", "+00:00"))
        floor_h = max(floor_h, gt.timestamp() / 3600.0 + 12.0)      # GLSEA's day: the analysis is for ~12 UTC
    except ValueError:
        pass
    names = list(sat)
    for k, name in enumerate(names):
        t, h = sat[name]
        ok = np.isfinite(t) & np.isfinite(h) & (h >= floor_h) & (h <= now_h + 1) & (h > best_h)
        out[ok], best_h[ok], src[ok] = t[ok], h[ok], 2 + k
    if glsea:
        for j in range(ny):
            for i in range(nx):
                if src[j, i] == 0:
                    v = sample_grid(glsea, lats[j], lons[i])
                    if v is not None:
                        out[j, i], src[j, i] = v, 1
    counts = {"glsea": int((src == 1).sum()), **{n: int((src == 2 + k).sum()) for k, n in enumerate(names)}}
    return out, src, counts


def upwelling(lats, lons, temp, rows=UPWELL_ROWS, near_km=(0.0, 2.0), off_km=(8.0, 15.0), flag_c=UPWELL_FLAG_C, min_run=10):
    """Nearshore vs offshore water along the Chicago shore: for each grid row in the band, the first water cell from
    the west is the shore; compare 0-2 km out with 8-15 km out. A colder nearshore by flag_c or more is upwelling."""
    dlon_km = (lons[1] - lons[0]) * 111.32 * math.cos(math.radians(float(np.mean(rows))))
    deltas, nears, offs = [], [], []
    for j, la in enumerate(lats):
        if not rows[0] <= la <= rows[1]:
            continue
        row = temp[j]
        fin = np.isfinite(row)
        run = np.convolve(fin.astype(int), np.ones(min_run, int), "valid") == min_run   # open water, not a lagoon or harbour
        if not run.any():
            continue
        i0 = int(np.argmax(run))
        km = (np.arange(len(row)) - i0) * dlon_km
        nm = row[(km >= near_km[0]) & (km <= near_km[1]) & np.isfinite(row)]
        om = row[(km >= off_km[0]) & (km <= off_km[1]) & np.isfinite(row)]
        if len(nm) and len(om):
            nears.append(float(np.mean(nm)))
            offs.append(float(np.mean(om)))
            deltas.append(offs[-1] - nears[-1])
    if not deltas:
        return None
    d = float(np.median(deltas))
    return {"near_c": round(float(np.median(nears)), 2), "off_c": round(float(np.median(offs)), 2), "delta_c": round(d, 2),
            "rows": len(deltas), "upwelling": d >= flag_c, "max_delta_c": round(float(np.max(deltas)), 2)}


def water_image(ee, now):
    """MODIS + Landsat newest clear-sky water temperature (deg C) and observation time (hours since epoch)."""
    start, end = ee.Date((now - dt.timedelta(days=4)).isoformat()), ee.Date((now + dt.timedelta(hours=1)).isoformat())
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    water = occ.gte(50)
    water_modis = water.focalMin(1200, "circle", "meters")          # keep 1 km pixels clear of the shore
    lon_h = -87.6 / 15.0

    def modis(cid, part):
        def f(img):
            good = img.select(f"QC_{part}").bitwiseAnd(3).lte(1)
            t = img.select(f"LST_{part}_1km").multiply(0.02).subtract(273.15)
            when = ee.Image.constant(ee.Number(img.date().millis()).divide(3.6e6)).add(img.select(f"{part}_view_time").multiply(0.1)).subtract(lon_h)
            return t.rename("t").addBands(when.rename("h").toFloat()).updateMask(good.And(water_modis)).toFloat()
        return ee.ImageCollection(cid).filterDate(start, end).map(f)

    # an empty, fully masked image keeps the mosaics' bands when no pass is in the window
    blank = ee.ImageCollection([ee.Image.constant([0, 0]).rename(["t", "h"]).toFloat().updateMask(ee.Image.constant(0))])
    md = modis("MODIS/061/MOD11A1", "Day").merge(modis("MODIS/061/MOD11A1", "Night")) \
        .merge(modis("MODIS/061/MYD11A1", "Day")).merge(modis("MODIS/061/MYD11A1", "Night")).merge(blank)
    m = md.qualityMosaic("h")

    def landsat(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(0b11111).eq(0)
        t = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        h = ee.Image.constant(ee.Number(img.date().millis()).divide(3.6e6)).toFloat()
        return t.rename("t").addBands(h.rename("h")).updateMask(clear.And(water)).toFloat()

    s, w, n, e = TEMP_BBOX
    box = ee.Geometry.Rectangle([w, s, e, n])
    ls = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")) \
        .filterBounds(box).filterDate(start, end).map(landsat).merge(blank)
    lsm = ls.qualityMosaic("h")
    lst = lsm.select("t").focalMean(250, "circle", "meters")       # 30 m pixels -> the 500 m grid without aliasing
    img = m.select(["t", "h"], ["modis_t", "modis_h"]).addBands(lst.rename("ls_t")).addBands(lsm.select("h").rename("ls_h")) \
        .addBands(water.rename("water"))
    return img


def build_water(ee, now, glsea):
    lats, lons = grid_axes(TEMP_BBOX, TEMP_D)
    img = water_image(ee, now)
    g = compute_grid(ee, img.unmask(-9999), TEMP_BBOX, TEMP_D)
    g = {k: np.where(v < -9000, np.nan, v) for k, v in g.items()}
    now_h = now.timestamp() / 3600.0
    sat = {"landsat": (g["ls_t"], g["ls_h"]), "modis": (g["modis_t"], g["modis_h"])}
    for k in sat:                                                  # computePixels gives masked pixels as 0 or nan
        t, h = sat[k]
        bad = ~np.isfinite(h) | (h < 1e5)
        sat[k] = (np.where(bad, np.nan, t), np.where(bad, np.nan, h))
    water = g["water"] > 0.5
    fused, src, counts = fuse_temperature(lats, lons, sat, glsea, now_h)
    fused[~water] = np.nan                                         # GLSEA's 1.4 km cells reach over the shore
    src[~water] = 0
    counts = {"glsea": int((src == 1).sum()), "landsat": int((src == 2).sum()), "modis": int((src == 3).sum())}
    times = {}
    for k, (t, h) in sat.items():
        hh = h[np.isfinite(h) & (now_h - h <= SAT_MAX_AGE_H)]
        if len(hh):
            times[k] = dt.datetime.fromtimestamp(float(np.max(hh)) * 3600, UTC).isoformat(timespec="minutes")
    grid = page_grid(lats, lons, fused)
    times = {k: v for k, v in times.items() if counts.get(k)}              # passes GLSEA already holds aren't shown
    lab = [f"{'Landsat' if k == 'landsat' else 'MODIS'} {v[5:10]}" for k, v in sorted(times.items(), key=lambda x: x[1], reverse=True)]
    if counts.get("glsea"):
        lab.append(f"GLSEA {str((glsea or {}).get('time', ''))[5:10]}")
    grid.update(time=max(list(times.values()) + [str((glsea or {}).get("time") or "")]), source="satellite fusion", label=" + ".join(lab))
    up = upwelling(lats, lons, fused)
    if up:
        near_src = src[(lats >= UPWELL_ROWS[0])[:, None] & (lats <= UPWELL_ROWS[1])[:, None] & np.isfinite(fused)]
        up["satellite_share"] = round(float(np.mean(near_src > 1)) if len(near_src) else 0.0, 2)
    return {"temp": grid, "counts": counts, "times": times, "upwelling": up}


# ---------------------------------------------------------------------------- Sentinel-1 / Sentinel-2
def nechad_fnu(rho):
    """Turbidity (FNU) from red (665 nm) water reflectance, Nechad et al. 2009/2016 (A 228.1, C 0.1641)."""
    rho = np.asarray(rho, float)
    return np.where(rho < 0.16, 228.1 * rho / (1 - rho / 0.1641), np.nan)


def build_s1(ee, now, days=12):
    s, w, n, e = MAP_BBOX
    pt = ee.Geometry.Point([NORTH_AVE[1], NORTH_AVE[0]])
    col = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(pt).filterDate((now - dt.timedelta(days=days)).isoformat(), now.isoformat()) \
        .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .sort("system:time_start", False)
    img = ee.Image(col.first())
    info = ee.Dictionary({"t": img.date().format("YYYY-MM-dd'T'HH:mm'Z'", "UTC"), "orbit": img.get("orbitProperties_pass"), "n": col.size()}).getInfo()
    if not info.get("n"):
        raise RuntimeError(f"no Sentinel-1 pass in {days} days")
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).gte(50)
    vv = img.select("VV").focalMedian(40, "circle", "meters").updateMask(water).rename("vv").addBands(img.select("angle").rename("angle"))
    lats, lons = grid_axes(MAP_BBOX, MAP_D)
    g = compute_grid(ee, vv.unmask(-99), MAP_BBOX, MAP_D)
    a = np.where(g["vv"] < -60, np.nan, g["vv"])
    out = byte_grid(lats, lons, a, -30.0, 0.0)
    out.update(time=info["t"], orbit=info.get("orbit"), angle_deg=round(float(np.nanmean(np.where(g["angle"] > 0, g["angle"], np.nan))), 1),
               median_db=round(float(np.nanmedian(a)), 2) if np.isfinite(a).any() else None, label=f"Sentinel-1 {info['t'][5:10]}",
               launches={k: _near(lats, lons, a, la, lo, 400) for k, (la, lo) in LAUNCHES.items()})
    return out


def build_s2(ee, now, days=10):
    pt = ee.Geometry.Point([NORTH_AVE[1], NORTH_AVE[0]])
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate((now - dt.timedelta(days=days)).isoformat(), now.isoformat()) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))

    def f(img):
        h = ee.Image.constant(ee.Number(img.date().millis()).divide(3.6e6)).toFloat()
        # water leaves almost nothing at 1.6 um: what the SWIR band still sees over water is glint/haze the
        # atmospheric correction left, taken off the red band too
        r = img.select("B4").subtract(img.select("B11")).divide(10000).max(0)
        return r.rename("r").addBands(h.rename("h")).updateMask(img.select("SCL").eq(6)).toFloat()

    n = col.size().getInfo()
    if not n:
        raise RuntimeError(f"no Sentinel-2 pass in {days} days")
    m = col.map(f).qualityMosaic("h")
    lats, lons = grid_axes(MAP_BBOX, MAP_D)
    g = compute_grid(ee, m.unmask(-1), MAP_BBOX, MAP_D)
    ok = (g["r"] >= 0) & (g["h"] > 1e5)
    fnu = np.where(ok, nechad_fnu(g["r"]), np.nan)
    if not np.isfinite(fnu).any():
        raise RuntimeError("no clear Sentinel-2 water pixels")
    hh = g["h"][ok]
    t = dt.datetime.fromtimestamp(float(np.max(hh)) * 3600, UTC).isoformat(timespec="minutes")
    out = byte_grid(lats, lons, fnu, 0.0, 60.0, log_scale=True)
    out.update(time=t, label=f"Sentinel-2 {t[5:10]}", median_fnu=round(float(np.nanmedian(fnu)), 1),
               launches={k: _near(lats, lons, fnu, la, lo, 400) for k, (la, lo) in LAUNCHES.items()})
    return out


def _near(lats, lons, a, lat, lon, r_m):
    """Median of valid cells within r_m of a point (None if none)."""
    dy = (lats[:, None] - lat) * 110570.0
    dx = (lons[None, :] - lon) * 111320.0 * math.cos(math.radians(lat))
    sel = (dx ** 2 + dy ** 2 <= r_m ** 2) & np.isfinite(a)
    return round(float(np.median(a[sel])), 1) if sel.any() else None


# ---------------------------------------------------------------------------- GFS / GOES
def newest_complete(counts, need):
    full = [(float(k), k) for k, v in counts.items() if v >= need]
    return max(full)[1] if full else None


def sample_points(ee, col, bands, scale, hour_prop):
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"id": k}) for k, (la, lo) in POINTS.items()])

    def per(img):
        hv = img.get(hour_prop)
        return img.select(bands).reduceRegions(pts, ee.Reducer.first(), scale).map(lambda f: f.set("fh", hv))
    return col.map(per).flatten().getInfo()["features"]


def gfs_series(features, init_ms):
    from .weathernext import met_dir
    rows = {}
    for f in features:
        p = f["properties"]
        rows.setdefault(p["id"], {})[int(p["fh"])] = p
    t0 = dt.datetime.fromtimestamp(init_ms / 1000, UTC)
    out = {}
    for pid, by in rows.items():
        hs = sorted(by)
        g = lambda b: [by[h].get(b) for h in hs]  # noqa: E731
        u, v = g("u_component_of_wind_10m_above_ground"), g("v_component_of_wind_10m_above_ground")
        out[pid] = {"t": [(t0 + dt.timedelta(hours=h)).isoformat() for h in hs],
                    "wspd": [None if a is None or b is None else round(math.hypot(a, b), 2) for a, b in zip(u, v)],
                    "wdir": [met_dir(a, b) for a, b in zip(u, v)],
                    "t2m_c": [None if x is None else round(x, 1) for x in g("temperature_2m_above_ground")],
                    "cloud": [None if x is None else round(x) for x in g("total_cloud_cover_entire_atmosphere")]}
    return out


def build_gfs(ee, now, hours=48):
    col = ee.ImageCollection("NOAA/GFS0P25").filterDate((now - dt.timedelta(hours=36)).isoformat(), (now + dt.timedelta(hours=1)).isoformat()) \
        .filter(ee.Filter.rangeContains("forecast_hours", 0, hours))
    counts = col.aggregate_histogram("creation_time").getInfo() or {}
    init = newest_complete(counts, hours + 1)
    if init is None:
        raise RuntimeError(f"no complete GFS run in 36 h ({len(counts)} partial)")
    run = col.filter(ee.Filter.eq("creation_time", int(float(init))))
    bands = ["u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground", "temperature_2m_above_ground",
             "total_cloud_cover_entire_atmosphere"]
    feats = sample_points(ee, run, bands, 27830, "forecast_hours")
    t0 = dt.datetime.fromtimestamp(float(init) / 1000, UTC)
    return {"init": t0.isoformat(), "points": gfs_series(feats, float(init))}


def cloud_fraction(bt13_k, refl02, tref_k, sun_up, dt_k=8.0, refl_min=0.30):
    """Cloudy if colder than the surface by dt_k at 10.3 um, or (by day) brighter than refl_min in the red band."""
    bt = np.asarray(bt13_k, float)
    c = bt < tref_k - dt_k
    if sun_up and refl02 is not None:
        c = c | (np.asarray(refl02, float) > refl_min)
    ok = np.isfinite(bt)
    return float(np.mean(c[ok])) if ok.any() else None


def _scale_offset(props, band, default):
    keys = {k.lower(): k for k in props}
    sc = next((props[keys[k]] for k in keys if band.lower() in k and "scale" in k), None)
    of = next((props[keys[k]] for k in keys if band.lower() in k and "offset" in k), None)
    return (float(sc), float(of or 0.0)) if sc is not None else default


def sun_elevation(lat, lon, t):
    d = t.timetuple().tm_yday + (t.hour + t.minute / 60) / 24
    g = 2 * math.pi / 365.25 * (d - 1)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = t.hour * 60 + t.minute + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)))


def build_goes(ee, now, tref_c):
    col = ee.ImageCollection("NOAA/GOES/19/MCMIPC").filterDate((now - dt.timedelta(hours=2)).isoformat(), (now + dt.timedelta(minutes=5)).isoformat()) \
        .sort("system:time_start", False)
    img = ee.Image(col.first())
    props = img.toDictionary().getInfo()
    t = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd'T'HH:mm'Z'", "UTC").getInfo()
    s13 = _scale_offset(props, "CMI_C13", None)
    s02 = _scale_offset(props, "CMI_C02", None)
    if s13 is None:
        raise RuntimeError("GOES scale/offset not found in image properties: " + ", ".join(sorted(props)[:40]))
    bbox = (41.65, -87.95, 42.20, -87.25)
    lats, lons = grid_axes(bbox, 0.02)
    g = compute_grid(ee, img.select(["CMI_C13", "CMI_C02"]).unmask(-9999), bbox, 0.02)
    bt = np.where(g["CMI_C13"] < -9000, np.nan, g["CMI_C13"] * s13[0] + s13[1])
    rf = None if s02 is None else np.where(g["CMI_C02"] < -9000, np.nan, g["CMI_C02"] * s02[0] + s02[1])
    tt = dt.datetime.strptime(t, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    sun_up = sun_elevation(NORTH_AVE[0], NORTH_AVE[1], tt) > 10
    tref = tref_c + 273.15
    dy = (lats[:, None] - NORTH_AVE[0]) * 110.57
    dx = (lons[None, :] - NORTH_AVE[1]) * 111.32 * math.cos(math.radians(NORTH_AVE[0]))
    near = dx ** 2 + dy ** 2 <= 15 ** 2
    loc = cloud_fraction(np.where(near, bt, np.nan), None if rf is None else np.where(near, rf, np.nan), tref, sun_up)
    area = cloud_fraction(bt, rf, tref, sun_up)
    return {"time": t, "scale_c13": list(s13), "sky_local_pct": None if loc is None else round(100 * loc), "sky_area_pct": None if area is None else round(100 * area),
            "tref_c": round(tref_c, 1), "day": sun_up, "bt_median_c": round(float(np.nanmedian(bt)) - 273.15, 1) if np.isfinite(bt).any() else None}


# ---------------------------------------------------------------------------- normals
def doy_window(t, d, days):
    a = t.timetuple().tm_yday
    b = d.replace(year=t.year).timetuple().tm_yday if not (d.month == 2 and d.day == 29) else 60
    k = abs(a - b)
    return min(k, 365 - k) <= days


def era5_normals(series, today, days=10, years=(1991, 2020), calib=None):
    """Wind (m/s, 10 m, scaled to the buoy) and air (C) percentiles for +-days around today's date."""
    ws, at = [], []
    for t, o in series.items():
        if years[0] <= t.year <= years[1] and doy_window(today, t, days):
            r = 1.0
            if calib and calib.get("sectors"):
                r = calib["sectors"][int(((o["wdir"] + 15) % 360) // 30)]
            ws.append(o["wspd"] * r)
            if o.get("atmp") is not None:
                at.append(o["atmp"])
    if len(ws) < 100:
        return None
    p = lambda a, q: round(float(np.percentile(a, q)), 2)  # noqa: E731
    return {"wind_ms": {"p10": p(ws, 10), "p50": p(ws, 50), "p90": p(ws, 90), "p99": p(ws, 99)},
            "air_c": {"p10": p(at, 10), "p50": p(at, 50), "p90": p(at, 90)} if at else None, "hours": len(ws),
            "window_days": days, "years": list(years)}


def lake_normal(ee, today, days=10, years=(1991, 2020)):
    base = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("lake_mix_layer_temperature")
    col = None
    for y in range(years[0], years[1] + 1):                         # one short date range per year keeps the query small
        c = dt.datetime(y, today.month, min(today.day, 28), tzinfo=UTC)
        part = base.filterDate((c - dt.timedelta(days=days)).isoformat(), (c + dt.timedelta(days=days + 1)).isoformat())
        col = part if col is None else col.merge(part)
    rows = col.getRegion(ee.Geometry.Point([-87.50, 41.95]), 11132).getInfo()
    i = rows[0].index("lake_mix_layer_temperature")
    v = np.array([r[i] for r in rows[1:] if r[i] is not None], float) - 273.15
    return None if len(v) < 30 else {"mean_c": round(float(v.mean()), 2), "sd_c": round(float(v.std()), 2), "days": int(len(v))}


# ---------------------------------------------------------------------------- pass log (learn branch)
def log_passes(earth, d):
    """Append each new Sentinel-1 / Sentinel-2 pass's launch values to sat/<s1|s2>_passes.jsonl: over months this
    pairs radar roughness at the launches with the wind at the time (from the buoy archive), i.e. how sheltered each
    launch really is, and clarity with storms and river outflow."""
    os.makedirs(os.path.join(d, "sat"), exist_ok=True)
    wrote = []
    for k, keys in (("s1", ("time", "orbit", "angle_deg", "median_db", "launches")), ("s2", ("time", "median_fnu", "launches"))):
        x = (earth or {}).get(k)
        if not x or not x.get("time"):
            continue
        path = os.path.join(d, "sat", f"{k}_passes.jsonl")
        seen = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    seen.add(json.loads(line)["time"])
                except (ValueError, KeyError):
                    pass
        if x["time"] in seen:
            continue
        with open(path, "a") as fh:
            fh.write(json.dumps({q: x.get(q) for q in keys}, separators=(",", ":")) + "\n")
        wrote.append(k)
    return wrote


# ---------------------------------------------------------------------------- main
def build(lakefront=None, era5_dir=None, era5_climate=None, prev=None, now=None):
    from .weathernext import init_ee
    now = now or dt.datetime.now(UTC)
    ee = init_ee()
    out = {"version": 1, "generated": now.isoformat(timespec="seconds"), "notes": [],
           "credits": "Earth Engine: MODIS (NASA LP DAAC), Landsat (USGS), Sentinel-1/2 (Copernicus, ESA), GOES-19 and GFS (NOAA), "
                      "ERA5 / ERA5-Land (Copernicus Climate Change Service, ECMWF), JRC Global Surface Water"}
    glsea = ((lakefront or {}).get("lake") or {}).get("temp")

    def part(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", name, e)
            out["notes"].append(f"{name}: {str(e)[:300]}")

    part("water", lambda: build_water(ee, now, glsea))
    part("s1", lambda: build_s1(ee, now))
    part("s2", lambda: build_s2(ee, now))
    part("gfs", lambda: build_gfs(ee, now))
    tref = None
    try:
        g = out["gfs"]["points"]["northave"]
        k = min(range(len(g["t"])), key=lambda i: abs(dt.datetime.fromisoformat(g["t"][i]) - now))
        tref = g["t2m_c"][k]
    except Exception:  # noqa: BLE001
        pass
    if tref is None and out.get("water") and out["water"].get("upwelling"):
        tref = out["water"]["upwelling"]["near_c"]
    if tref is not None:
        part("goes", lambda: build_goes(ee, now, tref))
    # normals change slowly: reuse last snapshot's for the same day
    pn = (prev or {}).get("normals")
    if pn and pn.get("date") == now.date().isoformat() and pn.get("era5") and pn.get("lake"):
        out["normals"] = pn
    else:
        n = {"date": now.date().isoformat()}
        try:
            if era5_dir and os.path.isdir(era5_dir):
                from . import era5
                ser = era5.load(era5_dir).get("45198") or {}
                cal = ((era5_climate or {}).get("calibration") or {}).get("45198")
                n["era5"] = era5_normals(ser, now, calib=cal)
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"normals (ERA5): {str(e)[:200]}")
        try:
            n["lake"] = lake_normal(ee, now)
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"normals (lake): {str(e)[:200]}")
        out["normals"] = n
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "log-pass":
        lp = argparse.ArgumentParser()
        lp.add_argument("cmd"); lp.add_argument("--earth", required=True); lp.add_argument("--dir", required=True)
        b = lp.parse_args()
        e = json.load(open(b.earth)) if os.path.exists(b.earth) and os.path.getsize(b.earth) else None
        print("logged passes:", log_passes(e, b.dir))
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="earth.json")
    ap.add_argument("--lakefront")
    ap.add_argument("--era5")
    ap.add_argument("--era5-climate")
    ap.add_argument("--prev")
    a = ap.parse_args()
    def rd(p):
        try:
            return json.load(open(p)) if p and os.path.exists(p) and os.path.getsize(p) else None
        except ValueError:
            return None
    o = build(rd(a.lakefront), a.era5, rd(a.era5_climate), rd(a.prev))
    with open(a.out, "w") as fh:
        json.dump(o, fh, separators=(",", ":"))
    w = o.get("water") or {}
    print(json.dumps({"water": {k: w.get(k) for k in ("counts", "times", "upwelling")}, "s1": (o.get("s1") or {}).get("label"),
                      "s2": {k: (o.get("s2") or {}).get(k) for k in ("label", "median_fnu", "launches")}, "gfs": (o.get("gfs") or {}).get("init"),
                      "goes": o.get("goes"), "normals": o.get("normals"), "notes": o["notes"]}, default=str))
