"""
Real lake bottom for the Chicago lakefront.

  nearshore  2020 USACE NCMP topobathy lidar DEM (1 m, IGLD85 metres; ~1 km offshore or laser extinction),
             NOAA Digital Coast dataset 9970 on the public S3 bucket noaa-nos-coastal-lidar-pds
  offshore   NOAA NCEI bathymetry mosaic (ArcGIS ImageServer export), aligned to the lidar where they overlap
  lake level NOAA CO-OPS water level (IGLD85) at Calumet Harbor, so depth = level - bed elevation

    python -m app.bathymetry --out bathy.json           # world grid (10 m over the lakefront map)
    python -m app.bathymetry --out bathy.json --model   # also the 4 m wave-model grid for the route

Output grids are in the shared local frame (metres east/north of State & Madison). Bed elevation is stored
relative to Lake Michigan low water datum (176.0 m IGLD85) in centimetres (int16), with a source map
(0 none, 1 lidar, 2 NCEI, 3 filled).
"""

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import gzip
import io
import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import lakefront as lf

log = logging.getLogger("klot.bathy")
UTC = dt.timezone.utc
LWD = 176.0                                    # Lake Michigan-Huron low water datum, m IGLD85
LIDAR_BUCKET = "noaa-nos-coastal-lidar-pds"
LIDAR_PREFIXES = ["dem/USACE_Lake_Mich_IL_IN_MI_WI_DEM_2020_9970/", "dem/USACE_Lake_Mich_DEM_2012_9265/"]
NCEI_SERVICES = [
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage",
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage",
]
COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
LEVEL_STATIONS = ["9087044", "9087031"]       # Calumet Harbor IL, Milwaukee WI (fallback)
# world grid = lakefront map; model grid = North Ave beach to past Diversey Harbor, out to ~10 m depth
MODEL_BBOX = (41.905, -87.648, 41.945, -87.598)
# corridor = the Lake Michigan Water Trail stretch paddled most: Monroe Harbor to Montrose, out past the lidar's
# reach (~1 km); the page draws its shoreline and nearshore bottom from it (the 10 m world grid covers the lane
# out to the Harrison-Dever crib)
# (north past Montrose to Kathy Osterman Beach, so the SWAN trail domain's north boundary sits ~2 km from Montrose Beach)
CORRIDOR_BBOX = (41.866, -87.656, 41.988, -87.578)
MODEL_BBOXES = {"route": MODEL_BBOX, "montrose": (41.950, -87.648, 41.975, -87.600), "corridor": CORRIDOR_BBOX}
ISGS_PAGES = ["https://clearinghouse.isgs.illinois.edu/data/coastal/chicago-beaches-chicago-park-district-topobathymetric-models"]
ISGS_DIR = os.environ.get("ISGS_DIR", "isgs_cache")


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": lf.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------- grids in the local frame
def local_grid(bbox, d):
    s, w, n, e = bbox
    x0, y0 = lf.xy(s, w); x1, y1 = lf.xy(n, e)
    nx, ny = int((x1 - x0) // d) + 1, int((y1 - y0) // d) + 1
    return {"x0": x0, "y0": y0, "dx": d, "nx": nx, "ny": ny}


def grid_lonlat(g):
    x = g["x0"] + np.arange(g["nx"]) * g["dx"]
    y = g["y0"] + np.arange(g["ny"]) * g["dx"]
    X, Y = np.meshgrid(x, y)                   # rows south -> north
    lon = lf.REF[1] + X / lf.KX
    lat = lf.REF[0] + Y / lf.KY
    return lon, lat


# ---------------------------------------------------------------------------- lidar
def lidar_tiles(prefix, bbox_utm, probe):
    """Keys of GeoTIFF tiles that overlap bbox_utm (xmin, ymin, xmax, ymax) in EPSG:6345."""
    import rasterio
    cli = s3()
    keys, pag = [], cli.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=LIDAR_BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    tifs = [k for k in keys if k.lower().endswith((".tif", ".tiff"))]
    probe["lidar_" + prefix] = {"objects": len(keys), "tifs": len(tifs), "sample": [k for k in keys if not k.lower().endswith((".tif", ".tiff"))][:20] + tifs[:5]}
    if not tifs:
        return []
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    xmin, ymin, xmax, ymax = bbox_utm

    def bounds(k):
        try:
            with rasterio.open(f"/vsis3/{LIDAR_BUCKET}/{k}") as ds:
                b = ds.bounds
                return k, (b.left, b.bottom, b.right, b.top), ds.crs.to_string() if ds.crs else None
        except Exception as e:  # noqa: BLE001
            return k, None, str(e)

    hits, crs_seen, errs = [], set(), 0
    with cf.ThreadPoolExecutor(16) as ex:
        for k, b, crs in ex.map(bounds, tifs):
            if b is None:
                errs += 1
                continue
            crs_seen.add(crs)
            if b[0] < xmax and b[2] > xmin and b[1] < ymax and b[3] > ymin:
                hits.append(k)
    probe["lidar_" + prefix].update({"overlap": len(hits), "crs": sorted(c for c in crs_seen if c)[:3], "header_errors": errs})
    return hits


def lidar_mosaic(g, probe):
    """Bed elevation (m IGLD85) sampled on grid g from the lidar DEM tiles, NaN where none."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from pyproj import Transformer
    lon, lat = grid_lonlat(g)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:6345", always_xy=True)
    ux, uy = tr.transform(lon, lat)
    pad = 20
    bb = (ux.min() - pad, uy.min() - pad, ux.max() + pad, uy.max() + pad)
    out = np.full(lon.shape, np.nan, np.float32)
    for prefix in LIDAR_PREFIXES:
        try:
            keys = lidar_tiles(prefix, bb, probe)
        except Exception as e:  # noqa: BLE001
            probe["lidar_" + prefix] = {"error": str(e)}
            continue
        if not keys:
            continue
        res = max(1.0, g["dx"] / 2)
        dss = [rasterio.open(f"/vsis3/{LIDAR_BUCKET}/{k}") for k in keys]
        try:
            arr, tf = merge(dss, bounds=bb, res=res, nodata=-9999.0, resampling=Resampling.average)
        finally:
            for d in dss:
                d.close()
        z = arr[0].astype(np.float32)
        z[(z <= -9000) | ~np.isfinite(z)] = np.nan
        cols = ((ux - tf.c) / tf.a).astype(int)
        rows = ((uy - tf.f) / tf.e).astype(int)
        ok = (rows >= 0) & (cols >= 0) & (rows < z.shape[0]) & (cols < z.shape[1])
        samp = np.full(lon.shape, np.nan, np.float32)
        samp[ok] = z[rows[ok], cols[ok]]
        fill = np.isnan(out) & np.isfinite(samp)
        out[fill] = samp[fill]
        probe["lidar_" + prefix]["coverage"] = round(float(np.isfinite(samp).mean()), 3)
        if np.isfinite(out).mean() > 0.5:
            break                                  # 2020 covers it; the 2012 survey only fills gaps otherwise
    return out


# ---------------------------------------------------------------------------- ISGS / Chicago Park District beach surveys
def _links(html, base):
    import re
    out = []
    for h in re.findall(r'href=["\']([^"\']+)["\']', html):
        out.append(urllib.parse.urljoin(base, h))
    return out


def isgs_rasters(probe):
    """Download the Park District topobathymetric DEMs (Montrose, Foster, 63rd, Rainbow); return local raster paths."""
    import re
    import zipfile
    os.makedirs(ISGS_DIR, exist_ok=True)
    seen, files, log_ = set(), [], []
    queue = list(ISGS_PAGES)
    while queue and len(seen) < 25:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        html = None
        for attempt in range(3):                   # the clearinghouse is slow to answer
            try:
                html = get(url, timeout=240).decode("utf-8", "replace"); break
            except Exception as e:  # noqa: BLE001
                log_.append(f"{url} (try {attempt + 1}): {e}"); time.sleep(10)
        if html is None:
            continue
        for h in _links(html, url):
            low = h.lower()
            if re.search(r"\.(zip|tif|tiff)(\?|$)", low) and ("dem" in low or "topobath" in low or "tbm" in low or low.endswith((".tif", ".tiff"))):
                files.append(h)
            elif "clearinghouse.isgs.illinois.edu/data/coastal/" in low and any(b in low for b in ("chicago", "montrose", "foster", "beach")) and h not in seen:
                queue.append(h)
    files = list(dict.fromkeys(files))
    probe["isgs"] = {"pages": len(seen), "candidates": files[:40], "log": log_[:10]}
    paths = []
    for f in files:
        name = os.path.join(ISGS_DIR, os.path.basename(urllib.parse.urlparse(f).path))
        try:
            if not os.path.exists(name):
                raw = get(f, timeout=900)
                if len(raw) > 1.5e9:
                    continue
                with open(name, "wb") as fh:
                    fh.write(raw)
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(name) as z:
                    for m in z.namelist():
                        if m.lower().endswith((".tif", ".tiff")) and ("dem" in m.lower() or "topobath" in m.lower() or "tbm" in m.lower() or "elev" in m.lower()):
                            z.extract(m, ISGS_DIR); paths.append(os.path.join(ISGS_DIR, m))
            else:
                paths.append(name)
        except Exception as e:  # noqa: BLE001
            log_.append(f"{f}: {e}")
    probe["isgs"]["rasters"] = [os.path.basename(p) for p in paths]
    return paths


def sample_raster(path, g, out=None):
    """Sample any georeferenced raster (any CRS) onto grid g with nearest-cell lookup; NaN where no data."""
    import rasterio
    from pyproj import Transformer
    lon, lat = grid_lonlat(g)
    res = np.full(lon.shape, np.nan, np.float32) if out is None else out
    with rasterio.open(path) as ds:
        tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        ux, uy = tr.transform(lon, lat)
        b = ds.bounds
        inside = (ux >= b.left) & (ux <= b.right) & (uy >= b.bottom) & (uy <= b.top)
        if not inside.any():
            return res, None
        # read decimated to roughly the target spacing (units may be feet)
        unit = (ds.crs.linear_units or "metre").lower()
        to_m = 0.3048006 if "foot" in unit or "feet" in unit or "us survey" in unit else 1.0
        step = max(1, int(g["dx"] / (abs(ds.transform.a) * to_m) / 2))
        h, w = max(1, ds.height // step), max(1, ds.width // step)
        from rasterio.enums import Resampling
        z = ds.read(1, out_shape=(h, w), resampling=Resampling.average, masked=True).astype(np.float32).filled(np.nan)
        tf = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        zunit = to_m                                   # vertical units follow horizontal for State Plane feet products
        cols = ((ux - tf.c) / tf.a).astype(int); rows = ((uy - tf.f) / tf.e).astype(int)
        ok = inside & (rows >= 0) & (cols >= 0) & (rows < h) & (cols < w)
        v = np.full(lon.shape, np.nan, np.float32); v[ok] = z[rows[ok], cols[ok]] * zunit
        fill = np.isfinite(v)
        res[fill] = v[fill]
        return res, {"crs": ds.crs.to_string(), "units": unit, "cells": int(fill.sum())}


# ---------------------------------------------------------------------------- NCEI offshore
def ncei_grid(g, probe):
    """Elevation (m) from the NCEI DEM mosaic on grid g (raw vertical reference; aligned later)."""
    import rasterio
    lon, lat = grid_lonlat(g)
    w, e, s, n = lon.min() - 0.002, lon.max() + 0.002, lat.min() - 0.002, lat.max() + 0.002
    W, H = min(2000, g["nx"]), min(2000, g["ny"])
    for url in NCEI_SERVICES:
        q = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326, "size": f"{W},{H}", "format": "tiff",
             "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "noData": -99999, "f": "image"}
        try:
            raw = get(url + "?" + urllib.parse.urlencode(q), timeout=180)
            with rasterio.MemoryFile(raw) as mf, mf.open() as ds:
                z = ds.read(1).astype(np.float32); tf = ds.transform
            z[(z < -9000) | ~np.isfinite(z)] = np.nan
            cols = ((lon - tf.c) / tf.a).astype(int); rows = ((lat - tf.f) / tf.e).astype(int)
            ok = (rows >= 0) & (cols >= 0) & (rows < z.shape[0]) & (cols < z.shape[1])
            out = np.full(lon.shape, np.nan, np.float32); out[ok] = z[rows[ok], cols[ok]]
            probe["ncei"] = {"url": url, "coverage": round(float(np.isfinite(out).mean()), 3),
                             "range": [float(np.nanmin(out)), float(np.nanmax(out))] if np.isfinite(out).any() else None}
            if np.isfinite(out).mean() > 0.2:
                return out
        except Exception as e:  # noqa: BLE001
            probe.setdefault("ncei_errors", {})[url] = str(e)[:300]
    return None


# ---------------------------------------------------------------------------- lake level
def lake_level(probe):
    for st in LEVEL_STATIONS:
        q = {"station": st, "product": "water_level", "datum": "IGLD", "date": "latest", "units": "metric",
             "time_zone": "gmt", "format": "json", "application": "chicago-storm-3d"}
        try:
            js = json.loads(get(COOPS + "?" + urllib.parse.urlencode(q), timeout=60))
            v = float(js["data"][-1]["v"])
            probe["level"] = {"station": st, "m_igld85": v, "t": js["data"][-1]["t"]}
            return v, st, js["data"][-1]["t"]
        except Exception as e:  # noqa: BLE001
            probe.setdefault("level_errors", {})[st] = str(e)[:200]
    return None, None, None


# ---------------------------------------------------------------------------- blend
def nearest_fill(z, mask):
    """Fill NaNs inside mask from the nearest finite cell (scipy EDT)."""
    from scipy import ndimage
    bad = ~np.isfinite(z)
    if not bad.any():
        return z
    _, (ri, ci) = ndimage.distance_transform_edt(bad, return_indices=True)
    out = z.copy()
    m = bad & mask
    out[m] = z[ri[m], ci[m]]
    return out


def blend(lidar, ncei, level, lake_mask=None):
    """Bed elevation (m IGLD85) and source map. NCEI is shifted to agree with the lidar where both exist."""
    z = lidar.copy()
    src = np.where(np.isfinite(lidar), 1, 0).astype(np.uint8)
    info = {}
    if ncei is not None:
        both = np.isfinite(lidar) & np.isfinite(ncei) & (lidar < level - 3.0)
        if both.sum() > 200:
            off = float(np.median(lidar[both] - ncei[both]))
        else:
            off = None
        info["ncei_overlap_cells"] = int(both.sum())
        # the NCEI mosaic stores lake bottoms as depths below LWD (negative, small) and land as elevations
        # above sea level (~176+); without lidar overlap, convert each kind separately
        if off is None:
            nz = np.where(ncei < 60, ncei + LWD, ncei); info["ncei_offset_m"] = "lake cells +176.0 (LWD), land as is"
        else:
            nz = np.where(ncei < 60, ncei + LWD, ncei)
            both2 = np.isfinite(lidar) & np.isfinite(nz) & (lidar < level - 3.0)
            off2 = float(np.median(lidar[both2] - nz[both2])) if both2.sum() > 200 else 0.0
            nz = nz + off2; info["ncei_offset_m"] = round(off2, 2)
        # feather NCEI into the lidar edge over ~150 m so there is no step at laser extinction
        gap = ~np.isfinite(z) & np.isfinite(nz)
        z[gap] = nz[gap]; src[gap] = 2
        # NCEI's nearshore cells are interpolated from 200-600 m track lines; in lidar gaps (harbours, turbid
        # water) its very shallow values are artefacts, so floor them at 1.5 m of water
        shallow = gap & (z > level - 1.5) & (z < level)
        z[shallow] = level - 1.5; info["ncei_shallow_floored"] = int(shallow.sum())
    water = (z < level) if lake_mask is None else lake_mask
    still = ~np.isfinite(z)
    if still.any():
        z = nearest_fill(z, np.ones_like(still))
        src[still] = 3
    info["coverage"] = {"lidar": round(float((src == 1).mean()), 3), "ncei": round(float((src == 2).mean()), 3), "filled": round(float((src == 3).mean()), 3)}
    return z, src, info


def smooth_seam(z, src, passes=6):
    """Light diffusion across the lidar/NCEI seam only."""
    from scipy import ndimage
    seam = ndimage.binary_dilation(src == 1, iterations=4) & ndimage.binary_dilation(src != 1, iterations=4)
    for _ in range(passes):
        zs = ndimage.uniform_filter(z, 5)
        z = np.where(seam, zs, z)
    return z


def pack(g, z, src):
    zc = np.clip(np.round((z - LWD) * 100), -32000, 32000).astype("<i2")
    enc = lambda a: base64.b64encode(gzip.compress(a.tobytes(), 6)).decode()
    return g | {"z_cm": enc(zc), "src": enc(src.astype(np.uint8)), "order": "rows south->north, cols west->east"}


def build_grid(bbox, d, level, probe, tag):
    g = local_grid(bbox, d)
    log.info("%s grid %dx%d at %.0f m", tag, g["nx"], g["ny"], d)
    p = probe.setdefault(tag, {})
    lid = lidar_mosaic(g, p)
    if ISGS_PATHS:
        isg = np.full(lid.shape, np.nan, np.float32); used = {}
        for path in ISGS_PATHS:
            try:
                isg, meta = sample_raster(path, g, isg)
                if meta:
                    used[os.path.basename(path)] = meta
            except Exception as e:  # noqa: BLE001
                used[os.path.basename(path)] = {"error": str(e)[:200]}
        both = np.isfinite(isg) & np.isfinite(lid)
        if both.sum() > 100:
            d = float(np.median(lid[both] - isg[both]))
            # small offsets are datum differences (NAVD88 vs IGLD85 ~ tens of cm); large ones mean bad units
            if abs(d) < 3:
                isg = isg + d
                lid = np.where(np.isfinite(isg), isg, lid)
                used["offset_to_lidar_m"] = round(d, 3)
            else:
                used["rejected_offset_m"] = round(d, 2)
        elif np.isfinite(isg).any() and np.nanmedian(isg) > 150:
            lid = np.where(np.isfinite(isg), isg, lid)
        used["cells"] = int(np.isfinite(isg).sum())
        p["isgs"] = used
    nc = ncei_grid(g, p)
    if not np.isfinite(lid).any() and nc is None:
        raise RuntimeError("no bathymetry source reachable")
    z, src, info = blend(lid, nc, level)
    z = smooth_seam(z, src)
    p.update(info)
    depth = level - z
    p["depth_stats"] = {"max_m": round(float(np.nanmax(depth)), 2), "water_frac": round(float((depth > 0).mean()), 3)}
    return pack(g, z, src)


ISGS_PATHS = []


def build(model=False):
    probe = {}
    level, st, lt = lake_level(probe)
    if level is None:
        level = LWD + 0.4                      # recent seasonal mean above LWD (approximate)
        probe["level_assumed"] = level
    out = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"),
           "datum": "bed elevation, cm relative to LWD 176.0 m IGLD85", "lwd_m": LWD,
           "level": {"m_igld85": round(level, 3), "station": st, "t": lt},
           "sources": ["2020 USACE NCMP topobathy lidar DEM (NOAA Digital Coast 9970)", "NOAA NCEI DEM mosaic", "NOAA CO-OPS water level"]}
    global ISGS_PATHS
    try:
        ISGS_PATHS = isgs_rasters(probe)
    except Exception as e:  # noqa: BLE001
        probe["isgs_error"] = str(e)[:300]
    out["world"] = build_grid(lf.MAP_BBOX, 10.0, level, probe, "world")
    if model:
        out["models"] = {k: build_grid(b, 4.0, level, probe, k) for k, b in MODEL_BBOXES.items()}
    out["probe"] = probe
    return out


def unpack(gr):
    z = np.frombuffer(gzip.decompress(base64.b64decode(gr["z_cm"])), "<i2").reshape(gr["ny"], gr["nx"]) / 100.0 + LWD
    src = np.frombuffer(gzip.decompress(base64.b64decode(gr["src"])), np.uint8).reshape(gr["ny"], gr["nx"])
    return z, src


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bathy.json")
    ap.add_argument("--model", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    d = build(a.model)
    with open(a.out, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    print(json.dumps(d["probe"], indent=1, default=str)[:6000])
    print(f"wrote {a.out} in {time.time() - t0:.0f} s")
