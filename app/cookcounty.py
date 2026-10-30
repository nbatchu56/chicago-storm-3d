"""
The 2023 USGS / Cook County echo-sounder survey of Diversey Harbor and the Lincoln Park lagoons.

USGS and the Cook County Bureau of Technology surveyed 58 water bodies in 2023 (single-beam echo sounder; "Bathymetry
and Water-Quality Data of Surface Water Bodies in Cook County, Illinois, 2023", ver. 2.0, DOI 10.5066/P1VVHKFP, CC0).
Diversey Harbor was run on May 23, 2023, and its water body polygon takes in the basin, the channel under the Fullerton
Avenue bridge and the South Lagoon down to the zoo; North Pond and South Pond are their own polygons. Cook County GIS
serves the products without a key:

  polygons   the surveyed shoreline of each water body (the harbour's walls, to the foot)
  raster     depth below that day's water surface, 3 ft pixels, whole feet (U8; 0 = no data), EPSG:3435 (Illinois East, ft)
  contours   2 ft contours (not used here)

Depth becomes bed elevation with the lake level at Calumet Harbor on the survey day (IGLD85): the harbour and the lagoon
are open to the lake through the Fullerton channel, so they sit at lake level. Everything is sampled onto the shared local
grid (metres east/north of State & Madison, app.lakefront.xy); the polygons are rasterised with an even-odd scanline fill
(holes kept), so no raster library is needed for the mask.
"""

import datetime as dt
import json
import logging
import os
import urllib.parse
import urllib.request

import numpy as np

from . import lakefront as lf

log = logging.getLogger("klot.cook")
GIS = "https://gis.cookcountyil.gov"
POLY_URL = GIS + "/traditional/rest/services/bathmetrySurveyData/MapServer/1/query"
RASTER_URL = GIS + "/imagery/rest/services/bathymetric_survey_3m_depth_surfaces_raster/ImageServer/exportImage"
COOPS = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
SURVEY_DATE = "2023-05-23"                      # Diversey Harbor's survey day (the data release's water-quality table)
SURVEY_LEVEL_FALLBACK = 176.81                  # m IGLD85, Calumet Harbor's mean that day (24 hourly heights)
SOURCE = "USGS / Cook County 2023 bathymetric survey (DOI 10.5066/P1VVHKFP)"
# the harbour, the Fullerton channel and the South Lagoon, North Pond and South Pond, with a margin
BBOX = (41.912, -87.641, 41.937, -87.625)       # S, W, N, E
FT = 0.3048
PX_FT = 3.0                                     # the raster's native pixel (feet)
COOK_DIR = os.environ.get("COOK_DIR", "cook_cache")
# channels no survey sees: the inlet under the Lake Shore Drive bridge (the survey boat stopped at the bridge's west
# face and the lidar returns the deck as land) and the gap between the jetties to the lake, confirmed by paddle tracks
# (the same strokes the page draws, app/static/world.html WATERWAYS): [name, width m, nominal depth m, [[lat, lon], ...]]
CHANNELS = [
    ["Diversey Harbor inlet under Lake Shore Drive", 16.0, 1.8, [[41.93252, -87.63242], [41.93250, -87.63162], [41.93259, -87.63132], [41.93267, -87.63117]]],
    ["Diversey Harbor mouth", 18.0, 2.5, [[41.93264, -87.63112], [41.93254, -87.63091], [41.93242, -87.63076], [41.93228, -87.63060], [41.93212, -87.63048]]],
]


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": lf.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cached(name, fetch):
    """Keep each download in COOK_DIR (the survey does not change between runs)."""
    os.makedirs(COOK_DIR, exist_ok=True)
    path = os.path.join(COOK_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            return fh.read()
    raw = fetch()
    with open(path, "wb") as fh:
        fh.write(raw)
    return raw


# ---------------------------------------------------------------------------- polygons
def fetch_polygons(bbox=BBOX):
    """Water-body polygons intersecting bbox, in local metres: [{name, acres, rings: [[[x, y], ...], ...]}, ...]
    (the first ring is the shoreline, the rest are holes)."""
    s, w, n, e = bbox
    q = {"where": "1=1", "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
         "spatialRel": "esriSpatialRelIntersects", "outFields": "Name,Acres", "outSR": 4326, "f": "geojson"}
    raw = cached("polys_%s.geojson" % "_".join(f"{v:.3f}" for v in bbox), lambda: get(POLY_URL + "?" + urllib.parse.urlencode(q)))
    return polygons_from_geojson(json.loads(raw))


def polygons_from_geojson(gj):
    out = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        p = f.get("properties") or {}
        for rings in polys:
            out.append({"name": (p.get("Name") or "").strip().title(), "acres": p.get("Acres"),
                        "rings": [[list(lf.xy(lat, lon)) for lon, lat in ring] for ring in rings]})
    return out


def rasterize(polys, g):
    """Cell centres of grid g inside any polygon (even-odd over all rings, so holes stay out). Bool (ny, nx)."""
    xs = g["x0"] + np.arange(g["nx"]) * g["dx"]
    ys = g["y0"] + np.arange(g["ny"]) * g["dx"]
    inside = np.zeros((g["ny"], g["nx"]), bool)
    for poly in polys:
        hit = np.zeros_like(inside)
        for ring in poly["rings"]:
            pts = np.asarray(ring, float)
            if len(pts) < 3:
                continue
            x0, y0 = pts[:, 0], pts[:, 1]
            x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
            for j, y in enumerate(ys):
                # edges straddling this row (half-open, so a vertex on the row counts once)
                m = (y0 > y) != (y1 > y)
                if not m.any():
                    continue
                xc = x0[m] + (y - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m])
                # crossings to the left of each cell: odd = inside
                cnt = (xc[None, :] < xs[:, None]).sum(axis=1)
                hit[j] ^= (cnt % 2 == 1)
        inside |= hit
    return inside


def channel_mask(g, channels=None):
    """Cells of grid g within a channel stroke: (inside bool, nominal depth m where inside, else NaN)."""
    xs = g["x0"] + np.arange(g["nx"]) * g["dx"]; ys = g["y0"] + np.arange(g["ny"]) * g["dx"]
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros(X.shape, bool); depth = np.full(X.shape, np.nan, np.float32)
    for name, w, d, pts in (CHANNELS if channels is None else channels):
        P = [lf.xy(la, lo) for la, lo in pts]
        for (ax, ay), (bx, by) in zip(P, P[1:]):
            vx, vy = bx - ax, by - ay; L2 = vx * vx + vy * vy or 1.0
            t = np.clip(((X - ax) * vx + (Y - ay) * vy) / L2, 0.0, 1.0)
            dist = np.hypot(X - (ax + t * vx), Y - (ay + t * vy))
            hit = dist <= w / 2
            depth[hit & (~inside | (depth < d))] = d; inside |= hit
    return inside, depth


# ---------------------------------------------------------------------------- depth raster
def fetch_depth(g, bbox=BBOX, probe=None):
    """Depth (m) below the survey-day surface on grid g, NaN outside the raster's data; needs rasterio and pyproj."""
    import rasterio
    from pyproj import Transformer
    s, w, n, e = bbox
    (xa, ya), (xb, yb) = lf.xy(s, w), lf.xy(n, e)
    wpx, hpx = int((xb - xa) / (PX_FT * FT)) + 2, int((yb - ya) / (PX_FT * FT)) + 2
    q = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 3435, "size": f"{wpx},{hpx}", "format": "tiff",
         "pixelType": "U8", "noData": 0, "interpolation": "RSP_NearestNeighbor", "f": "image"}
    raw = cached("depth_%s.tif" % "_".join(f"{v:.3f}" for v in bbox), lambda: get(RASTER_URL + "?" + urllib.parse.urlencode(q), timeout=240))
    with rasterio.MemoryFile(raw) as mf, mf.open() as ds:
        z = ds.read(1).astype(np.float32); tf = ds.transform; crs = ds.crs
    z[z <= 0] = np.nan                           # 0 = no data; depths are whole feet from 1
    x = g["x0"] + np.arange(g["nx"]) * g["dx"]; y = g["y0"] + np.arange(g["ny"]) * g["dx"]
    X, Y = np.meshgrid(x, y)
    lon, lat = lf.REF[1] + X / lf.KX, lf.REF[0] + Y / lf.KY
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ux, uy = tr.transform(lon, lat)
    cols = ((ux - tf.c) / tf.a).astype(int); rows = ((uy - tf.f) / tf.e).astype(int)
    ok = (rows >= 0) & (cols >= 0) & (rows < z.shape[0]) & (cols < z.shape[1])
    out = np.full(X.shape, np.nan, np.float32)
    out[ok] = z[rows[ok], cols[ok]] * FT
    if probe is not None:
        probe["raster"] = {"px": [int(wpx), int(hpx)], "crs": crs.to_string() if crs else None, "cells": int(np.isfinite(out).sum()),
                           "max_m": round(float(np.nanmax(out)), 2) if np.isfinite(out).any() else None}
    return out


def survey_level(probe=None):
    """Lake level (m IGLD85) at Calumet Harbor on the survey day: the daily mean of the hourly heights."""
    q = {"station": "9087044", "product": "hourly_height", "datum": "IGLD", "begin_date": SURVEY_DATE.replace("-", ""),
         "end_date": SURVEY_DATE.replace("-", ""), "units": "metric", "time_zone": "gmt", "format": "json", "application": "chicago-storm-3d"}
    try:
        js = json.loads(cached("level_%s.json" % SURVEY_DATE, lambda: get(COOPS + "?" + urllib.parse.urlencode(q), timeout=60)))
        v = [float(d["v"]) for d in js["data"] if d.get("v")]
        if len(v) >= 12:
            lvl = float(np.mean(v))
            if probe is not None:
                probe["level"] = {"m_igld85": round(lvl, 3), "hours": len(v)}
            return lvl
    except Exception as e:  # noqa: BLE001
        if probe is not None:
            probe["level_error"] = str(e)[:200]
    if probe is not None:
        probe["level"] = {"m_igld85": SURVEY_LEVEL_FALLBACK, "assumed": True}
    return SURVEY_LEVEL_FALLBACK


def fill_inside(depth, inside):
    """Raster gaps inside a polygon (the rim the echo sounder never reached, a dock) take the nearest surveyed depth;
    cells outside the polygons are left NaN."""
    from scipy import ndimage
    have = np.isfinite(depth) & inside
    if not have.any():
        return depth
    _, (ri, ci) = ndimage.distance_transform_edt(~have, return_indices=True)
    out = np.where(inside, depth[ri, ci], np.nan).astype(np.float32)
    return out


def bed_elevation(g, probe=None, bbox=BBOX, polys=None, depth=None, level=None):
    """Survey bed elevation (m IGLD85) on grid g inside the surveyed water bodies, NaN elsewhere, plus the inside mask
    and the metadata the lake-bottom file carries. polys/depth/level can be given (tests, reuse across grids)."""
    p = {} if probe is None else probe.setdefault("cook", {})
    polys = fetch_polygons(bbox) if polys is None else polys
    inside = rasterize(polys, g)
    if not inside.any():
        p["cells"] = 0
        return np.full((g["ny"], g["nx"]), np.nan, np.float32), inside, None
    depth = fetch_depth(g, bbox, p) if depth is None else depth
    level = survey_level(p) if level is None else level
    d = fill_inside(np.where(inside, depth, np.nan), inside)
    z = np.where(inside & np.isfinite(d), level - d, np.nan).astype(np.float32)
    p.update({"cells": int(np.isfinite(z).sum()), "polys": [q["name"] for q in polys],
              "depth_m": {"max": round(float(np.nanmax(d)), 2), "mean": round(float(np.nanmean(d)), 2)} if np.isfinite(d).any() else None})
    meta = {"source": SOURCE, "date": SURVEY_DATE, "level_m_igld85": round(level, 3),
            "polys": [{"name": q["name"], "acres": q.get("acres"), "rings": [[[round(x, 1), round(y, 1)] for x, y in r] for r in q["rings"]]} for q in polys]}
    return z, inside, meta


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=4.0)
    ap.add_argument("--out", default="cook_survey.json")
    a = ap.parse_args()
    from . import bathymetry as bm
    g = bm.local_grid(BBOX, a.dx); probe = {}
    z, inside, meta = bed_elevation(g, probe)
    print(json.dumps(probe, indent=1, default=str))
    with open(a.out, "w") as fh:
        json.dump({"grid": g, "meta": meta, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, fh)
    print(f"{int(inside.sum())} cells inside, {int(np.isfinite(z).sum())} with a bed; wrote {a.out}")
