"""
The Chicago River for the page: the lake → the Chicago Harbor Lock → the main stem → Wolf Point, the North Branch
(both sides of Goose Island, the North Avenue turning basin) to Belmont Avenue, the South Branch to Ping Tom Park.

    python -m app.river --out river.json [--cache DIR]          (needs: pip install laspy lazrs pyproj tifffile)

river.json holds, in the page's frame (metres east/north of State & Madison, heights in IGLD85 metres):
  bed       the river bottom on 2 m grids, one per reach box, from the USGS surveys (2022: lock → main stem → South
            Branch past Ping Tom, and the North Branch through both Goose Island channels to the North Ave basin;
            2023: North Ave → Belmont), packed like bathy.json (z in cm above LWD 176.0 m, int16, gzip, base64)
  polys     the river's water (OSM multipolygons and ways): the walls the page draws hard, class "river"
  quays     the sea walls and the Riverwalk edges (OSM man_made=quay, Riverwalk footways) with the bank height above
            the water from the USGS 3DEP 1 m lidar (bare earth: the quay top)
  bridges   every bridge over the river: name, kind (road / rail / foot), movable type, the deck outline, the deck
            height from the lidar at each abutment (the road surface), the piers' positions
  lock      the Chicago Harbor Lock chamber and gates
  buildings Cook County's lidar building footprints with heights near the river (OSM has heights for 0.4% of them)
  reaches   the paddling reaches (axis, width, the CSO reach name, the gage) for the page's current / wind / CSO model
  launches  where paddlers put in

Datums (derived 2026-11-02 from 12 months of 2003 lake readings at the Chicago Lock, USGS 04087440 in CCD feet,
against the same months' NOAA Calumet Harbor means in IGLD85): Chicago City Datum 0 = 176.40 m IGLD85 (σ 1.6 cm),
and IGLD85 − NAVD88 = −0.14 m at Chicago (CCD = 579.19 ft NAVD88 per USGS). The surveys are NAVD88 metres.
"""

import argparse
import base64
import datetime as dt
import gzip
import io
import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

from . import lakefront as lf

log = logging.getLogger("river")
UTC = dt.timezone.utc
UA = lf.UA
LWD = 176.0
CCD_IGLD = 176.40                      # Chicago City Datum 0 in IGLD85 metres
NAVD_TO_IGLD = -0.138                  # add to a NAVD88 height at Chicago to get IGLD85
RIVER_BBOX = (41.850, -87.700, 41.945, -87.600)       # S, W, N, E
BED_BOXES = {                                           # 2 m bed grids (S, W, N, E): the main stem + South Branch, and the North Branch
    "main": (41.850, -87.652, 41.898, -87.600),
    "north": (41.895, -87.700, 41.945, -87.636),
}
BED_DX = 2.0
LAZ = [  # USGS ScienceBase, the 0.5 m "reduced" point clouds (no login); the 2023 file carries no CRS: both are EPSG:6345
    {"name": "ChicagoSSCanal_bathymetry_2022-08.laz", "doi": "10.5066/P94TL2PV", "surveyed": "2022-08",
     "url": "https://www.sciencebase.gov/catalog/file/get/64419ae4d34ee8d4ade7c239?f=__disk__79%2Fab%2F91%2F79ab9116d06993ccf9033a19065908a350355698"},
    {"name": "NChicagoRivChannel_bathymetry_2023-07.laz", "doi": "10.5066/P98D7RAK", "surveyed": "2023-07",
     "url": "https://www.sciencebase.gov/catalog/file/get/6564eae7d34e3aa43a43c4db?f=__disk__10%2F24%2Fa0%2F1024a05354dcc54a6372f30caf4c35798bd4e0c8"},
]
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"   # 1 m bare earth, NAVD88 m
DEM_TILE = 400.0                       # m; tiles fetched on demand around the features that need heights
COOK_FOOT = "https://gis.cookcountyil.gov/traditional/rest/services/buildingFootprint_2022/MapServer/0/query"
OVERPASS = ([os.environ["OVERPASS_FIRST"]] if os.environ.get("OVERPASS_FIRST") else []) + lf.OVERPASS + ["https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
WATER_RELS = {13730103: "main", 3485552: "north", 13462125: "south", 17398850: "south"}     # OSM relations: the river's water
WATER_WAYS = {382710108: "canal", 401242573: "slip"}                                      # the North Branch Canal, Ogden Slip
LOCK_WAY = 164030677
BUILDING_BAND = 400.0                  # m from the river: footprints drawn with lidar heights
BUILDING_TALL = 30.0                   # m: taller ones anywhere in the river box

# the paddling reaches: a centreline (lat, lon) the page follows, the reach's CSO name (MWRD WATERWAY_REACH) and gage
REACHES = [
    {"id": "lock", "name": "The lock to Columbus Dr", "cso": "Chicago R", "gage": "05536123",
     "line": [[41.8888, -87.6090], [41.8885, -87.6150], [41.8880, -87.6205]]},
    {"id": "main", "name": "Main stem, Columbus Dr to Wolf Point", "cso": "Chicago R", "gage": "05536123",
     "line": [[41.8880, -87.6205], [41.8875, -87.6270], [41.8869, -87.6330], [41.8870, -87.6372]]},
    {"id": "south", "name": "South Branch, Wolf Point to Ping Tom Park", "cso": "SB Chicago R", "gage": "05536123",
     "line": [[41.8870, -87.6372], [41.8830, -87.6385], [41.8780, -87.6390], [41.8720, -87.6385], [41.8660, -87.6370], [41.8600, -87.6350], [41.8555, -87.6340]]},
    {"id": "north_low", "name": "North Branch, Wolf Point to Chicago Ave", "cso": "NBCR Lower (NSC Confluence)", "gage": "05536118",
     "line": [[41.8870, -87.6372], [41.8905, -87.6395], [41.8960, -87.6430]]},
    {"id": "goose_w", "name": "North Branch, west of Goose Island", "cso": "NBCR Lower (NSC Confluence)", "gage": "05536118",
     "line": [[41.8960, -87.6430], [41.9000, -87.6490], [41.9040, -87.6545], [41.9085, -87.6560], [41.9115, -87.6555]]},
    {"id": "goose_e", "name": "North Branch Canal, east of Goose Island", "cso": "NBCR Lower (NSC Confluence)", "gage": "05536118",
     "line": [[41.8960, -87.6430], [41.9000, -87.6480], [41.9040, -87.6500], [41.9090, -87.6510], [41.9115, -87.6555]]},
    {"id": "north_up", "name": "North Branch, North Ave to Belmont", "cso": "NBCR Upper (NSC Confluence)", "gage": "05536118",
     "line": [[41.9115, -87.6555], [41.9160, -87.6600], [41.9210, -87.6660], [41.9260, -87.6720], [41.9320, -87.6800], [41.9390, -87.6900]]},
]
LAUNCHES = [
    ["Urban Kayaks · Riverwalk", 41.8880, -87.6197, "main"],
    ["Kayak Chicago · North Ave turning basin", 41.9105, -87.6560, "goose_w"],
    ["Ping Tom boathouse", 41.8575, -87.6345, "south"],
]


def http(url, data=None, timeout=120, accept="*/*"):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------- the river bottom
def fetch_laz(cache, get=http):
    os.makedirs(cache, exist_ok=True)
    paths = []
    for f in LAZ:
        p = os.path.join(cache, f["name"])
        if not os.path.exists(p) or os.path.getsize(p) < 1e6:
            log.info("river: downloading %s", f["name"])
            with open(p, "wb") as fh:
                fh.write(get(f["url"], timeout=900))
        paths.append(p)
    return paths


def bed_points(paths, bbox=RIVER_BBOX):
    """The surveys' points inside bbox -> x, y (local m), z (IGLD85 m)."""
    import laspy
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:6345", "EPSG:4326", always_xy=True)
    xs, ys, zs = [], [], []
    for p in paths:
        las = laspy.read(p)
        lon, lat = tr.transform(np.asarray(las.x), np.asarray(las.y))
        z = np.asarray(las.z, np.float64) + NAVD_TO_IGLD
        m = (lat > bbox[0]) & (lat < bbox[2]) & (lon > bbox[1]) & (lon < bbox[3])
        xs.append((lon[m] - lf.REF[1]) * lf.KX); ys.append((lat[m] - lf.REF[0]) * lf.KY); zs.append(z[m])
        log.info("river: %s: %d points in the box", os.path.basename(p), int(m.sum()))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def local_grid(bbox, d):
    (xa, ya), (xb, yb) = lf.xy(bbox[0], bbox[1]), lf.xy(bbox[2], bbox[3])
    x0, y0 = math.floor(xa / d) * d, math.floor(ya / d) * d
    return {"x0": x0, "y0": y0, "dx": d, "nx": int((xb - x0) // d) + 1, "ny": int((yb - y0) // d) + 1}


def bed_grid(g, x, y, z, max_gap=3):
    """Mean bed height per cell; holes up to max_gap cells wide filled from their neighbours. NaN where no survey."""
    from scipy import ndimage
    i = np.floor((x - g["x0"]) / g["dx"]).astype(int); j = np.floor((y - g["y0"]) / g["dx"]).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < g["nx"]) & (j < g["ny"])
    n = np.zeros((g["ny"], g["nx"]), np.float64); s = np.zeros_like(n)
    np.add.at(n, (j[ok], i[ok]), 1); np.add.at(s, (j[ok], i[ok]), z[ok])
    zz = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    have = n > 0
    if not have.any():
        return zz, have
    # fill small holes: cells within max_gap of data, from the nearest data cell
    near = ndimage.binary_dilation(have, iterations=max_gap) & ~have
    if near.any():
        idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
        zz[near] = zz[idx[0][near], idx[1][near]]
    return zz, have | near


def pack(g, z, inside):
    """bathy.json's packing: z in cm above LWD (int16, -32768 where none), inside as bytes."""
    zc = np.where(np.isfinite(z) & inside, np.round((z - LWD) * 100), -32768).astype("<i2")
    enc = lambda a: base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()  # noqa: E731
    return {**g, "z_cm": enc(zc), "inside": enc(inside.astype(np.uint8)), "cells": int(inside.sum())}


def unpack(gr):
    z = np.frombuffer(gzip.decompress(base64.b64decode(gr["z_cm"])), "<i2").reshape(gr["ny"], gr["nx"]).astype(np.float32)
    inside = np.frombuffer(gzip.decompress(base64.b64decode(gr["inside"])), np.uint8).reshape(gr["ny"], gr["nx"]) > 0
    z = np.where(inside, z / 100.0 + LWD, np.nan)
    return z, inside


# ---------------------------------------------------------------------------- the river's shape (OpenStreetMap)
def overpass(q, get=http):
    last = None
    for host in OVERPASS:
        for attempt in range(3):
            try:
                return json.loads(get(host, data=q.encode(), timeout=180, accept="application/json"))
            except Exception as e:  # noqa: BLE001
                last = e; log.info("overpass %s: %s", host.split("/")[2], str(e)[:80]); time.sleep(8)
    raise RuntimeError(f"Overpass failed: {last}")


def queries(bbox=RIVER_BBOX):
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"
    h = "[out:json][timeout:120];"
    rels = ",".join(str(r) for r in WATER_RELS); ways = ",".join(str(w) for w in WATER_WAYS)
    return {
        "water": h + f"(rel(id:{rels});way(id:{ways}););out body geom;",
        "lock": h + f'(way({LOCK_WAY});node["waterway"="lock_gate"]({bb});way["waterway"="lock_gate"]({bb}););out geom tags;',
        "quays": h + f'(way["man_made"="quay"]({bb});way["man_made"="embankment"]({bb});way["highway"]["name"~"Riverwalk"]({bb}););out geom tags;',
        "bridges": h + f'(way["bridge"]({bb});way["man_made"="bridge"]({bb}););out geom tags;',
    }


def ring_xy(geom):
    return [list(lf.xy(p["lat"], p["lon"])) for p in geom]


def water_polys(js):
    """The river's water as rings in local metres, tagged by reach ("main", "north", "south", "canal", "slip")."""
    out = []
    for el in js.get("elements", []):
        if el.get("type") == "relation" and el["id"] in WATER_RELS:
            parts = [[tuple(lf.xy(p["lat"], p["lon"])) for p in m["geometry"]] for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry")]
            inner = [[list(lf.xy(p["lat"], p["lon"])) for p in m["geometry"]] for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role") == "inner" and m.get("geometry")]
            for ring in lf.join_rings(parts):
                out.append({"kind": WATER_RELS[el["id"]], "rings": [[list(p) for p in ring]] + [r for r in inner if len(r) > 3], "rel": el["id"],
                            "name": (el.get("tags") or {}).get("name")})
        elif el.get("type") == "way" and el["id"] in WATER_WAYS and el.get("geometry"):
            out.append({"kind": WATER_WAYS[el["id"]], "rings": [ring_xy(el["geometry"])], "way": el["id"], "name": (el.get("tags") or {}).get("name")})
    return out


def clip_rings(polys, bbox=RIVER_BBOX):
    """Keep only the rings' points inside the river box (the North Branch relation runs far north; the page needs
    the box only). Rings are cut at the box edge and closed along it."""
    (xa, ya), (xb, yb) = lf.xy(bbox[0], bbox[1]), lf.xy(bbox[2], bbox[3])
    out = []
    for p in polys:
        rings = []
        for ring in p["rings"]:
            r = clip_polygon(ring, xa, ya, xb, yb)
            if len(r) >= 4:
                rings.append(r)
        if rings:
            out.append({**p, "rings": rings})
    return out


def clip_polygon(ring, xa, ya, xb, yb):
    """Sutherland-Hodgman against the box."""
    def clip(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out
    pts = [tuple(p) for p in ring if len(p) >= 2]
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    ix = lambda a, b, x: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))  # noqa: E731
    iy = lambda a, b, y: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)  # noqa: E731
    for inside, inter in ((lambda p: p[0] >= xa, lambda a, b: ix(a, b, xa)), (lambda p: p[0] <= xb, lambda a, b: ix(a, b, xb)),
                          (lambda p: p[1] >= ya, lambda a, b: iy(a, b, ya)), (lambda p: p[1] <= yb, lambda a, b: iy(a, b, yb))):
        pts = clip(pts, inside, inter)
        if not pts:
            return []
    return [[round(x, 1), round(y, 1)] for x, y in pts]


def rasterize(rings, g):
    """Even-odd fill of rings (local m) on grid g -> bool (ny, nx)."""
    nx, ny, d = g["nx"], g["ny"], g["dx"]
    out = np.zeros((ny, nx), bool)
    ys = g["y0"] + np.arange(ny) * d
    for ring in rings:
        P = np.asarray(ring, float)
        if len(P) < 3:
            continue
        x0, y0 = P[:, 0], P[:, 1]; x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for jj, y in enumerate(ys):
            m = (y0 > y) != (y1 > y)
            if not m.any():
                continue
            xs = np.sort(x0[m] + (y - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m]))
            for k in range(0, len(xs) - 1, 2):
                a, b = int(np.ceil((xs[k] - g["x0"]) / d)), int(np.floor((xs[k + 1] - g["x0"]) / d))
                if b >= a:
                    out[jj, max(a, 0):min(b, nx - 1) + 1] ^= True
    return out


def point_in_rings(x, y, rings):
    inside = False
    for ring in rings:
        P = ring
        n = len(P)
        for i in range(n):
            ax, ay = P[i]; bx, by = P[(i + 1) % n]
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
    return inside


# ---------------------------------------------------------------------------- heights from the 3DEP lidar (bare earth)
class Dem:
    """1 m bare-earth heights (IGLD85 m) around points, fetched as 400 m tiles from the USGS 3DEP ImageServer and cached."""

    def __init__(self, cache, get=http):
        self.cache, self.get, self.tiles = cache, get, {}
        os.makedirs(os.path.join(cache, "dem"), exist_ok=True)

    def tile(self, tx, ty):
        key = (tx, ty)
        if key in self.tiles:
            return self.tiles[key]
        import tifffile
        path = os.path.join(self.cache, "dem", f"t_{tx}_{ty}.tif")
        if not os.path.exists(path):
            x0, y0 = tx * DEM_TILE, ty * DEM_TILE
            lat0, lon0 = lf.REF[0] + y0 / lf.KY, lf.REF[1] + x0 / lf.KX
            lat1, lon1 = lf.REF[0] + (y0 + DEM_TILE) / lf.KY, lf.REF[1] + (x0 + DEM_TILE) / lf.KX
            n = int(DEM_TILE)
            q = {"bbox": f"{lon0},{lat0},{lon1},{lat1}", "bboxSR": "4326", "imageSR": "4326", "size": f"{n},{n}", "format": "tiff",
                 "pixelType": "F32", "noData": "-9999", "interpolation": "RSP_BilinearInterpolation", "renderingRule": '{"rasterFunction":"None"}', "f": "image"}
            for attempt in range(3):
                try:
                    b = self.get(DEM + "?" + urllib.parse.urlencode(q), timeout=180)
                    if b[:2] in (b"II", b"MM"):
                        with open(path, "wb") as fh:
                            fh.write(b)
                        break
                except Exception as e:  # noqa: BLE001
                    log.info("dem tile %s,%s: %s", tx, ty, e)
                time.sleep(3)
        arr = None
        if os.path.exists(path):
            a = tifffile.imread(path).astype(np.float32)
            a[(a < -1000) | ~np.isfinite(a)] = np.nan
            arr = a + NAVD_TO_IGLD      # rows run north -> south over the tile (lat1 at row 0)
        self.tiles[key] = arr
        return arr

    def at(self, x, y):
        """Height at a local point (m), NaN where none."""
        tx, ty = math.floor(x / DEM_TILE), math.floor(y / DEM_TILE)
        a = self.tile(tx, ty)
        if a is None:
            return float("nan")
        n = a.shape[0]
        col = int((x - tx * DEM_TILE) / DEM_TILE * a.shape[1]); row = int((1 - (y - ty * DEM_TILE) / DEM_TILE) * n)
        col = min(max(col, 0), a.shape[1] - 1); row = min(max(row, 0), n - 1)
        return float(a[row, col])

    def robust(self, pts, lo=10, hi=90):
        """A trimmed mean of heights at several points (the quay top, a road deck), NaN if fewer than 3 are known."""
        v = np.array([self.at(x, y) for x, y in pts], float); v = v[np.isfinite(v)]
        if len(v) < 3:
            return float("nan")
        a, b = np.percentile(v, [lo, hi])
        return float(v[(v >= a) & (v <= b)].mean())


# ---------------------------------------------------------------------------- quays, bridges, the lock
def seg_len(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))


def resample(pts, step):
    out = [pts[0]]; acc = 0.0
    for i in range(1, len(pts)):
        ax, ay = pts[i - 1]; bx, by = pts[i]; L = math.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue
        s = step - acc
        while s <= L:
            out.append([ax + (bx - ax) * s / L, ay + (by - ay) * s / L]); s += step
        acc = L - (s - step)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


class Edges:
    """The river's edges as arrays, for fast distance / inside tests."""

    def __init__(self, polys):
        A, B, rings = [], [], []
        for p in polys:
            for ring in p["rings"]:
                P = np.asarray(ring, float)
                if len(P) < 3:
                    continue
                rings.append(P); A.append(P); B.append(np.roll(P, -1, axis=0))
        self.A = np.concatenate(A) if A else np.zeros((0, 2)); self.B = np.concatenate(B) if B else np.zeros((0, 2)); self.rings = rings
        self.lo = np.minimum(self.A, self.B); self.hi = np.maximum(self.A, self.B)

    def dist(self, x, y, rmax=25.0):
        m = (self.lo[:, 0] <= x + rmax) & (self.hi[:, 0] >= x - rmax) & (self.lo[:, 1] <= y + rmax) & (self.hi[:, 1] >= y - rmax)
        if not m.any():
            return None
        A, B = self.A[m], self.B[m]; D = B - A; L2 = (D * D).sum(1)
        t = np.clip(((x - A[:, 0]) * D[:, 0] + (y - A[:, 1]) * D[:, 1]) / np.where(L2 == 0, 1, L2), 0, 1)
        d = np.hypot(x - (A[:, 0] + D[:, 0] * t), y - (A[:, 1] + D[:, 1] * t)).min()
        return float(d) if d <= rmax else None

    def inside(self, pts):
        """Even-odd test of many points (n, 2) against all rings -> bool (n,)."""
        P = np.asarray(pts, float); out = np.zeros(len(P), bool)
        for R in self.rings:
            ax, ay = R[:, 0], R[:, 1]; bx, by = np.roll(ax, -1), np.roll(ay, -1)
            for k in range(len(P)):
                x, y = P[k]
                m = (ay > y) != (by > y)
                if m.any():
                    xi = ax[m] + (y - ay[m]) * (bx[m] - ax[m]) / (by[m] - ay[m])
                    out[k] ^= bool((x < xi).sum() & 1)
        return out


def nearest_water(x, y, polys, rmax=25.0):
    E = polys if isinstance(polys, Edges) else Edges(polys)
    return E.dist(x, y, rmax)


def quays(js, polys, dem, level):
    E = polys if isinstance(polys, Edges) else Edges(polys)
    """Walls along the water: OSM quay / embankment / Riverwalk ways within 25 m of the river, sampled every 10 m for
    the bank height above the water (the lidar's bare earth, which is the quay top)."""
    out = []
    for el in js.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        t = el.get("tags") or {}
        pts = ring_xy(el["geometry"])
        if len(pts) < 2:
            continue
        kind = "riverwalk" if "Riverwalk" in (t.get("name") or "") else t.get("man_made", "quay")
        near = [p for p in resample(pts, 10.0) if (E.dist(p[0], p[1]) or 99) <= 25]
        if len(near) < 2:
            continue
        hs = [dem.at(p[0], p[1]) for p in near]
        h = np.array(hs, float); h = h[np.isfinite(h)]
        top = float(np.median(h)) if len(h) else float("nan")
        out.append({"kind": kind, "name": t.get("name"), "pts": [[round(x, 1), round(y, 1)] for x, y in pts],
                    "top": round(top, 2) if np.isfinite(top) else None, "above": round(top - level, 2) if np.isfinite(top) else None})
    return out


def bridge_kind(t):
    if "railway" in t:
        return "rail"
    if t.get("highway") in ("footway", "path", "cycleway", "pedestrian", "steps"):
        return "foot"
    return "road"


def way_width(t, kind):
    w = t.get("width")
    try:
        if w:
            return float(re.match(r"[0-9.]+", w).group(0))
    except (AttributeError, ValueError):
        pass
    try:
        lanes = int(t.get("lanes", 0))
        if lanes:
            return lanes * 3.4 + 4.0
    except ValueError:
        pass
    return {"rail": 9.0, "foot": 4.0}.get(kind, 14.0)


def crossing(pts, polys):
    """The part of a way over the water: the first and last points of pts inside any river polygon, extended to the
    edges; None if it never crosses the water."""
    E = polys if isinstance(polys, Edges) else Edges(polys)
    fine = resample(pts, 3.0)
    # a quick box test first: most of the 3,500 bridges in the box are nowhere near the river
    P = np.asarray(fine); lo, hi = P.min(0), P.max(0)
    if not ((E.lo[:, 0] <= hi[0]) & (E.hi[:, 0] >= lo[0]) & (E.lo[:, 1] <= hi[1]) & (E.hi[:, 1] >= lo[1])).any():
        return None
    inside = list(E.inside(fine))
    if not any(inside):
        return None
    i0 = inside.index(True); i1 = len(inside) - 1 - inside[::-1].index(True)
    a, b = fine[i0], fine[i1]
    if math.hypot(b[0] - a[0], b[1] - a[1]) < 8:
        return None
    return a, b, i0, i1, fine


def bridges(js, polys, dem, level):
    """One record per bridge: the ways (road, rail, foot) sharing a name or an outline over the same crossing."""
    polys = polys if isinstance(polys, Edges) else Edges(polys)
    outlines, ways = [], []
    for el in js.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        t = el.get("tags") or {}
        pts = ring_xy(el["geometry"])
        if t.get("man_made") == "bridge" and pts[0] == pts[-1]:
            outlines.append({"pts": pts, "tags": t})
        elif t.get("bridge") and t.get("bridge") != "no" and ("highway" in t or "railway" in t):
            ways.append({"pts": pts, "tags": t})
    recs = []
    for w in ways:
        c = crossing(w["pts"], polys)
        if not c:
            continue
        a, b, i0, i1, fine = c
        t = w["tags"]; kind = bridge_kind(t)
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        name = t.get("name") or t.get("bridge:name") or ""
        # the deck height: the lidar's road surface 6-20 m past each end of the crossing where it is plausible (the bare-
        # earth DEM strips elevated decks and their approaches, leaving the ground below), else the river's typical closed
        # clearances (chicagoloopbridges.com: 18-31 ft at LWD, DuSable 17 ft, Lake Shore Dr 25 ft)
        ux, uy = b[0] - a[0], b[1] - a[1]; L = math.hypot(ux, uy) or 1; ux, uy = ux / L, uy / L
        ends = [[a[0] - ux * s, a[1] - uy * s] for s in (6, 10, 14, 20)] + [[b[0] + ux * s, b[1] + uy * s] for s in (6, 10, 14, 20)]
        lid = dem.robust(ends, 25, 75)
        nominal = LWD + known_clearance(name, kind, t)
        deck, how = (lid, "lidar") if np.isfinite(lid) and 3.5 <= lid - level <= 12 else (nominal, "nominal")
        rec = {"name": name, "kind": kind, "movable": t.get("bridge:movable"), "layer": t.get("layer"), "width": round(way_width(t, kind), 1),
               "a": [round(a[0], 1), round(a[1], 1)], "b": [round(b[0], 1), round(b[1], 1)], "span": round(L, 1),
               "deck": round(deck, 2), "clearance": round(deck - level, 2), "deck_from": how,
               "line": [[round(x, 1), round(y, 1)] for x, y in fine[max(i0 - 4, 0):i1 + 5]], "_c": (cx, cy)}
        recs.append(rec)
    # merge the ways of one bridge: the same name within 40 m, or an unnamed way (a sidewalk, a bikeway) within 25 m of any
    # bridge; named ways first so the sidewalks join their road
    merged = []
    for r in sorted(recs, key=lambda r: (r["name"] == "", {"road": 0, "rail": 1, "foot": 2}[r["kind"]], r["name"])):
        hit = None
        for m in merged:
            d = math.hypot(m["_c"][0] - r["_c"][0], m["_c"][1] - r["_c"][1])
            if (r["name"] and r["name"] == m["name"] and d < 40) or (not r["name"] and d < 25):
                hit = m; break
        if hit:
            hit["ways"] += 1
            if abs(hit["_c"][0] - r["_c"][0]) + abs(hit["_c"][1] - r["_c"][1]) > 3 and r["kind"] != "foot":
                hit["width"] = round(hit["width"] + r["width"], 1)
            if hit["deck_from"] == "nominal" and r["deck_from"] == "lidar":
                hit["deck"], hit["clearance"], hit["deck_from"] = r["deck"], r["clearance"], "lidar"
            continue
        merged.append({**r, "ways": 1})
    # outlines (man_made=bridge): the deck polygon, matched to the nearest merged bridge
    for o in outlines:
        P = np.asarray(o["pts"]); cx, cy = P[:, 0].mean(), P[:, 1].mean()
        best = min(merged, key=lambda m: math.hypot(m["_c"][0] - cx, m["_c"][1] - cy), default=None)
        if best and math.hypot(best["_c"][0] - cx, best["_c"][1] - cy) < 60:
            best["outline"] = [[round(x, 1), round(y, 1)] for x, y in o["pts"]]
            if not best["name"] and o["tags"].get("name"):
                best["name"] = o["tags"]["name"]
    for m in merged:
        m.pop("_c", None)
    merged = [m for m in merged if m["span"] >= 20 or m["name"]]      # tiny unnamed crossings are slips and dock inlets
    merged.sort(key=lambda m: (m["a"][0] + m["b"][0]) / 2)
    return merged


KNOWN_CLEARANCE_FT = {"michigan": 17.0, "dusable": 17.0, "jean baptiste": 17.0, "lake shore": 25.0}   # closed, above LWD (chicagoloopbridges.com)


def known_clearance(name, kind, tags):
    """Closed clearance above LWD (m) for a bridge without a usable lidar deck: a known figure, else the river's typical."""
    n = (name or "").lower()
    for k, ft in KNOWN_CLEARANCE_FT.items():
        if k in n:
            return ft * 0.3048
    if tags.get("bridge:movable") in ("bascule", "lift", "swing"):
        return {"road": 18.0, "rail": 16.0, "foot": 15.0}[kind] * 0.3048
    return {"road": 22.0, "rail": 18.0, "foot": 16.0}[kind] * 0.3048        # fixed spans sit higher


def lock(js):
    out = {"gates": []}
    for el in js.get("elements", []):
        t = el.get("tags") or {}
        if el.get("type") == "way" and el["id"] == LOCK_WAY and el.get("geometry"):
            out["chamber"] = ring_xy(el["geometry"]); out["name"] = t.get("name", "Chicago Harbor Lock")
        elif t.get("waterway") == "lock_gate":
            if el.get("type") == "node":
                out["gates"].append([round(v, 1) for v in lf.xy(el["lat"], el["lon"])])
            elif el.get("geometry"):
                out["gates"].append(ring_xy(el["geometry"]))
    return out


# ---------------------------------------------------------------------------- buildings (Cook County lidar footprints)
def cook_buildings(bbox, polys, get=http, band=BUILDING_BAND, tall=BUILDING_TALL):
    """Footprints with lidar heights (feet -> m) within band m of the river or at least tall m high."""
    polys = polys if isinstance(polys, Edges) else Edges(polys)
    s, w, n, e = bbox
    out, offset = [], 0
    while True:
        q = {"geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
             "outFields": "Height,Area_SQFT", "outSR": "4326", "resultOffset": offset, "resultRecordCount": 2000, "f": "json",
             "geometryPrecision": 6}
        js = json.loads(get(COOK_FOOT + "?" + urllib.parse.urlencode(q), timeout=180))
        feats = js.get("features", [])
        for f in feats:
            h = (f.get("attributes") or {}).get("Height")
            if h is None or h < 10:
                continue
            hm = float(h) * 0.3048
            rings = (f.get("geometry") or {}).get("rings") or []
            if not rings:
                continue
            ring = [list(lf.xy(p[1], p[0])) for p in rings[0]]
            if len(ring) < 4:
                continue
            cx = sum(p[0] for p in ring) / len(ring); cy = sum(p[1] for p in ring) / len(ring)
            if hm < tall and (polys.dist(cx, cy, rmax=band) or 1e9) > band:
                continue
            out.append({"h": round(hm, 1), "pts": [[round(x, 1), round(y, 1)] for x, y in simplify(ring, 1.5)]})
        if not js.get("exceededTransferLimit") or not feats:
            break
        offset += len(feats)
        time.sleep(0.5)
    return out


def simplify(pts, tol):
    """Douglas-Peucker on a closed ring (keeps the first point)."""
    def dp(P):
        if len(P) < 3:
            return P
        (x0, y0), (x1, y1) = P[0], P[-1]
        dx, dy = x1 - x0, y1 - y0; n = math.hypot(dx, dy) or 1e-12
        k, dmax = 0, -1.0
        for i in range(1, len(P) - 1):
            d = abs(dy * (P[i][0] - x0) - dx * (P[i][1] - y0)) / n
            if d > dmax:
                k, dmax = i, d
        if dmax <= tol:
            return [P[0], P[-1]]
        return dp(P[:k + 1])[:-1] + dp(P[k:])
    P = pts if pts[0] != pts[-1] else pts[:-1]
    if len(P) < 4:
        return P
    h = len(P) // 2
    return dp(P[:h + 1])[:-1] + dp(P[h:] + [P[0]])[:-1]


# ---------------------------------------------------------------------------- the river's water level
def gage_level(get=http, site="05536123"):
    """The main stem's stage now (CCD ft) -> IGLD85 m, and the 72254 water velocity (ft/s, + = away from the lake)."""
    u = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00065,72254&siteStatus=all"
    d = json.loads(get(u, timeout=60))
    out = {"site": site}
    for ts in d["value"]["timeSeries"]:
        code = ts["variable"]["variableCode"][0]["value"]; vals = ts["values"][0]["value"]
        if not vals:
            continue
        v = float(vals[-1]["value"]); t = vals[-1]["dateTime"]
        if code == "00065":
            out["stage_ft"] = v; out["level_m"] = round(CCD_IGLD + v * 0.3048, 3); out["t"] = t
        elif code == "72254":
            out["velocity_fps"] = v
    return out


def osm_features(cache, get=http, max_age_days=30):
    """The Overpass answers, kept in the cache for a month (the mirrors are slow and the river doesn't move)."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "osm.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400:
        js = json.load(open(path))
        if all(k in js for k in queries()):
            return js
    js = {}
    for k, q in queries().items():
        js[k] = overpass(q, get)
        log.info("river: overpass %s: %d elements", k, len(js[k].get("elements", [])))
    with open(path, "w") as fh:
        json.dump(js, fh)
    return js


# ---------------------------------------------------------------------------- build
def build(out_path, cache, get=http, skip_buildings=False):
    t0 = time.time()
    level = LWD + 0.4
    try:
        gl = gage_level(get); level = gl.get("level_m", level)
    except Exception as e:  # noqa: BLE001
        gl = {"error": str(e)[:200]}
    log.info("river: water level %.2f m IGLD85 (%s)", level, gl)
    js = osm_features(cache, get)
    polys = clip_rings(water_polys(js["water"]))
    log.info("river: %d water polygons", len(polys))
    E = Edges(polys)
    dem = Dem(cache, get)
    out = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds"), "bbox": RIVER_BBOX,
           "datum": {"ccd_igld85_m": CCD_IGLD, "navd88_to_igld85_m": NAVD_TO_IGLD, "lwd_m": LWD}, "level": {**gl, "used_m": round(level, 3)},
           "polys": polys, "lock": lock(js["lock"]), "quays": quays(js["quays"], E, dem, level), "bridges": bridges(js["bridges"], E, dem, level),
           "reaches": REACHES, "launches": LAUNCHES,
           "sources": ["USGS ScienceBase bathymetry 2022-08 (10.5066/P94TL2PV) and 2023-07 (10.5066/P98D7RAK)", "USGS 3DEP 1 m lidar (IL 4-county 2016)",
                       "Cook County GIS buildingFootprint_2022", "OpenStreetMap (ODbL)", "USGS NWIS 05536123"]}
    log.info("river: %d quays, %d bridges, lock %s", len(out["quays"]), len(out["bridges"]), "ok" if out["lock"].get("chamber") else "missing")
    # the bed
    paths = fetch_laz(cache, get)
    x, y, z = bed_points(paths)
    rings = [r for p in polys for r in p["rings"]]
    out["bed"] = {}
    for name, box in BED_BOXES.items():
        g = local_grid(box, BED_DX)
        zz, have = bed_grid(g, x, y, z)
        water = rasterize(rings, g)
        inside = have & water                      # the survey inside the mapped river only (stray bank returns dropped)
        # mapped water the survey missed (under bridges, the edges): the nearest surveyed bed within 12 m
        from scipy import ndimage
        miss = water & ~inside
        if miss.any() and inside.any():
            idx = ndimage.distance_transform_edt(~inside, return_distances=True, return_indices=True)
            dist, ind = idx
            near = miss & (dist <= 6)
            zz[near] = zz[ind[0][near], ind[1][near]]; inside = inside | near
        out["bed"][name] = pack(g, zz, inside)
        dep = level - zz[inside]
        log.info("river bed %s: %dx%d at %g m, %d cells, depth median %.1f max %.1f m", name, g["nx"], g["ny"], BED_DX, int(inside.sum()),
                 float(np.nanmedian(dep)), float(np.nanmax(dep)))
        out["bed"][name]["depth"] = {"median_m": round(float(np.nanmedian(dep)), 2), "max_m": round(float(np.nanmax(dep)), 2)}
    if not skip_buildings:
        try:
            out["buildings"] = cook_buildings(RIVER_BBOX, E, get)
            log.info("river: %d buildings", len(out["buildings"]))
        except Exception as e:  # noqa: BLE001
            out["buildings_error"] = str(e)[:300]
    out["took_s"] = round(time.time() - t0)
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


# ---------------------------------------------------------------------------- now: the gages, overflows and the lock (each snapshot)
CSO_EVENTS = "https://utility.arcgis.com/usrsvcs/servers/addf95280b254c1e8fe64dc8e35dfedc/rest/services/CSO/CSO_PossibleEvents/MapServer/0/query?where=1%3D1&outFields=*&f=json"
NTNI = "https://ndc.ops.usace.army.mil/ords/ntni/json_data/notices_by_district/LRC"
LOCK_SEASON = {"open": "04-15", "close": "11-15"}      # the Chicago Harbor Lock is closed to recreational traffic Nov 15 - Apr 15 (USACE notice 13968-1)
CSO_AFTER_H = 48                                       # hours after an overflow ends that the reach stays red
GAGES = {"05536123": "main", "05536118": "north", "05536121": "lock"}
REVERSAL_FPS = -0.3                                    # a sustained velocity toward the lake past this is a reversal, not lock sloshing


def gage_series(get=http, days=7):
    """Hourly stage (CCD ft) at the three gages and velocity (ft/s) at Columbus Dr for the past days: {site: {t: [...], ...}}."""
    u = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={','.join(GAGES)}&parameterCd=00065,72254&period=P{days}D&siteStatus=all"
    d = json.loads(get(u, timeout=90))
    out = {}
    for ts in d["value"]["timeSeries"]:
        site = ts["sourceInfo"]["siteCode"][0]["value"]; code = ts["variable"]["variableCode"][0]["value"]
        vals = [(v["dateTime"], float(v["value"])) for v in ts["values"][0]["value"] if v.get("value") not in (None, "", "-999999")]
        if not vals:
            continue
        # one value an hour (the last reading in each hour)
        hourly = {}
        for t, v in vals:
            hourly[t[:13]] = (t, v)
        rows = sorted(hourly.values())
        rec = out.setdefault(site, {"name": GAGES[site], "t": [r[0] for r in rows]})
        key = "stage_ft" if code == "00065" else "velocity_fps"
        if rec["t"] != [r[0] for r in rows]:          # the two parameters on different hours: align on the stage's hours
            byt = dict(rows); rec[key] = [byt.get(t) for t in rec["t"]]
        else:
            rec[key] = [r[1] for r in rows]
        rec["now"] = rec.get("now", {}); rec["now"][key] = rows[-1][1]; rec["now"]["t"] = rows[-1][0]
    for site, rec in out.items():
        if "stage_ft" in rec:
            rec["level_m"] = [round(CCD_IGLD + v * 0.3048, 3) if v is not None else None for v in rec["stage_ft"]]
    return out


def cso_events(prev, now=None, get=http):
    """MWRD's open 'possible CSO events' by reach, carried with first-seen / last-seen times so the page can hold a reach
    red for CSO_AFTER_H hours after an overflow ends. prev: the previous snapshot's record (events list), or None."""
    now = now or dt.datetime.now(UTC)
    try:
        d = json.loads(get(CSO_EVENTS, timeout=60)); feats = d.get("features", []); err = None
    except Exception as e:  # noqa: BLE001
        feats, err = [], str(e)[:200]
    open_now = {}
    for f in feats:
        a = f.get("attributes") or {}
        key = f"{a.get('WATERWAY_REACH', '?')}|{a.get('LOCATION', '')}"
        open_now[key] = {"reach": a.get("WATERWAY_REACH"), "location": a.get("LOCATION"), "owner": a.get("OWNER"), "city": a.get("OUTFALL_CITY"),
                         "reversal": bool(re.search(r"lake|crcw|reversal", f"{a.get('LOCATION', '')} {a.get('WATERWAY_REACH', '')}", re.I))}
    events = {e["key"]: e for e in (prev or []) if e.get("key")}
    ts = now.isoformat(timespec="seconds")
    for key, e in open_now.items():
        if key in events and events[key].get("open"):
            events[key]["last_seen"] = ts
        else:
            events[key] = {**e, "key": key, "first_seen": ts, "last_seen": ts, "open": True}
    for key, e in events.items():
        if key not in open_now and e.get("open"):
            e["open"] = False; e["ended"] = ts
    keep = [e for e in events.values() if e.get("open") or (e.get("ended") and (now - dt.datetime.fromisoformat(e["ended"])).total_seconds() < 14 * 86400)]
    return {"events": keep, "checked": ts, "error": err, "source": "MWRD Possible CSO Events (ArcGIS)"}


def lock_notices(get=http, now=None):
    """USACE Chicago District notices naming the Chicago Harbor / River, and the lock's season."""
    now = now or dt.datetime.now(UTC)
    out = {"season": LOCK_SEASON, "notices": []}
    y = now.year
    op = dt.datetime.fromisoformat(f"{y}-{LOCK_SEASON['open']}T12:00:00+00:00"); cl = dt.datetime.fromisoformat(f"{y}-{LOCK_SEASON['close']}T12:00:00+00:00")
    out["open_now"] = op <= now < cl
    out["next"] = {"opens": f"{y + 1 if now >= op else y}-{LOCK_SEASON['open']}", "closes": f"{y}-{LOCK_SEASON['close']}" if now < cl else f"{y + 1}-{LOCK_SEASON['close']}"}
    try:
        d = json.loads(get(NTNI, timeout=60))
        for it in d.get("items", []):
            if re.search(r"chicago (harbor|river)", it.get("waterways", ""), re.I):
                out["notices"].append({k: it.get(k) for k in ("noticeno", "issuedate", "begindate", "enddate", "waterways", "noticelink")})
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:200]
    return out


def build_now(out_path, prev_path=None, get=http):
    prev = None
    if prev_path and os.path.exists(prev_path):
        try:
            prev = json.load(open(prev_path)).get("cso", {}).get("events")
        except Exception:  # noqa: BLE001
            prev = None
    out = {"version": 1, "generated": dt.datetime.now(UTC).isoformat(timespec="seconds")}
    try:
        out["gages"] = gage_series(get)
    except Exception as e:  # noqa: BLE001
        out["gages_error"] = str(e)[:200]
    out["cso"] = cso_events(prev, get=get)
    out["lock"] = lock_notices(get)
    g = (out.get("gages") or {}).get("05536123") or {}
    vel = [v for v in (g.get("velocity_fps") or [])[-6:] if v is not None]
    out["reversal"] = bool(vel) and len(vel) >= 3 and max(vel) < REVERSAL_FPS
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.environ.get("RIVER_CACHE", "river_cache"))
    ap.add_argument("--no-buildings", action="store_true")
    ap.add_argument("--now", action="store_true", help="the live record (gages, overflows, the lock) instead of the geometry")
    ap.add_argument("--prev", help="with --now: the previous snapshot's river_now.json (carries the overflow history)")
    a = ap.parse_args()
    if a.now:
        d = build_now(a.out, a.prev)
        g = (d.get("gages") or {}).get("05536123", {}).get("now")
        print(json.dumps({"gage": g, "cso_open": sum(1 for e in d["cso"]["events"] if e.get("open")), "lock_open": d["lock"]["open_now"], "reversal": d["reversal"]}))
        return
    d = build(a.out, a.cache, skip_buildings=a.no_buildings)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in d.items() if k in ("polys", "quays", "bridges", "buildings", "level", "took_s")}, default=str))
    print({k: v["depth"] for k, v in d["bed"].items()}, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
