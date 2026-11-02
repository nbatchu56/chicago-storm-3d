"""The Chicago River build: clipping, rasterizing, the edge tests, the bed grid, bridges and clearances, the datum chain (no network)."""
import json
import math

import numpy as np

from app import lakefront as lf
from app import river


def test_datums():
    # Chicago City Datum 0 = 579.19 ft NAVD88 = 176.54 m; in the page's IGLD85 frame 176.40 m (12 months of 2003 lake readings)
    assert abs(579.19 * 0.3048 + river.NAVD_TO_IGLD - river.CCD_IGLD) < 0.01
    # the Aug 2022 survey's water surface (175.9 m NAVD88) lands at the river's usual stage (CCD -2 ft) in the page's frame
    assert abs((175.9 + river.NAVD_TO_IGLD) - (river.CCD_IGLD - 2.0 * 0.3048)) < 0.05


def test_clip_polygon_and_rasterize():
    ring = [[0, 0], [100, 0], [100, 50], [0, 50]]
    c = river.clip_polygon(ring, 20, -10, 60, 30)
    xs = [p[0] for p in c]; ys = [p[1] for p in c]
    assert min(xs) == 20 and max(xs) == 60 and min(ys) == 0 and max(ys) == 30 and len(c) == 4
    assert river.clip_polygon(ring, 200, 200, 300, 300) == []
    g = {"x0": 0.0, "y0": 0.0, "dx": 2.0, "nx": 60, "ny": 30}
    m = river.rasterize([ring, [[40, 10], [60, 10], [60, 30], [40, 30]]], g)       # a hole (even-odd)
    assert m[5, 10] and not m[10, 25] and m[20, 10] is np.False_ or not m[27, 10]
    assert abs(m.sum() - (100 * 50 - 20 * 20) / 4) < 60


def test_edges_inside_and_distance():
    polys = [{"kind": "main", "rings": [[[0, 0], [1000, 0], [1000, 60], [0, 60]], [[400, 20], [500, 20], [500, 40], [400, 40]]]}]
    E = river.Edges(polys)
    ins = E.inside([[10, 30], [450, 30], [450, 50], [-5, 30]])
    assert list(ins) == [True, False, True, False]
    assert abs(E.dist(500, 70) - 10) < 1e-9 and E.dist(500, 200) is None and E.dist(500, 50, 5) is None
    assert river.nearest_water(500, 70, polys) == E.dist(500, 70)


def test_bed_grid_fills_small_holes_only():
    g = {"x0": 0.0, "y0": 0.0, "dx": 2.0, "nx": 20, "ny": 10}
    xs, ys, zs = [], [], []
    for i in range(20):
        for j in range(10):
            if 8 <= i <= 9 and j == 5:
                continue                                          # a 2-cell hole
            if i >= 15:
                continue                                          # no survey east of x = 30
            xs.append(i * 2 + 1); ys.append(j * 2 + 1); zs.append(170 + i * 0.1)
    z, inside = river.bed_grid(g, np.array(xs), np.array(ys), np.array(zs), max_gap=3)
    assert inside[5, 8] and inside[5, 9] and abs(z[5, 8] - 170.7) < 0.2
    assert inside[5, 16] and inside[5, 17] and not inside[5, 19]      # filled 3 cells out, no further
    pk = river.pack(g, z, inside); z2, in2 = river.unpack(pk)
    assert (in2 == inside).all() and np.nanmax(np.abs(z2[in2] - z[inside])) < 0.006 and pk["cells"] == int(inside.sum())


def test_known_clearance_and_bridge_kind():
    assert abs(river.known_clearance("North Michigan Avenue", "road", {"bridge:movable": "bascule"}) - 17 * 0.3048) < 1e-9
    assert abs(river.known_clearance("", "road", {"bridge:movable": "bascule"}) - 18 * 0.3048) < 1e-9
    assert river.known_clearance("", "rail", {}) > river.known_clearance("", "rail", {"bridge:movable": "bascule"})
    assert river.bridge_kind({"railway": "rail", "bridge": "yes"}) == "rail" and river.bridge_kind({"highway": "footway"}) == "foot"
    assert river.way_width({"lanes": "4"}, "road") == 4 * 3.4 + 4.0 and river.way_width({}, "rail") == 9.0


class FlatDem:
    def __init__(self, h):
        self.h = h

    def at(self, x, y):
        return self.h

    def robust(self, pts, lo=10, hi=90):
        return self.h


def _way(wid, pts_ll, tags):
    return {"type": "way", "id": wid, "tags": tags, "geometry": [{"lat": la, "lon": lo} for la, lo in pts_ll]}


def test_bridges_merge_sidewalks_and_use_lidar_when_plausible():
    # a river 60 m wide running east-west at lat 41.8880, from lon -87.63 to -87.62
    la0, la1 = 41.8880 - 30 / lf.KY, 41.8880 + 30 / lf.KY
    polys = [{"kind": "main", "rings": [[list(lf.xy(la0, -87.63)), list(lf.xy(la0, -87.62)), list(lf.xy(la1, -87.62)), list(lf.xy(la1, -87.63))]]}]
    lon = -87.625; road = [(41.8870, lon), (41.8890, lon)]
    side = [(41.8870, lon + 8 / lf.KX), (41.8890, lon + 8 / lf.KX)]
    far = [(41.8870, -87.64), (41.8890, -87.64)]                  # no water there
    js = {"elements": [_way(1, road, {"highway": "primary", "bridge": "movable", "bridge:movable": "bascule", "name": "West Test Street", "lanes": "4"}),
                       _way(2, side, {"highway": "footway", "bridge": "yes"}), _way(3, far, {"highway": "primary", "bridge": "yes"})]}
    level = 175.8
    B = river.bridges(js, polys, FlatDem(level + 5.0), level)
    assert len(B) == 1 and B[0]["name"] == "West Test Street" and B[0]["ways"] == 2 and B[0]["kind"] == "road"
    assert B[0]["deck_from"] == "lidar" and abs(B[0]["clearance"] - 5.0) < 1e-6 and 55 < B[0]["span"] < 66
    # the bare-earth lidar reads the ground under an elevated deck: implausible, so the typical closed clearance stands in
    B = river.bridges(js, polys, FlatDem(level + 0.4), level)
    assert B[0]["deck_from"] == "nominal" and abs(B[0]["clearance"] - (river.LWD + 18 * 0.3048 - level)) < 0.01   # the typical, quoted at LWD (rounded to cm)


def test_quays_near_the_water_only():
    la = 41.8880
    polys = [{"kind": "main", "rings": [[list(lf.xy(la - 30 / lf.KY, -87.63)), list(lf.xy(la - 30 / lf.KY, -87.62)), list(lf.xy(la + 30 / lf.KY, -87.62)), list(lf.xy(la + 30 / lf.KY, -87.63))]]}]
    near = [(la + 34 / lf.KY, -87.63), (la + 34 / lf.KY, -87.62)]
    far = [(la + 300 / lf.KY, -87.63), (la + 300 / lf.KY, -87.62)]
    js = {"elements": [_way(1, near, {"man_made": "quay"}), _way(2, far, {"man_made": "quay"}), _way(3, near, {"highway": "footway", "name": "Chicago Riverwalk"})]}
    Q = river.quays(js, polys, FlatDem(177.3), 175.8)
    assert len(Q) == 2 and {q["kind"] for q in Q} == {"quay", "riverwalk"} and all(abs(q["above"] - 1.5) < 1e-6 for q in Q)


def test_gage_level_parse():
    js = {"value": {"timeSeries": [
        {"variable": {"variableCode": [{"value": "00065"}]}, "values": [{"value": [{"value": "-2.10", "dateTime": "2026-11-01T12:00:00.000-06:00"}]}]},
        {"variable": {"variableCode": [{"value": "72254"}]}, "values": [{"value": [{"value": "-0.11", "dateTime": "2026-11-01T12:00:00.000-06:00"}]}]}]}}
    g = river.gage_level(get=lambda u, timeout=0: json.dumps(js).encode())
    assert abs(g["level_m"] - (river.CCD_IGLD - 2.1 * 0.3048)) < 1e-3 and g["velocity_fps"] == -0.11


def test_simplify_keeps_corners():
    ring = [[0, 0], [50, 0.3], [100, 0], [100, 50], [0, 50]]
    s = river.simplify(ring, 1.0)
    assert len(s) == 4 and [0, 0] in s and [100, 50] in s
