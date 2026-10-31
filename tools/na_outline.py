"""
North America's outline for the jet stream map (the Radar map card): coastlines, the Great Lakes, country and
state/province borders from Natural Earth 1:50m (public domain), cropped to the jet box, simplified and rounded.

    python tools/na_outline.py --out app/static/na_outline.json
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import jet  # noqa: E402

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LAYERS = {"coast": "ne_50m_coastline.geojson", "lakes": "ne_50m_lakes.geojson",
          "countries": "ne_50m_admin_0_boundary_lines_land.geojson", "states": "ne_50m_admin_1_states_provinces_lines.geojson"}
TOL = {"coast": 0.08, "lakes": 0.05, "countries": 0.1, "states": 0.15}


def lines_of(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "LineString":
        return [c]
    if t in ("MultiLineString", "Polygon"):
        return list(c)
    if t == "MultiPolygon":
        return [r for p in c for r in p]
    return []


def dp(pts, tol):
    """Douglas-Peucker in degrees."""
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0; n = (dx * dx + dy * dy) ** 0.5 or 1e-12
    k, dmax = 0, -1.0
    for i in range(1, len(pts) - 1):
        d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / n
        if d > dmax:
            k, dmax = i, d
    if dmax <= tol:
        return [pts[0], pts[-1]]
    return dp(pts[: k + 1], tol)[:-1] + dp(pts[k:], tol)


def crop(line, box):
    s, w, n, e = box
    out, cur = [], []
    for x, y in line:
        if w - 2 <= x <= e + 2 and s - 2 <= y <= n + 2:
            cur.append((x, y))
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"source": "Natural Earth 1:50m (public domain)", "box": jet.BOX}
    for k, f in LAYERS.items():
        g = json.load(urllib.request.urlopen(NE + f, timeout=120))
        keep = []
        for ft in g["features"]:
            p = ft.get("properties") or {}
            if k == "states" and (p.get("adm0_a3") or p.get("ADM0_A3")) not in ("USA", "CAN", "MEX"):
                continue
            for ln in lines_of(ft["geometry"]):
                for seg in crop(ln, jet.BOX):
                    s = dp(seg, TOL[k])
                    if len(s) >= 2:
                        keep.append([v for x, y in s for v in (round(x, 2), round(y, 2))])
        res[k] = keep
        print(k, len(keep), "lines", sum(len(x) for x in keep) // 2, "points")
    json.dump(res, open(a.out, "w"), separators=(",", ":"))
    print(os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
