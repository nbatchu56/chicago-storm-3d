"""
The Lincoln Park South Lagoon's 1000 m course line for the page: the lagoon's centreline from the south face of the
Fullerton Avenue bridge southward, from the 2023 survey's shoreline polygon (bathy.json, survey.polys "Diversey Harbor").

    python tools/lagoon_course.py --bathy bathy.json            # prints the LAGOON_COURSE constant (lat, lon every 50 m)

Rowing on the lagoon (Lincoln Park Boat Club's Chicago Sprints) is a 1,000 m course between the Fullerton bridge and the
zoo; the page draws the line with a mark every 100 m and a label every 250 m (app/static/world.html, LAGOON_COURSE).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import lakefront as lf  # noqa: E402

BRIDGE_Y = 4825.0          # the south face of the Fullerton bridge (local m north; the deck spans ~4825-4860)
BRIDGE_X = -390.0          # the channel's centre under it
LENGTH = 1000.0
STEP = 5.0                 # sampling along the lagoon (m)
EVERY = 50.0               # waypoints kept (m)


def spans_at(ring, y):
    """x intervals where the horizontal line at y is inside the ring (even-odd)."""
    pts = np.asarray(ring, float); x0, y0 = pts[:, 0], pts[:, 1]; x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    m = (y0 > y) != (y1 > y)
    xs = np.sort(x0[m] + (y - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m]))
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def centreline(ring, y_start=BRIDGE_Y, x_start=BRIDGE_X, step=STEP, length=LENGTH):
    """Midpoints of the lagoon's width every step m south of the bridge, following the span nearest the last point."""
    pts, x = [], x_start
    y = y_start
    while True:
        sp = spans_at(ring, y)
        if not sp:
            break
        a, b = min(sp, key=lambda s: abs((s[0] + s[1]) / 2 - x))
        if not (a - 15 <= x <= b + 15):          # the channel ended (the span found is another arm)
            break
        x = (a + b) / 2; pts.append((x, y))
        y -= step
    P = np.array(pts)
    if len(P) > 7:                               # a light smoothing of the midpoints (the banks are ragged at 2 m)
        k = np.ones(7) / 7; P[:, 0] = np.convolve(np.pad(P[:, 0], 3, mode="edge"), k, mode="valid")
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
    return P, d


def waypoints(P, d, every=EVERY, length=LENGTH):
    out = []
    for s in np.arange(0, min(length, d[-1]) + 1e-6, every):
        x, y = np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])
        out.append((round(s), round(lf.REF[0] + y / lf.KY, 6), round(lf.REF[1] + x / lf.KX, 6)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bathy", required=True)
    a = ap.parse_args()
    b = json.load(open(a.bathy))
    poly = next(p for p in b["survey"]["polys"] if p["name"].lower().startswith("diversey"))
    P, d = centreline(poly["rings"][0])
    wp = waypoints(P, d)
    print(f"// the lagoon's centreline south of the Fullerton bridge: {d[-1]:.0f} m to the end; {len(wp)} waypoints every {EVERY:.0f} m")
    print("const LAGOON_COURSE = [" + ", ".join(f"[{la}, {lo}]" for _, la, lo in wp) + "];")


if __name__ == "__main__":
    main()
