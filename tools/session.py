"""Turn a paddle into a session file (sessions/<label>.json) the page learns from.

    python tools/session.py --gpx paddle.gpx --rating glass --notes "..." [--spot "41.9145,-87.6235,150,choppy,rebound off the hook"] \
        [--photo "7:10 AM looking east: ..."] [--title "..."] --out sessions/

Kept: the route every ~5 s (position, speed) and a GPS roughness index ("bump"); heart rate is never stored.
bump = how much the board speed jitters around its 30 s mean, over 20 s, relative to that mean. On flat water it stays
low (Oct 21 glass: median 0.05, 95th percentile ~0.17 on a watch logging every ~2 s); sustained chop knocks the speed
about. The page calibrates it against your calm and glass sessions and marks clearly bumpier stretches as chop found by GPS.
Ratings: glass, calm, light chop, choppy, rough, very rough. --spot: lat,lon,radius m,rating,note (repeatable).
"""
import argparse
import datetime as dt
import json
import math
import os
import re

import numpy as np

ORIGIN = (41.8820, -87.6278)
KX = 111.32 * math.cos(math.radians(ORIGIN[0])); KY = 110.57
RATINGS = ("glass", "calm", "light chop", "choppy", "rough", "very rough")


def parse_time(s):
    s = s.strip().replace("Z", "+0000")
    s = re.sub(r"([+-]\d\d):(\d\d)$", r"\1\2", s)
    for f in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError(s)


def read_gpx(path):
    txt = open(path).read()
    pts = []
    for m in re.finditer(r"<trkpt[^>]*lat=\"([-\d.]+)\"[^>]*lon=\"([-\d.]+)\"[^>]*>(.*?)</trkpt>", txt, re.S):
        body = m.group(3); t = re.search(r"<time>(.*?)</time>", body)
        if not t:
            continue
        sp = re.search(r"<(?:\w+:)?speed>([\d.]+)</", body)
        pts.append((parse_time(t.group(1)).timestamp(), float(m.group(1)), float(m.group(2)), float(sp.group(1)) if sp else None))
    pts.sort()
    t = np.array([p[0] for p in pts]); la = np.array([p[1] for p in pts]); lo = np.array([p[2] for p in pts])
    x = (lo - ORIGIN[1]) * KX * 1000; y = (la - ORIGIN[0]) * KY * 1000
    v = np.array([p[3] if p[3] is not None else np.nan for p in pts], float)
    if np.isnan(v).all():
        d = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3); v = np.concatenate([[d[0] if len(d) else 0], d])
    return t, x, y, np.nan_to_num(v)


def bump_index(t, v):
    """Speed jitter around the 30 s mean, over 20 s, relative to that mean (time windows: works at any logging rate)."""
    out = np.full(len(t), np.nan); mean30 = np.empty(len(t))
    for i in range(len(t)):
        w = np.abs(t - t[i]) <= 15; mean30[i] = v[w].mean()
    hp = v - mean30
    for i in range(len(t)):
        w = np.abs(t - t[i]) <= 10
        if w.sum() >= 4 and mean30[i] > 0.6:          # moving, with enough samples
            out[i] = hp[w].std() / max(mean30[i], 0.5)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpx", required=True); ap.add_argument("--rating", required=True, choices=RATINGS)
    ap.add_argument("--quality", default=""); ap.add_argument("--notes", default=""); ap.add_argument("--title")
    ap.add_argument("--spot", action="append", default=[]); ap.add_argument("--photo", action="append", default=[])
    ap.add_argument("--extra", help="JSON file merged into the session (archive readings, lessons)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t, x, y, v = read_gpx(a.gpx); b = bump_index(t, v)
    t0 = dt.datetime.fromtimestamp(t[0], dt.timezone.utc); label = f"paddle-{t0.date().isoformat()}"
    keep, last = [], -1e9
    for i in range(len(t)):
        if t[i] - last >= 5:
            last = t[i]; keep.append(i)
    pts = [[round(t[i] - t[0]), round(float(x[i]), 1), round(float(y[i]), 1), round(float(v[i]), 2), None,
            None if np.isnan(b[i]) else round(float(b[i]), 3)] for i in keep]
    spots = []
    for s in a.spot:
        la, lo, r, rating, *note = s.split(",")
        assert rating.strip() in RATINGS, rating
        spots.append({"lat": float(la), "lon": float(lo), "r_m": float(r), "rating": rating.strip(), "note": ",".join(note).strip()})
    iso = lambda s: dt.datetime.fromtimestamp(s, dt.timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: E731
    sess = {"label": label, "title": a.title or f"Your paddle, {t0.strftime('%b')} {t0.day} · {a.rating}",
            "window": {"start": iso(t[0]), "end": iso(t[-1])},
            "observed": {"rating": a.rating, "quality": a.quality, "by": "paddler", "notes": a.notes, "photos": a.photo, "spots": spots},
            "track": {"name": f"{t0.strftime('%b')} {t0.day} paddle", "start": iso(t[0]), "source": "GPX (route, speed and a GPS roughness index; heart rate not stored)",
                      "fields": ["t_s", "x_m", "y_m", "speed_ms", "unused", "bump"], "pts": pts}}
    if a.extra:
        for k, val in json.load(open(a.extra)).items():
            sess.setdefault(k, val)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, label + ".json")
    json.dump(sess, open(p, "w"), separators=(",", ":"))
    ok = b[~np.isnan(b)]
    print(f"{p}: {len(pts)} points, bump median {np.median(ok):.3f}, p95 {np.percentile(ok, 95):.3f}")


if __name__ == "__main__":
    main()
