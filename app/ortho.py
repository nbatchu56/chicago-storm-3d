"""
Aerial imagery of Diversey Harbor and the Lincoln Park lagoon for the page: Cook County's 2024 orthophoto.

Cook County GIS serves its county-wide orthophotos (6 in / 0.15 m, RGB+NIR) from an ArcGIS ImageServer without a key.
One export covers the harbour, the Fullerton channel and the South Lagoon (BOX) as a JPEG in plain lat/lon
(EPSG:4326, so the page can drape it with its linear lat/lon-to-metres mapping), about 0.55 m per pixel, ~2-3 MB.
The server keeps its pixels square in degrees and widens the box to fit the size asked for, so the export is made with
f=json and the extent it reports is what the page uses (the box asked for would misplace the photo by up to ~170 m).
The image hardly changes, so a previous snapshot's copy is reused for a month.

    python -m app.ortho --out data/ortho/diversey.jpg --meta ortho.json [--prev prev_ortho.json --prev-image prev.jpg]

ortho.json lists the images for tools/build_world.py (--ortho): [{src, lat0, lon0, lat1, lon1, w, h, source, year, fetched}].
"""

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import urllib.parse
import urllib.request

from . import lakefront as lf

log = logging.getLogger("klot.ortho")
UTC = dt.timezone.utc
SERVICE = "https://gis.cookcountyil.gov/imagery/rest/services/CookOrtho2024/ImageServer/exportImage"
SOURCE = "Cook County GIS 2024 orthophoto"
YEAR = 2024
BOX = (41.9150, -87.6400, 41.9350, -87.6260)   # S, W, N, E: Diversey Harbor, the channel, the South Lagoon, North Pond
MAX_PX = (15000, 4100)                          # the service's export limits (width, height)
TARGET_M = 0.55                                 # metres per pixel asked for (the limits allow no finer over this box)
REUSE_DAYS = 30
MIN_BYTES = 200_000                             # a real export of this box is a few MB; less is an error page


def size_for(box=BOX, target_m=TARGET_M):
    s, w, n, e = box
    (xa, ya), (xb, yb) = lf.xy(s, w), lf.xy(n, e)
    wpx, hpx = int(round((xb - xa) / target_m)), int(round((yb - ya) / target_m))
    k = max(wpx / MAX_PX[0], hpx / MAX_PX[1], 1.0)
    return int(wpx / k), int(hpx / k)


def export_url(box=BOX, size=None, quality=82):
    s, w, n, e = box
    wpx, hpx = size or size_for(box)
    q = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326, "size": f"{wpx},{hpx}", "format": "jpg",
         "compressionQuality": quality, "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    return SERVICE + "?" + urllib.parse.urlencode(q)


def meta_for(path, extent, size, box=BOX, fetched=None):
    """extent: (S, W, N, E) the image really covers, as the server reported it."""
    s, w, n, e = extent
    return {"src": path, "lat0": s, "lon0": w, "lat1": n, "lon1": e, "w": size[0], "h": size[1], "box": list(box), "extent": "server",
            "source": SOURCE, "year": YEAR, "fetched": fetched or dt.datetime.now(UTC).isoformat(timespec="seconds")}


def reusable(prev_meta, prev_image, box=BOX, now=None):
    """The previous snapshot's image is used again when it is the same request and source, under REUSE_DAYS old, and
    its extent came from the server (an older copy placed by the box asked for is fetched again)."""
    if not prev_meta or not prev_image or not os.path.exists(prev_image) or os.path.getsize(prev_image) < MIN_BYTES:
        return False
    m = prev_meta[0] if isinstance(prev_meta, list) else prev_meta
    try:
        age = (now or dt.datetime.now(UTC)) - dt.datetime.fromisoformat(m["fetched"])
    except (KeyError, ValueError, TypeError):
        return False
    same = m.get("box") == list(box) and m.get("extent") == "server" and m.get("source") == SOURCE
    return same and dt.timedelta(0) <= age < dt.timedelta(days=REUSE_DAYS)


def _get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": lf.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def fetch(out, box=BOX, timeout=300, get=_get):
    """Export, download, and return (bytes, the extent the image really covers as (S, W, N, E), (w, h))."""
    meta, _ = get(export_url(box).replace("f=image", "f=json"), timeout)
    m = json.loads(meta)
    ext = m["extent"]
    raw, ctype = get(m["href"], timeout)
    if len(raw) < MIN_BYTES or "image" not in ctype:
        raise RuntimeError(f"ortho export returned {len(raw)} bytes of {ctype}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(raw)
    return len(raw), (ext["ymin"], ext["xmin"], ext["ymax"], ext["xmax"]), (m["width"], m["height"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the JPEG, beside the page (e.g. out/data/ortho/diversey.jpg)")
    ap.add_argument("--meta", required=True, help="ortho.json for build_world --ortho")
    ap.add_argument("--rel", default="data/ortho/diversey.jpg", help="the image's path relative to the page")
    ap.add_argument("--prev", help="the previous snapshot's ortho.json")
    ap.add_argument("--prev-image", help="the previous snapshot's JPEG")
    a = ap.parse_args()
    prev = None
    if a.prev and os.path.exists(a.prev):
        try:
            prev = json.load(open(a.prev))
        except ValueError:
            prev = None
    if reusable(prev, a.prev_image):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        shutil.copyfile(a.prev_image, a.out)
        m = dict(prev[0] if isinstance(prev, list) else prev, src=a.rel)
        log.info("reused the previous ortho (%s)", m.get("fetched"))
    else:
        n, ext, size = fetch(a.out)
        m = meta_for(a.rel, ext, size)
        log.info("fetched %d KB from %s", n // 1000, SOURCE)
    with open(a.meta, "w") as fh:
        json.dump([m], fh)
    print(json.dumps(m))


if __name__ == "__main__":
    main()
