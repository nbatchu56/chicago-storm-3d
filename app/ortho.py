"""
Aerial imagery of the lakefront for the page: Cook County's 2025 orthophoto, in 1 km tiles around the launches.

Cook County GIS serves its county-wide orthophotos (6 in / 0.15 m, RGB+NIR) from ArcGIS ImageServers without a key.
The page gets a grid of 1 km x 1 km tiles (metres from State & Madison, edges at x = -50 + 1000 i, y = 250 + 1000 j)
over Monroe Harbor and the shore from North Ave to Belmont (Diversey Harbor, the Fullerton channel, the lagoons),
each a JPEG in plain lat/lon (EPSG:4326, so the page can drape it with its linear lat/lon-to-metres mapping). The
server's pixels are square in degrees, so a tile 4096 px wide is ~0.24 m east-west and ~0.33 m north-south; a few MB.
The server widens the box to fit the size asked for, so each export is made with f=json and the extent it reports is
what the page uses. The photo hardly changes, so a previous snapshot's tile is reused for a month.

    python -m app.ortho --out-dir data/ortho --meta ortho.json [--prev prev_ortho.json --prev-dir prev/data/ortho]

ortho.json lists the tiles for tools/build_world.py (--ortho): [{id, src, lat0, lon0, lat1, lon1, w, h, box, source, year,
fetched}]. The page loads only the tiles near its view, and only while the Aerial overlay is on.
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
YEAR = 2025
SERVICE = f"https://gis.cookcountyil.gov/imagery/rest/services/CookOrtho{YEAR}/ImageServer/exportImage"
SOURCE = f"Cook County GIS {YEAR} orthophoto"
TILE_M = 1000.0
GRID0 = (-50.0, 250.0)                          # x, y (m) of the grid's corner: tile (i, j) spans x0 + 1000 i .. +1000
# (i, j): the shore from North Ave (y ~3.3 km) to the north end of Belmont Harbor (~8.2 km) with the lagoons west of it, and Monroe
# Harbor with the Grant Park shore; the open lake east of Diversey and Belmont is left out (the page draws water live)
TILES = [(-1, 3), (0, 3), (-1, 4), (0, 4), (-1, 5), (0, 5), (-1, 6), (-1, 7),
         (0, -2), (1, -2), (0, -1), (1, -1)]
MAX_PX = (15000, 4100)                          # the service's export limits (width, height)
TILE_PX = 4096                                  # the longer side (east-west): a texture size every GPU takes
REUSE_DAYS = 30
MIN_BYTES = 40_000                              # a real tile is a few MB (calm open water compresses to a few hundred KB)


def tile_id(i, j):
    return f"t{i:+d}{j:+d}".replace("+", "p").replace("-", "m")


def tile_box(i, j):
    """(S, W, N, E) of tile (i, j)."""
    x0, y0 = GRID0[0] + TILE_M * i, GRID0[1] + TILE_M * j
    w, s = lf.REF[1] + x0 / lf.KX, lf.REF[0] + y0 / lf.KY
    e, n = lf.REF[1] + (x0 + TILE_M) / lf.KX, lf.REF[0] + (y0 + TILE_M) / lf.KY
    return (round(s, 6), round(w, 6), round(n, 6), round(e, 6))


def size_for(box, px=TILE_PX):
    """Pixels square in degrees (as the server keeps them), the longer side px, within the export limits."""
    s, w, n, e = box
    d = max(e - w, n - s) / px
    wpx, hpx = int(round((e - w) / d)), int(round((n - s) / d))
    k = max(wpx / MAX_PX[0], hpx / MAX_PX[1], 1.0)
    return int(wpx / k), int(hpx / k)


def export_url(box, size=None, quality=80):
    s, w, n, e = box
    wpx, hpx = size or size_for(box)
    q = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326, "size": f"{wpx},{hpx}", "format": "jpg",
         "bandIds": "0,1,2", "compressionQuality": quality, "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    return SERVICE + "?" + urllib.parse.urlencode(q)


def meta_for(tid, path, extent, size, box, fetched=None):
    """extent: (S, W, N, E) the image really covers, as the server reported it."""
    s, w, n, e = extent
    return {"id": tid, "src": path, "lat0": s, "lon0": w, "lat1": n, "lon1": e, "w": size[0], "h": size[1], "box": list(box),
            "extent": "server", "source": SOURCE, "year": YEAR, "fetched": fetched or dt.datetime.now(UTC).isoformat(timespec="seconds")}


def reusable(prev_tile, prev_image, box, now=None):
    """A previous snapshot's tile is used again when it is the same request and source, under REUSE_DAYS old, and its
    extent came from the server."""
    if not prev_tile or not prev_image or not os.path.exists(prev_image) or os.path.getsize(prev_image) < MIN_BYTES:
        return False
    try:
        age = (now or dt.datetime.now(UTC)) - dt.datetime.fromisoformat(prev_tile["fetched"])
    except (KeyError, ValueError, TypeError):
        return False
    same = prev_tile.get("box") == list(box) and prev_tile.get("extent") == "server" and prev_tile.get("source") == SOURCE
    return same and dt.timedelta(0) <= age < dt.timedelta(days=REUSE_DAYS)


def _get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": lf.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def fetch(out, box, timeout=300, get=_get):
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


def build(out_dir, rel_dir="data/ortho", prev=None, prev_dir=None, tiles=TILES, get=_get, now=None):
    """Fetch (or reuse) every tile; returns the ortho.json list. A tile that fails is left out (the rest still go)."""
    prev_by_id = {t.get("id"): t for t in (prev or []) if isinstance(t, dict) and t.get("id")}
    metas = []
    for i, j in tiles:
        tid, box = tile_id(i, j), tile_box(i, j)
        name = f"{tid}.jpg"
        out, rel = os.path.join(out_dir, name), f"{rel_dir}/{name}"
        p = prev_by_id.get(tid)
        pimg = os.path.join(prev_dir, name) if prev_dir else None
        if reusable(p, pimg, box, now=now):
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(pimg, out)
            metas.append(dict(p, src=rel))
            log.info("%s: reused (%s)", tid, p.get("fetched"))
            continue
        try:
            n, ext, size = fetch(out, box, get=get)
        except Exception as e:                              # noqa: BLE001 - one bad tile shouldn't sink the others
            log.warning("%s: %s", tid, e)
            continue
        metas.append(meta_for(tid, rel, ext, size, box))
        log.info("%s: fetched %d KB", tid, n // 1000)
    return metas


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="where the tiles go, beside the page (e.g. out/data/ortho)")
    ap.add_argument("--meta", required=True, help="ortho.json for build_world --ortho")
    ap.add_argument("--rel-dir", default="data/ortho", help="the tiles' folder relative to the page")
    ap.add_argument("--prev", help="the previous snapshot's ortho.json")
    ap.add_argument("--prev-dir", help="the previous snapshot's tiles")
    a = ap.parse_args()
    prev = None
    if a.prev and os.path.exists(a.prev):
        try:
            prev = json.load(open(a.prev))
        except ValueError:
            prev = None
    metas = build(a.out_dir, a.rel_dir, prev if isinstance(prev, list) else None, a.prev_dir)
    with open(a.meta, "w") as fh:
        json.dump(metas, fh)
    print(f"{len(metas)} of {len(TILES)} tiles")


if __name__ == "__main__":
    main()
