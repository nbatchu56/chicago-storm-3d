"""The aerial imagery tiles: the grid, the export request and the reuse rule (no network)."""
import datetime as dt
import json
import urllib.parse as up

from app import lakefront as lf
from app import ortho


def test_tile_grid():
    s, w, n, e = ortho.tile_box(-1, 5)
    (x0, y0), (x1, y1) = lf.xy(s, w), lf.xy(n, e)
    assert abs(x0 + 1050) < 1 and abs(x1 + 50) < 1 and abs(y0 - 5250) < 1 and abs(y1 - 6250) < 1   # 1 km, on the grid
    ids = [ortho.tile_id(i, j) for i, j in ortho.TILES]
    assert len(set(ids)) == len(ids) and ortho.tile_id(-1, 5) == "tm1p5" and ortho.tile_id(0, -2) == "tp0m2"
    # the launches and harbours the page cares about are each inside a tile
    for lat, lon in [(41.9265, -87.6327), (41.9143, -87.6225), (41.8800, -87.6120), (41.9400, -87.6370), (41.9215, -87.6340)]:
        assert any(b[0] <= lat <= b[2] and b[1] <= lon <= b[3] for b in (ortho.tile_box(i, j) for i, j in ortho.TILES)), (lat, lon)


def test_export_size_and_url():
    box = ortho.tile_box(0, 3)
    w, h = ortho.size_for(box)
    assert w == ortho.TILE_PX and 2900 <= h <= 3200 and w <= ortho.MAX_PX[0] and h <= ortho.MAX_PX[1]   # square in degrees
    q = dict(up.parse_qsl(up.urlparse(ortho.export_url(box)).query))
    assert q["bboxSR"] == "4326" and q["imageSR"] == "4326" and q["format"] == "jpg" and q["size"] == f"{w},{h}" and q["bandIds"] == "0,1,2"
    assert q["bbox"] == f"{box[1]},{box[0]},{box[3]},{box[2]}" and "CookOrtho2025" in ortho.SERVICE


def test_reuse_rule(tmp_path):
    img = tmp_path / "prev.jpg"; img.write_bytes(b"x" * (ortho.MIN_BYTES + 1))
    now = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc)
    box = ortho.tile_box(-1, 5)
    fresh = ortho.meta_for("tm1p5", "data/ortho/tm1p5.jpg", box, (4096, 3053), box, fetched="2026-10-15T00:00:00+00:00")
    old = dict(fresh, fetched="2026-09-01T00:00:00+00:00")
    assert ortho.reusable(fresh, str(img), box, now=now)
    assert not ortho.reusable(old, str(img), box, now=now)
    assert not ortho.reusable(fresh, str(tmp_path / "missing.jpg"), box, now=now)
    assert not ortho.reusable(fresh, str(img), ortho.tile_box(-1, 4), now=now)                 # another tile's request
    assert not ortho.reusable(dict(fresh, source="Cook County GIS 2024 orthophoto"), str(img), box, now=now)   # last year's photo
    assert not ortho.reusable({k: v for k, v in fresh.items() if k != "extent"}, str(img), box, now=now)
    small = tmp_path / "small.jpg"; small.write_bytes(b"x" * 100)
    assert not ortho.reusable(fresh, str(small), box, now=now)


def _fake_get(fail_ids=()):
    calls = []

    def get(url, timeout):
        calls.append(url)
        if "f=json" in url:
            bbox = [float(v) for v in dict(up.parse_qsl(up.urlparse(url).query))["bbox"].split(",")]
            if any(ortho.tile_box(i, j)[1] == bbox[0] and ortho.tile_box(i, j)[0] == bbox[1] for i, j in fail_ids):
                raise OSError("server said no")
            return json.dumps({"href": f"https://x/{len(calls)}.jpg", "width": 4096, "height": 3053,
                               "extent": {"xmin": bbox[0], "ymin": bbox[1] - 1e-5, "xmax": bbox[2], "ymax": bbox[3] + 1e-5}}).encode(), "application/json"
        return b"j" * (ortho.MIN_BYTES + 5), "image/jpeg"
    return get, calls


def test_build_fetches_reuses_and_skips(tmp_path):
    tiles = [(-1, 4), (-1, 5), (0, 5)]
    get, calls = _fake_get(fail_ids=[(0, 5)])
    metas = ortho.build(str(tmp_path / "a"), tiles=tiles, get=get)
    assert [m["id"] for m in metas] == ["tm1p4", "tm1p5"] and metas[0]["src"] == "data/ortho/tm1p4.jpg"   # the failed tile is left out
    assert metas[1]["lat0"] == ortho.tile_box(-1, 5)[0] - 1e-5 and metas[1]["extent"] == "server"      # the server's extent
    assert (tmp_path / "a" / "tm1p4.jpg").exists()
    # the next snapshot reuses both and tries the missing one again
    get2, calls2 = _fake_get()
    metas2 = ortho.build(str(tmp_path / "b"), prev=metas, prev_dir=str(tmp_path / "a"), tiles=tiles, get=get2)
    assert [m["id"] for m in metas2] == ["tm1p4", "tm1p5", "tp0p5"] and len(calls2) == 2
    assert metas2[0]["fetched"] == metas[0]["fetched"] and (tmp_path / "b" / "tm1p5.jpg").exists()
