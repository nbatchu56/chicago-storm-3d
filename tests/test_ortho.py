"""The aerial imagery export: its request and the reuse rule (no network)."""
import datetime as dt
import urllib.parse as up

from app import ortho


def test_export_size_and_url():
    w, h = ortho.size_for()
    assert w <= ortho.MAX_PX[0] and h <= ortho.MAX_PX[1] and 2000 <= w <= 2200 and 3900 <= h <= 4100   # ~0.55 m over 1.16 x 2.21 km
    q = dict(up.parse_qsl(up.urlparse(ortho.export_url()).query))
    assert q["bboxSR"] == "4326" and q["imageSR"] == "4326" and q["format"] == "jpg" and q["size"] == f"{w},{h}"
    assert q["bbox"] == "-87.64,41.915,-87.626,41.935"


def test_reuse_rule(tmp_path):
    img = tmp_path / "prev.jpg"; img.write_bytes(b"x" * (ortho.MIN_BYTES + 1))
    now = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc)
    ext = (41.91166, -87.64, 41.93834, -87.626)
    fresh = [ortho.meta_for("data/ortho/diversey.jpg", ext, (2110, 4021), fetched="2026-10-15T00:00:00+00:00")]
    old = [ortho.meta_for("data/ortho/diversey.jpg", ext, (2110, 4021), fetched="2026-09-01T00:00:00+00:00")]
    assert ortho.reusable(fresh, str(img), now=now)
    assert not ortho.reusable(old, str(img), now=now)
    assert not ortho.reusable(fresh, str(tmp_path / "missing.jpg"), now=now)
    moved = [dict(fresh[0], box=[41.9, -87.64, 41.935, -87.626])]
    assert not ortho.reusable(moved, str(img), now=now)
    placed_by_request = [{k: v for k, v in fresh[0].items() if k != "extent"}]     # an older copy without the server's extent
    assert not ortho.reusable(placed_by_request, str(img), now=now)
    small = tmp_path / "small.jpg"; small.write_bytes(b"x" * 100)
    assert not ortho.reusable(fresh, str(small), now=now)


def test_fetch_uses_the_servers_extent(tmp_path):
    import json
    calls = []

    def get(url, timeout):
        calls.append(url)
        if "f=json" in url:
            return json.dumps({"href": "https://x/img.jpg", "width": 2110, "height": 4021,
                               "extent": {"xmin": -87.64, "ymin": 41.91166, "xmax": -87.626, "ymax": 41.93834}}).encode(), "application/json"
        return b"j" * (ortho.MIN_BYTES + 5), "image/jpeg"
    n, ext, size = ortho.fetch(str(tmp_path / "o.jpg"), get=get)
    assert ext == (41.91166, -87.64, 41.93834, -87.626) and size == (2110, 4021) and calls[1] == "https://x/img.jpg"
    m = ortho.meta_for("data/ortho/diversey.jpg", ext, size)
    assert m["lat0"] == 41.91166 and m["lat1"] == 41.93834 and m["extent"] == "server"
