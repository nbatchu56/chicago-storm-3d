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
    fresh = [ortho.meta_for("data/ortho/diversey.jpg", fetched="2026-10-15T00:00:00+00:00")]
    old = [ortho.meta_for("data/ortho/diversey.jpg", fetched="2026-09-01T00:00:00+00:00")]
    assert ortho.reusable(fresh, str(img), now=now)
    assert not ortho.reusable(old, str(img), now=now)
    assert not ortho.reusable(fresh, str(tmp_path / "missing.jpg"), now=now)
    moved = [dict(fresh[0], lat0=41.9)]
    assert not ortho.reusable(moved, str(img), now=now)
    small = tmp_path / "small.jpg"; small.write_bytes(b"x" * 100)
    assert not ortho.reusable(fresh, str(small), now=now)
