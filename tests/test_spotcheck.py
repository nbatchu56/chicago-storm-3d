"""Beach forecasts logged by each snapshot, and photo spot checks matched against them (synthetic data only)."""
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from app import learn  # noqa: E402
import spotcheck  # noqa: E402

UTC = dt.timezone.utc


def sup_at(start, rating):
    t0 = dt.datetime.fromisoformat(start)
    hourly = [{"t": (t0 + dt.timedelta(hours=k)).isoformat(), "wind": 3.0, "gust": 4.0, "dir": 200, "offshore": True,
               "hs_offshore": 0.1, "hs_paddle": 0.02, "tp": 0.5, "rating": rating, "sky": 20, "chop": 0.0,
               "learned": {"rating": rating, "hs_paddle": 0.03}} for k in range(20)]
    return {"start": start, "generated": start, "spots": [{"id": "north-ave", "lat": 41.9145, "lon": -87.6235, "hourly": hourly}]}


def test_log_spots_keeps_6_hours(tmp_path):
    path = learn.log_spots(sup_at("2026-10-28T10:00:00+00:00", "glassy"), str(tmp_path))
    assert path.endswith("spots-2026-10.jsonl")
    e = json.loads(open(path).read().splitlines()[0])
    rows = e["spots"]["north-ave"]["hourly"]
    assert len(rows) == 6 and "chop" not in rows[0]
    assert rows[0]["rating"] == "glassy" and rows[0]["learned_rating"] == "glassy"
    assert learn.log_spots({"start": "2026-10-28T10:00:00+00:00", "spots": []}, str(tmp_path)) is None


def test_pick_prefers_newest_forecast_before_the_photo(tmp_path):
    for start, rating in (("2026-10-28T04:00:00+00:00", "choppy"), ("2026-10-28T10:00:00+00:00", "glassy"),
                          ("2026-10-28T13:00:00+00:00", "good")):
        learn.log_spots(sup_at(start, rating), str(tmp_path))
    entries = spotcheck.read_forecasts(str(tmp_path / "scorecard" / "spots-2026-10.jsonl"))
    t = dt.datetime(2026, 10, 28, 12, 38, tzinfo=UTC)
    f = spotcheck.pick(entries, "north-ave", t)
    assert f["rating"] == "glassy" and f["lead_h"] == 2.0 and "issued_after_photo" not in f
    assert spotcheck.pick(entries[2:], "north-ave", t) is None          # the 13:00 run doesn't cover 12:00
    assert spotcheck.pick(entries, "oak-st", t) is None
    f = spotcheck.pick(entries, "north-ave", t.replace(hour=18))
    assert f["rating"] == "good" and f["lead_h"] == 5.0


def test_verdict_scale():
    assert spotcheck.verdict("calm", "glassy") == "match"
    assert spotcheck.verdict("calm", "good") == "match"
    assert spotcheck.verdict("glass", "choppy") == "forecast rougher"
    assert spotcheck.verdict("rough", "good") == "forecast calmer"
    assert spotcheck.verdict("choppy", "offshore") == "match"
