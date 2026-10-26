"""Offline tests for the NWS alert watcher's dispatch decision (synthetic alerts and runs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import alert_watch as aw  # noqa: E402


def feat(event, sent, status="Actual", mtype="Alert"):
    return {"properties": {"event": event, "sent": sent, "status": status, "messageType": mtype, "id": event}}


def test_filters_to_wind_and_wave_alerts():
    gj = {"features": [feat("Heat Advisory", "2026-10-26T14:00:00-05:00"),
                       feat("Small Craft Advisory", "2026-10-26T09:09:00-05:00"),
                       feat("Gale Warning", "2026-10-26T10:00:00-05:00", status="Test"),
                       feat("Beach Hazards Statement", "2026-10-26T11:00:00-05:00", mtype="Cancel")]}
    assert [a["event"] for a in aw.relevant_alerts(gj)] == ["Small Craft Advisory"]


def test_decide():
    alerts = aw.relevant_alerts({"features": [feat("Gale Warning", "2026-10-26T15:00:00Z")]})
    old, new = aw.ts("2026-10-26T14:00:00Z"), aw.ts("2026-10-26T15:05:00Z")
    done_before = [{"id": 1, "status": "completed", "created_at": "2026-10-26T13:58:00Z"}]
    assert aw.decide([], old, [])[0] is False
    assert aw.decide(alerts, old, done_before)[0] is True
    assert aw.decide(alerts, None, [])[0] is True
    assert aw.decide(alerts, new, done_before)[0] is False                                     # already reflected
    assert aw.decide(alerts, old, [{"id": 2, "status": "in_progress", "created_at": "2026-10-26T13:00:00Z"}])[0] is False
    assert aw.decide(alerts, old, [{"id": 3, "status": "completed", "created_at": "2026-10-26T15:01:00Z"}])[0] is False
