"""Start a fresh snapshot when the NWS issues a wind/wave alert for the Chicago lakefront.

Run every ~10 minutes by .github/workflows/alert-watch.yml. Standard library only (no installs).

Decision: dispatch snapshot.yml when the newest *relevant* active alert was sent after the current
`snapshot` branch commit, and no snapshot run has been started since that alert (queued, running or
finished). Re-issued or upgraded alerts (e.g. Small Craft Advisory -> Gale Warning) have a new `sent`
time and trigger again; cancellations and expiries don't.

Zones: LMZ741 / LMZ742 (nearshore Lake Michigan, Wilmette to Calumet Harbor), ILZ104 (Central Cook,
the forecast zone for the whole lakefront) and ILC031 (Cook County, for county/polygon warnings).
"""
import argparse
import datetime as dt
import json
import logging
import os
import urllib.request

log = logging.getLogger("alert_watch")

ZONES = ("LMZ741", "LMZ742", "ILZ104", "ILC031")
RELEVANT = {
    "Small Craft Advisory", "Gale Watch", "Gale Warning", "Storm Watch", "Storm Warning",
    "Hurricane Force Wind Watch", "Hurricane Force Wind Warning", "Special Marine Warning",
    "Beach Hazards Statement", "Lakeshore Flood Watch", "Lakeshore Flood Advisory",
    "Lakeshore Flood Warning", "Lakeshore Flood Statement",
    "Wind Advisory", "High Wind Watch", "High Wind Warning", "Severe Thunderstorm Warning",
}
UA = "chicago-storm-3d alert watch (github.com/nbatchu56/chicago-storm-3d)"
GH = "https://api.github.com"


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def relevant_alerts(geojson):
    out = []
    for f in geojson.get("features", []):
        p = f.get("properties", {})
        if p.get("event") in RELEVANT and p.get("status") == "Actual" and p.get("messageType") != "Cancel" and p.get("sent"):
            out.append({"event": p["event"], "sent": ts(p["sent"]), "id": p.get("id", "")})
    return sorted(out, key=lambda a: a["sent"])


def decide(alerts, snapshot_time, runs):
    """Return (dispatch?, reason). `runs` are snapshot.yml runs: dicts with status and created_at."""
    if not alerts:
        return False, "no relevant alerts"
    newest = alerts[-1]
    label = f"{newest['event']} sent {newest['sent']:%Y-%m-%d %H:%M %Z}"
    if snapshot_time is not None and snapshot_time >= newest["sent"]:
        return False, f"snapshot already newer than {label}"
    for r in runs:
        if r.get("status") != "completed":
            return False, f"snapshot run {r.get('id')} already {r.get('status')}"
        if ts(r["created_at"]) >= newest["sent"]:
            return False, f"snapshot run {r.get('id')} already started after {label}"
    return True, label


def _req(url, token=None, data=None, accept="application/json"):
    h = {"User-Agent": UA, "Accept": accept}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=h, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read().decode("utf-8", "replace")
    return json.loads(body) if body else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="decide but don't dispatch")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_TOKEN")

    alerts = relevant_alerts(_req(f"https://api.weather.gov/alerts/active?zone={','.join(ZONES)}", accept="application/geo+json"))
    for a in alerts:
        log.info("active: %s (sent %s)", a["event"], a["sent"].isoformat())
    try:
        snap = ts(_req(f"{GH}/repos/{repo}/branches/snapshot", token)["commit"]["commit"]["committer"]["date"])
    except Exception as e:  # no snapshot branch yet
        log.info("snapshot branch unavailable: %s", e)
        snap = None
    runs = _req(f"{GH}/repos/{repo}/actions/workflows/snapshot.yml/runs?per_page=10", token).get("workflow_runs", [])

    go, why = decide(alerts, snap, runs)
    log.info("%s: %s", "DISPATCH" if go else "skip", why)
    if go and not args.dry_run:
        _req(f"{GH}/repos/{repo}/actions/workflows/snapshot.yml/dispatches", token, data=json.dumps({"ref": "main"}).encode())
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(f"Started a snapshot for: {why}\n")


if __name__ == "__main__":
    main()
