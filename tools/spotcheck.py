"""Photo spot checks: a running log of what the lake looked like vs what the page forecast for that beach and hour.

Not training data (a handful of photos can't fit anything); it's a long-term check that the beach ratings match
what people actually see, especially on days the buoys miss. Each check is one line in spotchecks/log.jsonl, with
a small copy of the photo (location/camera metadata stripped) in spotchecks/photos/.

    python tools/spotcheck.py add --time 2026-10-28T07:38-05:00 --lat 41.9157 --lon -87.6229 --observed calm \\
        --waves "fine ripples, no whitecaps" --wind "light, glassy patches" --sky "mostly cloudy" --photo IMG_7169.jpg
    python tools/spotcheck.py report

The forecast attached is the newest one issued before the photo, from the `learn` branch's
scorecard/spots-YYYY-MM.jsonl (every snapshot logs the next 6 h at each beach), or from files passed with
--forecast (spots-*.jsonl, sup.json or a built world.html). If only a later run covers that hour, it's used and
flagged issued_after_photo. Observed ratings: glass, calm, light chop, choppy, rough, very rough.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "spotchecks", "log.jsonl")
UTC = dt.timezone.utc
OBSERVED = {"glass": 0, "calm": 0.5, "light chop": 1, "choppy": 2, "rough": 3, "very rough": 4}
FORECAST = {"glassy": 0, "good": 1, "choppy": 2, "offshore": 2, "rough": 3, "no-go": 4}


def parse_t(s):
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise SystemExit("--time needs a UTC offset, e.g. 2026-10-28T07:38-05:00")
    return t.astimezone(UTC)


def km(a, b):
    kx = 111.32 * math.cos(math.radians(a[0]))
    return math.hypot((a[1] - b[1]) * kx, (a[0] - b[0]) * 110.57)


def entries_from_sup(sup):
    """A sup.json-shaped dict -> one spots-log entry (same shape app.learn.log_spots writes)."""
    out ={"issued": sup["start"], "generated": sup.get("generated"), "spots": {}}
    for s in sup.get("spots") or []:
        rows = []
        for h in s.get("hourly") or []:
            r = {k: h[k] for k in ("t", "wind", "gust", "dir", "offshore", "hs_offshore", "hs_paddle", "tp", "rating", "sky") if h.get(k) is not None}
            lr = h.get("learned") or {}
            if lr.get("rating"):
                r["learned_rating"], r["learned_hs_paddle"] = lr["rating"], lr.get("hs_paddle")
            rows.append(r)
        out["spots"][s["id"]] = {"lat": s.get("lat"), "lon": s.get("lon"), "name": s.get("name"), "hourly": rows}
    return out


def read_forecasts(path):
    txt = open(path, encoding="utf-8").read()
    if path.endswith(".html"):
        m = re.search(r'<script[^>]*id="sup-data"[^>]*>(.*?)</script>', txt, re.S)
        return [entries_from_sup(json.loads(m.group(1)))] if m else []
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    return [entries_from_sup(json.loads(txt))]


def learn_branch_forecasts(t, ref):
    out = []
    months = {t.strftime("%Y-%m"), (t.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")}
    for m in sorted(months):
        try:
            txt = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:scorecard/spots-{m}.jsonl"], capture_output=True, text=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        out += [json.loads(line) for line in txt.splitlines() if line.strip()]
    return out


def pick(entries, spot, t):
    """Newest forecast issued at or before t whose hourly covers t's hour; else the earliest later one (flagged)."""
    hour = t.replace(minute=0, second=0, microsecond=0)
    hits = []
    for e in entries:
        s = (e.get("spots") or {}).get(spot)
        row = next((r for r in (s or {}).get("hourly", []) if parse_t(r["t"]) == hour), None)
        if row:
            hits.append((parse_t(e["issued"]), e, row))
    before = [h for h in hits if h[0] <= t]
    if before:
        iss, e, row = max(before, key=lambda h: h[0]); after = False
    elif hits:
        iss, e, row = min(hits, key=lambda h: h[0]); after = True
    else:
        return None
    f = {"issued": e["issued"], "generated": e.get("generated"), "hour": row["t"], "lead_h": round((hour - iss).total_seconds() / 3600, 1)}
    f.update({k: v for k, v in row.items() if k != "t"})
    if after:
        f["issued_after_photo"] = True
    return f


def verdict(observed, forecast_rating):
    if forecast_rating not in FORECAST:
        return None
    d = FORECAST[forecast_rating] - OBSERVED[observed]
    return "match" if abs(d) <= 0.5 else ("forecast rougher" if d > 0 else "forecast calmer")


def save_photo(src, dst, long_side=1600):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((long_side, long_side))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "JPEG", quality=82, optimize=True)      # no exif passed: GPS/camera tags are dropped


def add(a):
    t = parse_t(a.time)
    spots = {}
    entries = []
    for p in a.forecast:
        entries += read_forecasts(p)
    if not a.no_learn_branch:
        entries += learn_branch_forecasts(t, a.learn_ref)
    for e in entries:
        for sid, s in (e.get("spots") or {}).items():
            if s.get("lat") is not None:
                spots.setdefault(sid, (s["lat"], s["lon"]))
    spot = a.spot
    if not spot:
        if not spots:
            raise SystemExit("no forecasts found to pick the nearest beach from; pass --spot or --forecast")
        spot = min(spots, key=lambda s: km((a.lat, a.lon), spots[s]))
    local = parse_t(a.time).astimezone(dt.datetime.fromisoformat(a.time.replace("Z", "+00:00")).tzinfo)
    cid = f"{local.date().isoformat()}-{spot}-{local.strftime('%H%M')}"
    rec = {"id": cid, "t": t.isoformat().replace("+00:00", "Z"), "local": local.isoformat(timespec="minutes"),
           "lat": a.lat, "lon": a.lon, "spot": spot,
           "spot_km": round(km((a.lat, a.lon), spots[spot]), 2) if spot in spots and a.lat is not None else None,
           "by": a.by, "source": a.source,
           "observed": {"rating": a.observed, **{k: v for k, v in (("waves", a.waves), ("wind", a.wind), ("sky", a.sky), ("notes", a.notes)) if v}}}
    if a.hs_est is not None:
        rec["observed"]["hs_est_m"] = a.hs_est
    f = pick(entries, spot, t)
    rec["forecast"] = f
    rec["verdict"] = verdict(a.observed, f.get("rating")) if f else None
    if f and f.get("learned_rating"):
        rec["verdict_learned"] = verdict(a.observed, f["learned_rating"])
    if a.photo:
        rel = os.path.join("spotchecks", "photos", cid + ".jpg")
        save_photo(a.photo, os.path.join(ROOT, rel))
        rec["photo"] = rel
    log = a.log
    old = [json.loads(line) for line in open(log)] if os.path.exists(log) else []
    old = [r for r in old if r["id"] != cid]           # re-adding the same check replaces it
    old.append(rec)
    old.sort(key=lambda r: r["t"])
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        for r in old:
            fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    if not f:
        print("warning: no logged forecast covers that beach and hour; the check is kept without one")


def report(a):
    recs = [json.loads(line) for line in open(a.log)] if os.path.exists(a.log) else []
    if not recs:
        print("no spot checks yet"); return
    print(f"{'when (local)':17} {'beach':10} {'seen':11} {'forecast':9} {'lead':>5}  verdict")
    tally, by_spot = {}, {}
    for r in recs:
        f = r.get("forecast") or {}
        v = r.get("verdict") or "no forecast"
        lead = "" if f.get("lead_h") is None else "%.0fh" % f["lead_h"] + ("*" if f.get("issued_after_photo") else "")
        print(f"{r['local'][:16]:17} {r['spot']:10} {r['observed']['rating']:11} {f.get('rating', '-'):9} {lead:>5}  {v}")
        tally[v] = tally.get(v, 0) + 1
        by_spot.setdefault(r["spot"], {}).setdefault(v, 0); by_spot[r["spot"]][v] += 1
    print("\n" + ", ".join(f"{k}: {n}" for k, n in sorted(tally.items())) + f"  ({len(recs)} checks; * = forecast issued after the photo)")
    for s, d in sorted(by_spot.items()):
        print(f"  {s}: " + ", ".join(f"{k} {n}" for k, n in sorted(d.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--time", required=True, help="when the photo was taken, ISO with UTC offset")
    p.add_argument("--lat", type=float); p.add_argument("--lon", type=float)
    p.add_argument("--spot", help="beach id (montrose, north-ave, oak-st, ...); default: nearest to --lat/--lon")
    p.add_argument("--observed", required=True, choices=list(OBSERVED))
    p.add_argument("--waves"); p.add_argument("--wind"); p.add_argument("--sky"); p.add_argument("--notes")
    p.add_argument("--hs-est", type=float, help="rough wave height you'd guess from the photo, m")
    p.add_argument("--photo"); p.add_argument("--by", default="paddler"); p.add_argument("--source", default="photo")
    p.add_argument("--forecast", action="append", default=[], help="spots-*.jsonl, sup.json or world.html")
    p.add_argument("--learn-ref", default="origin/learn"); p.add_argument("--no-learn-branch", action="store_true")
    p.add_argument("--log", default=LOG)
    p = sub.add_parser("report"); p.add_argument("--log", default=LOG)
    a = ap.parse_args()
    if a.cmd == "add" and not a.spot and (a.lat is None or a.lon is None):
        ap.error("give --spot or --lat/--lon")
    (add if a.cmd == "add" else report)(a)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    main()
