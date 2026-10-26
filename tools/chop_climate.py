"""Paddling-season wave climate offshore of the Chicago lakefront, for the chop trouble-spot map.

Reads the learn branch's hourly buoy backfill (backfill/pairs_<buoy>_<year>.csv.gz: obs_hs, obs_tp, obs_dir) and
counts daylight hours (6 AM - 8 PM Chicago) from May through October by offshore condition. The Chicago buoy (45198,
~9 m deep off the lakefront) is used where it reported; the Wilmette buoy (45174) fills the other hours.

    git show origin/learn:backfill/... | or: python tools/chop_climate.py --backfill learn/backfill --out tools/chop_climate.json

Output: {"hours": N, "seasons": "2015-2026", "bins": {"dir|tp|hs": hours}} with dir (from) in 5 deg, tp in 0.5 s and
hs in 0.05 m steps. Calm hours (no reported waves) count toward the total as hs 0.
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import gzip
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", required=True, help="folder with pairs_<buoy>_<year>.csv.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    years = set()
    for buoy in ("45174", "45198"):          # 45198 second: it overwrites 45174 where both reported
        for p in sorted(glob.glob(os.path.join(a.backfill, f"pairs_{buoy}_*.csv.gz"))):
            with gzip.open(p, "rt") as fh:
                for r in csv.DictReader(fh):
                    t = dt.datetime.fromisoformat(r["t"])
                    local_h = (t.hour - 5) % 24          # CDT; the season is all daylight time
                    if not (5 <= t.month <= 10 and 6 <= local_h < 20):
                        continue
                    try:
                        hs = float(r["obs_hs"])
                    except ValueError:
                        continue
                    tp = float(r["obs_tp"]) if r["obs_tp"] not in ("", "nan") else 3.0
                    d = float(r["obs_dir"]) if r["obs_dir"] not in ("", "nan") else float(r["wdir"] or 0)
                    rows[r["t"]] = (round(d / 5) * 5 % 360, round(tp * 2) / 2, round(hs * 20) / 20)
                    years.add(t.year)
    bins = collections.Counter(f"{d}|{tp}|{hs}" for d, tp, hs in rows.values())
    out = {"hours": len(rows), "seasons": f"{min(years)}-{max(years)}", "months": "May-Oct", "local_hours": "6 AM-8 PM",
           "buoys": "45198 where it reported, else 45174", "bins": dict(sorted(bins.items()))}
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{len(rows)} hours, {len(bins)} bins, seasons {out['seasons']}")


if __name__ == "__main__":
    main()
