"""Offline tests for the learned wave correction and the forecast scorecard (synthetic data only)."""
import datetime as dt
import gzip
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import learn  # noqa: E402
from app import sup_waves as sw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sup import NOW, make_get  # noqa: E402

UTC = dt.timezone.utc


def synth_pairs(years=(2019, 2020, 2021, 2022), seed=1):
    """Model 40% too high in NE winds, right elsewhere, with noise."""
    rnd = random.Random(seed)
    out = []
    for y in years:
        t = dt.datetime(y, 5, 1, tzinfo=UTC)
        for k in range(3000):
            wdir = rnd.choice([30, 45, 60, 180, 200, 270, 300])
            u = rnd.uniform(3, 13)
            pred = 0.1 + 0.12 * u
            truth = pred / 1.4 if 20 <= wdir <= 70 else pred
            obs = truth * math.exp(rnd.gauss(0, 0.15))
            out.append({"t": (t + dt.timedelta(hours=k)).isoformat(), "buoy": "45198" if k % 2 else "45174", "obs_hs": obs, "obs_tp": 4.0,
                        "obs_dir": None, "pred_hs": pred, "pred_tp": 4.0, "pred_dir": None, "u": u, "wdir": wdir, "swell_frac": 0.0, "wind_src": "x"})
    return out


def test_fit_recovers_direction_bias():
    c = learn.fit_pairs(synth_pairs())
    ne = learn.factor(c, 45, 8.0, m=0.1 + 0.12 * 8.0)
    sw_ = learn.factor(c, 200, 8.0, m=0.1 + 0.12 * 8.0)
    assert 0.6 < ne < 0.85, ne        # ~1/1.4
    assert 0.9 < sw_ < 1.12, sw_
    assert learn.factor(None, 45, 8.0, m=1.0) == 1.0
    assert learn.factor(c, 45, 8.0, m=0.01) == 1.0        # no model sea to scale
    lo, hi = c["hs"]["ratio_clip"]
    assert lo <= learn.factor(c, 45, 30.0, m=0.06) <= hi


def test_cross_validation_reports_held_out_gain():
    cv = learn.cross_validate(synth_pairs())
    assert cv["years_scored"] == 4
    assert cv["overall"]["learned"]["rmse_m"] < cv["overall"]["raw"]["rmse_m"]
    assert cv["improves"]


def test_no_gain_when_model_is_right():
    rnd = random.Random(3)
    ps = synth_pairs()
    for p in ps:
        p["obs_hs"] = p["pred_hs"] * math.exp(rnd.gauss(0, 0.15))
    cv = learn.cross_validate(ps)
    assert not cv["improves"]


def ndbc_year(station, year, wspd=8.0, wdir=30, wvht=0.9):
    head = "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft\n"
    body = []
    t = dt.datetime(year, 6, 1, tzinfo=UTC)
    for k in range(24 * 10):
        tt = t + dt.timedelta(hours=k)
        hs = f"{wvht:.2f}" if station.startswith("45") else "99.00"
        body.append(f"{tt:%Y %m %d %H} 00 {wdir:3d} {wspd:4.1f} {wspd*1.3:4.1f} {hs} 4.00 3.50 {wdir:3d} 1015.0 15.0 14.0 999.0 99.0 99.00")
    return gzip.compress((head + "\n".join(body) + "\n").encode())


def test_backfill_and_fit_from_archive_files(tmp_path):
    def get(url, timeout=60):
        for st in ("45174", "45198", "chii2"):
            for y in (2021, 2022):
                if f"/{st}h{y}.txt.gz" in url:
                    return ndbc_year(st.upper(), y)
        raise OSError("404 " + url)
    s = learn.backfill(str(tmp_path), [2021, 2022], get=get, now=dt.datetime(2026, 10, 1, tzinfo=UTC))
    assert s["45174"]["2021"] > 200 and s["45198"]["2022"] > 200
    pairs = learn.read_pairs(sorted((tmp_path / "backfill").glob("*.csv.gz")))
    assert all(p["wind_src"] in ("45174", "45198") for p in pairs)   # each buoy's own anemometer first
    c = learn.fit_dir(str(tmp_path))
    assert (tmp_path / "correction.json").exists() and len(c["hs"]["coef"]) == len(learn.FEATURES)
    assert c["left_out_hours"] > 0 and c["hours_by_buoy"]["45174"] == 0      # 45174 before 2025: not fitted
    assert c["cv"]["years_scored"] == 2


def test_sup_build_adds_learned_hours_and_buoy_forecasts(tmp_path, monkeypatch):
    c = learn.fit_pairs(synth_pairs())
    p = tmp_path / "correction.json"
    p.write_text(json.dumps(c))
    monkeypatch.setenv("LEARN_PATH", str(p))
    data = sw.build(now=NOW, hours=12, get=make_get(8.0, 40))          # NE wind: the learned model should cut the waves
    h = data["spots"][0]["hourly"][3]
    assert h["learned"]["factor"] < 0.9
    assert h["learned"]["hs_offshore"] < h["hs_offshore"] * 1.01 or data["learned"]["calibration"] > data["calibration"]
    # the spot's factor comes from its nearer buoy's sea (Montrose and North Ave: Chicago buoy 45198)
    assert sw.nearest_buoy(41.9145, -87.6235) == "45198" and sw.nearest_buoy(42.13, -87.66) == "45174"
    bf = data["buoy_fc"]
    assert set(bf) == {"45174", "45198"} and len(bf["45174"]["raw"]) == 13
    assert all(a < b for a, b in zip(bf["45198"]["learned"][2:], bf["45198"]["raw"][2:]) if b > 0.2)
    # the buoy-nudged forecasts agree with the buoy now (both start from the 1.1 m it reports)
    assert abs(bf["45174"]["product"][0] - 1.1) < 0.15 and abs(bf["45174"]["product_learned"][0] - 1.1) < 0.15


def test_sup_build_without_correction_has_no_learned_fields(monkeypatch):
    monkeypatch.delenv("LEARN_PATH", raising=False)
    data = sw.build(now=NOW, hours=6, get=make_get(8.0, 40))
    assert "learned" not in data["spots"][0]["hourly"][0] and data["learned"] is None
    assert "learned" not in data["buoy_fc"]["45198"] and len(data["buoy_fc"]["45198"]["raw"]) == 7


def test_log_and_score_by_lead(tmp_path):
    t0 = dt.datetime(2026, 10, 20, 12, tzinfo=UTC)
    glwu = {"cycle": "2026102012", "points": {"45198": {"t": [(t0 + dt.timedelta(hours=k)).isoformat() for k in range(60)], "hs": [0.82] * 60}}}
    for i in range(3):                                   # three runs, an hour apart
        ti = (t0 + dt.timedelta(hours=i)).isoformat()
        sup = {"start": ti, "generated": ti, "learned": {"x": 1},
               "buoy_fc": {"45198": {"t0": ti, "raw": [1.0] * 49, "learned": [0.8] * 49, "product": [0.9] * 49, "nws": [None] * 49}}}
        path = learn.log_forecast(sup, str(tmp_path), glwu)
    assert path.endswith("forecasts-2026-10.jsonl")
    obs = {"45198": {t0 + dt.timedelta(hours=k): 0.8 for k in range(60)}, "45174": {}}
    card = learn.score(str(tmp_path), obs_hourly=obs, now=t0 + dt.timedelta(days=3))
    st = card["stats"]
    assert st["learned"][0]["rmse_m"] == 0.0 and abs(st["raw"][5]["bias_m"] - 0.2) < 1e-9
    assert abs(st["glwu"][2]["bias_m"] - 0.02) < 1e-6
    assert "nws" not in card["methods"] and card["forecasts"] == 3
    assert learn.summarize(None, card)["scorecard"]["forecasts"] == 3


def test_glwu_index_and_ranges():
    from app import glwu
    listing = '<a href="glwu.grlc_2p5km.t01z.grib2">x</a> <a href="glwu.grlc_2p5km.t01z.grib2.idx">x</a> <a href="glwu.grlc_2p5km.t07z.grib2">'
    assert glwu.pick_file(listing) == ("glwu.grlc_2p5km.t07z.grib2", 7)
    idx = "1:0:d=2026102401:HTSGW:surface:anl:\n2:100:d=2026102401:PERPW:surface:anl:\n3:180:d=2026102401:WIND:surface:anl:\n" \
          "4:260:d=2026102401:DIRPW:surface:anl:\n5:300:d=2026102401:HTSGW:surface:1 hour fcst:\n6:400:d=2026102401:HTSGW:surface:49 hour fcst:\n"
    rows = glwu.parse_idx(idx, size=500)
    assert rows[0] == (0, 99, "HTSGW", 0) and rows[-1] == (400, 499, "HTSGW", 49)
    recs, merged = glwu.wanted_ranges(rows)
    assert [r[2:] for r in recs] == [("HTSGW", 0), ("PERPW", 0), ("DIRPW", 0), ("HTSGW", 1)]
    assert [(m[0], m[1]) for m in merged] == [(0, 179), (260, 399)]      # 0-179 and 260-399, WIND and the 49 h record skipped
    i, d = glwu.nearest([41.9, 41.89, 42.0], [272.44, 272.437, 272.3], [9999, 0.5, 0.3], 9999, 41.892, -87.563)
    assert i == 1 and d < 1
