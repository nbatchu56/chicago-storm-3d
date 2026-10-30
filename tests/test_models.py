"""Offline checks for the lake-bottom, SWAN, FUNWAVE and wind-correction pieces."""
import base64
import gzip
import json
import math

import numpy as np

from app import bathymetry as bm
from app import funwave_case as fc
from app import sup_waves as sw
from app import swan_cases as sc


def fake_bathy(level=176.5):
    """Plane beach: land west, deepening east to ~10 m, on the route and world grids."""
    out = {"level": {"m_igld85": level}, "models": {}}
    for name, bbox, d in (("world", None, 10.0), ("route", bm.MODEL_BBOX, 4.0)):
        g = bm.local_grid(bbox or bm.MODEL_BBOX, d)
        x = g["x0"] + np.arange(g["nx"]) * d
        shore = -400.0
        depth = np.clip((x - shore) * 0.01, -3, 10)
        z = level - np.tile(depth, (g["ny"], 1))
        src = np.ones_like(z, np.uint8)
        packed = bm.pack(g, z, src)
        if name == "world":
            out["world"] = packed
        else:
            out["models"][name] = packed
    return out


def test_pack_roundtrip_and_blend():
    lid = np.full((40, 50), np.nan, np.float32); lid[:, :25] = 172.0
    nc = np.full((40, 50), -5.0, np.float32)                     # NCEI style: depth below LWD
    z, src, info = bm.blend(lid, nc, 176.5)
    assert (src[:, :25] == 1).all() and (src[:, 25:] == 2).all()
    assert abs(z[0, 40] - (176.0 - 5.0 + info["ncei_offset_m"])) < 1e-3
    g = {"x0": 0, "y0": 0, "dx": 4, "nx": 50, "ny": 40}
    z2, s2 = bm.unpack(bm.pack(g, z, src))
    assert np.abs(z2 - z).max() < 0.006 and (s2 == src).all()


def test_swan_inputs(tmp_path):
    b = fake_bathy()
    p = tmp_path / "bathy.json"; p.write_text(json.dumps(b))
    lake = {"osm": [{"kind": "breakwater", "pts": [[0, 4000], [300, 4200]], "closed": False},
                    {"kind": "pier", "pts": [[0, 4500], [20, 4500]], "closed": False}]}   # short dock: ignored
    lp = tmp_path / "lake.json"; lp.write_text(json.dumps(lake))
    meta = sc.prepare(str(p), str(lp), "route", [45, 90], str(tmp_path / "runs"), periods=[4.0], heights=[1.0])
    assert len(meta["cases"]) == 2 and [o["kind"] for o in meta["obstacles"]] == ["breakwater"]
    txt = (tmp_path / "runs" / "route" / "d045_t4p0_h1p0" / "INPUT").read_text()
    assert "BOUNDSPEC SIDE E CCW CONSTANT PAR 1.00 4.00 45.0" in txt and "DIFFRACTION 1 0.2" in txt and "PROJECT 'chicago' 'c045'" in txt
    bottom = np.loadtxt(tmp_path / "runs" / "route" / "bottom.txt")
    assert bottom.shape == (meta["grid"]["ny"], meta["grid"]["nx"])
    assert bottom[:, -1].mean() > 8 and bottom[:, 0].mean() < 0                 # deep east, land west (depth below LWD)


def test_funwave_domain(tmp_path):
    p = tmp_path / "bathy.json"; p.write_text(json.dumps(fake_bathy()))
    meta = fc.prepare(str(p), "rough", str(tmp_path / "fw"))
    d = np.loadtxt(tmp_path / "fw" / "depth.txt")
    assert d.shape == (meta["n"], meta["m"])
    assert d[:, 0].mean() > d[:, -1].mean()                                     # model i=0 is the offshore (east) edge
    txt = (tmp_path / "fw" / "input.txt").read_text()
    assert "ThetaPeak = -45.0" in txt and "Mglob = %d" % meta["m"] in txt


def test_wind_corrections():
    assert abs(sw.u10(10.0, "45198") - 10.0 * (10 / 3.2) ** (1 / 7)) < 1e-9
    assert sw.u10(10.0, "CHII2") == 10.0
    assert sw.stability(12.0, 16.4) > 1.05 > 1.0 > sw.stability(20.0, 12.0)    # cold air over warm water grows waves faster
    assert sw.stability(None, 10) == 1.0


def test_funwave_fill_holes():
    D = np.full((60, 60), 4.5, np.float32); S = np.ones((60, 60), np.uint8)
    D[20:40, 20:40] = 1.6; S[20:40, 20:40] = 2                  # NCEI false shoal inside surveyed harbour water
    D[:, :3] = -1.0                                              # shore on the west edge
    F = fc.fill_holes(D, S)
    assert abs(F[30, 30] - 4.5) < 0.05 and (F[:, :3] == -1.0).all()
    S2 = S.copy(); S2[:, 3:8] = 2                                # a hole against the shore stays as it is
    D2 = D.copy(); D2[:, 3:8] = 1.0
    assert (fc.fill_holes(D2, S2)[:, 3:8] == 1.0).all()


def test_monroe_walls_vertical():
    (x0, y0), (x1, y1) = bm.lf.xy(41.8740, -87.6175), bm.lf.xy(41.8790, -87.6165)   # Grant Park seawall
    g = {"x0": x0 - 200, "y0": y0 - 200, "dx": 8.0, "nx": 80, "ny": 120}
    lake = {"osm": [{"kind": "breakwater", "pts": [[x0, y0], [x0, y1]], "closed": False},
                    {"kind": "breakwater", "pts": [[x0 + 100, y0], [x0 + 100, y1]], "closed": False}]}
    ob = sc.obstacles(lake, g, np.full((120, 80), 5.0))
    assert [o["refl"] for o in ob] == [sc.REFL["vertical"]] * 2
    far = {"x0": g["x0"], "y0": g["y0"] + 20000, "dx": 8.0, "nx": 80, "ny": 120}
    lake2 = {"osm": [{"kind": "breakwater", "pts": [[x0, y0 + 20000], [x0, y1 + 20000]], "closed": False}]}
    assert sc.obstacles(lake2, far, np.full((120, 80), 5.0))[0]["refl"] == sc.REFL["breakwater"]


def test_coastline_across_harbour_mouth_is_open():
    """OSM closes the coastline across a harbour mouth; with water along it and on both sides it is no wall"""
    g = {"x0": 0.0, "y0": 0.0, "dx": 8.0, "nx": 60, "ny": 60}
    depth = np.full((60, 60), 6.0); depth[:, :10] = -2.0             # land to the west, deep water elsewhere
    fine = ({"x0": 0.0, "y0": 0.0, "dx": 4.0, "nx": 120, "ny": 120}, np.kron(depth, np.ones((2, 2))))
    mouth = {"kind": "coastline", "pts": [[300.0, 100.0], [300.0, 300.0]]}            # across open water
    shore = {"kind": "coastline", "pts": [[78.0, 100.0], [78.0, 300.0]]}              # the real waterline
    ob = sc.obstacles({"osm": [mouth, shore]}, g, depth, fine)
    xs = sorted({round(p[0]) for o in ob for p in o["pts"]})
    assert ob and all(x < 100 for x in xs)


def test_swan_case_groups(tmp_path):
    p = tmp_path / "bathy.json"; p.write_text(json.dumps(fake_bathy()))
    meta = sc.prepare(str(p), None, "route", [0], str(tmp_path / "runs"), cases=sc.CASE_GROUPS["storm"])
    assert {(c["tp"], c["hs"]) for c in meta["cases"]} == set(sc.CASES[6:])
    assert set(sc.CASE_GROUPS["base"]) | set(sc.CASE_GROUPS["storm"]) == set(sc.CASES)
    txt = (tmp_path / "runs" / "route" / "d000_t8p5_h5p0" / "INPUT").read_text()
    assert "CIRCLE 36 0.06 1.2 28" in txt and "BOUNDSPEC SIDE N CCW CONSTANT PAR 5.00 8.50 0.0" in txt


def test_swan_collect_merge(tmp_path):
    from scipy.io import savemat
    g = {"x0": 0.0, "y0": 0.0, "dx": 10.0, "nx": 4, "ny": 3}

    def run(root, cases, hs_val):
        d = root / "lane"; d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"domain": "lane", "grid": g, "level_m_igld85": 176.5, "cases": cases}))
        (d / "obstacles.json").write_text("[]")
        for c in cases:
            (d / c["id"]).mkdir()
            savemat(d / c["id"] / "out.mat", {"Hsig": np.full((3, 4), hs_val * c["hs"]), "Dir": np.zeros((3, 4)), "Yp": np.arange(3)[:, None] * np.ones((1, 4))})
    a = [{"id": sc.case_id(0, tp, hs), "dir": 0, "tp": tp, "hs": hs} for tp, hs in sc.CASE_GROUPS["base"]]
    b = [{"id": sc.case_id(0, tp, hs), "dir": 0, "tp": tp, "hs": hs} for tp, hs in sc.CASE_GROUPS["storm"]]
    run(tmp_path / "r1", a, 0.5); old = sc.collect(str(tmp_path / "r1"), str(tmp_path / "s1.json"))
    run(tmp_path / "r2", b, 0.8); new = sc.collect(str(tmp_path / "r2"), str(tmp_path / "s2.json"), merge=old)
    cs = new["domains"]["lane"]["cases"]
    assert len(cs) == len(sc.CASES) and [c["id"] for c in cs] == sorted(c["id"] for c in cs)
    r = np.frombuffer(gzip.decompress(base64.b64decode(new["domains"]["lane"]["ratio"])), np.uint8).reshape(len(cs), 3, 4)
    for k, c in enumerate(cs):
        assert r[k].max() == (80 if (c["tp"], c["hs"]) in sc.CASE_GROUPS["storm"] else 50)


def test_levels_hourly():
    import datetime as dt
    import urllib.parse as up
    from app import levels as lv

    def get(url, timeout=90):
        q = dict(up.parse_qsl(up.urlparse(url).query))
        b = dt.datetime.strptime(q["begin_date"], "%Y%m%d %H:%M"); e = dt.datetime.strptime(q["end_date"], "%Y%m%d %H:%M")
        step = dt.timedelta(hours=1) if q.get("interval") == "h" else dt.timedelta(minutes=6)
        rows, t = [], b
        while t <= e:
            rows.append({"t": t.strftime("%Y-%m-%d %H:%M"), "v": "176.500" if step.seconds == 3600 else "176.700"}); t += step
        return json.dumps({"data": rows})
    d = lv.build(days=30, get=get)
    assert d["station"] == "9087044" and d["step_s"] == 3600
    assert d["cm"][0] == 50 and d["cm"][-1] == 70            # recent hours from the 6-minute series
    assert len(d["cm"]) == 30 * 24 + 1


# ---------------------------------------------------------------- the 2023 harbour survey (app.cookcounty)
from app import cookcounty as cook  # noqa: E402


def square_poly(x0, y0, x1, y1, hole=None):
    rings = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
    if hole:
        a, b, c, d = hole
        rings.append([[a, b], [c, b], [c, d], [a, d], [a, b]])
    return {"name": "Test Harbor", "acres": 1.0, "rings": rings}


def test_cookcounty_polygons_from_geojson():
    lat, lon = cook.lf.REF
    gj = {"features": [{"properties": {"Name": "DIVERSEY HARBOR", "Acres": 41.4},
                        "geometry": {"type": "Polygon", "coordinates": [[[lon, lat], [lon + 0.001, lat], [lon + 0.001, lat + 0.001], [lon, lat]]]}},
                       {"properties": {"Name": "NORTH POND"},
                        "geometry": {"type": "MultiPolygon", "coordinates": [[[[lon, lat], [lon + 0.002, lat], [lon, lat + 0.002], [lon, lat]]], [[[lon, lat], [lon - 0.002, lat], [lon, lat - 0.002], [lon, lat]]]]}}]}
    ps = cook.polygons_from_geojson(gj)
    assert [p["name"] for p in ps] == ["Diversey Harbor", "North Pond", "North Pond"]
    assert ps[0]["rings"][0][0] == [0.0, 0.0] and abs(ps[0]["rings"][0][1][0] - 0.001 * cook.lf.KX) < 0.2 and ps[0]["rings"][0][2][1] > 100


def test_cookcounty_rasterize_with_hole():
    g = {"x0": 0.0, "y0": 0.0, "dx": 2.0, "nx": 50, "ny": 40}
    inside = cook.rasterize([square_poly(10, 10, 60, 50, hole=(30, 20, 40, 30))], g)
    assert inside.shape == (40, 50)
    assert inside[10, 10] and inside[20, 25]            # (y=20, x=20) and (y=40, x=50): inside the ring
    assert not inside[12, 17] and not inside[13, 16]   # (y=24, x=34), (y=26, x=32): in the hole
    assert not inside[2, 2] and not inside[30, 45]     # outside
    assert 25 * 20 - 5 * 5 - 30 < inside.sum() < 25 * 20 - 5 * 5 + 30


def test_blend_survey_wins_and_seam_stays_sharp():
    lid = np.full((40, 50), 178.0, np.float32)          # lidar: dry park everywhere (boats, docks, the lagoon's surface)
    lid[:, 40:] = 172.0                                 # and lake to the east
    sv = np.full((40, 50), np.nan, np.float32); sv[5:35, 5:30] = 174.5   # the survey: 2.3 m of water in the basin
    z, src, info = bm.blend(lid, None, 176.8, survey=sv)
    assert (src[5:35, 5:30] == 4).all() and src[0, 0] == 1 and src[0, 45] == 1 and info["survey_cells"] == 30 * 25
    assert z[20, 20] == 174.5 and z[0, 0] == 178.0
    zs = bm.smooth_seam(z, src)
    assert zs[20, 20] == 174.5 and zs[5, 5] == 174.5 and zs[4, 5] == 178.0   # no smearing at the wall


def test_cookcounty_bed_elevation_fills_rim():
    g = {"x0": 0.0, "y0": 0.0, "dx": 2.0, "nx": 50, "ny": 40}
    polys = [square_poly(10, 10, 60, 50)]
    depth = np.full((40, 50), np.nan, np.float32); depth[8:22, 8:27] = 3.0; depth[10, 10] = 6.0   # the rim has no soundings
    z, inside, meta = cook.bed_elevation(g, polys=polys, depth=depth, level=176.8)
    assert inside[10, 10] and np.isfinite(z[inside]).all() and np.isnan(z[~inside]).all()
    assert abs(z[10, 10] - (176.8 - 6.0)) < 1e-5 and abs(z[24, 29] - 173.8) < 1e-5   # the rim takes the nearest sounding
    assert meta["level_m_igld85"] == 176.8 and meta["polys"][0]["name"] == "Test Harbor" and meta["date"] == cook.SURVEY_DATE


def test_model_bboxes_carry_spacing():
    assert bm.MODEL_BBOXES["harbor"] == (cook.BBOX, 2.0) and bm.MODEL_BBOXES["corridor"][1] == 4.0
    g = bm.local_grid(cook.BBOX, 2.0)
    assert bm.overlaps(g, cook.BBOX) and not bm.overlaps(bm.local_grid((41.95, -87.648, 41.975, -87.60), 4.0), cook.BBOX)


# ---------------------------------------------------------------- the harbor domain: wind library and surveyed walls
def test_swan_wind_inputs(tmp_path):
    b = fake_bathy()
    b["survey"] = {"polys": [{"name": "Diversey Harbor", "rings": [[[-500, 5500], [-400, 5500], [-400, 4700], [-500, 4700], [-500, 5500]]]},
                             {"name": "North Pond", "rings": [[[-800, 5000], [-700, 5000], [-700, 4900], [-800, 5000]]]}]}
    p = tmp_path / "bathy.json"; p.write_text(json.dumps(b))
    sc.DOMAINS["test_harbor"] = {"grid": "route", "dx": 4.0, "wind": True, "min_opening": 12.0}
    try:
        meta = sc.prepare(str(p), None, "test_harbor", [0, 90], str(tmp_path / "runs"), cases=sc.CASE_GROUPS["wind"][:2])
    finally:
        del sc.DOMAINS["test_harbor"]
    assert [c["id"] for c in meta["cases"]] == ["w000_u04p0", "w000_u07p0", "w090_u04p0", "w090_u07p0"] and meta["cases"][0] == {"id": "w000_u04p0", "dir": 0, "u10": 4.0}
    kinds = [o["kind"] for o in meta["obstacles"]]
    assert "quay" in kinds and "bank" in kinds and len(meta["obstacles"]) >= 2          # the pond is left out
    txt = (tmp_path / "runs" / "test_harbor" / "w090_u07p0" / "INPUT").read_text()
    assert "WIND 7.00 90.0" in txt and "GEN3 KOMEN AGROW" in txt and "PROJECT 'chicago' 'w090'" in txt
    assert "BOUNDSPEC" not in txt and "OFF QUADRUPL" not in txt and "RTP" in txt and "OBSTACLE TRANSM 0.00 REFL 0.50" in txt
    assert sc.case_id(45, 4.0, 1.0) == "d045_t4p0_h1p0" and len(sc.CASE_GROUPS["wind"]) == 4 and len(sc.WIND_DIRS) == 24


def test_survey_obstacles_split_at_the_lagoon():
    g = {"x0": -1000.0, "y0": 3000.0, "dx": 2.0, "nx": 600, "ny": 1600}
    bathy = {"survey": {"polys": [{"name": "Diversey Harbor", "rings": [[[-500, 5500], [-400, 5500], [-400, 4000], [-500, 4000], [-500, 5500]]]}]}}
    obs = sc.survey_obstacles(bathy, g)
    assert {o["kind"] for o in obs} == {"quay", "bank"} and all(o["pts"][0][0] >= 0 for o in obs)
    quay = [o for o in obs if o["kind"] == "quay"]; bank = [o for o in obs if o["kind"] == "bank"]
    assert quay[0]["refl"] == sc.REFL_QUAY and bank[0]["refl"] == sc.REFL_BANK
    assert sum(len(o["pts"]) - 1 for o in obs) >= 4                                     # every edge of the ring is a wall segment


def test_swan_collect_wind(tmp_path):
    from scipy.io import savemat
    g = {"x0": 0, "y0": 0, "dx": 2, "nx": 4, "ny": 3}
    d = tmp_path / "r" / "harbor"; d.mkdir(parents=True)
    cases = [{"id": sc.case_id(0, "wind", u), "dir": 0, "u10": u} for u in (4.0, 7.0)] + [{"id": sc.case_id(0, 4.0, 1.0), "dir": 0, "tp": 4.0, "hs": 1.0}]
    json.dump({"domain": "harbor", "grid": g, "level_m_igld85": 176.5, "cases": cases}, open(d / "meta.json", "w")); json.dump([], open(d / "obstacles.json", "w"))
    for c in cases:
        (d / c["id"]).mkdir()
        hs = np.full((3, 4), 0.3 if "u10" in c else 0.5); savemat(d / c["id"] / "out.mat", {"Hsig": hs, "Dir": np.full((3, 4), 90.0), "RTpeak": np.full((3, 4), 2.2), "Yp": np.arange(3)[:, None] * np.ones((1, 4))})
    out = sc.collect(str(tmp_path / "r"), str(tmp_path / "s.json"))
    h = out["domains"]["harbor"]
    assert [c["id"] for c in h["cases"]] == ["d000_t4p0_h1p0"] and [c["u10"] for c in h["wind"]["cases"]] == [4.0, 7.0]
    whs = np.frombuffer(gzip.decompress(base64.b64decode(h["wind"]["hs"])), np.uint8).reshape(2, 3, 4)
    wtp = np.frombuffer(gzip.decompress(base64.b64decode(h["wind"]["tp"])), np.uint8).reshape(2, 3, 4)
    assert whs.max() == 30 and wtp.max() == 22
    # a merge keeps the wind cases this run did not redo
    (d / cases[1]["id"] / "out.mat").unlink()
    out["domains"]["trail"] = {"grid": {"x0": 0, "y0": 0, "dx": 8, "nx": 2, "ny": 2}, "cases": [], "ratio": "", "dir": ""}   # another domain, not rerun
    out2 = sc.collect(str(tmp_path / "r"), str(tmp_path / "s2.json"), merge=out)
    assert [c["u10"] for c in out2["domains"]["harbor"]["wind"]["cases"]] == [4.0, 7.0] and "trail" in out2["domains"]
