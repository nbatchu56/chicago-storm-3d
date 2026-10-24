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
