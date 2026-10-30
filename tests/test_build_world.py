"""The page assembly: what of the model products reaches the page."""
import base64
import gzip
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import build_world as bw  # noqa: E402


def packed(nx, ny, dx, src):
    enc = lambda a: base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()  # noqa: E731
    z = np.full((ny, nx), -250, "<i2")
    return {"x0": 0.0, "y0": 0.0, "dx": dx, "nx": nx, "ny": ny, "z_cm": enc(z), "src": enc(src.astype(np.uint8))}


def test_thin_bathy_keeps_survey_and_surveyed_cells():
    src = np.zeros((100, 100), np.uint8); src[:, :10] = 1; src[30:40, 40:50] = 4     # lidar strip west, a surveyed basin far east
    b = {"version": 1, "level": {"m_igld85": 176.8}, "world": packed(100, 100, 10.0, src), "models": {"corridor": packed(100, 100, 4.0, src)},
         "survey": {"polys": [{"name": "Diversey Harbor", "rings": [[[0, 0], [1, 0], [1, 1]]]}], "date": "2023-05-23"},
         "probe": {"world": {"coverage": {"lidar": 0.2}}}}
    out = bw.thin_bathy(b)
    assert "models" not in out and out["survey"]["polys"][0]["name"] == "Diversey Harbor" and out["probe"] == {"world": {"coverage": {"lidar": 0.2}}}
    det = out["detail"]
    z = np.frombuffer(gzip.decompress(base64.b64decode(det["z_cm"])), "<i2").reshape(det["ny"], det["nx"])
    s = np.frombuffer(gzip.decompress(base64.b64decode(det["src"])), np.uint8).reshape(det["ny"], det["nx"])
    assert (s[30:40, 40:50] == 4).all() and z[35, 45] == -250        # the surveyed basin survives the lidar band crop
    assert z[90, 95] == bw.NODATA and s[90, 95] == 0                 # over 160 m from either source: dropped
