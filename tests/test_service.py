"""
Offline tests for the live service.

The AWS archive is replaced by a stand-in that lists KLOT-style keys for the last 30 minutes and
"downloads" Py-ART's bundled real NEXRAD Level II sample file, so the full path runs:
list -> download -> Py-ART gridding (+ cache) -> motion -> forecast -> cloud -> scene -> HTTP.
Run:  pytest -q tests
"""
import datetime as dt
import gzip
import importlib
import json
import os
import shutil
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
UTC = dt.timezone.utc


def sample_file():
    import pyart
    return os.path.join(os.path.dirname(pyart.testing.__file__), "data",
                        "example_nexrad_archive_msg31_compressed.ar2v")


class FakeS3:
    """Minimal stand-in for the public NEXRAD bucket."""
    def __init__(self, site="KLOT", n=7):
        now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
        self.keys = [f"{(now - dt.timedelta(minutes=5 * k)):%Y/%m/%d}/{site}/{site}"
                     f"{(now - dt.timedelta(minutes=5 * k)):%Y%m%d_%H%M%S}_V06" for k in range(n)]
        self.keys.append(f"{now:%Y/%m/%d}/{site}/{site}{now:%Y%m%d_%H%M%S}_V06_MDM")  # must be skipped
        self.downloads = 0

    def get_paginator(self, _name):
        fake = self
        class P:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [{"Key": k} for k in fake.keys if k.startswith(Prefix)]}]
        return P()

    def download_file(self, bucket, key, path):
        self.downloads += 1
        shutil.copy(sample_file(), path)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NDBC_STATIONS", "")         # no network in tests
    monkeypatch.setenv("N_SCANS", "4")
    monkeypatch.setenv("LEAD_MIN", "30")
    return tmp_path


def fresh_modules():
    for m in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[m]
    return importlib.import_module("app.pipeline")


def test_aws_pipeline_with_real_level2(env, monkeypatch):
    pl = fresh_modules()
    fake = FakeS3()
    monkeypatch.setattr(pl.nc, "_s3", lambda: fake)
    p = pl.Pipeline()
    t0 = time.time(); assert p.run_once(), p.status["last_error"]; first = time.time() - t0
    st = p.public_status()
    assert st["ready"] and st["frames"] == 4 + 3 and len(st["scans"]) == 4
    assert st["wind"]["source"].startswith("scenario")
    scene = json.load(open(p.scene_path))
    assert scene["meta"]["site"] == "KLOT" and scene["meta"]["live"] is True
    assert scene["meta"]["grid"]["nz"] == 30
    assert len(gzip.decompress(open(p.scene_gz_path, "rb").read())) == os.path.getsize(p.scene_path)
    assert fake.downloads == 4
    cached = [f for f in os.listdir(p.cfg.cache_dir) if f.endswith(".grid.npz")]
    assert len(cached) == 4
    # same scans in the archive: no rebuild at all
    updated = p.status["updated"]
    assert not p.run_once() and p.status["updated"] == updated and p.status["last_error"] is None
    # a new scan arrives: only that one is downloaded; the rest come from the grid cache
    now = dt.datetime.now(UTC) + dt.timedelta(minutes=1)
    fake.keys.insert(0, f"{now:%Y/%m/%d}/KLOT/KLOT{now:%Y%m%d_%H%M%S}_V06")
    monkeypatch.setattr(pl.dt, "datetime", type("D", (dt.datetime,), {"now": staticmethod(lambda tz=None: now + dt.timedelta(seconds=30))}))
    t0 = time.time(); assert p.run_once(), p.status["last_error"]; second = time.time() - t0
    assert fake.downloads == 5
    assert not any(f.endswith("_V06") for f in os.listdir(p.cfg.cache_dir))   # raw volumes removed after gridding
    print(f"first run {first:.1f}s, cached run {second:.1f}s")


def test_failure_keeps_last_scene(env, monkeypatch):
    pl = fresh_modules()
    fake = FakeS3()
    monkeypatch.setattr(pl.nc, "_s3", lambda: fake)
    p = pl.Pipeline(); assert p.run_once()
    good = os.path.getmtime(p.scene_path)
    fake.keys = []                                   # archive returns nothing
    assert not p.run_once()
    st = p.public_status()
    assert st["ready"] and "Found only 0" in st["last_error"] and st["failures"] == 1
    assert os.path.getmtime(p.scene_path) == good


def test_demo_source(env, monkeypatch):
    monkeypatch.setenv("SOURCE", "demo")
    pl = fresh_modules()
    p = pl.Pipeline(); assert p.run_once(), p.status["last_error"]
    assert p.status["has_precip"] is True and p.status["frames"] == 7


def test_http(env, monkeypatch):
    monkeypatch.setenv("SOURCE", "demo")
    monkeypatch.setenv("SCHEDULER", "external")
    monkeypatch.setenv("REFRESH_TOKEN", "s3cret")
    monkeypatch.setenv("SUP", "off")
    fresh_modules()
    from fastapi.testclient import TestClient
    import app.main as main
    def offline():
        raise RuntimeError("offline in tests")
    monkeypatch.setattr(main.sup_waves_mod, "build", offline)
    main.pipeline.lock.acquire()                     # hold off the startup run so we can see "not ready"
    with TestClient(main.app) as c:
        assert c.get("/healthz").json() == {"ok": True}
        assert c.get("/readyz").status_code == 503
        assert c.get("/api/scene.json").status_code == 503
        main.pipeline.lock.release()
        assert c.post("/api/refresh").status_code == 403
        r = c.post("/api/refresh", headers={"X-Refresh-Token": "s3cret"})
        assert r.status_code == 200 and r.json()["status"]["ready"]
        r = c.get("/api/scene.json", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
        assert r.json()["meta"]["live"] is True
        etag = r.headers["etag"]
        assert c.get("/api/scene.json", headers={"If-None-Match": etag}).status_code == 304
        st = c.get("/api/status").json()
        assert st["ready"] and st["stale"] is False and st["scan_age_min"] < 10
        html = c.get("/").text
        assert "Chicago Storm 3D" in html and "pollLive" in html
        # SUP forecast: page always served; data once built (stubbed here, no network in tests)
        sup_page = c.get("/sup")
        assert "Chicago SUP Waves" in sup_page.text and sup_page.text.startswith("<!doctype html>")
        assert 'charset="utf-8"' in sup_page.text and "utf-8" in sup_page.headers["content-type"]
        assert c.get("/api/sup.json").status_code == 503
        monkeypatch.setattr(main.sup_waves_mod, "build", lambda: {"generated": "2026-10-23T18:00:00+00:00", "sources": {"errors": []}, "spots": [{"id": "x", "hourly": []}]})
        assert main.sup.run_once()
        assert c.get("/api/sup.json").json()["generated"].startswith("2026-10-23")
        assert c.get("/api/status").json()["sup"]["updated"].startswith("2026-10-23")
