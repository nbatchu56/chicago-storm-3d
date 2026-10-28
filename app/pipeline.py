"""
Live radar pipeline: KLOT Level II scans -> 3-D scene JSON for the LakeView viewer.

Each run:
  1. lists the newest volume scans for the site in the public NOAA/Unidata archive on AWS,
  2. downloads only the ones not already cached,
  3. grids each new volume once (Py-ART) and caches the 3-D grid as .npz,
  4. tracks echo motion, advects every level forward (0-2 h), infers clouds,
  5. adds the current lake wind from NDBC (falls back to a fixed scenario),
  6. writes scene.json atomically and updates status.

Runs in a background thread inside the web service, or once from the command line:
    python -m app.pipeline --once
"""

import datetime as dt
import gzip
import json
import logging
import os
import shutil
import threading
import time
import traceback

import numpy as np

from . import radar_to_scene as r2s
from . import chicago_radar_nowcast as nc

log = logging.getLogger("klot.pipeline")
UTC = dt.timezone.utc


def env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except ValueError:
        log.warning("Bad value for %s=%r, using %r", name, v, default)
        return default


class Config:
    def __init__(self):
        self.source = env("SOURCE", "aws")                 # aws | demo
        self.site = env("RADAR_SITE", "KLOT").upper()
        self.interval_min = env("INTERVAL_MIN", 5.0, float)
        self.n_scans = env("N_SCANS", 6, int)
        self.lookback_min = env("LOOKBACK_MIN", 75, int)
        self.lead_min = env("LEAD_MIN", 120, int)
        self.step_min = env("STEP_MIN", 10, int)
        self.cloud_base_km = env("CLOUD_BASE_KM", 1.2, float)
        # NDBC_STATIONS="" (set but empty) turns the buoy lookup off; unset uses the defaults
        self.ndbc = [s.strip() for s in os.environ.get("NDBC_STATIONS", "45007,45174,CHII2").split(",") if s.strip()]
        self.wind_speed = env("WIND_SPEED", 10.0, float)    # fallback scenario wind
        self.wind_dir = env("WIND_DIR", 20.0, float)
        self.wind_duration_h = env("WIND_DURATION_H", 8.0, float)
        self.stale_min = env("STALE_MIN", 30, int)
        self.data_dir = env("DATA_DIR", "/tmp/klot-data")
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.keep_hours = env("CACHE_HOURS", 3.0, float)

    def public(self):
        return {"source": self.source, "site": self.site, "interval_min": self.interval_min,
                "n_scans": self.n_scans, "lead_min": self.lead_min, "step_min": self.step_min,
                "ndbc_stations": self.ndbc}


class NoNewScans(Exception):
    """The archive has no volume newer than the current scene."""


class Pipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self._last_keys = None
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        self.scene_path = os.path.join(self.cfg.data_dir, "scene.json")
        self.scene_gz_path = self.scene_path + ".gz"
        self.lock = threading.Lock()
        self.status = {"ready": os.path.exists(self.scene_path), "running": False, "updated": None,
                       "latest_scan": None, "scans": [], "frames": 0, "last_error": None,
                       "last_run_seconds": None, "next_run": None, "runs": 0, "failures": 0,
                       "wind": None, "has_precip": None, "config": self.cfg.public()}
        self._stop = threading.Event()
        self._load_previous_status()

    # ------------------------------------------------------------------ helpers
    def _load_previous_status(self):
        p = os.path.join(self.cfg.data_dir, "status.json")
        if os.path.exists(p) and os.path.exists(self.scene_path):
            try:
                with open(p) as fh:
                    old = json.load(fh)
                for k in ("updated", "latest_scan", "scans", "frames", "wind", "has_precip"):
                    self.status[k] = old.get(k)
                self.status["ready"] = True
            except (OSError, ValueError):
                pass

    def _write_atomic(self, path, data_bytes):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data_bytes)
        os.replace(tmp, path)

    def _lake_wind(self):
        from .chicago_nearshore_waves import fetch_ndbc
        for st in self.cfg.ndbc:
            try:
                obs = fetch_ndbc(st)
                age_ok = True
                try:  # realtime2 rows start "YYYY MM DD hh mm"
                    y, mo, d, h, mi = (int(x) for x in obs["time"].split()[:5])
                    age = dt.datetime.now(UTC) - dt.datetime(y, mo, d, h, mi, tzinfo=UTC)
                    age_ok = age < dt.timedelta(hours=3)
                except ValueError:
                    pass
                if age_ok:
                    return {"speed": round(obs["wspd10"], 1), "dir": obs["wdir"],
                            "duration_h": self.cfg.wind_duration_h,
                            "source": f"NDBC {st.upper()} {obs['time']} UTC"}
            except Exception as e:  # noqa: BLE001 - network/parse problems fall through to next station
                log.info("NDBC %s unavailable: %s", st, e)
        return {"speed": self.cfg.wind_speed, "dir": self.cfg.wind_dir,
                "duration_h": self.cfg.wind_duration_h, "source": "scenario (no buoy reachable)"}

    def _grid_cached(self, path):
        npz = path + ".grid.npz"
        if os.path.exists(npz):
            z = np.load(npz)
            t = dt.datetime.fromtimestamp(float(z["t"]), UTC)
            return t, z["vol"].astype(float), (float(z["rx"]), float(z["ry"]))
        t, vol, (rx, ry), _ = r2s.grid_volume(path)
        np.savez_compressed(npz, t=t.timestamp(), vol=vol.astype(np.float32), rx=rx, ry=ry)
        import gc; gc.collect()   # release the Py-ART radar object before the next volume
        return t, vol, (rx, ry)

    def _prune_cache(self, keep_names):
        cutoff = time.time() - self.cfg.keep_hours * 3600
        for name in os.listdir(self.cfg.cache_dir):
            p = os.path.join(self.cfg.cache_dir, name)
            base = name.replace(".grid.npz", "")
            if base not in keep_names and os.path.getmtime(p) < cutoff:
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ------------------------------------------------------------------ one run
    def _build_aws(self):
        cfg = self.cfg
        scans = nc.list_scans(cfg.site, dt.datetime.now(UTC), cfg.n_scans, cfg.lookback_min)
        keys = [k for _, _, k in scans]
        fresh_enough = self.status.get("updated") and (
            dt.datetime.now(UTC) - dt.datetime.fromisoformat(self.status["updated"])) < dt.timedelta(minutes=30)
        if keys == self._last_keys and fresh_enough and os.path.exists(self.scene_path):
            raise NoNewScans()
        self._last_keys = keys
        times, vols, radar_xy = [], [], (0.0, 0.0)
        for _, bucket, key in scans:
            path = os.path.join(cfg.cache_dir, key.rsplit("/", 1)[-1])
            if not os.path.exists(path + ".grid.npz"):
                path = nc.download(bucket, key, cfg.cache_dir)
            t, v, radar_xy = self._grid_cached(path)
            if os.path.exists(path) and os.path.exists(path + ".grid.npz"):
                os.remove(path)   # raw volume no longer needed once gridded
            times.append(t); vols.append(v)
        self._prune_cache({k.rsplit("/", 1)[-1] for _, _, k in scans})
        order = np.argsort(times)
        times = [times[i] for i in order]; vols = [vols[i] for i in order]
        source = f"NEXRAD {cfg.site} Level II"
        return times, vols, radar_xy, source

    def _build_demo(self):
        cfg = self.cfg
        ax = r2s.axes(r2s.EXT_HALF)
        X, Y = np.meshgrid(ax, ax)
        state = r2s.synthetic_state()
        # the demo storm repeats every 3 hours so the "live" demo keeps moving
        now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
        now = now - dt.timedelta(minutes=now.minute % 5)
        phase = ((now.timestamp() / 60) % 180) - 60
        times = [now + dt.timedelta(minutes=5 * k) for k in range(-cfg.n_scans + 1, 1)]
        vols = [r2s.synthetic_volume(phase + (t - now).total_seconds() / 60, X, Y, state) for t in times]
        return times, vols, r2s.xy_from_downtown(*nc.SITES["KLOT"]), "Demo storm (synthetic, repeating)"

    def run_once(self):
        if not self.lock.acquire(blocking=False):
            return False  # a run is already in progress
        t0 = time.time()
        self.status["running"] = True
        try:
            times, vols, radar_xy, source = self._build_aws() if self.cfg.source == "aws" else self._build_demo()
            wind = self._lake_wind() if self.cfg.source == "aws" else {
                "speed": self.cfg.wind_speed, "dir": self.cfg.wind_dir,
                "duration_h": self.cfg.wind_duration_h, "source": "scenario"}
            scene = r2s.build_scene(times, vols, source, self.cfg.site if self.cfg.source == "aws" else "DEMO",
                                    radar_xy, self.cfg.lead_min, self.cfg.step_min, wind, self.cfg.cloud_base_km)
            has_precip = bool(any(np.isfinite(v[:8]).any() for v in vols))
            scene["meta"]["live"] = True
            scene["meta"]["has_precip"] = has_precip
            body = json.dumps(scene, separators=(",", ":")).encode()
            self._write_atomic(self.scene_path, body)
            self._write_atomic(self.scene_gz_path, gzip.compress(body, 6))
            now = dt.datetime.now(UTC)
            self.status.update({
                "ready": True, "updated": now.isoformat(timespec="seconds"),
                "latest_scan": times[-1].isoformat(timespec="seconds"),
                "scans": [t.isoformat(timespec="seconds") for t in times],
                "frames": len(scene["frames"]), "last_error": None, "wind": wind,
                "has_precip": has_precip, "scene_bytes": len(body),
            })
            self._write_atomic(os.path.join(self.cfg.data_dir, "status.json"),
                               json.dumps(self.status, default=str).encode())
            log.info("Scene updated: %d frames, latest scan %s, %.1f s", len(scene["frames"]),
                     self.status["latest_scan"], time.time() - t0)
            return True
        except NoNewScans:
            self.status["checked"] = dt.datetime.now(UTC).isoformat(timespec="seconds")
            self.status["last_error"] = None
            return False
        except Exception as e:  # noqa: BLE001 - keep serving the last good scene
            self.status["failures"] += 1
            self.status["last_error"] = f"{type(e).__name__}: {e}"
            log.error("Pipeline run failed: %s\n%s", e, traceback.format_exc())
            return False
        finally:
            self.status["runs"] += 1
            self.status["last_run_seconds"] = round(time.time() - t0, 1)
            self.status["running"] = False
            self.lock.release()

    # ------------------------------------------------------------------ scheduler
    def loop(self):
        while not self._stop.is_set():
            self.run_once()
            # align to the interval so runs land a fixed time after each scan cycle
            period = max(60.0, self.cfg.interval_min * 60)
            wait = period - (time.time() % period) + 20  # +20 s: archive upload lag
            self.status["next_run"] = (dt.datetime.now(UTC) + dt.timedelta(seconds=wait)).isoformat(timespec="seconds")
            self._stop.wait(wait)

    def start(self):
        th = threading.Thread(target=self.loop, name="klot-pipeline", daemon=True)
        th.start()
        return th

    def stop(self):
        self._stop.set()

    def public_status(self):
        st = dict(self.status)
        if st.get("latest_scan"):
            age = (dt.datetime.now(UTC) - dt.datetime.fromisoformat(st["latest_scan"])).total_seconds() / 60
            st["scan_age_min"] = round(age, 1)
            st["stale"] = age > self.cfg.stale_min
        return st


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the live LakeView scene")
    ap.add_argument("--once", action="store_true", help="run one update and exit")
    a = ap.parse_args()
    p = Pipeline()
    if a.once:
        ok = p.run_once()
        print(json.dumps(p.public_status(), indent=2, default=str))
        raise SystemExit(0 if ok else 1)
    p.loop()
