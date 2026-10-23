# Chicago Storm 3D: live KLOT radar service

A single container that:

1. pulls the newest **KLOT (Chicago/Romeoville) NEXRAD Level II** volume scans from NOAA's public archive on AWS every 5 minutes (no account or API key needed),
2. grids every tilt into a 3-D box over Chicago (Py-ART), tracks storm motion and extrapolates a **0–2 h forecast**, and infers clouds from the echoes,
3. adds the current **lake wind** from NDBC buoys for the wave model,
4. serves the **3-D viewer** at `/`, which loads the newest scene and refreshes itself when new scans arrive.

```
app/
  main.py                    web service (FastAPI): viewer, scene, status, health, refresh
  pipeline.py                scheduled radar pipeline (background thread)
  radar_to_scene.py          gridding, forecast, cloud inference, encoding
  chicago_radar_nowcast.py   archive access, motion tracking, advection
  chicago_nearshore_waves.py NDBC buoy reader
  static/index.html          the viewer (live mode)
tests/test_service.py        offline tests (stand-in archive serving a real Level II file)
Dockerfile, docker-compose.yml, fly.toml, render.yaml
```

## Run it locally

```bash
docker compose up --build          # then open http://localhost:8080
```

Without Docker (Python 3.13, Linux x86_64 or macOS):

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080 --workers 1
```

The first scene appears 1–2 minutes after start (6 downloads + gridding); after that each 5-minute run only downloads and grids the one new volume (a few seconds). `SOURCE=demo` runs a repeating synthetic storm when you want to try it without radar access.

## Deploy

The service must stay running between visitors (the pipeline is a background thread), use **one** worker process, and have **2 GB RAM** (gridding a full volume peaks near 0.9 GB). Py-ART publishes x86_64 Linux wheels only, so deploy on x86_64 (the Dockerfile pins `linux/amd64`).

### Fly.io (recommended: has a Chicago region)

```bash
fly launch --copy-config --no-deploy      # pick a unique app name when asked
fly volumes create storm_data --size 1 --region ord
fly deploy
fly open
```

`fly.toml` keeps one machine always running in `ord` with 2 GB and a volume for the cache.

### Render

Push this folder to a GitHub repo, then **New → Blueprint** and pick the repo. `render.yaml` creates a Docker web service with a 1 GB disk. Use a paid instance with at least 1–2 GB RAM (the free tier sleeps when idle, which stops the pipeline).

### Any Linux server (DigitalOcean, Lightsail, a home server)

```bash
git clone <your repo> && cd klot-service
docker compose up -d --build
```

Put Caddy or nginx in front for HTTPS if it's public.

### Google Cloud Run (or other scale-to-zero hosts)

Background threads don't run while an instance is idle, so use an external schedule:

```bash
gcloud run deploy chicago-storm-3d --source . --region us-central1 --memory 2Gi \
  --min-instances 1 --max-instances 1 --set-env-vars SCHEDULER=external,REFRESH_TOKEN=<secret>
gcloud scheduler jobs create http storm-refresh --schedule "*/5 * * * *" \
  --uri https://<service-url>/api/refresh --http-method POST --headers X-Refresh-Token=<secret>
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SOURCE` | `aws` | `aws` = live radar, `demo` = synthetic storm |
| `RADAR_SITE` | `KLOT` | NEXRAD site (KMKX Milwaukee, KIWX N. Indiana also work; the 3-D box stays centred on Chicago) |
| `INTERVAL_MIN` | `5` | minutes between pipeline runs |
| `N_SCANS` | `6` | past volumes used (motion tracking + playback) |
| `LOOKBACK_MIN` | `75` | how far back to look for those volumes |
| `LEAD_MIN` / `STEP_MIN` | `120` / `10` | forecast length and frame step |
| `CLOUD_BASE_KM` | `1.2` | cloud base for the inferred cloud field |
| `NDBC_STATIONS` | `45007,45174,CHII2` | buoys/stations tried in order for lake wind |
| `WIND_SPEED` / `WIND_DIR` / `WIND_DURATION_H` | `10` / `20` / `8` | fallback lake wind when no station reports |
| `STALE_MIN` | `30` | status reports `stale` when the newest scan is older |
| `DATA_DIR` | `/data` (Docker) | scene, status and grid cache |
| `CACHE_HOURS` | `3` | how long old grids are kept |
| `SCHEDULER` | `internal` | `external` = only update on `POST /api/refresh` |
| `REFRESH_TOKEN` | – | required for `POST /api/refresh` |
| `CORS_ORIGINS` | – | comma list of sites allowed to fetch the API from a browser |

## Endpoints

| | |
|---|---|
| `GET /` | 3-D viewer (live) |
| `GET /api/scene.json` | newest scene (gzip, ETag) |
| `GET /api/status` | newest scan time, scan age, `stale`, last error, next run, wind source |
| `GET /healthz` | process is up |
| `GET /readyz` | 200 once a scene exists |
| `POST /api/refresh` | run now (`X-Refresh-Token` header) |

## How it behaves

- **Freshness:** Level II volumes complete every 4–10 minutes (faster in storm mode) and appear in the archive a few minutes later, so the newest scan shown is typically 5–15 minutes old. The page shows the scan time and turns red when data is stale (radar outage, archive delay).
- **Failures:** if a run fails (network, a bad file), the last good scene keeps being served and `/api/status` shows the error. Runs with no new volume are skipped.
- **Clear weather:** the scene still builds; the page notes there's no precipitation on radar.
- **Forecast limits:** extrapolation is good for roughly 30–60 minutes and weakens after; it can't create new storms or capture rapid growth/decay. Clouds are inferred from echoes; radar does not see non-precipitating cloud.

## Tests

Every push runs GitHub Actions (`.github/workflows/ci.yml`): the offline tests, a Docker build with an endpoint check, and a live smoke test that processes the real newest KLOT scans from AWS.


```bash
pip install pytest httpx
PYART_QUIET=1 pytest -q tests
```

The tests replace the AWS archive with a stand-in that serves a real NEXRAD Level II file, so listing, download, gridding, caching, forecast, failure handling and every HTTP endpoint run offline.

Data: NOAA NEXRAD Level II via the NOAA Open Data Dissemination program on AWS (`unidata-nexrad-level2`, with `noaa-nexrad-level2` as fallback for older dates); NDBC buoy observations.
