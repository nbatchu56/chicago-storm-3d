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

## Chicago Lake World (`/world`)

One 3-D world for the lakefront: the radar volume (clouds, rain and snow shafts, lightning) over the real OpenStreetMap shoreline, harbours, piers and buildings, with satellite lake temperature and ice, and paddler-scale waves (refraction and shoaling toward the nearest beach) under a board you can ride along a GPX track. The camera goes from a 100 km overview down to standing on the board.

Conditions come from one of four sources on a single timeline:

| Source | Weather | Wind and waves | Water |
|---|---|---|---|
| Live | latest KLOT volumes | nearest NDBC/GLOS station; SUP model's waves for the hour | NOAA GLSEA (ACSPO) |
| Forecast | radar extrapolation for 2 h, then NWS sky cover | NWS wind, SUP wave model (48 h) | GLSEA |
| Historical | NEXRAD archive volumes for the window | NDBC observations (realtime or stdmet archive) | GLSEA for that day, ice analysis in season |
| Experimental | generated storms: fair-weather cumulus, showers, thunderstorms, squall line, steady rain, lake-effect snow | your wind, duration and the real fetch (CEM growth) | your water temperature and ice cover |

Historical periods: `GET /api/history?start=2026-10-23T12:20Z&end=2026-10-23T12:51Z` builds a period (202 while building, then cached under `DATA_DIR/history`), or run the **History snapshot** workflow, or `python -m app.history --start ... --end ... --out file.json` and load the file in the page.

## Lakefront Local refresh and GitHub minutes

The page is a published Claude artifact, which can't fetch data itself. The **Radar snapshot** workflow runs hourly from 5 AM to 10 PM Chicago time (about 2 minutes each) and builds `world.html` on the `snapshot` branch; a Claude scheduled task republishes it at :20. The page's **Refresh now** button fires that task on demand (it starts a new snapshot when the last one is over 20 minutes old). With the repo private this uses roughly 18 runs/day × 2 min ≈ 1,100 Actions minutes a month; CI runs tests on code pushes only (the Docker build and live smoke test run on demand). The SWAN, FUNWAVE, bathymetry and validation workflows are run by hand (SWAN ≈ 2 h of runner time, FUNWAVE ≈ 1.7 h).

## Wave models and the lake bottom

The lakefront waves are built in layers, each checked against data:

| Layer | What | Source / model | Workflow |
|---|---|---|---|
| Lake bottom | 10 m grid over the map, 4 m grids for the route (North Ave–Diversey) and Montrose | 2020 USACE NCMP topobathy lidar (1 m, IGLD85), NOAA NCEI mosaic offshore (aligned to the lidar), ISGS/Chicago Park District beach surveys where reachable; lake level from NOAA CO-OPS Calumet Harbor | **Bathymetry** → `bathy` branch |
| Open-lake waves | Hour-by-hour growth with memory (decay 6 h, swell after wind shifts) | USACE CEM II-2 growth over the real fetch, wind adjusted to 10 m height and for air–sea stability | (in `app.sup_waves`) |
| Validation | Hindcast at buoys 45198 (Chicago) and 45174 (Wilmette) from wind only | Last 30 days: RMSE ≈ 0.21 m, r ≈ 0.87–0.89; Oct 16 peak within 5 cm at 45198, ~7 h late | **Validate waves vs buoy** → `validate` branch |
| Nearshore | Refraction, shoaling, breaking, friction, triads, diffraction, partial reflection off breakwaters, groynes, long piers and seawall-type revetments | SWAN 41.51 (TU Delft), stationary, 10 m grids; 12 directions × 3 periods × 2 heights per domain | **SWAN wave library** → `swan` branch |
| Harbour entrance | Individual waves: diffraction around the heads, reflection off the walls, crossing chop, breaking | FUNWAVE-TVD (fully nonlinear Boussinesq), 1.5 m grid, ~1 km tiles: Diversey, Belmont, North Ave, Fullerton, Monroe Harbor | **FUNWAVE harbour entrance** → `funwave` branch |

**Chop overlay** (View → Chop): wave energy density E = ρgHs²/16 (613 J/m² per m² of wave height, log scale 10–1000 J/m²) from SWAN everywhere it runs, so reflection off walls shows as extra energy in front of them. Inside a FUNWAVE tile (pick a condition under "Walls, wave by wave") magenta hatching marks crossing seas: from the phase-resolved surface, crossing = 1 − |⟨η_t ∇η⟩| / √(⟨η_t²⟩⟨|∇η|²⟩) over ~40 m. That's 0 for a clean run of waves, about 0.3 for two trains at right angles and about 0.9 against a wall that reflects 90%. It is faded out in the surf, and the readout gives the energy and crossing % at the view centre. The Monroe Harbor tile covers the gap between the Easterly and Southerly Inner Breakwaters and the Grant Park seawall it faces, with an Oct 3, 2026 race-day case.

**Chicago SUP launches** (in the conditions box): a chop rating for the Montrose, North Ave and Diversey launches, now and for the next 12 hours (one square per hour). It is computed in 1.5–2 m of water off each launch. The rating starts from the wave energy there (the SWAN library plus local wind chop; harbour water the library leaves at zero only gets the chop the wind raises over the basin). Short, steep waves count for more (× 4 s / period), and so does any FUNWAVE crossing shown. The bands are Calm < 25, Light chop < 70, Choppy < 160, Rough < 350 J/m², Very rough above that.

The paddling forecast (`/sup`) uses the surveyed cross-shore profiles for North Ave and Oak St (others keep the idealized profile). The world page blends the SWAN library for the current offshore condition, uses the real depth for breaking, and can show the FUNWAVE wave field over the Diversey Harbor entrance.

To serve these from the deployed service, copy `bathy.json`, `swan.json` and `funwave.json` from those branches into `DATA_DIR` (they are served at `/api/bathy.json`, `/api/swan.json`, `/api/funwave.json`), and set `BATHY_PATH` for the forecast.

Assumptions to know: inside Diversey Harbor the lidar has no returns, so unsurveyed harbour water is treated as 2 m deep; the NOAA chart's interpolated nearshore cells are floored at 1.5 m; SWAN treats obstacles as fully opaque with 35–50% reflection (85% for known vertical walls: Navy Pier, Monroe Harbor's seawall and inner breakwaters).

## Stand-up paddling forecast (`/sup`)

`/sup` is a paddling forecast for eight lakefront launches (Montrose, North Avenue, Oak Street, Ohio Street, 31st Street, 63rd Street, Rainbow, Calumet), rebuilt every 30 minutes (`SUP_INTERVAL_MIN`):

- wind and gusts from the NWS hourly gridpoint forecast, recent hours from NDBC buoys/stations;
- fetch over the lake ray-cast from each beach, wind duration from the wind history, USACE growth curves for open-lake waves, matched to the nearest reporting wave buoy now;
- shoaling, refraction and breaking to the paddle zone (~2.5 m deep), plus local chop;
- breakwater/pier shelter per spot (approximate), an offshore-wind safety flag, water temperature;
- an hourly SUP rating, best daylight windows, the NWS nearshore marine text, and a 3-D water simulation of any hour.

`python -m app.sup_waves --out sup.json` builds it once. `SUP=off` disables it.

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
| `NDBC_STATIONS` | `45007,45174,CHII2` | buoys/stations tried in order for lake wind; set it empty to turn the lookup off |
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
