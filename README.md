# LakeView: Chicago's lakefront, live in 3-D

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

## The 3-D lakefront page (`/world`)

One 3-D world for the lakefront: the radar volume (clouds, rain and snow shafts, lightning) over the real OpenStreetMap shoreline, harbours, piers and buildings, with satellite lake temperature and ice, and paddler-scale waves (refraction and shoaling toward the nearest beach) under a board you can ride along a GPX track. The camera goes from a 100 km overview down to standing on the board.

Conditions come from one of four sources on a single timeline:

| Source | Weather | Wind and waves | Water |
|---|---|---|---|
| Live | latest KLOT volumes | nearest NDBC/GLOS station; SUP model's waves for the hour | NOAA GLSEA (ACSPO) |
| Forecast | radar extrapolation for 2 h, then NWS sky cover | NWS wind, SUP wave model (48 h) | GLSEA |
| Historical | NEXRAD archive volumes for the window | NDBC observations (realtime or stdmet archive) | GLSEA for that day, ice analysis in season |
| Experimental | generated storms: fair-weather cumulus, showers, thunderstorms, squall line, steady rain, lake-effect snow | your wind, duration and the real fetch (CEM growth) | your water temperature and ice cover |

Historical periods: `GET /api/history?start=2026-10-23T12:20Z&end=2026-10-23T12:51Z` builds a period (202 while building, then cached under `DATA_DIR/history`), or run the **History snapshot** workflow, or `python -m app.history --start ... --end ... --out file.json` and load the file in the page.

**Severe weather, 2022–2026** (Experimental → History → *Severe weather*): `tools/events.json` lists 23 lakefront events with the 60–90 minutes when each peaked at the shore (derechos, tornado days, hail, the May 2025 dust storm, floods, the Dec 2022 blizzard, lake-effect and lake-enhanced snow, the arctic blasts with the most nearshore ice, gales) with a headline and the NWS summary. The **History snapshot** workflow with no start builds the whole catalog as a matrix (or `label` = a comma list); the arctic events take their lake temperature and ice analysis from the ice-peak date (`--lake-date`). Each period is its own file under `data/history/`, fetched when picked, and applies its own satellite lake day while shown.

**Lake ice**: the water shader sorts ice into shore-fast sheet (cracks, pressure ridges, snow after a long cold spell), pancake ice at a wavy edge (plates with raised rims, rafting as the concentration grows), floes with dark leads offshore, and grease ice or nilas in calm new water; a sheet or pack kills the waves, grease the ripples. The kind and thickness follow the NOAA concentration, the wave height and the cold spell's freezing degree-days (Stefan: 3.3 cm × √FDD; from the catalog, a guess from the analysis, or the lab's *Cold spell* slider). The map's looks texture carries a distance-to-shore field, so the shelf ice at the beaches and in the harbours is placed by real distance where the 1.3 km analysis grid is blank.

**Diversey Outlook** (`outlook.html`, built by `build_world.py --also` with `meta.start.kiosk`): the same page in kiosk mode — one fixed view over Diversey Harbor, no controls, playing the next 24 hours as one even 10-second loop (`kiosk.span`; without it, from now to 3 hours after the next sunrise); it reloads itself every 30 minutes and, when the build it shows is over 6 hours old, fires the refresh task (only for the page's owner signed in to claude.ai). Published as its own artifact by the refresh task, with LakeView's data files copied server-side.

## Page refresh and GitHub minutes

The page is a published Claude artifact, which can't fetch data itself. The **Radar snapshot** workflow runs hourly from 5 AM to 10 PM Chicago time (about 2 minutes each) and builds `world.html` and `outlook.html` on the `snapshot` branch; a Claude scheduled task republishes both (the past periods under `data/history/` only when they change). The page's **Refresh now** button fires that task on demand (it starts a new snapshot when the last one is over 20 minutes old). With the repo private this uses roughly 18 runs/day × 2–3 min (the GLWU download and scorecard add under a minute) ≈ 1,100 Actions minutes a month; CI runs tests on code pushes only (the Docker build and live smoke test run on demand). The SWAN, FUNWAVE, bathymetry and validation workflows are run by hand (SWAN ≈ 2 h of runner time for the trail and lane; the harbor domain's offshore and wind cases are another few hours spread over its matrix; FUNWAVE ≈ 1.7 h).

## Wave models and the lake bottom

The lakefront waves are built in layers, each checked against data:

| Layer | What | Source / model | Workflow |
|---|---|---|---|
| Lake bottom | 10 m grid over the map, 4 m grids for the route (North Ave–Diversey), Montrose and the corridor (Monroe Harbor–Montrose), a 2 m grid for Diversey Harbor and the lagoon | 2020 USACE NCMP topobathy lidar (1 m, IGLD85), NOAA NCEI mosaic offshore (aligned to the lidar), ISGS/Chicago Park District beach surveys where reachable; inside Diversey Harbor, the channel under Fullerton, the South Lagoon, North Pond and South Pond the **2023 USGS / Cook County echo-sounder survey** (`app/cookcounty.py`: 3 ft depth raster and the surveyed shoreline polygons, with the lake level on the survey day, May 23, 2023); lake level from NOAA CO-OPS Calumet Harbor | **Bathymetry** → `bathy` branch |
| Open-lake waves | Hour-by-hour growth with memory (decay 6 h, swell after wind shifts) | USACE CEM II-2 growth over the real fetch, wind adjusted to 10 m height and for air–sea stability | (in `app.sup_waves`) |
| Validation | Hindcast at buoys 45198 (Chicago) and 45174 (Wilmette) from wind only | Last 30 days: RMSE ≈ 0.21 m, r ≈ 0.87–0.89; Oct 16 peak within 5 cm at 45198, ~7 h late | **Validate waves vs buoy** → `validate` branch |
| Nearshore | Refraction, shoaling, breaking, friction, triads, diffraction, partial reflection off breakwaters, groynes, long piers and seawall-type revetments | SWAN 41.51 (TU Delft), stationary; the trail (8 m, Monroe Harbor–Montrose) inside the lane (20 m, out past the crib); 12 directions × 11 (period, height) cases per domain | **SWAN wave library** → `swan` branch |
| Diversey Harbor and the lagoon | What the lake's sea does through the mouth (diffraction, reflection off the surveyed walls: quay walls 0.5, the lagoon's banks 0.25) and, separately, the chop the wind raises over the basin's own fetch (no incoming waves; 24 directions × 4, 7, 11, 16 m/s, third-generation growth); the page adds the two in quadrature inside the harbour | SWAN on the 2 m harbor grid (the inlet under LSD is 8 cells wide); `--domain harbor`, `--group wind` | **SWAN wave library** (domains `harbor`, the `wind` job) → `swan` branch |
| Harbour entrance | Individual waves: diffraction around the heads, reflection off the walls, crossing chop, breaking | FUNWAVE-TVD (fully nonlinear Boussinesq), 1.5 m grid, ~1 km tiles: Diversey (the mouth to the Fullerton sill, on the 2 m survey bottom), Belmont, North Ave, Fullerton, Monroe Harbor | **FUNWAVE harbour entrance** → `funwave` branch |

**Short waves** (the near-camera patch): the 28 refracted, shoaled components carry the energetic part of the sea; waves shorter than about half the main peak's wavelength come from a continuous wind-wave spectrum (Pierson–Moskowitz shape for the local wind, cos² spread about it) synthesized by FFT on a 64 m tile, 30 times a second: a 32 × 32 grid (2 m) moves the surface and the board, a 128 × 128 grid (0.5 m) gives slopes and the crest bunching (Jacobian) that breaks up whitecaps. What a pixel can't resolve goes into the sun glitter's roughness.

**Chop overlay** (View → Chop): wave energy density E = ρgHs²/16 (613 J/m² per m² of wave height, log scale 10–1000 J/m²) from SWAN everywhere it runs, so reflection off walls shows as extra energy in front of them. Inside a FUNWAVE tile (pick a condition under "Walls, wave by wave") magenta hatching marks crossing seas: from the phase-resolved surface, crossing = 1 − |⟨η_t ∇η⟩| / √(⟨η_t²⟩⟨|∇η|²⟩) over ~40 m. That's 0 for a clean run of waves, about 0.3 for two trains at right angles and about 0.9 against a wall that reflects 90%. It is faded out in the surf, and the readout gives the energy and crossing % at the view centre. The Monroe Harbor tile covers the gap between the Easterly and Southerly Inner Breakwaters and the Grant Park seawall it faces, with an Oct 3, 2026 race-day case.

**Conditions box**: wind and waves as two big numbers, then sky, air and water temperature. In live and forecast modes it also shows active NWS alerts for the lakefront zones (LMZ741/742, ILZ104, Cook County), today's SPC Day 1 severe outlook at North Avenue Beach, and the latest Midway / O'Hare METAR (live only). These are fetched by `app/wx_now.py` during the snapshot's lakefront step (stored as `wx` in `lakefront.json`), and each part is skipped if its source is unreachable.

**Chicago SUP launches** (in the conditions box): a chop rating for the Montrose, North Ave and Diversey launches, now and for the next 12 hours (one square per hour). It is computed in 1.5–2 m of water off each launch. The rating starts from the wave energy there (the SWAN library plus local wind chop; harbour water the library leaves at zero only gets the chop the wind raises over the basin). Short, steep waves count for more (× 4 s / period), and so does any FUNWAVE crossing shown. The bands are Calm < 25, Light chop < 70, Choppy < 160, Rough < 350 J/m², Very rough above that.

The paddling forecast (`/sup`) uses the surveyed cross-shore profiles for North Ave and Oak St (others keep the idealized profile). The world page blends the SWAN library for the current offshore condition, uses the real depth for breaking, and can show the FUNWAVE wave field over the Diversey Harbor entrance.

To serve these from the deployed service, copy `bathy.json`, `swan.json` and `funwave.json` from those branches into `DATA_DIR` (they are served at `/api/bathy.json`, `/api/swan.json`, `/api/funwave.json`), and set `BATHY_PATH` for the forecast.

Assumptions to know: the NOAA chart's interpolated nearshore cells are floored at 1.5 m; SWAN treats obstacles as fully opaque with 35–50% reflection (85% for known vertical walls: Navy Pier, Monroe Harbor's seawall and inner breakwaters; 50% for Diversey Harbor's surveyed quay walls, 25% for the lagoon's banks); the survey's depths are whole feet below the May 2023 surface, so the sill under the Fullerton bridge (1–2 ft that day) goes nearly dry at low water.

**Diversey Harbor on the page**: the survey's shoreline polygons are the water mask (`bathy.survey.polys`): inside them the water is certain, within 30 m outside no map heuristic may add water, and the edge is drawn hard (no blur, no shoreline noise). The harbour's waves come from the harbor SWAN domain: the lake's sea through the mouth plus the wind's own chop (the Water card's *Harbor chop* and *Through the mouth* rows, with the fetch), and the Diversey launch rating uses them. *Satellite → Aerial 2024* drapes Cook County's 2024 orthophoto (6 in, `app/ortho.py`, one ~2 MB JPEG at 0.55 m beside the page, refreshed monthly) over the harbour and the lagoon. The **lagoon's 1000 m rowing course** (Lincoln Park Boat Club's Chicago Sprints) is drawn as a straight 1000 m line from the south face of the Fullerton bridge down the lagoon (at least ~10 m off either bank) with a mark every 100 m (`tools/lagoon_course.py`; Settings → *Lagoon 1000 m course*). `node tools/page_probe.mjs --site <built page dir> --harbor` checks a built page's harbour edge headless (Playwright) and takes screenshots.

## El Niño tracker (Experimental → El Niño)

`app/enso.py now` runs in every snapshot: NOAA CPC's ONI, RONI, weekly Niño 1+2/3/3.4/4 and SOI, NOAA PSL's MEI v2, the
newest OISST equatorial Pacific anomaly map, the weekly Niño-3.4 trend (4 and 12 weeks), the current event against the
strong ones since 1979 (1982-83, 1997-98, 2015-16, 2023-24) season by season, and the cold season so far at O'Hare.
`app/enso.py climate` (the *El Niño composites* workflow, monthly → `enso` branch) classes the winters since 1979 by
their Dec-Feb ONI and compares them off Chicago: O'Hare temperature, nights ≤ 0 °F and snow (NCEI GHCN-D), Lake
Michigan's maximum ice (GLERL), days with ice at the Diversey nearshore cell (GLERL gridded ice), December lake
temperature (GLSEA), Calumet Harbor's Dec-Apr level change and ERA5 gale hours at buoy 45198. In strong El Niño winters
O'Hare ran about 5 °F warmer with a third of the sub-zero nights, the lake's ice peaked near 23% against 37%, the Chicago
shore had ~13 ice days against ~33 (open water and waves through more of the winter), and onshore gales were more
frequent. *View → El Niño* draws the gridded difference (Jan-Mar ice, strong or any El Niño; December water) on the lake.
The El Niño symbol (`ensoIcon()` in the page: a warm sun-disc riding a Pacific swell with an eastward arrow) marks the
tracker, the layer's legend, Upsets cards that fit the El Niño winter pattern (southerly gales, rain events, Nov-Mar)
and an "El Niño-mild" pill on the Air card when an El Niño cold-season day runs 5 °F or more above usual.

## Wind, the jet stream and the paddling light

- **Wind & gusts layer** (View → Wind & gusts): NOAA HRRR 3 km hourly 10 m wind and surface gust (`app/hrrr.py`, NOMADS grib
  filter, no key) over the radar map's region, turned from HRRR's Lambert grid to true north and resampled to a 3 km grid in
  the page's frame. Streaks flow with the wind over whatever you look at, coloured by speed; gust patches drift downwind
  and make the streaks surge and brighten as far as HRRR's gust over its mean wind. Observed hours are nudged to the
  lakefront stations and buoys (each station's miss spread over ~20 km). Outside HRRR's hours the streaks follow the readout.
- **Jet stream** (Radar map card → Jet stream): NOAA GFS 1° 250 hPa winds over North America (`app/jet.py now`), every 3 h
  to +48 h: bands shaded by speed with flowing streaks, the nearest jet streak's core speed, and Chicago's quadrant
  (under the core, left/right × entrance/exit). Outline: Natural Earth 1:50m (`tools/na_outline.py`).
- **Jet in the model** (`app/jet.py climate`, monthly on the enso branch): NCEP/NCAR reanalysis 6-hourly 250 hPa
  (2016-2025) through the same quadrant metrics against hourly gusts at CHII2, 45174 and 45198; wind-stratified ratios
  of the gust factor and of the chance of a 30 mph gust, day-block bootstrap with a Bonferroni-corrected 90% interval,
  and no change under 3% is applied. 2016-2025 found **no quadrant that makes the lakefront gustier than the surface
  wind already says** (gust factor 1-2% lower under any jet streak; 30 mph gusts 8% less often under the core), so the
  gusts and the Upsets are unchanged; the hook re-reads the study every month.
- **Subtropical jet and El Niño** (Experimental → El Niño): the latitude of the 250 hPa zonal-wind maximum 15-38° N over
  120-80° W, monthly 1948+, by winter ENSO phase: strong El Niño winters 28.3° N, La Niña 34.0° N (r = −0.62 with the
  Dec-Feb ONI); the next 48 h from GFS against the month's 1991-2020 normal.
- **Paddling light** (compass box): green / yellow / red for the spot at the view's centre and the timeline's time; the
  worst factor wins. Go needs local chop under 70 J/m², wind under 7 mph, gusts under 16, offshore (SW-NW) wind under 9,
  water 50 °F or warmer, no rain, no NWS advisory, daylight. Stay off at rough chop (≥160 J/m²), wind ≥16, gusts ≥21,
  offshore ≥13, water under 40 °F, a thunderstorm core on radar, or an NWS warning / small craft advisory. Tap it for the
  reasons and the full table (`PADDLE` in `world.html`).

## Learned correction and scorecard (experimental)

The wave model's misses are fairly consistent (too high in some winds, too low in others), so it can learn from the buoys:

- **Backfill** (`Learn waves` workflow, `python -m app.learn backfill`): reruns the open-lake model over years of NDBC archive at Wilmette (45174) and Chicago (45198), driven by observed wind only, and keeps every hour's model-vs-buoy pair (`learn` branch, `backfill/`).
- **Fit** (`python -m app.learn fit`): wave height is a small regression (model output statistics) of the buoy's height on the model's height and how that varies with wind direction, the wind direction, the 10 m wind speed and the swell share. Period is a least-squares factor by wind direction × speed. Scored by leaving one year out at a time (`correction.json`, `report.txt`), with the model's timing lag reported too. 45174 read about 20% higher waves through 2024 than from 2025 on, for the same wind (45198 didn't change), so its earlier years stay in the backfill but are left out of the fit. On held-out years (45198 2021–2026, 45174 2025–2026) the first fit cut the typical error from 0.24 to 0.18 m and on 1 m+ hours from 0.55 to 0.38 m; a plain direction × speed factor gained almost nothing.
- **Use**: when `LEARN_PATH` points at `correction.json`, the correction is worked out hour by hour at the buoys (from the model's sea there) and carried to each spot from its nearer buoy as a factor (0.33–3), like the live buoy nudge, since the regression's wind terms belong to the buoys' fetch. `sup.json` gets a `learned` version of every spot-hour (offshore and paddle-zone waves, period, rating), and the live buoy nudge fades toward the learned model instead of the raw one. On the page: Models & accuracy → *Learned correction (experimental)*, off by default.
- **Scorecard**: every snapshot logs its 48 h forecast at both buoys (model, model + learned, the buoy-nudged forecast with and without it, the NWS wave grid, and NOAA's Great Lakes Wave model GLWU from NOMADS when it's reachable) to `scorecard/forecasts-YYYY-MM.jsonl`, then scores the last 45 days against what the buoys measured by lead time (`scorecard.json`, shown under the toggle).
- **Photo spot checks** (`tools/spotcheck.py`): each snapshot also logs the next 6 h of every beach's forecast to `scorecard/spots-YYYY-MM.jsonl`. A photo from the lakefront (time and place from the phone's photo info) becomes one line in `spotchecks/log.jsonl`: what the water looked like (glass … very rough), the newest beach forecast issued before the photo, and whether they agree; a small copy of the photo with its metadata stripped goes in `spotchecks/photos/`. It isn't fed to the model. `python tools/spotcheck.py report` gives the running tally, the only check so far on the paddle-zone ratings themselves.

The buoys are offshore (~9–15 m deep) and come out for the winter, so the correction is learned on open-lake waves only; the paddle-zone numbers downstream of it are still unverified.

## Google Earth Engine layers

With a service-account key in the repo secret `GEE_SERVICE_ACCOUNT_KEY` (a Cloud project registered for Earth Engine; the account needs the Earth Engine Resource Viewer and Service Usage Consumer roles), the snapshot adds `earth.json` (`app/earth.py`). Each part is optional, and without the key the step is skipped:

| Layer | Source | On the page |
|---|---|---|
| Water temperature | Newest clear-sky pixel from the last 72 h of MODIS LST (Terra + Aqua, day + night, 1 km), VIIRS LST (Suomi NPP VNP21A1, day + night, 1 km), Landsat 8/9 thermal (100 m) or ECOSTRESS (ISS, 70 m, passes at varying hours), NOAA GLSEA elsewhere; JRC surface-water mask; 0.005° grid | Replaces GLSEA as the water-temperature layer (View → Water temp) and the water number |
| Upwelling | That grid: water 0–2 km from shore vs 8–15 km out, rows 41.85–42.06° N | "Upwelling" pill when the shore water is 2.5 °C (≈4.5 °F) or more colder |
| Wind on the water | Sentinel-1 VV backscatter (newest pass, ≤12 days), relative to the pass's median | View → *Wind on the water*: dark = sheltered/calm, bright = wind-roughened |
| Water clarity | Sentinel-2 red-band turbidity (Nechad), water pixels from the newest clear pass (≤10 days) | View → *Water clarity*; FNU at the view point and the launches |
| Clouds now | GOES-19 10.3 µm vs the GFS surface temperature (plus visible reflectance by day): a ~2 km cloud mask and a cloud-top height from the lapse rate | Live mode within 45 min of the image: the sky layer follows the satellite's cloud field (also over radar, which sees only clouds that rain) at that height |
| Second-opinion wind | NOAA GFS 0.25°, newest complete run, 48 h at North Ave and both buoys | "GFS model wind" in the readouts; the scorecard runs the wave model on it ("model on GFS wind") |
| Normals | ERA5 hourly 1991–2020 (±10 days, cached on the `learn` branch), ERA5-Land lake temperature | "Windier than 9 in 10 hours", air/water "vs usual" pills, usual wind |

Each new Sentinel-1 and Sentinel-2 pass's values at the launches are appended to `sat/` on the `learn` branch, to pair radar roughness with the wind at the time (how sheltered each launch is by wind direction) and clarity with storms. WeatherNext 3 (`wn3.json`), when the key has access, shows its wind with the 10–90% ensemble spread and is scored like GFS.

The *Earth Engine probe* workflow lists which datasets the key can read and how fresh they are. `app/weathernext.py` (Google DeepMind WeatherNext 3, experimental) runs in the snapshot too but needs separate dataset access for the service account; it takes the newest hourly run for the first 48 h and extends it with the newest 6-hourly run, which goes out 15 days (`WN3_HOURS`); that carries the LakeView Upsets tab past the NWS hourly forecast.

**ERA5 in the learned correction**: the *Learn waves* workflow caches ERA5 hourly wind at the open-lake cells next to both buoys (1991 on, `era5/` on the `learn` branch). The backfill fills hours with buoy waves but no measured wind with ERA5 wind scaled to read like the buoy (ratio by direction sector over the overlap hours), and runs the wave model on 35 years of ERA5 wind alone (`era5_climate.json`): how many big-wave hours each wind direction brings and how few of them the buoys were out for, i.e. where the fit extrapolates. The fit report adds held-out skill on north-wind hours with 1 m+ waves, and the hindcast's monthly wave percentiles become the page's "usual lake waves". ERA5 adds wind, not wave observations, so it can't by itself teach the correction about storms the buoys missed.

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
