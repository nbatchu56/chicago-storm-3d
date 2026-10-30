# Diversey Harbor and the lagoon: rigid walls, surveyed bottom, aerial imagery, a harbor wave model, the 1000 m course
Ticket: none (requested in chat, 2026-10-30)

## Why
The harbor's water edge on the page comes from OpenStreetMap (which has no water polygon for Diversey Harbor or the
Lincoln Park South Lagoon), patched by lidar heuristics, floating-dock carving and hand-typed strokes, then blurred
~5 m and dithered with noise by the ground shader. The harbor has rigid walls that have not moved; the edge should be
exact. The lagoon reads as dry land (no lidar return through its water; NCEI never surveyed it). Neither SWAN nor
FUNWAVE covers the inner basin or the lagoon, so harbor water is drawn as "sheltered: 25% of the lake" with a fixed
300 m fetch chop.

## Sources found (verified live 2026-10-30)
- **USGS / Cook County 2023 single-beam echo-sounder survey** (DOI 10.5066/P1VVHKFP, CC0; "Bathymetry and Water-Quality
  Data of Surface Water Bodies in Cook County, Illinois, 2023", v2.0 Mar 2025). Served by Cook County GIS without a key:
  - water-body polygons: `https://gis.cookcountyil.gov/traditional/rest/services/bathmetrySurveyData/MapServer/1`
    ("DIVERSEY HARBOR" 41.4 ac, 1034 vertices, lat 41.9165-41.9330: the basin, the Fullerton channel and the lagoon
    to the zoo; "NORTH POND"; "SOUTH POND")
  - 2 ft contours: `.../bathmetrySurveyData/MapServer/0` (fields contour_ft, collection_year)
  - depth raster, 3 ft pixels, U8 feet: `https://gis.cookcountyil.gov/imagery/rest/services/bathymetric_survey_3m_depth_surfaces_raster/ImageServer`
    (EPSG 3435; export at native resolution in that SR; an export resampled to 4326 showed row artefacts)
  - sampled: harbor 9-24 ft (20-24 ft only near the entrance), **Fullerton bridge channel 31-35 m wide and 2-3 ft deep
    (a sill)**, lagoon 8-11 ft just south of the bridge, 5-10 ft mid, 2-6 ft at the south end.
- **Cook County 2024 orthophoto**, 0.15 m (0.5 ft) RGB+NIR, `https://gis.cookcountyil.gov/imagery/rest/services/CookOrtho2024/ImageServer`
  (exportImage, max 15000x4100, anonymous, CORS reflected; no copyright/licence text; also CookOrtho2025).
- 2020 USACE lidar (already used): harbor bed 21-24 ft near the entrance where it has returns; the lagoon is nodata.
- Rowing course: Lincoln Park Boat Club's Chicago Sprints run 1,000 m "between Lake Shore Drive and the zoo"; the lagoon
  is ~1,040 m along its axis from the Fullerton bridge (41.9256) to the south end (41.9165), 47-72 m wide.

## Phases

### Phase 1: the survey in the lake bottom (Python)
Verified when: `bathy.json` built on the branch has the harbor basin, the Fullerton sill and the lagoon as water with the
survey's depths (sill 0.6-0.9 m, lagoon mid 1.5-3 m, entrance 6-7 m), source code 4, and the tests pass.
Files to touch: `app/cookcounty.py` (new), `app/bathymetry.py`, `tools/build_world.py`, `tests/test_models.py`, `README.md`

#### Step 1.1: fetch the survey (Size: S)
- Acceptance: `app.cookcounty.fetch(bbox)` returns the water-body polygons (local metres, holes kept) and the depth
  raster at native 3 ft in EPSG:3435 resampled to a requested local grid; cached to a directory (`COOK_DIR`) like ISGS.
- Verification: `pytest tests/test_models.py -k cookcounty` (synthetic raster + polygon fixtures; no network)
- Files: `app/cookcounty.py`, `tests/test_models.py`
- Test seams: the module's pure functions (`to_local`, `rasterize`, `sample_grid`)
- Blocked by: none

#### Step 1.2: blend the survey into the grids (Size: M)
- Acceptance: inside a survey polygon the bed is `level_survey - depth` (source 4), overriding lidar (boats and docks
  read as land there) and NCEI; a new 2 m model grid "harbor" (41.912,-87.641,41.937,-87.625) is built with `--model`;
  `bathy.json` carries `survey: {polys, year, level_m_igld85, source}`.
- Verification: `pytest tests/test_models.py -k "blend or survey"`; then the Bathymetry workflow on the branch and a
  probe of the published grid at the sill / lagoon / entrance points above.
- Files: `app/bathymetry.py`, `tests/test_models.py`
- Test seams: `blend()` with a synthetic survey; `build_grid` through a fake fetcher
- Blocked by: 1.1

#### Step 1.3: the page's bathy file keeps the survey (Size: XS)
- Acceptance: `thin_bathy` keeps `survey.polys` (and `crop_detail` keeps source-4 cells in the 4 m band).
- Verification: `pytest tests/test_build_world.py` (new, small)
- Files: `tools/build_world.py`, `tests/test_build_world.py`
- Blocked by: 1.2

### Phase 2: rigid walls on the page
Verified when: on the page, 3 m inside the survey polygons the mask reads harbor water (class 2) at >= 99% of sampled
points and 3 m outside (land by the ortho) it reads land at >= 95%; the edge is drawn without blur or noise; the
hand-typed Diversey channel strokes are gone.
Files to touch: `app/static/world.html` (buildMap, ground shader looks.a, WATERWAYS/WATER_AREAS, legend)

#### Step 2.1: survey polygons become the water mask (Size: M)
- Acceptance: `buildMap` rasterises `BATHY.survey.polys` as class 2 before the heuristics; inside a 30 m band outside
  the polygons the heuristics (lidar fix, dock carving, waterways, specks) cannot add water; the LSD inlet and the mouth
  (outside the polygon) keep their WATERWAYS/WATER_AREAS entries, the two "Channel south of Diversey Harbor" strokes go.
- Verification: `node tools/page_probe.mjs --harbor` (new Playwright probe: loads `/world` headless, samples `maskAt`
  along the polygon offsets, prints the two percentages; fails under 99/95)
- Files: `app/static/world.html`, `tools/page_probe.mjs` (new)
- Blocked by: 1.3

#### Step 2.2: a hard edge (Size: S)
- Acceptance: the looks texture's unused alpha marks a hard edge (255) within ~8 m of a survey polygon; there the wet
  fraction is written exact (no blur) and the shader's `edgeN` noise is multiplied by `1 - looks.a`; the near-camera
  water patch respects the same band.
- Verification: screenshot `tools/page_probe.mjs --shot harbor` viewed by eye (before/after in the PR)
- Files: `app/static/world.html`
- Blocked by: 2.1

### Phase 3: aerial imagery layer
Verified when: a "Aerial 2024" button in the Satellite section drapes the Cook County orthophoto over the harbor and
lagoon box (land and water), the ortho is a side file in the snapshot refreshed at most monthly, and the page credits it.
Files to touch: `app/ortho.py` (new), `tools/build_world.py`, `.github/workflows/snapshot.yml`, `app/static/world.html`

#### Step 3.1: fetch and cache the ortho (Size: S)
- Acceptance: `python -m app.ortho --out data/ortho/diversey.jpg --meta ortho.json` exports the box
  (41.9150,-87.6400)-(41.9350,-87.6260) at 0.5 m as JPEG (~2-3 MB) with `{lat0,lon0,lat1,lon1,w,h,source,year,fetched}`;
  `--prev` reuses the previous snapshot's file when under 30 days old.
- Verification: `pytest tests/test_ortho.py` (URL building, reuse rule); one real export by hand, viewed
- Files: `app/ortho.py`, `tests/test_ortho.py`
- Blocked by: none

#### Step 3.2: the page drapes it (Size: M)
- Acceptance: `ortho-data` block lists the images; the button loads the JPEG into the shared satellite texture slot
  (no new sampler: the ground shader is at the 16-unit limit); inside the box land takes the ortho colour and water
  the ortho tinted like the other satellite layers; outside the box nothing changes; the note credits
  "Cook County GIS 2024 orthophoto".
- Verification: `tools/page_probe.mjs --shot ortho` screenshot; `pytest tests/test_build_world.py`
- Files: `app/static/world.html`, `tools/build_world.py`, `.github/workflows/snapshot.yml`
- Blocked by: 3.1

### Phase 4: harbor SWAN library (wind chop and lake penetration)
Verified when: `swan.json` has domain "harbor" (2 m, x -700..150, y 3750..6100 m) with the offshore cases (base+storm,
12 directions) and 96 wind cases; for 11 m/s from the N the basin centre shows Hs within 30% of the CEM fetch estimate
(~0.25 m) and the lagoon south of the sill shows < 5% of offshore Hs for every offshore case.
Files to touch: `app/swan_cases.py`, `.github/workflows/swan.yml`, `tests/test_models.py`

#### Step 4.1: domain and wind cases (Size: M)
- Acceptance: `DOMAINS["harbor"]` (grid "harbor", dx 2); `CASE_GROUPS["wind"]` = 24 directions x (4, 7, 11, 16 m/s);
  `swn()` for a wind case writes `WIND`, `GEN3`, no boundary spectrum, outputs HSIGN DIR TM01 RTP; `collect` stores wind
  cases as Hs cm (uint8), Tp ds (uint8) and dir with `kind: "wind"`; harbor walls come from the survey-blended bottom
  (dry cells), floating docks transparent; the mouth and inlet stay open (opening rule: >= 4 cells, so 8 m at 2 m).
- Verification: `pytest tests/test_models.py -k "harbor or wind"`; one timing run of a single case on Actions
  (`swan.yml` with domains=harbor, dirs=0, groups=w11) before the full matrix — the matrix shape is decided from it
- Files: `app/swan_cases.py`, `tests/test_models.py`
- Blocked by: 1.2

#### Step 4.2: workflow matrix (Size: S)
- Acceptance: `swan.yml` gains `domains` (default "trail,lane"); harbor runs as its own matrix (dirs x groups incl. wind);
  `merge` keeps the trail/lane library; the collect step handles the wind layout.
- Verification: the workflow run on the branch publishes `swan.json` with all three domains; probe values above
- Files: `.github/workflows/swan.yml`
- Blocked by: 4.1

### Phase 5: FUNWAVE through the mouth to the sill
Verified when: the diversey tile (y_south 4750 m, to the Fullerton sill) runs rough/storm/east/glass on the
survey-blended 2 m bottom and the inner basin's Hs is a small fraction of the offshore case; the page shows the tile.
Files to touch: `app/funwave_case.py`, `tests/test_models.py`

#### Step 5.1: tile and depth source (Size: S)
- Acceptance: `TILES["diversey"].y_south = 4750`; `depth_grid` prefers the finest model grid covering the tile (harbor
  2 m); dry cells inside the harbor come from the survey (no boat "land").
- Verification: `pytest tests/test_models.py -k funwave`; `funwave.yml` run (tiles=diversey) on the branch
- Files: `app/funwave_case.py`, `tests/test_models.py`
- Blocked by: 1.2

### Phase 6: the harbor's waves on the page
Verified when: inside the harbor the overlays (Wave height, Chop, Energy flow) and the Water card use the harbor
library: lake penetration from the harbor domain's offshore cases (the finest domain wins, as today) plus the wind
chop case blended for the current wind, combined in quadrature; the fixed "25% sheltered" factor and the 300 m fetch
chop no longer apply inside a harbor SWAN domain; the Diversey launch rating uses it; the Diversey Outlook kiosk shows
the same water.
Files to touch: `app/static/world.html`

#### Step 6.1: wind-chop library on the page (Size: M)
- Acceptance: `setSwan` reads `kind: "wind"` domains; `swanWindAt(xk, yk, u10, dir)` blends by direction (circular)
  and speed; `swanUpdate` writes the combined field (penetration ratio x Hs_off, wind Hs) into the one RGBA texture
  (R ratio-equivalent, G/B direction) so the shaders need no new sampler.
- Verification: `tools/page_probe.mjs --harbor-waves`: prints Hs at the basin centre for N 11 m/s vs the CEM number
- Blocked by: 4.2, 2.1

#### Step 6.2: conditions, launch rating, legend, About (Size: S)
- Acceptance: `conditions()` inside a harbor domain replaces the "sheltered basin" branch with the modelled systems;
  the Water card shows "Harbor: wind chop x ft (fetch), lake waves in y ft"; legends and the About note name the
  survey and the model; `calm` in the shaders follows the modelled Hs where the harbor library exists.
- Verification: screenshot + the probe; `pytest tests/test_service.py`
- Blocked by: 6.1

### Phase 7: the lagoon's 1000 m course
Verified when: a line runs from the south face of the Fullerton bridge 1000 m south along the lagoon centreline, ticks
every 100 m and labels at 250/500/750/1000 m, 0.3 m above the water, toggled in Settings (default on), with a
"Lincoln Park lagoon" camera preset.
Files to touch: `app/static/world.html`, `tools/lagoon_course.py` (new: centreline from the survey polygon -> waypoints)

#### Step 7.1: course line (Size: S)
- Acceptance: as above; the waypoints are a constant in the page (derived once by the tool, source noted).
- Verification: screenshot from the preset; waypoint spacing check in the tool's test
- Blocked by: 1.2 (polygon), 2.1

### Phase 8: ship
Verified when: CI green, the three model branches rebuilt from the branch, a snapshot built and both artifacts
republished by the refresh task show the new harbor; README and the plan's execution record updated.
Files: `README.md`, `plans/`

#### Step 8.1: workflows on the branch, snapshot, PR (Size: S)
- Blocked by: all

## Decision log
| Decision | Chosen option | Why |
| --- | --- | --- |
| Harbor outline source | USGS/Cook County 2023 survey polygons | The water body's surveyed shoreline is the wall; no tracing, no OSM dependence |
| Lagoon depth | The same survey's 3 ft depth raster | A real echo-sounder survey exists (2023); the uniform fallback is not needed |
| Imagery | Cook County 2024 ortho baked into the snapshot as a side JPEG | 0.15 m, no key, no live third-party fetch from the page; NAIP via Earth Engine is 0.3-0.6 m and needs the GEE secret |
| Harbor model grid | 2 m bathy model grid + 2 m SWAN domain | The inlet under LSD is ~16 m wide (8 cells at 2 m, 4 at 4 m, where SWAN's diffraction goes unstable) |
| Wind chop | SWAN wind-only cases on the harbor domain; lake waves from the same domain's offshore cases; quadrature on the page | One product of wind x offshore cases would be 10x the runs; the two seas are nearly independent inside |
| FUNWAVE | Extend the diversey tile south to the sill; no FUNWAVE in the lagoon | The 2-3 ft sill 1 km from the mouth stops lake waves; wind chop there is SWAN's |
| Page textures | No new samplers (ortho and wind field reuse existing slots) | The ground shader sits at the 16-unit limit |
| Harbor offshore cases | Run without SWAN's diffraction term | Every 2 m case blew up with it (0 and 3 smoothing steps alike) and fell back to a no-diffraction rerun, doubling the time; the 25° spread and reflections carry the sea through the mouth, FUNWAVE resolves the diffraction |
| The inlet under LSD | Known channels (page WATERWAYS strokes) cut into the lake bottom as source 5; obstacle lines split at them | The survey stops at the bridge's west face and the lidar returns the deck as land: every model had the harbour sealed (timing run: 0% inside) |
| Survey ring edges over water | Not walls | The polygon closes across the mouth where the sounder stopped |

## Not yet specified
- Phase 1: the lake level on the survey's dates (needed to turn depth into bed elevation): the USGS release's metadata
  or the 2023 open-water mean at Calumet Harbor; looked up at pickup.
- Phase 4: the SWAN matrix shape (runs per job) follows the single-case timing run.
- Phase 6: fog — how the near-camera patch shader takes the wind field (it reads the same SWAN texture; check its
  calm/shelter path at pickup).

## Always / Never
- Always: Diversey Harbor, the Fullerton channel, the South Lagoon; both LakeView and Diversey Outlook; tests for every
  Python change; the workflows run from the branch before the PR.
- Never: Belmont, Montrose or Monroe harbors; North Pond / South Pond as wave domains (they get the survey polygon and
  depth only); imagery outside the harbor box; a new texture sampler in the ground shader; changes to the radar pipeline.

## Progress
| Phase | Status |
| --- | --- |
| 1 survey in the lake bottom | done (bathy.yml run 41096808739 on the branch) |
| 2 rigid walls | done (probe: 99.2% inside / 99.4% outside, from 19% / 88%) |
| 3 aerial imagery | done (0.55 m JPEG, 2.0 MB; Aerial 2024 button) |
| 4 harbor SWAN | pending |
| 5 FUNWAVE to the sill | pending |
| 6 harbor waves on the page | pending |
| 7 1000 m course | done (21 waypoints from the survey polygon; ribbon, 100 m ticks, 250 m labels) |
| 8 ship | pending |
