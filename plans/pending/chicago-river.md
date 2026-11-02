# The Chicago River: lake → lock → main stem, North Branch to Belmont, South Branch to Ping Tom

Asked 2026-11-02: "expand our project into the chicago river; require higher detail from lake michigan -> lock -> river,
north branch up to belmont avenue; south until tom ping park".

## Decisions (user, gate 1)
- One world, extended: LakeView's area grows west to the North Branch at Belmont and south to Ping Tom Park.
- Detail: 2 m river bottom (USGS 2022/2023 surveys), bridges as 3-D decks, the Riverwalk and walls at 1 m.
- Physics: current from the USGS gages, sewer overflows (CSO) in the paddling light, boat wakes and traffic by
  hour, wind in the canyon.
- Extent: both sides of Goose Island (North Branch + North Branch Canal), the North Ave turning basin, Ogden Slip,
  the South Branch past Ping Tom's lagoon.

## What the research found (2026-11-02)
- **River bottom**: USGS ScienceBase LAZ point clouds (0.5 m, EPSG:6345, NAVD88 m, no login). 2022 file covers the
  lock → main stem → South Branch (past Ping Tom) and the North Branch through both Goose Island channels to the
  North Ave turning basin (56 MB reduced); 2023 file covers North Ave → Belmont and beyond (16 MB). Read with laspy +
  lazrs; the 2023 file carries no CRS, assign 6345. Water surface Aug 2022 ≈ 175.9 m NAVD88.
- **Lidar**: the 2020 USACE lakeshore DEM stops at Columbus Dr. Inland: USGS 3DEP 1 m bare earth (IL 4-county 2016,
  NAVD88) by ImageServer exportImage, max 8000 px a call; Cook County DEM2022 (1 ft, EPSG:6455) as an alternative.
  No DSM, so bridge decks come from the road elevation at each abutment (bare earth includes road surfaces).
- **Buildings**: OSM heights are 0.4% complete there. Cook County buildingFootprint_2022 (lidar Height in ft,
  74k footprints in the box, paged 2000 a call) is the source; render those within ~400 m of the river or ≥ 30 m
  tall, footprints simplified.
- **OSM** (Overpass via maps.mail.ru mirror; overpass-api.de resets through the sandbox proxy): river water as
  multipolygon relations (13730103 main stem, 3485552 North Branch, 13462125 + 17398850 South Branch), the canal
  way 382710108, Ogden Slip 401242573; 146 movable-bridge ways (140 bascule), layer set, no clearance tags; the
  lock way 164030677 with lock_gate nodes; 166 Riverwalk ways; river edges tagged man_made=quay (167).
- **Gages (USGS NWIS IV, JSON)**: 05536123 Columbus Dr: stage (CCD ft) + water velocity 72254 (ft/s, positive = away
  from the lake; 30-day mean 0.01 ft/s, only large sustained negatives are a reversal); 05536118 Grand Ave and
  05536121 the lock: stage only. No live discharge anywhere on the reaches; velocity × section gives the flow.
- **CSO**: MWRD "Possible CSO Events" ArcGIS layer (public JSON, rows only while an event is open, by
  WATERWAY_REACH); outfall inventory FeatureServer (441 points, reach names: Chicago R, NBCR Lower/Upper,
  SB Chicago R, SF SB Chicago R); reversals PDF (last 2023-07-02/03). Alerts otherwise e-mail/SMS only.
- **Lock**: USACE NTNI notices JSON (district LRC): winter closure 15 Nov – 15 Apr, nightly closures; no live
  open/closed feed. Rules 33 CFR 207.420: green light, hold the wall ropes, commercial first, VHF 14.
- **Traffic** (no free live AIS): main stem Mar 15 – Nov 23, 9–21 h, 6–10 tour/taxi passes an hour each way at
  peak; South Branch water taxi every 15–30 min 7–19 h; North Branch tours to Goose Island 10–18, taxi Fri–Sun;
  above Chicago Ave rare (barges 2–3 a day); Nov–Mar near zero. Whole river no-wake. Volpe 2000 counts.
- **Current**: Columbus Dr annual mean 125 cfs (≈ 0.01 m/s); summer 300–465 cfs; reversal record −2,450 cfs
  ≈ 0.2 m/s (0.4 mph). Normal current is ≤ 0.1 mph.
- **Canyon wind**: no Chicago measurements; Oke/Nakamura skimming regime: along-axis k ≈ 0.7 (1.3–1.5 at gaps and
  tower corners), cross-axis k ≈ 0.2–0.5, oblique interpolated; gust factor 1.5 near corners.

## Page limits
- The world mask texture is capped at 3584 px wide (2560 low-memory); the extended box (11.5 × 14 km) at 4096 px is
  2.8 m per pixel. Walls therefore come from geometry, not the mask: quay strips with lidar heights (1 m), the
  water polygon's hard edge in the looks alpha as the harbor does today.
- Buildings are one merged mesh; the Loop adds ~10k footprints: LOD by distance, simplified footprints.
- First load grows (bottom +~3 MB, map +~2 MB, buildings +~3 MB): split files, the river bottom as its own side file.

## Plan (one PR per phase, each republished)
1. **Data** (`app/river.py`, bathymetry, lakefront): extend MAP_BBOX / LB to (41.850, -87.700) – (41.990, -87.560);
   river bottom grid at 2 m from the two LAZ files (NAVD88 → IGLD85 → the page's LWD frame; seam at North Ave),
   src code 6 "USGS river survey"; 3DEP 1 m bare earth for the banks, bridge abutments and the Riverwalk; Cook County
   footprints with heights near the river; OSM river water, canal, slip, lock, quays, Riverwalk, bridges (river
   crossings only). Tests on the parsers and the datum chain.
2. **Geometry** (world.html): the river as water in the mask (class 4 "river") with hard edges; quay walls as strips;
   bridge decks (deck at the abutment road height, piers, bascule leaves closed) and the lock walls/gates; buildings
   with lidar heights; Go-to views (the lock, Wolf Point, Goose Island, Ping Tom); launches (Urban Kayaks Riverwalk,
   Kayak Chicago North Ave, Ping Tom boathouse). Page probe checks of the mask at the river edge and screenshots.
3. **Physics and the light**: current along each reach from the Columbus Dr velocity (sign = direction; scaled by
   section area per reach), drawn as streaks on the water; canyon wind k(θ) per reach from the building heights
   along it, feeding the chop; wake chop from the traffic table by reach/season/hour; CSO: the MWRD layer polled by
   the snapshot and by the page on load, red for the reach with an open event and 48 h after, caution from forecast
   rain (≥ 0.5 in / 6 h); lock: closed Nov 15 – Apr 15 (NTNI), reversal = red at the lock reach. The paddling light
   gains the river rows; Upsets gains CSO and reversal events. Conditions in the river use the river's own wave
   model: no lake waves west of the lock, chop from canyon wind + wakes.
4. **Finish**: README, plan archive, the Outlook unchanged (its view is the harbor), performance check on a phone-size
   viewport.

## Open assumptions (say if any is wrong)
- Bridge decks drawn closed (no lifts); deck height = bank road elevation; clearance shown from that minus stage.
- Rendering the Loop at full building density is not required: footprints within 400 m of the river plus towers.
- CSO history: only what the snapshot sees from now on (the events layer keeps no history; the synopsis reports
  are not machine-readable), so "48 h after" works from first sighting.
- The first load of LakeView grows by roughly a third.

## Decision log
- 2026-11-02: plan written; gate-1 answers above.
