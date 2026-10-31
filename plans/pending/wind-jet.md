# Wind & gusts layer, jet stream view, jet in the model

Asked 2026-10-30: "add layer: visualize the wind & gusts dynamically / add option: toggle jet stream- build jetstream into model".

## Decisions (user, gate 1)
- Wind data: HRRR 3 km hourly, 10 m wind + surface gust, fetched with each snapshot from NOMADS (no key).
- Display: animated streaks flowing with the wind, coloured by speed, brightening in gusts; follows the timeline;
  buoys and stations nudge it in observed hours.
- Jet display: on the Radar map card only: 250 hPa over North America as flowing bands, core speed, Chicago's position.
- Jet in the model: (a) gust and storm risk, a jet streak overhead / left-exit / right-entrance raises the gust factor
  and flags Upsets events, quantified against past hours at the buoys before it changes any number; (b) El Niño link,
  the subtropical jet's latitude in the El Niño tracker.

## Data
- HRRR: filter_hrrr_2d.pl, UGRD/VGRD 10 m + GUST surface, box 41.1-42.8 N, 88.5-86.7 W (the radar map's region).
  Newest complete cycle f00-f18 (f00-f48 on 00/06/12/18), hours 19-48 from the newest synoptic cycle, the past 12 h
  from each earlier cycle's f01. Winds are grid-relative on the Lambert grid: rotated to earth-relative
  (angle = sin(38.5°)·(lon − (−97.5))). Resampled to a 3 km grid in the page's local km frame. → wind.json.
- GFS 1° 250 hPa u/v, 15-65 N, 140-50 W, newest cycle f000-f048 every 3 h + the cycle 12 h earlier f000-f009. → jet.json,
  with per-time Chicago metrics: speed overhead, polar-jet axis latitude and distance, streak (local max ≥ 50 m/s
  within 1500 km) and Chicago's quadrant relative to it (entrance/exit × left/right).
- History for the quantification: NCEP/NCAR reanalysis 6-hourly 250 hPa u/v (PSL OPeNDAP, 2.5°) 2015-2025, with
  NDBC hourly WSPD/GST at Chicago-area stations; gust factor G/U and P(gust ≥ 30 mph) by jet category.
  The adjustment is applied only where the bootstrap 90% interval excludes "no effect".
- STJ: NCEP R1 monthly 250 hPa u, 1979+, latitude of the zonal-wind max 15-40 N averaged 120-80 W, winter (DJF)
  composites by ONI phase; the current month from the GFS analyses. → enso branch (jet_climate.json).

## Page
- Wind layer: toggle button; CPU-advected particle trails (three.js LineSegments) over the view, from the HRRR hour
  pair around the timeline time, coloured by speed (mph ramp), bright and faster where gust/wind is high; observed
  hours nudged by station vectors (Gaussian weight, 20 km). Legend entry.
- Radar map card: "Jet" button switches the card to North America with flowing 250 hPa bands, core speed and
  Chicago's quadrant, following the timeline.
- Conditions: gust factor × the measured jet effect (when significant), noted in the source line. Upsets: "Jet streak"
  flag over hours in a significant quadrant. El Niño tracker: STJ latitude panel.

## Decision log
- 2026-10-30: plan written; gate answers above.
- 2026-10-31: built. HRRR rotation checked against O'Hare/Midway obs (NE both). Gust study 2016-2025, 13,917 samples:
  gust factor 0.978-0.991 of baseline in every quadrant, 30 mph gusts ×0.92 under the core; nothing raises gusts →
  no numbers change (the readouts and Upsets hooks stay wired to the monthly study). Multiple-comparison correction
  added after a single marginal right-exit result (×1.08, 90% lo 1.004) appeared. STJ: strong EN 28.3° N, LN 34.0° N.
- 2026-10-31: user asked mid-build for a green/yellow/red paddling status in the compass box with the criteria shown;
  proposed and built. User set: go needs wind < 7 mph; water < 50 °F caution, < 40 °F stay off.
