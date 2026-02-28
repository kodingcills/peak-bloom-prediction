# TECHNICAL_DEBRIEF.md

## Executive Summary
Phase 1 stabilization patch v2.0 was applied with rate-limit hardening, per-site incomplete markers, and refresh pre-downstream completeness checks. Cold-start now aborts cleanly before labels/features/gates when ERA5 is incomplete, with actionable site summaries. Phase 1 gates currently fail because no silver weather artifacts were produced in this sandbox due DNS/network resolution failures.

## Files Changed
- config/settings.py
- src/ingestion/era5_fetcher.py
- refresh_data.py
- .env.example

## Evidence Anchors

### New env vars in settings
- Code: config/settings.py:58 OPENMETEO_DELAY_SECONDS
- Code: config/settings.py:59 OPENMETEO_MAX_ATTEMPTS
- Code: config/settings.py:64 OPENMETEO_429_MIN_SLEEP_SECONDS
- Code: config/settings.py:70 OPENMETEO_429_MAX_SLEEP_SECONDS

### 429 handling + clamp behavior
- Code: src/ingestion/era5_fetcher.py:51 _fetch_json
- Code: src/ingestion/era5_fetcher.py:63 `_clamp(..., OPENMETEO_429_MIN_SLEEP_SECONDS, OPENMETEO_429_MAX_SLEEP_SECONDS)`
- Code: src/ingestion/era5_fetcher.py:114 non-429 backoff clamp to 60s
- Log: `python refresh_data.py` output includes
  - `Open-Meteo request failed (attempt 1/3)... Sleeping 2.43s...`
  - `Open-Meteo request failed (attempt 2/3)... Sleeping 4.33s...`
  - `Open-Meteo request failed (attempt 3/3)... Sleeping 8.20s...`

### _INCOMPLETE marker behavior
- Code: src/ingestion/era5_fetcher.py:150 incomplete_path definition
- Code: src/ingestion/era5_fetcher.py:179-183 marker write on chunk failure
- Code: src/ingestion/era5_fetcher.py:216-217 and 223-224 marker removal on full completion
- Artifact:
  - data/silver/weather/washingtondc/_INCOMPLETE.txt (514B)
  - data/silver/weather/kyoto/_INCOMPLETE.txt (506B)
  - data/silver/weather/liestal/_INCOMPLETE.txt (507B)
  - data/silver/weather/vancouver/_INCOMPLETE.txt (512B)
  - data/silver/weather/nyc/_INCOMPLETE.txt (505B)

### refresh_data early-exit checks
- Code: refresh_data.py:44 _validate_era5_inputs
- Code: refresh_data.py:78 _validate_asos_inputs
- Code: refresh_data.py:145-157 ERA5 completeness summary + sys.exit(1)
- Code: refresh_data.py:173-184 ASOS completeness summary + sys.exit(1)
- Log: `python refresh_data.py` output includes
  - `ERA5 completeness check failed for required sites:`
  - per-site lines with `consolidated_exists` and `incomplete_marker`
  - `ERA5 ingestion incomplete. Aborting Phase 1 before downstream steps.`

## Command Transcript + Outcomes
1. `python -m compileall .`
   - Outcome: success (compiled project tree).
2. `python -c "import config.settings as s; print('loaded settings ok', s.OPENMETEO_MAX_ATTEMPTS, s.OPENMETEO_DELAY_SECONDS)"`
   - Outcome: `loaded settings ok 3 1.5`.
3. `rm -rf data/silver data/processed data/gold && python refresh_data.py`
   - Outcome: failed cleanly with exit code 1 after ERA5 completeness failure; labels/features not executed.
4. `python refresh_data.py --step era5 --sites kyoto`
   - Outcome: failed cleanly with exit code 1; kyoto reported with `_INCOMPLETE` marker.
5. `python -m src.validation.run_all_gates --phase 1`
   - Outcome: failed immediately at `assert_inference_cutoff_utc` because silver weather artifacts are missing.

## Artifact Inventory (current run)
- Produced failure markers:
  - data/silver/weather/washingtondc/_INCOMPLETE.txt (514B)
  - data/silver/weather/kyoto/_INCOMPLETE.txt (506B)
  - data/silver/weather/liestal/_INCOMPLETE.txt (507B)
  - data/silver/weather/vancouver/_INCOMPLETE.txt (512B)
  - data/silver/weather/nyc/_INCOMPLETE.txt (505B)
- Missing due network failure:
  - data/silver/weather/*/*_consolidated.parquet
  - data/gold/features.parquet

## Gate Results
| Gate | Result | Evidence |
|---|---|---|
| assert_inference_cutoff_utc | FAIL | `python -m src.validation.run_all_gates --phase 1` -> `assert_inference_cutoff_utc: FAIL` |
| assert_historical_window_end | NOT RUN | runner stopped on first failure |
| assert_labels_complete | NOT RUN | runner stopped on first failure |
| assert_seas5_members | NOT RUN | runner stopped on first failure |
| assert_silver_utc | NOT RUN | runner stopped on first failure |
| assert_gold_schema | NOT RUN | runner stopped on first failure |

## Minimal Fix (spec-compliant)
Run Phase 1 in an environment with outbound DNS/network access so ERA5 and ASOS artifacts can be produced, then re-run:
1. `python refresh_data.py`
2. `python -m src.validation.run_all_gates --phase 1`
