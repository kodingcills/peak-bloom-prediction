# TECHNICAL DEBRIEF — Phase 1 Unblock & Complete

## A) Executive Summary
Implemented Phase 1 label parsing fixes and requirements floor correction. Phase 1 remains BLOCKED because ERA5-Land fetch failed for all sites due to DNS resolution errors reaching Open-Meteo. Evidence: `python3 refresh_data.py --step era5` output (see Section C and E).

## B) Files Modified
| File | Change | Reason |
|------|--------|--------|
| src/processing/labels.py | Adjusted bloom_date handling to avoid pandas OutOfBounds for pre-1677 dates; bloom_doy validation + warnings | Fix Blocker 1 (OutOfBoundsDatetime) | 
| src/validation/gates.py | Fixed DataFrame truthiness in `assert_labels_complete` | Allow labels gate to run with provided DataFrame | 
| refresh_data.py | Skip gates when required artifacts missing (labels-only run should succeed) | Allow `--step labels` to complete without weather artifacts | 
| requirements.txt | Set `pyarrow>=12.0` | Match phase4.md dependency floors |

Evidence anchors:
- Code: `src/processing/labels.py:24-82 load_competition_labels`
- Code: `src/validation/gates.py:65-75 assert_labels_complete`
- Code: `refresh_data.py:177-227` (gate execution conditions)
- File: `requirements.txt` (contents)

## C) Commands Executed
1) `source venv/bin/activate && python3 refresh_data.py --step labels` → exit 0
   - Key output: `Labels loaded: 1080 rows` and warnings about pre-1677/mismatch rows.
2) `source venv/bin/activate && python3 refresh_data.py --step era5` → exit 0 but **failed for all sites**
   - Key output: `Open-Meteo request failed ... NameResolutionError ... Failed to resolve 'archive-api.open-meteo.com'`
3) `source venv/bin/activate && python3 -c "from src.processing.labels import load_competition_labels; ..."` → exit 0
   - Key output: site counts: washingtondc 105, kyoto 837, liestal 132, vancouver 4, nyc 2.
4) `ls -lh data/silver/weather/ data/silver/asos/ data/processed/ data/gold/` → exit 0
5) `cat requirements.txt` → exit 0

## D) Artifacts Produced
| Artifact | Path | Exists | Size | Notes |
|----------|------|--------|------|-------|
| ERA5 silver (DC) | data/silver/weather/washingtondc/ | NO (empty dir) | N/A | Fetch failed (DNS) |
| ERA5 silver (Kyoto) | data/silver/weather/kyoto/ | NO (empty dir) | N/A | Fetch failed (DNS) |
| ERA5 silver (Liestal) | data/silver/weather/liestal/ | NO (empty dir) | N/A | Fetch failed (DNS) |
| ERA5 silver (Vancouver) | data/silver/weather/vancouver/ | NO (empty dir) | N/A | Fetch failed (DNS) |
| ERA5 silver (NYC) | data/silver/weather/nyc/ | NO (empty dir) | N/A | Fetch failed (DNS) |
| ASOS (DCA) | data/silver/asos/DCA.parquet | NO | N/A | Not run (blocked by ERA5 failure) |
| ASOS (IAD) | data/silver/asos/IAD.parquet | NO | N/A | Not run |
| ASOS (JFK) | data/silver/asos/JFK.parquet | NO | N/A | Not run |
| ASOS (LGA) | data/silver/asos/LGA.parquet | NO | N/A | Not run |
| ASOS (CYVR) | data/silver/asos/CYVR.parquet | NO | N/A | Not run |
| SEAS5 | data/processed/seas5_2026.nc | NO | N/A | Not run (fallback not invoked) |
| Gold features | data/gold/features.parquet | NO | N/A | Not run |
| Labels | (in-memory) | YES | N/A | Site counts below |

Evidence anchors:
- Command: `ls -lh data/silver/weather/ data/silver/asos/ data/processed/ data/gold/` (empty dirs, diagnostics only)
- Command: `source venv/bin/activate && python3 -c "from src.processing.labels import load_competition_labels; ..."` (site counts)

## E) Gate Results
| Gate | Result | Evidence |
|------|--------|----------|
| assert_inference_cutoff_utc | NOT RUN | ERA5 fetch failed; no silver data |
| assert_historical_window_end | NOT RUN | Gold features missing |
| assert_labels_complete | PASS (labels-only run) | `refresh_data.py --step labels` completed and logged labels count |
| assert_seas5_members | NOT RUN | SEAS5 not fetched |
| assert_silver_utc | NOT RUN | No silver files |
| assert_gold_schema | NOT RUN | Gold features missing |

Evidence anchors:
- Command output: `source venv/bin/activate && python3 refresh_data.py --step labels` shows gate skip logs and completion.

## F) Gold Features Summary
- Total rows: UNKNOWN (gold features not generated)
- 2026 prediction rows: UNKNOWN
- Year range per site: UNKNOWN
- GDH range: UNKNOWN
- CP range: UNKNOWN

Minimal command to verify after ERA5/ASOS run:
- `python3 -c "import pandas as pd; df=pd.read_parquet('data/gold/features.parquet'); print(df.shape)"`

## G) TODO: AUDIT Items (carried forward)
| Item | Location | Status |
|------|----------|--------|
| Open-Meteo inter-chunk delay | config/settings.py:57 | Still present |
| SEAS5 ensemble dimension detection | src/ingestion/seas5_fetcher.py:20 | Still present |
| SEAS5 product name params | src/ingestion/seas5_fetcher.py:54 | Still present |

## H) Blocking Issues for Phase 2
- ERA5-Land fetch failed for all sites due to DNS resolution errors (Open-Meteo endpoint). Phase 1 not complete, so Phase 2 cannot start.

Evidence anchors:
- Command: `source venv/bin/activate && python3 refresh_data.py --step era5`
  - Key output: `Failed to resolve 'archive-api.open-meteo.com'`

## Addendum — ERA5 Date Range Fix

### Problem
ERA5-Land starts at 1950 but config had pre-1950 label years (DC 1921, Liestal 1894) and feature builder crashed on missing weather data for pre-1950 years. Evidence: `context/ARCHITECTURE.md` data source years and missing Liestal consolidated parquet.

### Fix Applied
| File | Change |
|------|--------|
| config/settings.py | Added note clarifying era5_start is weather start (1950). |
| src/processing/features.py | Skip label years without weather data; raise if consolidated file missing; enforce 2026 data exists. |

Evidence anchors:
- Code: `config/settings.py:73-129` (SITES + note)
- Code: `src/processing/features.py:45-123 build_gold_features`

### Result
- Features build now skips pre-1950 label years instead of crashing once weather data exists.
- Liestal consolidated parquet missing; requires re-fetch by user.


## Addendum — Silver Audit + ASOS Fix

### Silver Data Audit Results
| Site | File Exists | Size | Years | 2026 Hours | Temp Range | Temp Nulls |
|------|------------|------|-------|------------|------------|------------|
| washingtondc | YES | 6.9 MB | 1950-2026 (77) | 1416 | -24.3 to 41.3 | 0 (0.00%) |
| kyoto | YES | 6.9 MB | 1950-2026 (77) | 1416 | -10.8 to 39.4 | 0 (0.00%) |
| liestal | YES | 6.9 MB | 1950-2026 (77) | 1416 | -26.0 to 35.6 | 0 (0.00%) |
| vancouver | YES | 6.9 MB | 1950-2026 (77) | 1416 | -18.6 to 36.3 | 0 (0.00%) |
| nyc | YES | 6.9 MB | 1950-2026 (77) | 1416 | -25.4 to 40.4 | 0 (0.00%) |

Evidence: `source venv/bin/activate && python3 /tmp/silver_audit.py` output.

### ASOS Fix
- Problem: Iowa Mesonet "M" strings cause mixed dtypes → pyarrow crash
- Fix: coerce tmpf/dwpf/relh to numeric after CSV read; replace "M"/"T" with NA; set low_memory=False
- Stations missing after audit: CYVR, IAD, LGA (DCA/JFK present)

Evidence: `src/ingestion/asos_fetcher.py` changes; audit output shows missing stations.

### MLflow Cleanup
- Removed from requirements.txt
- Removed mlruns/ and aimlflow_repo/
- Import references: none found (rg)

### Gate Results (post-fix)
| Gate | Result | Evidence |
|------|--------|----------|
| assert_inference_cutoff_utc | NOT RUN | gates depend on features build |
| assert_historical_window_end | NOT RUN | features missing |
| assert_labels_complete | NOT RUN | not executed in this prompt |
| assert_seas5_members | NOT RUN | fallback not invoked yet |
| assert_silver_utc | NOT RUN | gates pending |
| assert_gold_schema | NOT RUN | features missing |
