# phase1.md — Data Acquisition & Silver Layer

## Objective
Fetch all external data, normalize to UTC, validate, and persist as Silver-layer artifacts. After this phase, the project can run fully offline.

## Prerequisites
- Python 3.11+ environment
- `pip install pandas numpy pyarrow requests cdsapi xarray netcdf4`
- `~/.cdsapirc` configured (or set `SEAS5_FALLBACK_MODE=true`)
- GMU competition CSVs in `data/raw/gmu/` (washingtondc.csv, kyoto.csv, liestal.csv, vancouver.csv, nyc.csv)

---

## Step 1: `config/settings.py`

Create the central configuration module. All constants live here.

### Contents:
```python
# Site dataclass with fields: name, loc_id, lat, lon, alt, species, bloom_pct, era5_start, asos_stations (list)
# SITES dict keyed by site_key string
# API endpoints and parameters (see ARCHITECTURE.md for exact values)
# Temporal constants: COMPETITION_YEAR=2026, INFERENCE_CUTOFF_DOY=59, ERA5_CHUNK_YEARS=10
# Feature constants: GDH_BASE_C=4.5, CP_MIN/OPT/MAX, CHILL_START_DOY=274, WARM_START_DOY=1
# Path constants: RAW_DIR, SILVER_WX, SILVER_ASOS, PROCESSED, GOLD_DIR
# Env-driven: SEAS5_FALLBACK_MODE, N_JOBS
```

Refer to the site table and API details in `ARCHITECTURE.md` for exact values.

---

## Step 2: `src/ingestion/era5_fetcher.py`

### Function: `fetch_era5_site(site_key, site_config, output_dir, force=False) -> Path`

Fetches ERA5-Land hourly data for one site via Open-Meteo Archive API.

**Logic:**
1. Compute decadal chunks: `[1950-1959, 1960-1969, ..., 2020-2026]`. Start year = `site_config.era5_start`.
2. For each chunk:
   - Output file: `{output_dir}/{site_key}/{site_key}_{start}_{end}.parquet`
   - If file exists and `force=False`, skip.
   - Build URL: `https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}&hourly=temperature_2m,relative_humidity_2m,soil_temperature_0_to_7cm&timezone=UTC`
   - For 2026 chunk: end_date = `2026-02-28` (not Dec 31)
   - GET with timeout=120s, retry 3x with backoff
   - Parse JSON → `response["hourly"]` → DataFrame
   - Column `time` → `timestamp`, parse as UTC-aware datetime
   - Save as Parquet
   - Sleep `OPENMETEO_DELAY` between chunks
3. After all chunks: concatenate → deduplicate on timestamp → sort → save consolidated file `{site_key}_consolidated.parquet`
4. Return path to site directory.

### Function: `fetch_all_era5(output_dir=None, force=False) -> dict`
Loop over `SITES`, call `fetch_era5_site` for each, collect results dict.

**Error handling:** Log and continue on per-site failure. Return status dict with `ok`/`error` per site.

---

## Step 3: `src/ingestion/asos_fetcher.py`

### Function: `fetch_asos_station(station_id, start_year, end_year, output_dir, force=False) -> Path`

Fetches ASOS hourly data from Iowa Mesonet for a single station.

**Logic:**
1. Output file: `{output_dir}/{station_id}.parquet`
2. If exists and not force, skip.
3. Build URL params:
   ```
   station={station_id}
   data=tmpf,dwpf,relh
   tz=Etc/UTC
   format=onlycomma
   latlon=yes
   elev=yes
   year1={start_year}&month1=1&day1=1
   year2=2026&month2=2&day2=28
   missing=M
   trace=T
   report_type=3&report_type=4
   ```
4. GET request → parse CSV response with pandas
5. Convert `tmpf` (°F) → `temperature_2m` (°C): `(tmpf - 32) * 5/9`
6. Parse `valid` column as UTC-aware timestamp
7. Drop rows where temperature is NaN or "M"
8. Save as Parquet

### Function: `fetch_all_asos(output_dir=None, force=False) -> dict`
Loop over all SITES that have `asos_stations`, fetch each station.

---

## Step 4: `src/ingestion/seas5_fetcher.py`

### Function: `fetch_seas5(output_path=None, force=False) -> Path`

Fetches SEAS5 50-member ensemble forecast via CDS API.

**Logic:**
1. Output file: `{PROCESSED}/seas5_2026.nc`
2. If exists and not force, skip.
3. Check `SEAS5_FALLBACK_MODE`. If true, log warning and return None.
4. Use `cdsapi.Client()` (reads `~/.cdsapirc` automatically)
5. Request:
   ```python
   client.retrieve("seasonal-original-single-levels", {
       "product_type": "monthly_mean",  # or "ensemble" depending on availability
       "variable": "2m_temperature",
       "year": "2026",
       "month": "02",
       "leadtime_month": ["1", "2", "3"],
       "system": "51",
       "area": [50, -125, 24, 140],  # bounding box covering all 5 sites
       "format": "netcdf",
   }, output_path)
   ```
   Note: The exact CDS API parameters for SEAS5 can be finicky. If the above fails, try `seasonal-monthly-single-levels` as an alternative product name. Leave a `# TODO: AUDIT — verify SEAS5 product name and parameters` comment.
6. After download, open with xarray and validate: assert 50 unique ensemble members exist.
7. Return path.

**IMPORTANT:** SEAS5 retrieval is the most fragile step. The CDS API queue can take hours. Implement a progress log. If retrieval fails after 3 attempts, set a flag file `data/processed/SEAS5_FETCH_FAILED` and continue — Phase 3 will use fallback mode.

---

## Step 5: `src/processing/labels.py`

### Function: `load_competition_labels(raw_dir) -> pd.DataFrame`

Loads and validates all GMU competition CSVs.

**Logic:**
1. Load each CSV from `data/raw/gmu/`: washingtondc.csv, kyoto.csv, liestal.csv, vancouver.csv, nyc.csv
2. Each has columns: `location, lat, long, alt, year, bloom_date, bloom_doy`
3. Concatenate into single DataFrame
4. Parse `bloom_date` as datetime
5. Validate `bloom_doy` matches `bloom_date.day_of_year`
6. Add column `site_key` mapped from `location` (e.g., `newyorkcity` → `nyc`)
7. Return combined DataFrame

### Function: `load_supplementary_labels(raw_dir) -> pd.DataFrame`

Load japan.csv, meteoswiss.csv, south_korea.csv into a single DataFrame with the same schema. Add `is_supplementary = True` column. This data is NOT used in Phase 1 but will be available for Phase 2 transfer learning.

---

## Step 6: `src/processing/features.py`

### Function: `compute_gdh(hourly_df, year, cutoff_doy=59) -> float`

Compute Growing Degree Hours for a single site-year.

**Logic:**
1. Filter hourly_df to: Jan 1 of `year` through DOY `cutoff_doy` of `year`
2. For each hour: `gdh_hour = max(0, temperature_2m - GDH_BASE_C)`
3. Return sum of all `gdh_hour` values
4. If >48 hours missing in window, log warning but still compute on available data

### Function: `compute_chill_portions(hourly_df, year, cutoff_doy=59) -> float`

Compute Chill Portions for a single site-year.

**Logic:**
1. Filter hourly_df to: Oct 1 of `year-1` (DOY 274) through DOY `cutoff_doy` of `year`
2. For each hour where CP_MIN_C ≤ T ≤ CP_MAX_C:
   `cp_hour = max(0, 1 - ((T - CP_OPT_C) / ((CP_MAX_C - CP_OPT_C)))**2)`
   Else: `cp_hour = 0`
3. Return sum

### Function: `build_gold_features(silver_weather_dir, labels_df, output_path) -> pd.DataFrame`

Build the complete site-year feature matrix.

**Logic:**
1. For each (site, year) pair in labels_df:
   a. Load consolidated weather parquet for that site
   b. Compute GDH for that year
   c. Compute CP for that year
   d. Record: `site_key, year, gdh, cp, bloom_doy`
2. Also compute features for 2026 (no bloom_doy — this is the prediction target)
3. Save as `data/gold/features.parquet`
4. Return DataFrame

---

## Step 7: `src/validation/gates.py`

Implement these validation functions. Each raises `AssertionError` on failure.

```python
def assert_historical_window_end(features_df):
    """All feature years use data only up to DOY 59."""

def assert_inference_cutoff_utc(weather_dir):
    """No weather timestamps after 2026-02-28 23:59:59 UTC in any silver file."""

def assert_labels_complete(labels_df):
    """All 5 competition sites present. Max year >= 2024 for each."""

def assert_seas5_members(nc_path):
    """SEAS5 NetCDF contains exactly 50 unique ensemble members."""
    # Skip if SEAS5_FALLBACK_MODE

def assert_silver_utc(weather_dir):
    """All parquet files in silver layer have UTC-aware timestamps."""

def assert_gold_schema(features_df):
    """Gold features have columns: site_key, year, gdh, cp, bloom_doy (nullable for 2026)."""
```

---

## Step 8: `refresh_data.py`

The Phase 1 CLI entry point.

```python
"""
Phase 1: Data Acquisition & Silver Layer
Usage:
  python refresh_data.py                    # Run all steps
  python refresh_data.py --step era5        # ERA5-Land only
  python refresh_data.py --step asos        # ASOS only
  python refresh_data.py --step seas5       # SEAS5 only
  python refresh_data.py --step labels      # Label processing only
  python refresh_data.py --step features    # Gold feature build only
  python refresh_data.py --force            # Overwrite existing files
  python refresh_data.py --sites dc,kyoto   # Subset of sites
"""
```

**Execution order:**
1. Parse args
2. Create directory structure if not exists
3. Run selected steps in order: era5 → asos → seas5 → labels → features
4. Run ALL validation gates
5. Print summary table:
   ```
   ╔═══════════════════════════════════╗
   ║ PHASE 1 — DATA ACQUISITION       ║
   ╠═══════════════════════════════════╣
   ║ ERA5-Land    ✓  5/5 sites         ║
   ║ ASOS         ✓  5/5 stations      ║
   ║ SEAS5        ✓  50 members        ║
   ║ Labels       ✓  243 site-years    ║
   ║ Gold Features✓  248 rows          ║
   ║ Gates        ✓  6/6 passed        ║
   ╚═══════════════════════════════════╝
   ```

---

## Validation Checklist (Phase 1 Exit Criteria)

- [ ] `data/silver/weather/` contains consolidated parquet for all 5 sites
- [ ] All silver timestamps are UTC-aware
- [ ] No silver data extends past 2026-02-28 23:59:59 UTC
- [ ] `data/silver/asos/` contains parquet for DCA, IAD, JFK, LGA, CYVR
- [ ] `data/processed/seas5_2026.nc` exists (or SEAS5_FALLBACK flag set)
- [ ] `data/gold/features.parquet` contains all site-years with GDH and CP
- [ ] Gold features for 2026 have null bloom_doy (prediction target)
- [ ] `refresh_data.py --help` works and documents all flags
- [ ] Full cold-start run completes: `rm -rf data/silver data/processed data/gold && python refresh_data.py`
