# Cherry Blossom Dataset Audit Report (2026-02-26)

## Executive Summary
Audit performed on all datasets used for peak bloom DOY prediction. The pipeline shows high integrity for core historical features, but critical gaps exist for the 2026 forecast horizon.

## CHECK 0 — Key integrity (CRITICAL)
- **Status:** PASS
- **Command:** `python3 -c "import pandas as pd; ... check_integrity('features_train.csv')"`
- **Findings:** 
  - Zero duplicate `(site, date)` pairs in training or forecast files.
  - Date ranges: Train (1950/51 - 2025-08-31), Forecast (2025-09-01 - 2026-02-21).
  - Missingness: `oni_90d` and `oni_30d` are 99% missing in the 2026 forecast file.

## CHECK 1 — Temperature sanity pivot (CRITICAL)
- **Status:** PASS
- **Command:** `python3 -c "import pandas as pd; ... temperature_sanity('features_train.csv')"`
- **Findings:**
  - Median TMAX values align perfectly between `features_train.csv` and raw historical CSVs.
  - No biological outliers found in monthly medians (e.g., Kyoto July median = 32.3C).

## CHECK 2 — cp/gdd audit (CRITICAL)
- **Status:** PASS
- **Findings:**
  - `cp` and `gdd` are non-decreasing in the 2026 forecast.
  - Yearly max values are stable across the last 3 bio-years per site.

## CHECK 3 — Consecutive-day TMAX jumps (WARNING)
- **Status:** INFO
- **Findings:**
  - 153 jumps > 15C in `features_train.csv`.
  - NYC raw data has 70 such jumps, confirming continental weather patterns. No jumps found in Vancouver.

## CHECK 4 — Forecast cutoff + padding contract (CRITICAL)
- **Status:** **FAIL**
- **Findings:**
  - **CRITICAL:** `features_2026_padded.csv` is missing.
  - Max date in `features_2026_forecast.csv` is 2026-02-21.
  - **Impact:** Model cannot predict bloom dates beyond Feb 21 without padding.

## CHECK 5 — PRCP outliers (WARNING)
- **Status:** INFO
- **Findings:**
  - 91 days with PRCP > 100mm. Most in Washington DC (max 155mm).
  - Distributions are reasonable for the respective climates.

## CHECK 6 — Target file audit (CRITICAL)
- **Status:** **WARNING**
- **Findings:**
  - Kyoto: 837 years, but 377 missing years in historical sequence.
  - Vancouver: 4 years (2022-2025).
  - NYC: 2 years (2024-2025).
  - **Impact:** Prediction for Vancouver/NYC relies on extreme extrapolation or hierarchical pooling.

## CHECK 7 — Teleconnection integrity (WARNING)
- **Status:** **WARNING**
- **Findings:**
  - ONI data for Oct, Nov, Dec 2025 contains placeholders (-99.9).
  - AO/NAO data is current up to 2026-01-31.

## CHECK 8 — Date gaps (WARNING)
- **Status:** PASS
- **Findings:**
  - No gaps > 3 days found in `features_train.csv` for any site.

## CHECK 9 — Cross-file bio_year consistency (CRITICAL)
- **Status:** PASS
- **Findings:**
  - All relevant target years (since 1950) have matching feature coverage in `features_train.csv`.

## CHECK 10 — Lag feature value (INFO)
- **Status:** INFO
- **Findings:**
  - Global correlation `corr(bloom_doy, lag1)` is 0.2114.
  - Per-site correlations are low (0.14 - 0.18).

## CHECK 11 — Seasonal anomalies sanity (WARNING)
- **Status:** INFO
- **Findings:**
  - Liestal April 2026 anomaly is -5.5C (significant cold).
  - Most other anomalies are within [-3, 4]C.

## CHECK 12 — Vancouver & NYC raw climate spot-check (INFO)
- **Status:** PASS
- **Findings:**
  - Row counts and median temperatures match expectations. Vancouver: 25k rows, NYC: 27k rows.

---

## Top 5 NEW Issues
1. **Missing Padded Forecast:** `features_2026_padded.csv` is absent, blocking 2026 predictions.
2. **ONI Placeholders:** Oct-Dec 2025 ONI values are -99.9, which may skew recent trend analysis.
3. **Extreme Target Sparsity:** Vancouver (4 yrs) and NYC (2 yrs) have insufficient history for standalone models.
4. **Forecast Stale Date:** Forecast features end on 2026-02-21, missing the last 5 days of observed weather.
5. **Significant Cold Anomaly:** Liestal April forecast (-5.5C) may cause the model to predict an unusually late bloom.

## Prioritized Action Plan
### Immediate (Next 24 Hours)
1. **Run Padding:** Execute `feature_engineer.py` (or the relevant script) to generate `features_2026_padded.csv` up to 2026-05-31.
2. **Patch ONI:** Manually update Oct-Dec 2025 ONI values from NOAA's latest bulletin.
3. **Fetch Latest Weather:** Update `features_2026_forecast.csv` with weather up to 2026-02-26 using `patch_recent_weather.py`.

### Long-term
1. **Lag Implementation:** Integrate `bloom_doy_lag1` into the training pipeline as a feature.
2. **Hierarchical Refinement:** Fine-tune the Bayesian model to weight Vancouver/NYC targets less and rely more on global priors.
3. **Anomaly Sensitivity:** Run sensitivity tests on the Liestal -5.5C anomaly to ensure the model doesn't overreact.
