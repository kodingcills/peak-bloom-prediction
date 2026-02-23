# Repo Technical Brief: Peak Bloom Prediction (2026)

## 1. Inventory & Categorization

### Pipeline Scripts (Core Logic)
- `collect_targets_and_indices.py`: Data ingestion for historical bloom targets and macro-climate indices (ONI, AMO).
- `teleconnections.py`: Fetches daily AO and NAO indices from NOAA CPC.
- `patch_recent_weather.py`: Fills the 2025-2026 winter gap (up to Feb 21) using Open-Meteo Archive API.
- `data_qa_cleaner.py`: Performs station-level QA, temporal continuity enforcement, and DOY-based imputation.
- `generate_climatology.py`: Computes smoothed "modern normals" (1995–2025) for forecast padding.
- `feature_engineer.py`: The "Bio-Transformer". Implements DCM chill, GDD accumulation, and teleconnection merging.
- `data_sanitizer.py`: Handles teleconnection persistence and decay for the forecast window.
- `model_stacker.py`: The "God-Tier" orchestrator. Implements Hierarchical Bayesian + XGBoost Stacker.

### Utilities & Audits
- `noaa_daily_fetcher.py`: Data integrity audit for climate record completeness.
- `integrity_probe.py`: Audit for GDD monotonicity and teleconnection NaNs.
- `data_sanitizer.py`: Heuristic-based feature cleaning.

### Tests
- `tests/test_gates.py`: Unit tests for rescaling, monotonicity, cutoff enforcement, and gap filling.

### Context (Foundational Mandates)
- `context/SYSTEM_CONTEXT_INDEX.md`: Global constants and precedence rules.
- `context/MODELING_ASSUMPTIONS.md`: Structural contract for hybrid modeling.
- `context/BIOLOGICAL_BACKGROUND.md`: Physiological constraints (Chill before Heat).
- `context/RESIDUAL_MODEL_POLICY.md`: Firewall rules for ML training.
- `context/REACHABILITY_GUARDRAILS.md`: Safety specs for mechanistic triggers.

---

## 2. File-by-File Purpose

| File | Purpose | Key Functions | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| `feature_engineer.py` | Bio-feature generation | `dynamic_chill_model`, `process_site` | `*_historical_climate.csv` | `features_train.csv` |
| `model_stacker.py` | Modeling & Forecast | `build_hierarchical_model`, `simulate_bloom_doy` | `features_train.csv` | `submission_2026.csv` |
| `patch_recent_weather.py` | Gap filling | `patch_city_data` | `COORDS` map | Updated climate CSVs |
| `data_qa_cleaner.py` | Data Cleaning | `clean_station_data` | Raw climate CSVs | Pristine climate CSVs |
| `teleconnections.py` | Index Fetching | `fetch_cpc_teleconnections_final` | NOAA FTP | `*_daily.csv` |

---

## 3. Pipeline Map

1.  **Ingestion**: `collect_targets_and_indices.py` + `teleconnections.py` (Raw data).
2.  **Patching**: `patch_recent_weather.py` (Fills gap to Feb 21, 2026).
3.  **Cleaning**: `data_qa_cleaner.py` (QA + Imputation).
4.  **Climatology**: `generate_climatology.py` (Normals for future padding).
5.  **Features**: `feature_engineer.py` (GDD, Chill Portions, Teleconnections).
6.  **Sanitization**: `data_sanitizer.py` (Teleconnection decay/persistence).
7.  **Modeling**: `model_stacker.py` (Bayesian Fit -> Residual XGBoost -> 2026 Forecast).

---

## 4. Mathematics & Formulas

### Growing Degree Days (GDD)
- **Formula**: $GDD_t = \max(T_{avg,t} - T_{base}, 0)$
- **Implementation**: `model_stacker.py`: `compute_bio_thermal_path`
- **Logic**: Cumulative sum starting from Sept 1st (Bio-Year).

### Dynamic Chill Model (DCM)
- **Concept**: Models chill portions (CP) via a two-step biochemical reaction (intermediary → portion).
- **Implementation**: `feature_engineer.py`: `dynamic_chill_model`
- **Equations**: Fishman & Erez (1987). Uses hourly sine-interpolation:
  $T_h = \frac{T_{max} + T_{min}}{2} + \frac{T_{max} - T_{min}}{2} \sin(\frac{\pi(h-8)}{9})$

### Vapor Pressure Deficit (VPD)
- **Formula**: $VPD = SVP(T_{max}) - SVP(T_{min})$
- **SVP**: $0.6108 \cdot \exp(\frac{17.27 \cdot T}{T + 237.3})$
- **Implementation**: `model_stacker.py`: `compute_bio_thermal_path`

### Photoperiod
- **Formula**: Civil twilight photoperiod using Cooper (1969).
- **Implementation**: `feature_engineer.py`: `calculate_photoperiod`

### Hierarchical Bayesian Model
- **Structure**: $GDD_{bloom} \sim 	ext{Normal}(\mu_{GDD}, \sigma_{obs})$
- **Linear Synergy**: $\mu_{GDD} = a_{site} + b_{site} \cdot 	ext{Chill}_{standardized}$
- **Priors**:
  - $a_{site} \sim 	ext{Normal}(	ext{Anchor}_{site}, \sigma_a)$
  - $b_{site} \sim 	ext{Normal}(\mu_b, \sigma_b)$
- **Standardization**: Chill is z-scored per site to ensure $b_{site}$ represents sensitivity to 1-SD of chill.

### Mechanistic Simulation (Tripwire)
- **Rule**: Bloom occurs on first DOY $t \ge 60$ where $GDD_{cum}(t) \ge a_{site} + b_{site} \cdot 	ext{Chill}_{cum}(t)$.
- **Implementation**: `model_stacker.py`: `simulate_bloom_doy`

### Residual Stacker
- **Model**: XGBoost Regressor.
- **Filtering**: MAD (Median Absolute Deviation) filter. Rows where $|resid| > 	ext{median} + 2.5 \cdot 	ext{MAD}$ are excluded to prevent learning from catastrophic failures.

---

## 5. Validation Strategy

- **Method**: Expanding Window Validation (2015–2024).
- **Leakage Prevention**:
  - Training data strictly < test year.
  - Forecast features cut off at Feb 28th.
  - Teleconnections decay to 0 after 14-day persistence to avoid using future info.
- **Metric**: Mean Absolute Error (MAE) across 5 sites.

---

## 6. Tests & Gates (`test_gates.py`)

1.  `test_smart_rescale_tenths`: Detects and fixes $0.1^\circ C$ scaling.
2.  `test_gdd_monotonicity`: Ensures $GDD_{cum}$ never decreases.
3.  `test_cutoff_enforcement`: Ensures no "observed" data after Feb 28th.
4.  `test_feb_gap_completeness`: Validates the Feb 22-28 linear decay bridge.
5.  `test_reachability_sanity`: Verifies GDD anchors are reachable by May 31 (DOY 151).

---

## 7. Known Issues & Bottlenecks

- **Data Gap**: Vancouver has a significant historical imputation gap (181 days).
- **Teleconnection Decay**: AO/NAO decay to climatology (0) after 14 days might lose late-spring signals.
- **NYC Proxy**: NYC relies on USA-NPN Yoshino data which requires a +0.5 day offset calibration.
- **Reachability Risk**: Fragile headroom (mean headroom < 5 days) in cold springs could lead to DOY 151 sentinels.

---

## 8. Next Steps (Minimal Patch Plan)

1.  **Vancouver Weighting**: Implement a $0.2$ precision weight for Vancouver in the hierarchical model to account for data uncertainty.
2.  **NYC Offset**: Explicitly apply the $+0.5$ day offset to the final NYC prediction in `model_stacker.py`.
3.  **VPD Integration**: Incorporate 14-day rolling VPD into the Bayesian core (instead of just XGBoost) to test if it improves mechanistic stability.
4.  **Reachability CI**: Add a standalone `tests/test_reachability.py` to enforce the 5% max unreachable rate per `REACHABILITY_GUARDRAILS.md`.

---

## 9. Assumption Register

| ID | Assumption | Source |
| :--- | :--- | :--- |
| A-01 | Temperature is the dominant driver of bloom timing. | `MODELING_ASSUMPTIONS.md` |
| A-02 | Chill must be met before GDD becomes effective. | `BIOLOGICAL_BACKGROUND.md` |
| A-03 | AO/NAO anomalies decay to zero after 14 days. | `data_sanitizer.py` |
| A-04 | Modern Normals (1995+) capture climate-shifted baselines. | `generate_climatology.py` |

## 10. Risk Register

| ID | Risk | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| R-01 | Arctic Oscillation (AO) flip in March. | High | Residual XGBoost layer for atmospheric noise. |
| R-02 | GDD threshold unreachability in late springs. | Critical | `REACHABILITY_GUARDRAILS.md` + Smooth Likelihood. |
| R-03 | Station micro-climate divergence. | Medium | Site-specific Bayesian intercepts ($a_{site}$). |
| R-04 | Unit scaling error (Tenths vs Celsius). | Critical | `smart_rescale` heuristic + Unit-Safe Audit. |
