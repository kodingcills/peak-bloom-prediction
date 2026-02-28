# phase3.md — Inference & Bimodal Cluster Selection

## Objective
Generate the 5 final bloom DOY predictions for 2026 using the trained model, SEAS5 ensemble forecasts, and Empirical Bayes shrinkage. Resolve potential bimodality in ensemble predictions via GMM + BIC.

## Prerequisites
- Phase 1 & 2 complete
- `data/gold/features.parquet` (2026 rows have GDH/CP through DOY 59)
- `data/processed/seas5_2026.nc` (or fallback mode)
- `data/processed/shrinkage_weights.json`
- `data/processed/bias_coefficients.json`
- `data/processed/cv_results.parquet` (for model coefficients)

---

## Step 1: `src/modeling/seas5_processor.py`

### Function: `extract_site_forecasts(nc_path, site_config) -> pd.DataFrame`

Extract SEAS5 ensemble temperature trajectories for a single site.

**Logic:**
1. Open NetCDF with xarray
2. Select nearest grid point to `(site_config.lat, site_config.lon)`
3. Extract all 50 ensemble members' 2m_temperature for lead months 1-3 (Mar/Apr/May 2026)
4. Convert to daily mean temperature per member
5. Return DataFrame: columns = `[member, date, temperature_2m]`

### Function: `ensemble_to_bloom_predictions(site_forecasts, site_key, model_coeffs, bias_coeffs, observed_gdh, observed_cp) -> np.ndarray`

Convert 50 ensemble temperature trajectories into 50 predicted bloom DOYs.

**Logic:**
1. For each ensemble member m (1..50):
   a. Append member m's forecast temperatures to the observed Jan 1 – Feb 28 temperatures
   b. Continue GDH accumulation from the observed GDH at DOY 59 using forecast temps
   c. Find the DOY where accumulated GDH crosses the site's bloom threshold (estimated from training)
   d. Apply bias correction using site's β₀, β₁
   e. Record predicted bloom DOY for member m
2. Return array of 50 predicted bloom DOYs

**Fallback (SEAS5_FALLBACK_MODE):** Instead of SEAS5 ensemble, use the last 30 years of observed post-DOY-59 temperature trajectories as pseudo-ensemble members. This gives you ~30 "scenarios" instead of 50.

---

## Step 2: `src/modeling/gmm_selector.py`

### Function: `fit_bimodal(bloom_predictions, site_key) -> dict`

Fit GMM and select between unimodal and bimodal distribution.

**Logic:**
1. Input: array of 50 (or ~30 in fallback) predicted bloom DOYs
2. Fit `GaussianMixture(n_components=1)` → compute BIC₁
3. Fit `GaussianMixture(n_components=2)` → compute BIC₂
4. **k selection:** Choose k with lower BIC.
   - BIC formula: `BIC = ln(n) * k_params - 2 * ln(L_hat)`
   - k=1 has 2 params (mean, variance); k=2 has 5 params (2 means, 2 variances, 1 mixing weight)
5. **Tie-breaking:** If `|BIC₁ - BIC₂| / |BIC₁| < SEAS5_NEUTRAL_THRESHOLD (0.3)`, default to k=1 (conservative).
6. **Cluster selection (if k=2):**
   a. Compute each cluster's mean bloom DOY
   b. Select cluster whose mean is closer to site's climatological mean bloom DOY
   c. Exception: if one cluster has >70% membership weight and the other <30%, select the dominant cluster regardless
7. Return: `{"k": k, "bic_1": BIC₁, "bic_2": BIC₂, "selected_mean": float, "selected_std": float, "all_predictions": array}`

**CONSTRAINT:** Never fit k > 2. This is hardcoded. (Problem 22)

**CONSTRAINT:** No artificial noise injection into predictions. Variance comes only from the SEAS5 ensemble spread. (Problem 23)

---

## Step 3: `src/modeling/predictor.py`

### Function: `generate_predictions(features_2026, model_coeffs, shrinkage_weights, gmm_results) -> pd.DataFrame`

Produce the final 5 predictions.

**Logic:**
For each site:
1. Get GMM-selected bloom DOY mean → `model_pred`
2. Get global mean bloom DOY from training data → `global_mean`
3. Get site shrinkage weight → `w_s`
4. Apply shrinkage: `final_pred = w_s * model_pred + (1 - w_s) * global_mean`
5. Round to nearest integer
6. Clip to reasonable range: `[60, 140]` (DOY — early March to late May)

Return DataFrame:
```
location,year,bloom_doy
washingtondc,2026,<int>
kyoto,2026,<int>
liestal,2026,<int>
vancouver,2026,<int>
newyorkcity,2026,<int>
```

### Function: `save_submission(predictions_df, output_path="submission.csv")`
Write CSV. Validate exactly 5 rows. Validate all bloom_doy are integers.

---

## Step 4: Diagnostic Outputs

Generate and save to `data/processed/diagnostics/`:

1. **`ensemble_distributions.json`** — Per-site: 50 predicted bloom DOYs, GMM params, BIC values
2. **`prediction_summary.json`** — Per-site: model_pred, global_mean, weight, final_pred
3. **`prediction_intervals.json`** — Per-site: 10th/25th/50th/75th/90th percentiles from ensemble

---

## Step 5: Validation Gates (add to `src/validation/gates.py`)

```python
def assert_gmm_k_range(gmm_results):
    """All sites have k ∈ {1, 2}. No higher."""

def assert_no_noise_injection(gmm_results):
    """Ensemble spread comes only from SEAS5 members, not artificial noise."""

def assert_submission_schema(submission_path):
    """submission.csv has exactly 5 rows, columns [location, year, bloom_doy], all bloom_doy are int."""

def assert_predictions_reasonable(submission_df):
    """All bloom_doy in [60, 140] range."""

def assert_shrinkage_applied(prediction_summary):
    """Vancouver and NYC predictions differ from raw model prediction (shrinkage was applied)."""
```

---

## Validation Checklist (Phase 3 Exit Criteria)

- [ ] SEAS5 ensemble processed for all 5 sites (or fallback used)
- [ ] GMM fit with k ∈ {1,2} for all sites, BIC logged
- [ ] No artificial noise in ensemble predictions
- [ ] Shrinkage applied — Vancouver and NYC predictions pulled toward global mean
- [ ] `submission.csv` contains exactly 5 integer DOY predictions
- [ ] All predictions in [60, 140] range
- [ ] Diagnostic JSONs saved
- [ ] Full pipeline reproducible: `python refresh_data.py && quarto render analysis.qmd` produces identical `submission.csv`
