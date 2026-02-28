# phase2.md — Fold-Safe Engineering & Cross-Validation

## Objective
Build a leakage-free model estimation pipeline using Site-Year Holdout (SYH) CV. Estimate regression coefficients, bias corrections, and Empirical Bayes shrinkage weights — all fold-safe. Produce MAE diagnostics.

## Prerequisites
- Phase 1 complete (all validation gates passed)
- `data/gold/features.parquet` exists
- `data/silver/weather/` and `data/silver/asos/` populated

## Depends On
- `config/settings.py`
- `src/processing/features.py` (GDH, CP computation)
- `src/validation/gates.py`

---

## Step 1: `src/processing/bias_correction.py`

### Function: `estimate_bias(asos_dir, weather_dir, site_key, exclude_year=None) -> dict`

Estimate ERA5-Land → ground-truth temperature bias for sites with ASOS stations.

**Logic:**
1. Load ASOS parquet for site's stations (e.g., DCA for washingtondc)
2. Load ERA5-Land consolidated parquet for same site
3. Inner-join on nearest hour (tolerance: 30 min)
4. If `exclude_year` is not None, filter out that year (fold-safety)
5. Fit OLS: `T_asos ~ β₀ + β₁ * T_era5`
6. Return `{"beta0": β₀, "beta1": β₁, "r2": R², "n": sample_count}`

**Fold-safety requirement:** When called during SYH CV, `exclude_year` MUST be passed. The assert in `gates.py` (`assert_bias_fold_safe`) verifies this.

For sites WITHOUT ASOS (Kyoto, Liestal): bias correction is skipped. Return `{"beta0": 0, "beta1": 1.0}` (identity transform).

---

## Step 2: `src/processing/warming_velocity.py`

### Function: `compute_warming_velocity(hourly_df, year, window_center_doy, window_days=14) -> float`

Compute the rate of warming in a window around expected bloom.

**Logic:**
1. Define window: `[window_center_doy - window_days, window_center_doy]`
2. Filter hourly_df to this DOY range for the given year
3. Compute daily mean temperature for each DOY in window
4. Fit linear regression: `T_daily_mean ~ slope * DOY + intercept`
5. Return `slope` (°C per day)

**CRITICAL:** `window_center_doy` must come from the TRAINING fold's mean bloom DOY, not the global mean. This is the fold-safe requirement.

**Edge case:** If the window extends past DOY 59 (Feb 28) for 2026 inference, the post-cutoff portion must use SEAS5 forecast temperatures, not observations. Flag this in the output with `uses_forecast=True`.

---

## Step 3: `src/modeling/syh_cv.py`

### Class: `SYHCrossValidator`

Implements the full Site-Year Holdout cross-validation loop.

```python
class SYHCrossValidator:
    def __init__(self, features_df, labels_df, sites_config):
        """
        Args:
            features_df: Gold feature matrix (site_key, year, gdh, cp)
            labels_df: Bloom labels (site_key, year, bloom_doy)
            sites_config: Dict of SiteConfig objects
        """
    
    def run(self) -> pd.DataFrame:
        """
        For each unique year Y in the label data:
          1. Split: train = all rows where year != Y, test = all rows where year == Y
          2. Re-estimate bias corrections on train only
          3. Re-calculate mean_bloom_doy per site on train only
          4. Compute warming velocity with fold-safe window center
          5. Re-derive Empirical Bayes weights on train only
          6. Fit regression: bloom_doy ~ gdh + cp + warming_velocity (on train)
          7. Predict test year for all sites
          8. Record: site_key, year, predicted_doy, actual_doy, residual
        
        Returns DataFrame of all CV predictions with residuals.
        """
    
    def compute_mae(self, cv_results) -> dict:
        """Compute per-site and overall MAE from CV results."""
```

**Regression model (within each fold):**
```
bloom_doy_s = α_s + β_gdh * GDH_s + β_cp * CP_s + β_wv * WV_s + ε
```
Where `α_s` is a site-specific intercept (fixed effect for each site to handle species/threshold differences).

---

## Step 4: `src/modeling/empirical_bayes.py`

### Function: `compute_shrinkage_weights(cv_results, epsilon=0.01) -> dict`

Compute Empirical Bayes precision weights per site.

**Logic:**
```
w_s = σ²_global / (σ²_global + (σ²_s + ε) / N_s)
```
Where:
- `σ²_global`: variance of residuals across ALL sites in training data
- `σ²_s`: variance of residuals for site s
- `N_s`: number of training observations for site s
- `ε`: stability floor

**Return:** `{"washingtondc": 0.92, "kyoto": 0.89, ..., "vancouver": 0.34, "nyc": 0.21}`

Sites with many observations → high weight (trust model). Sites with few → low weight (trust prior).

### Function: `apply_shrinkage(model_pred, global_mean, weight) -> float`
```
final_pred = weight * model_pred + (1 - weight) * global_mean
```

---

## Step 5: Validation Gates (add to `src/validation/gates.py`)

```python
def assert_bias_fold_safe(cv_log):
    """Verify that bias correction was re-estimated for every held-out year."""

def assert_window_safe(cv_log):
    """Verify warming velocity window center was derived from training fold only."""

def assert_precision_fold_safe(cv_log):
    """Verify shrinkage weights were derived from training data only."""

def assert_vancouver_weight_stable(weights):
    """assert std(weights across folds for Vancouver) < 0.15"""

def assert_cv_no_leakage(cv_results):
    """For each test prediction, verify the test year was excluded from all training components."""
```

---

## Output Artifacts

| Artifact | Path | Contents |
|----------|------|----------|
| CV results | `data/processed/cv_results.parquet` | All fold predictions with residuals |
| MAE summary | `data/processed/mae_summary.json` | Per-site and overall MAE |
| Shrinkage weights | `data/processed/shrinkage_weights.json` | Per-site w_s values |
| Bias coefficients | `data/processed/bias_coefficients.json` | Per-site β₀, β₁ values |

---

## Validation Checklist (Phase 2 Exit Criteria)

- [ ] SYH CV completes for all years without leakage assertions firing
- [ ] Per-site MAE is computed and logged
- [ ] Vancouver weight std across folds < 0.15
- [ ] NYC weight is < 0.5 (heavy shrinkage expected with 2 data points)
- [ ] Bias correction coefficients have sensible signs (β₁ close to 1.0)
- [ ] Warming velocity is only computed on data ≤ DOY 59 for 2026 inference
- [ ] All intermediate artifacts saved to `data/processed/`
