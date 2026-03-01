# PHASE 2 IMPLEMENTATION BLUEPRINT
# Phenology Engine v1.7 — Fold-Safe Modeling & Cross-Validation

**Classification:** Execution-safe algorithmic specification
**Derived from:** ARCHITECTURE.md (authoritative), phase2.md, agent.md
**Produces:** CLI agent prompt (Section 11)

---

# SECTION 1 — Phase 2 Role in Global Architecture

## 1.1 What Phase 2 Consumes

| Artifact | Path | Schema | Producer |
|----------|------|--------|----------|
| Gold features | `data/gold/features.parquet` | `site_key, year, gdh, cp, bloom_doy` | Phase 1 |
| Silver weather | `data/silver/weather/{site}/*_consolidated.parquet` | `timestamp(UTC), temperature_2m, ...` | Phase 1 |
| Silver ASOS | `data/silver/asos/{station}.parquet` | `timestamp(UTC), temperature_2m(°C), ...` | Phase 1 |
| Site config | `config/settings.py` | SITES dict with lat, lon, ASOS stations | Phase 1 |

The gold features contain 239 rows: 234 training site-years (with bloom_doy) + 5 prediction rows (2026, bloom_doy=NaN).

Training data distribution:
- washingtondc: ~76 years (1950-2025)
- kyoto: ~75 years (post-1950 labels only)
- liestal: ~76 years (1950-2025)
- vancouver: 4 years (2022-2025)
- nyc: 2 years (2024-2025)

## 1.2 What Phase 2 Produces

| Artifact | Path | Contents |
|----------|------|----------|
| CV results | `data/processed/cv_results.parquet` | Per-(site,year) predictions, residuals, fold metadata |
| MAE summary | `data/processed/mae_summary.json` | Per-site and overall MAE |
| Shrinkage weights | `data/processed/shrinkage_weights.json` | Per-site w_s values |
| Bias coefficients | `data/processed/bias_coefficients.json` | Per-site β₀, β₁ (full-data fit for Phase 3 use) |
| Trained model coefficients | `data/processed/model_coefficients.json` | Full-data regression coefficients for Phase 3 |

## 1.3 Why Phase 3 Cannot Function Without Phase 2

Phase 3 requires:
1. **Model coefficients** to convert SEAS5 temperature trajectories → predicted bloom DOYs
2. **Shrinkage weights** to pull Vancouver/NYC predictions toward global mean
3. **Bias coefficients** to correct ERA5→ground-truth temperature shift
4. **Mean bloom DOY per site** (training-derived) for warming velocity window centering and GMM cluster selection

Without these, Phase 3 has no mapping from weather → bloom date.

## 1.4 Dependency Graph

```
Phase 1                Phase 2                    Phase 3
─────────             ─────────                  ─────────
features.parquet ──┐
                   ├──► SYH CV ──► cv_results ──► (diagnostic only)
silver/weather ────┤            ├─► mae_summary ──► (diagnostic only)
silver/asos ───────┤            ├─► shrinkage_weights ──► apply_shrinkage()
                   │            ├─► bias_coefficients ──► bias-correct SEAS5
                   │            └─► model_coefficients ──► ensemble_to_bloom()
                   │
                   └──► Full-data refit ──────────────────► Phase 3 inference
```

---

# SECTION 2 — Mathematical Foundations

## 2.1 Bias Correction Model

### Formulation

For each site s with ASOS station data, we estimate a linear mapping from ERA5-Land temperature to ground-truth station temperature:

```
T_asos(h) = β₀ˢ + β₁ˢ · T_era5(h) + ε(h)
```

where h indexes hourly observations in the overlap period (ASOS starts ~2000).

### OLS Estimator

Given N overlapping hourly observations for site s (excluding held-out year Y):

Let X = [1, T_era5] (N×2 matrix), y = T_asos (N×1 vector).

```
[β₀ˢ, β₁ˢ]ᵀ = (XᵀX)⁻¹Xᵀy
```

This is standard OLS. R² provides a diagnostic on the strength of the ERA5-ASOS relationship.

### Why Per-Fold Estimation Is Mandatory

Consider the leakage scenario:

1. Fit bias correction on ALL years including year Y
2. Apply bias-corrected temperatures for year Y
3. Compute features (GDH, WV) for year Y using corrected temperatures
4. Predict bloom_doy for year Y

Step 1 allows information from year Y's temperature patterns to influence the correction applied in Step 2. The bias coefficients β₀, β₁ absorb the mean temperature structure of year Y, and when applied back to year Y, they reduce apparent temperature error. This systematically inflates the model's apparent accuracy.

**Quantitative leakage bound:** If year Y has an anomalous warm winter (T_era5 systematically high), the globally-estimated β₁ shifts toward accommodating this anomaly. When β₁ is then applied to year Y's ERA5 data, the correction is partially tuned to year Y, reducing the effective residual variance by O(1/N_overlap) where N_overlap is the ASOS-ERA5 overlap sample size. For N_overlap ~ 200,000 hourly observations, this leakage is small (~0.0005%) but violates the zero-leakage invariant.

### Sites Without ASOS

Kyoto and Liestal have no ASOS stations. For these sites:
```
β₀ˢ = 0, β₁ˢ = 1.0  (identity transform)
```

No bias correction is applied. ERA5-Land is used directly. This is documented as an accepted limitation.

---

## 2.2 Warming Velocity (WV)

### Definition

Warming velocity captures the rate of temperature increase in a window near expected bloom time:

```
WV(s, y) = slope of linear regression: T̄_daily(d) ~ a + b·d
```

over d ∈ [d_center - 14, d_center], where:
- T̄_daily(d) = mean of hourly temperatures on day-of-year d in year y at site s
- d_center = mean bloom DOY for site s, computed from TRAINING FOLD ONLY

### Slope Estimator

For the 14 daily observations (d₁, T̄₁), ..., (d₁₄, T̄₁₄):

```
b = [Σᵢ(dᵢ - d̄)(T̄ᵢ - T̄̄)] / [Σᵢ(dᵢ - d̄)²]
```

where d̄ = mean of the day indices, T̄̄ = mean of daily mean temperatures.

Units: °C per day.

### Why Window Center Must Be Fold-Safe

The window center is `mean_bloom_doy(s, training_years)`. If we compute this from ALL years including the held-out year Y:

```
d_center_LEAKED = mean(bloom_doy(s, all years))
d_center_SAFE = mean(bloom_doy(s, all years \ {Y}))
```

The difference is:

```
d_center_LEAKED - d_center_SAFE = (bloom_doy(s, Y) - d_center_SAFE) / N_s
```

This directly encodes the test label `bloom_doy(s, Y)` into the feature computation. For sites with small N_s (Vancouver: N=4), this leakage is ~25% of one standard deviation of bloom_doy — **a significant information leak**.

**Concrete failure example:**

Vancouver 2023: bloom_doy = 96 (late). 
Vancouver training mean (2022, 2024, 2025) = (86 + 83 + 93)/3 = 87.3
Vancouver global mean including 2023 = (86 + 96 + 83 + 93)/4 = 89.5

Leaked window center is 89.5, safe is 87.3. The 2.2-day shift moves the WV window later, toward the actual late bloom — encoding the answer into the feature.

### DOY 59 Boundary

For historical years in CV: the WV window (typically DOY ~75-90 for most sites) uses observed temperatures from the silver weather data. This data exists for all historical years.

For 2026 inference (Phase 3): the window extends past DOY 59. Post-cutoff temperatures must come from SEAS5 forecasts. Phase 2 does NOT handle this — it only processes historical years where full weather data exists.

**Key implication:** In Phase 2, WV is computed purely from silver weather observations. In Phase 3, WV for 2026 will use SEAS5 forecast data. This is intentional — the Phase 2 model learns the relationship between WV (from observations) and bloom_doy, and Phase 3 plugs in WV (from forecasts).

---

## 2.3 Site-Year Holdout Cross-Validation (SYH)

### Dataset Indexing

Define the labeled dataset as:

```
D = {(sᵢ, yᵢ, x_i, bloom_doy_i) : i = 1, ..., N}
```

where sᵢ ∈ {DC, Kyoto, Liestal, Vancouver, NYC}, yᵢ ∈ {1950, ..., 2025}, and x_i = (gdh_i, cp_i).

N = 234 (training site-years).

### Unique Years

Let Y = {y : ∃ (s, y) ∈ D} be the set of unique years. |Y| ≈ 76.

### Fold Construction

For each holdout year Y ∈ Y:

```
D_train(Y) = {(s, y, x, bloom) ∈ D : y ≠ Y}
D_test(Y)  = {(s, y, x, bloom) ∈ D : y = Y}
```

Note: |D_test(Y)| varies. For early years (Y=1955), only DC, Kyoto, Liestal have labels (~3 test points). For recent years (Y=2024), all 5 sites have labels (5 test points).

### Why Holding Out Only One Site Leaks Information

Consider an alternative: hold out (site=DC, year=2024) but keep (site=Kyoto, year=2024) in training.

The regression model learns from Kyoto-2024's features (GDH, CP, WV) and bloom_doy. Weather in 2024 is correlated across sites — a warm European/Asian winter typically co-occurs with a warm North American winter (teleconnection patterns). Kyoto-2024's bloom_doy encodes information about 2024's global temperature anomaly, which is correlated with DC-2024's bloom_doy.

**Formally:** Let Z_Y be a latent "global temperature anomaly" for year Y. Then:

```
bloom_doy(DC, Y) = f(GDH_DC_Y, ...) + γ·Z_Y + ε_DC
bloom_doy(Kyoto, Y) = f(GDH_Kyoto_Y, ...) + γ·Z_Y + ε_Kyoto
```

If Kyoto-Y is in training while DC-Y is in test, the model observes Z_Y through Kyoto-Y and partially reconstructs it for DC-Y. This violates independence.

**SYH eliminates this** by removing ALL sites for year Y, ensuring Z_Y is unobserved in training.

---

## 2.4 Regression Model

### Full Specification

```
bloom_doy(s, y) = α_s + β_gdh · GDH(s, y) + β_cp · CP(s, y) + β_wv · WV(s, y) + ε(s, y)
```

Parameters:
- α_s: site-specific intercept (5 values: one per site)
- β_gdh: coefficient for growing degree hours (shared across sites)
- β_cp: coefficient for chill portions (shared across sites)
- β_wv: coefficient for warming velocity (shared across sites)
- ε(s,y) ~ N(0, σ²): residual

Total parameters: 5 intercepts + 3 slopes = 8.

### Design Matrix

Encode as a standard linear regression with dummy variables:

```
X = [I_DC, I_Kyoto, I_Liestal, I_Vancouver, I_NYC, GDH, CP, WV]
```

where I_s are indicator columns. Note: NO global intercept — the 5 site indicators absorb it.

### OLS Estimation

```
β̂ = (XᵀX)⁻¹Xᵀy
```

With N_train ≈ 229-234 and p = 8, the system is well-identified (N >> p).

### Identifiability Concerns for Sparse Sites

When holdout year Y is 2024 or 2025, NYC's training data reduces to 1 observation. A site intercept estimated from 1 point has zero residual — the model perfectly fits that point, learning nothing generalizable.

**This is not a bug.** The Empirical Bayes shrinkage in Section 2.5 handles this: NYC's prediction will be heavily shrunk toward the global mean, so the noisy site intercept has minimal influence on the final prediction.

### Expected Coefficient Signs

- β_gdh < 0: more accumulated heat by DOY 59 → closer to bloom threshold → earlier bloom → lower DOY
- β_cp: ambiguous. Sufficient chill breaks dormancy sooner, but chill is often non-limiting. Expected weak or slightly negative.
- β_wv < 0: faster warming → earlier bloom → lower DOY

If any coefficient has an unexpected sign, this should be flagged as a diagnostic warning (not a gate failure).

---

## 2.5 Empirical Bayes Shrinkage

### Hierarchical Interpretation

We model bloom_doy predictions as arising from a two-level hierarchy:

**Level 1 (site-specific):** The regression model produces a site-specific prediction f̂(s, 2026).

**Level 2 (global prior):** All sites share a common underlying bloom distribution with mean μ_global.

The Empirical Bayes framework combines these:

```
pred_final(s) = w_s · f̂(s) + (1 - w_s) · μ_global
```

### Weight Derivation

The optimal weight balances the precision of the site-specific estimate against the global prior. 

Define:
- σ²_global: variance of bloom_doy residuals across ALL training sites and years
- σ²_s: variance of the site-specific prediction (estimated from CV residuals for site s)
- N_s: number of training observations for site s

The precision (inverse variance) of the site-specific prediction is proportional to N_s / σ²_s.
The precision of the global mean is proportional to 1 / σ²_global.

The optimal Bayes weight:

```
w_s = σ²_global / (σ²_global + (σ²_s + ε) / N_s)
```

### Variance Decomposition

- Numerator `σ²_global`: the total pool of predictive uncertainty across all sites. Larger σ²_global → the global mean is less informative → more weight on site-specific estimate.

- Denominator adds `(σ²_s + ε) / N_s`: the site-specific estimation uncertainty. For large N_s (DC: 76), this term is small → w_s ≈ 1 → trust the model. For small N_s (NYC: 2), this term is large → w_s << 1 → trust the global mean.

### ε Stabilization

The floor ε > 0 prevents pathological behavior:

1. **Division stability:** If σ²_s = 0 (model perfectly fits all site-s training points), then without ε, w_s = N_s · σ²_global / (N_s · σ²_global) = 1 regardless of N_s. This would give NYC w=1 if the model happens to perfectly fit its 2 points — overfitting masquerading as confidence.

2. **Calibration:** ε should be calibrated to represent a minimum plausible prediction variance. A reasonable approach: set ε = median(σ²_s across all sites) / 10, or use LOO residual variance as a lower bound.

3. **Conservative default:** ε = 1.0 (1 day² of irreducible variance) is a safe starting point. We can tune via LOO.

### Expected Behavior by Site

| Site | N_s | Expected w_s | Interpretation |
|------|-----|-------------|----------------|
| washingtondc | ~76 | ~0.90-0.95 | Trust model strongly |
| kyoto | ~75 | ~0.90-0.95 | Trust model strongly |
| liestal | ~76 | ~0.90-0.95 | Trust model strongly |
| vancouver | 4 | ~0.30-0.50 | Significant shrinkage |
| nyc | 2 | ~0.15-0.30 | Heavy shrinkage toward global mean |

### Geometric Intuition

Imagine the number line of possible bloom DOYs. The global mean is a "gravity well" at DOY ~90. Each site's model prediction is a point on this line, connected to the global mean by a spring. The spring stiffness is inversely proportional to w_s:

- DC's spring is very slack (high w_s) → prediction stays near the model output
- NYC's spring is very stiff (low w_s) → prediction pulled strongly toward global mean

This is James-Stein shrinkage: we ALWAYS improve total MSE by shrinking, especially for the smallest groups.

---

# SECTION 3 — Execution Order (ABSOLUTE)

## 3.1 Pre-Loop Setup

```
LOAD gold_features from data/gold/features.parquet
SPLIT into training_df (year ≠ 2026) and prediction_df (year == 2026)
LOAD silver weather data paths for all 5 sites
LOAD silver ASOS data for sites with stations
SET random seed = 42
EXTRACT unique_years = sorted(training_df['year'].unique())
INIT cv_results = []
INIT cv_fold_metadata = []
```

## 3.2 Per-Fold Loop

```
FOR each Y in unique_years:

    ┌─── STEP 1: Data Split ───┐
    │ train = training_df[year ≠ Y]
    │ test  = training_df[year == Y]
    │ IF test is empty: CONTINUE (no labels this year)
    └──────────────────────────┘

    ┌─── STEP 2: Fold-Safe Bias Correction ───┐
    │ FOR each site s with ASOS:
    │   Load ERA5 hourly for site s
    │   Load ASOS hourly for site s
    │   EXCLUDE year Y from both
    │   Fit OLS: T_asos ~ β₀ + β₁·T_era5
    │   Store: bias_s_Y = {β₀, β₁}
    │ FOR sites without ASOS:
    │   bias_s_Y = {β₀=0, β₁=1}
    └──────────────────────────────────────────┘

    ┌─── STEP 3: Fold-Safe Mean Bloom DOY ───┐
    │ FOR each site s:
    │   site_train = train[site_key == s]
    │   IF |site_train| > 0:
    │     mean_bloom_s_Y = mean(site_train['bloom_doy'])
    │   ELSE:
    │     mean_bloom_s_Y = mean(train['bloom_doy'])  # global fallback
    └────────────────────────────────────────┘

    ┌─── STEP 4: Compute Warming Velocity ───┐
    │ FOR each (site, year) in UNION(train, test):
    │   Load silver weather for site
    │   window_center = mean_bloom_s_Y  (from step 3)
    │   window = [window_center - 14, window_center]
    │   Extract daily mean temps in window for this year
    │   Fit slope: T̄_daily ~ a + b·doy
    │   wv(site, year) = b
    │ APPEND wv to feature vectors
    └────────────────────────────────────────┘

    ┌─── STEP 5: Fit Regression ───┐
    │ X_train = [site_dummies, gdh, cp, wv] for train rows
    │ y_train = train['bloom_doy']
    │ Fit OLS: y_train = X_train · β
    │ Store: β_fold_Y = estimated coefficients
    └──────────────────────────────┘

    ┌─── STEP 6: Predict Test ───┐
    │ X_test = [site_dummies, gdh, cp, wv] for test rows
    │ y_pred = X_test · β_fold_Y
    │ y_actual = test['bloom_doy']
    │ residual = y_actual - y_pred
    │ APPEND to cv_results: {site, year, y_pred, y_actual, residual, fold_Y}
    └────────────────────────────┘

END FOR
```

## 3.3 Post-Loop: Shrinkage Weight Estimation

```
COMPUTE σ²_global = var(all residuals in cv_results)
FOR each site s:
    site_residuals = cv_results[site_key == s]
    σ²_s = var(site_residuals) if |site_residuals| >= 2, else σ²_global
    N_s = |training_df[site_key == s]|
    w_s = σ²_global / (σ²_global + (σ²_s + ε) / N_s)
SAVE shrinkage_weights.json
```

## 3.4 Post-Loop: Full-Data Refit for Phase 3

```
REFIT bias correction on ALL years (no holdout) → save bias_coefficients.json
REFIT mean_bloom_doy on ALL years → save per-site means
RECOMPUTE WV on ALL years with full-data mean_bloom_doy
REFIT regression on ALL training data → save model_coefficients.json
```

This full-data refit produces the coefficients Phase 3 will use for 2026 inference.

## 3.5 Post-Loop: MAE Computation

```
FOR each site s:
    site_cv = cv_results[site_key == s]
    mae_s = mean(|site_cv['residual']|)
overall_mae = mean(|cv_results['residual']|)
SAVE mae_summary.json
```

---

# SECTION 4 — Full Phase 2 Pseudocode

## 4.1 estimate_bias()

```python
def estimate_bias(
    asos_path: Path,          # Path to ASOS parquet for one station
    era5_path: Path,          # Path to ERA5 consolidated parquet for matching site
    exclude_year: int | None, # Year to exclude (fold-safety)
) -> dict:
    """
    Returns: {"beta0": float, "beta1": float, "r2": float, "n_obs": int}
    """
    # Load data
    asos_df = read_parquet(asos_path)  # columns: timestamp(UTC), temperature_2m
    era5_df = read_parquet(era5_path)  # columns: timestamp(UTC), temperature_2m

    # Round both to nearest hour for join
    asos_df["hour"] = asos_df["timestamp"].dt.floor("h")
    era5_df["hour"] = era5_df["timestamp"].dt.floor("h")

    # Inner join on hour
    merged = inner_join(asos_df, era5_df, on="hour", suffixes=("_asos", "_era5"))

    # Exclude held-out year
    IF exclude_year IS NOT None:
        merged = merged[merged["hour"].dt.year != exclude_year]

    # Drop rows where either temp is NaN
    merged = merged.dropna(subset=["temperature_2m_asos", "temperature_2m_era5"])

    # Fit OLS
    X = np.column_stack([np.ones(len(merged)), merged["temperature_2m_era5"]])
    y = merged["temperature_2m_asos"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # R²
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    RETURN {"beta0": beta[0], "beta1": beta[1], "r2": r2, "n_obs": len(merged)}
```

## 4.2 compute_warming_velocity()

```python
def compute_warming_velocity(
    weather_path: Path,      # ERA5 consolidated parquet for site
    year: int,               # Year to compute WV for
    window_center_doy: int,  # Fold-safe mean bloom DOY
    window_days: int = 14,   # Window length
) -> dict:
    """
    Returns: {"wv": float, "r2": float, "n_days": int, "uses_forecast": bool}
    """
    df = read_parquet(weather_path)

    window_start = window_center_doy - window_days
    window_end = window_center_doy

    # Filter to target year and DOY range
    year_data = df[(df["timestamp"].dt.year == year)]
    year_data["doy"] = year_data["timestamp"].dt.dayofyear
    window_data = year_data[(year_data["doy"] >= window_start) & (year_data["doy"] <= window_end)]

    # Compute daily mean temperature
    daily_means = window_data.groupby("doy")["temperature_2m"].mean().reset_index()
    daily_means.columns = ["doy", "t_mean"]

    IF len(daily_means) < 3:
        RETURN {"wv": NaN, "r2": NaN, "n_days": len(daily_means), "uses_forecast": False}

    # Linear regression: t_mean ~ a + b * doy
    X = np.column_stack([np.ones(len(daily_means)), daily_means["doy"]])
    y = daily_means["t_mean"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # R²
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    RETURN {
        "wv": beta[1],          # slope: °C per day
        "r2": r2,
        "n_days": len(daily_means),
        "uses_forecast": False   # Always False in Phase 2 (historical data)
    }
```

## 4.3 SYHCrossValidator.run()

```python
class SYHCrossValidator:
    def __init__(self, gold_features_path, silver_weather_dir, silver_asos_dir, sites_config):
        self.features = read_parquet(gold_features_path)
        self.training = self.features[self.features["year"] != 2026]
        self.weather_dir = silver_weather_dir
        self.asos_dir = silver_asos_dir
        self.sites = sites_config

        # Determinism
        np.random.seed(42)

    def run(self) -> pd.DataFrame:
        unique_years = sorted(self.training["year"].unique())
        cv_results = []
        fold_log = []   # for gate verification

        FOR Y in unique_years:
            train = self.training[self.training["year"] != Y]
            test  = self.training[self.training["year"] == Y]

            IF len(test) == 0:
                CONTINUE

            # --- Fold-safe bias correction ---
            bias_coeffs = {}
            FOR site_key, site_cfg in self.sites.items():
                IF site_cfg.asos_stations:
                    # Use first station as primary
                    asos_path = self.asos_dir / f"{site_cfg.asos_stations[0]}.parquet"
                    era5_path = self.weather_dir / site_key / f"{site_key}_consolidated.parquet"
                    IF asos_path.exists() AND era5_path.exists():
                        bias_coeffs[site_key] = estimate_bias(asos_path, era5_path, exclude_year=Y)
                    ELSE:
                        bias_coeffs[site_key] = {"beta0": 0.0, "beta1": 1.0}
                ELSE:
                    bias_coeffs[site_key] = {"beta0": 0.0, "beta1": 1.0}

            fold_log.append({"year": Y, "bias_exclude_year": Y})

            # --- Fold-safe mean bloom DOY per site ---
            mean_bloom = {}
            FOR site_key in self.sites:
                site_train = train[train["site_key"] == site_key]
                IF len(site_train) > 0:
                    mean_bloom[site_key] = site_train["bloom_doy"].mean()
                ELSE:
                    mean_bloom[site_key] = train["bloom_doy"].mean()

            fold_log[-1]["mean_bloom_from_train_only"] = True

            # --- Compute WV for all site-years in this fold ---
            all_fold_data = pd.concat([train, test])
            wv_values = {}
            FOR _, row in all_fold_data.iterrows():
                sk = row["site_key"]
                yr = row["year"]
                era5_path = self.weather_dir / sk / f"{sk}_consolidated.parquet"
                wv_result = compute_warming_velocity(
                    era5_path, yr, window_center_doy=int(mean_bloom[sk])
                )
                wv_values[(sk, yr)] = wv_result["wv"]

            # --- Build regression matrices ---
            # Add WV to train/test
            train_wv = [wv_values.get((r["site_key"], r["year"]), np.nan)
                        for _, r in train.iterrows()]
            test_wv  = [wv_values.get((r["site_key"], r["year"]), np.nan)
                        for _, r in test.iterrows()]

            # Site dummy encoding (no intercept — site dummies absorb it)
            site_names = sorted(self.sites.keys())

            X_train = np.column_stack([
                np.array([(1 if r["site_key"] == s else 0) for s in site_names]
                         for _, r in train.iterrows()),
                train["gdh"].values,
                train["cp"].values,
                np.array(train_wv),
            ])
            y_train = train["bloom_doy"].values

            X_test = np.column_stack([
                np.array([(1 if r["site_key"] == s else 0) for s in site_names]
                         for _, r in test.iterrows()),
                test["gdh"].values,
                test["cp"].values,
                np.array(test_wv),
            ])
            y_test = test["bloom_doy"].values

            # Handle NaN in WV (fill with 0 — neutral contribution)
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            # --- Fit OLS ---
            beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

            # --- Predict ---
            y_pred = X_test @ beta
            residuals = y_test - y_pred

            # --- Record ---
            FOR i, (_, row) in enumerate(test.iterrows()):
                cv_results.append({
                    "site_key": row["site_key"],
                    "year": row["year"],
                    "predicted_doy": float(y_pred[i]),
                    "actual_doy": float(y_test[i]),
                    "residual": float(residuals[i]),
                    "fold_holdout_year": Y,
                })

        RETURN pd.DataFrame(cv_results), fold_log
```

## 4.4 compute_shrinkage_weights()

```python
def compute_shrinkage_weights(
    cv_results: pd.DataFrame,
    training_df: pd.DataFrame,
    epsilon: float = 1.0,
) -> dict:
    """
    Returns: {site_key: {"w": float, "sigma2_s": float, "n_s": int}}
    """
    all_residuals = cv_results["residual"].values
    sigma2_global = np.var(all_residuals, ddof=1)

    weights = {}
    FOR site_key in training_df["site_key"].unique():
        site_residuals = cv_results[cv_results["site_key"] == site_key]["residual"].values
        n_s = len(training_df[training_df["site_key"] == site_key])

        IF len(site_residuals) >= 2:
            sigma2_s = np.var(site_residuals, ddof=1)
        ELSE:
            sigma2_s = sigma2_global  # fallback for sites with <2 CV results

        w_s = sigma2_global / (sigma2_global + (sigma2_s + epsilon) / n_s)

        weights[site_key] = {
            "w": float(w_s),
            "sigma2_s": float(sigma2_s),
            "sigma2_global": float(sigma2_global),
            "n_s": int(n_s),
            "epsilon": float(epsilon),
        }

    RETURN weights
```

## 4.5 Full-Data Refit (for Phase 3)

```python
def refit_full_model(training_df, weather_dir, asos_dir, sites_config):
    """
    Refit all Phase 2 components on FULL training data (no holdout).
    Produces coefficients that Phase 3 will use for 2026 inference.
    """
    # Bias correction: full data
    bias_coefficients = {}
    FOR site_key, cfg in sites_config.items():
        IF cfg.asos_stations:
            asos_path = asos_dir / f"{cfg.asos_stations[0]}.parquet"
            era5_path = weather_dir / site_key / f"{site_key}_consolidated.parquet"
            bias_coefficients[site_key] = estimate_bias(asos_path, era5_path, exclude_year=None)
        ELSE:
            bias_coefficients[site_key] = {"beta0": 0.0, "beta1": 1.0}

    # Mean bloom DOY: full data
    mean_bloom = {}
    FOR site_key in sites_config:
        site_data = training_df[training_df["site_key"] == site_key]
        IF len(site_data) > 0:
            mean_bloom[site_key] = float(site_data["bloom_doy"].mean())
        ELSE:
            mean_bloom[site_key] = float(training_df["bloom_doy"].mean())

    # WV: full data with full-data mean bloom
    wv_values = {}
    FOR _, row in training_df.iterrows():
        sk, yr = row["site_key"], row["year"]
        era5_path = weather_dir / sk / f"{sk}_consolidated.parquet"
        wv_result = compute_warming_velocity(era5_path, yr, int(mean_bloom[sk]))
        wv_values[(sk, yr)] = wv_result["wv"]

    # Regression: full data
    site_names = sorted(sites_config.keys())
    X = build_design_matrix(training_df, wv_values, site_names)
    y = training_df["bloom_doy"].values
    X = np.nan_to_num(X, nan=0.0)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    model_coefficients = {
        "site_intercepts": {s: float(beta[i]) for i, s in enumerate(site_names)},
        "beta_gdh": float(beta[len(site_names)]),
        "beta_cp": float(beta[len(site_names) + 1]),
        "beta_wv": float(beta[len(site_names) + 2]),
        "feature_order": site_names + ["gdh", "cp", "wv"],
    }

    global_mean_bloom = float(training_df["bloom_doy"].mean())

    SAVE bias_coefficients → data/processed/bias_coefficients.json
    SAVE model_coefficients → data/processed/model_coefficients.json
    SAVE mean_bloom → data/processed/mean_bloom_doy.json
    SAVE {"global_mean": global_mean_bloom} → data/processed/global_mean.json

    RETURN bias_coefficients, model_coefficients, mean_bloom, global_mean_bloom
```

---

# SECTION 5 — Fold Safety Proof

## 5.1 Leakage Points

| Component | Information That Could Leak | How |
|-----------|---------------------------|-----|
| Bias correction β₀, β₁ | Year Y's temperature distribution | If estimated on full data including Y |
| mean_bloom_doy(s) | Year Y's bloom_doy | If computed including Y |
| Warming velocity | Year Y's bloom_doy (via window center) | If window_center uses mean including Y |
| σ²_global (shrinkage) | Year Y's residual | If computed from all CV folds including Y's prediction |

## 5.2 How Architecture Prevents Leakage

1. **Bias correction:** `estimate_bias(exclude_year=Y)` removes all year-Y hourly data from the OLS fit.

2. **mean_bloom_doy:** Computed as `mean(train[site_key == s]['bloom_doy'])` where `train` excludes year Y.

3. **Warming velocity:** Window center derived from fold-safe mean_bloom_doy (point 2). The weather observations for year Y ARE used (temperature is not the label), but the window LOCATION is fold-safe.

4. **Shrinkage weights:** Computed AFTER the full CV loop from the aggregated residuals. This is NOT per-fold — it's a post-hoc summary statistic. This is acceptable because shrinkage weights are applied at INFERENCE time (Phase 3), not during CV prediction. The CV predictions themselves do not use shrinkage.

## 5.3 Incorrect Implementation Counterexample

**Bug:** Computing mean_bloom_doy ONCE before the CV loop:
```python
# WRONG
mean_bloom = training_df.groupby("site_key")["bloom_doy"].mean()
for Y in unique_years:
    # mean_bloom includes year Y → LEAKED
```

**Consequence for Vancouver (N=4):**
- Holdout Y=2023, bloom_doy=96 (latest)
- Correct mean (excl 2023): (86+83+93)/3 = 87.3
- Leaked mean (incl 2023): (86+96+83+93)/4 = 89.5
- Window shift: 2.2 days → WV computed from warmer part of spring → systematically biases prediction toward later bloom
- This makes the prediction for 2023 artificially better (closer to actual 96)
- **Optimistic MAE bias for Vancouver: ~1-2 days**

---

# SECTION 6 — Determinism Specification

## 6.1 Random Seeds

```python
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
```

Although the pipeline is primarily deterministic (OLS, linear regression), seed control ensures:
- Any future stochastic extensions (e.g., bootstrap) are reproducible
- NumPy operations with potential floating-point ordering effects are stable

## 6.2 Ordering Guarantees

- `unique_years` MUST be sorted ascending: `sorted(training_df['year'].unique())`
- Sites in design matrix MUST be sorted alphabetically: `sorted(sites_config.keys())` → `[kyoto, liestal, newyorkcity, vancouver, washingtondc]`
  - NOTE: NYC's site_key in the data is "nyc" (from config) but the location_id in labels is "newyorkcity". Verify which key the gold features use and sort accordingly.
- CV results appended in year-then-site order

## 6.3 Parallelism

`n_jobs = 1` for the CV loop. No parallel fold execution. This eliminates race conditions and ensures deterministic output ordering.

For bias correction within a fold: sequential over sites. No parallel station fetches.

## 6.4 Numeric Stability

- Use `np.linalg.lstsq(rcond=None)` for OLS (handles near-singular matrices gracefully)
- Variance computations use `ddof=1` (Bessel's correction)
- Check for `σ²_global = 0` (would mean perfect CV fit — impossible in practice but guard against it)
- WV: check for constant temperature in window (ss_tot = 0 → R² undefined → set WV = 0)

---

# SECTION 7 — Computational Complexity

## 7.1 Time Complexity

| Operation | Complexity | Dominant Factor |
|-----------|-----------|-----------------|
| CV loop | O(|Y| × |S| × H) | |Y|≈76 folds × |S|=5 sites × H=hourly data I/O |
| Bias correction per fold | O(N_overlap) | N_overlap ≈ 200K hourly obs per station |
| WV computation per site-year | O(H_window) | H_window ≈ 14×24 = 336 hours |
| OLS regression per fold | O(N_train × p²) | N_train ≈ 230, p = 8 → trivial |
| Shrinkage weights | O(|S| × N_cv) | One-time, trivial |

**Bottleneck:** I/O for loading parquet files in the WV computation loop. Each fold reloads ERA5 data for each site to compute WV.

**Mitigation:** Pre-load all silver weather data into memory at the start. 5 sites × ~7 MB each = ~35 MB. Fits easily in memory.

## 7.2 Expected Runtime

- Bias correction: ~76 folds × 3 ASOS sites × ~1s = ~4 min
- WV computation: ~76 folds × ~234 site-years × ~0.01s = ~3 min (if pre-loaded)
- OLS: trivial (<1s total)
- **Total: ~5-10 minutes** (well within 15-min SLA)

## 7.3 Memory Hotspots

- Weather data pre-loaded: ~35 MB (5 sites)
- ASOS data pre-loaded: ~15 MB (5 stations)
- CV results DataFrame: ~76 × 5 × 6 columns ≈ negligible
- **Peak: ~50 MB** — no issues.

---

# SECTION 8 — Validation Gates Derivation

## 8.1 assert_bias_fold_safe

**What it prevents:** Bias correction information leaking from test year to training.

**Implementation:** The CV fold_log records `{"year": Y, "bias_exclude_year": Y}` for each fold. The gate verifies that for every fold entry, `bias_exclude_year == year`. If any fold has `bias_exclude_year != year` or `bias_exclude_year == None`, the gate fails.

**Failure mode prevented:** Accidentally passing `exclude_year=None` to `estimate_bias()` inside the CV loop.

## 8.2 assert_window_safe

**What it prevents:** Warming velocity window center derived from data including the test year's bloom_doy.

**Implementation:** The fold_log records `mean_bloom_from_train_only = True/False`. The gate verifies all entries are True. Additionally, the gate can recompute mean_bloom from the recorded training indices and compare to the stored value.

**Failure mode prevented:** Computing mean_bloom_doy once before the loop instead of per-fold.

## 8.3 assert_precision_fold_safe

**What it prevents:** Shrinkage weights computed from data that includes test predictions.

**Implementation:** Verify that shrinkage weights are computed AFTER the full CV loop, from aggregated residuals, and that they are applied only at inference time (Phase 3), not during CV predictions. The gate checks that no cv_results row has a "shrunk_prediction" field — only raw model predictions.

**Failure mode prevented:** Applying shrinkage during CV (which would make CV MAE non-representative).

## 8.4 assert_vancouver_weight_stable

**What it prevents:** Excessive variance in Vancouver's shrinkage weight across bootstrap/perturbation.

**Implementation:** Compute w_vancouver from the full CV residuals. Then compute leave-one-fold-out variants of w_vancouver (drop one fold's residuals and recompute). Assert that the standard deviation of these variants < 0.15.

**Why 0.15:** Vancouver has 4 labels. Removing one fold removes up to 25% of Vancouver's CV data. If the weight swings by more than 0.15, the weight estimate is unstable and the shrinkage is unreliable.

## 8.5 assert_cv_no_leakage

**What it prevents:** Any form of information from the test year reaching the training procedure.

**Implementation:** For each CV fold, verify:
1. No training row has year == holdout_year
2. Bias correction excluded the holdout year
3. mean_bloom_doy was computed from training years only
4. The fold's coefficient vector differs from adjacent folds (proving it was re-estimated)

**Meta-check:** Compute "forward-looking residual correlation." If residual(s, Y) is correlated with residual(s, Y+1) at r > 0.5, flag as suspicious (could indicate temporal leakage).

---

# SECTION 9 — Expected Artifacts

## 9.1 cv_results.parquet

Schema:
```
site_key:          string    # "washingtondc", "kyoto", etc.
year:              int       # Held-out year
predicted_doy:     float     # Model prediction (no shrinkage)
actual_doy:        float     # Ground truth bloom_doy
residual:          float     # actual - predicted
fold_holdout_year: int       # Same as year (for verification)
```

Expected rows: ~360-380 (each site-year is tested once; only years where site has a label)

## 9.2 mae_summary.json

```json
{
    "overall_mae": 5.23,
    "per_site": {
        "washingtondc": {"mae": 4.1, "n_folds": 76},
        "kyoto": {"mae": 5.8, "n_folds": 75},
        "liestal": {"mae": 6.2, "n_folds": 76},
        "vancouver": {"mae": 4.5, "n_folds": 4},
        "nyc": {"mae": 3.9, "n_folds": 2}
    }
}
```

Note: NYC/Vancouver MAEs are unreliable due to tiny fold counts. Report them but flag this.

## 9.3 shrinkage_weights.json

```json
{
    "washingtondc": {"w": 0.93, "sigma2_s": 18.2, "sigma2_global": 42.5, "n_s": 76, "epsilon": 1.0},
    "kyoto":        {"w": 0.92, ...},
    "liestal":      {"w": 0.91, ...},
    "vancouver":    {"w": 0.38, ...},
    "nyc":          {"w": 0.22, ...}
}
```

## 9.4 bias_coefficients.json (full-data fit)

```json
{
    "washingtondc": {"beta0": 0.5, "beta1": 0.98, "r2": 0.97, "n_obs": 180000},
    "kyoto":        {"beta0": 0.0, "beta1": 1.0, "r2": null, "n_obs": null},
    "liestal":      {"beta0": 0.0, "beta1": 1.0, "r2": null, "n_obs": null},
    "vancouver":    {"beta0": 0.3, "beta1": 0.99, "r2": 0.96, "n_obs": 190000},
    "nyc":          {"beta0": 0.4, "beta1": 0.97, "r2": 0.95, "n_obs": 175000}
}
```

## 9.5 model_coefficients.json (full-data fit)

```json
{
    "site_intercepts": {"kyoto": 105.2, "liestal": 112.3, "nyc": 98.1, "vancouver": 95.7, "washingtondc": 100.4},
    "beta_gdh": -0.023,
    "beta_cp": -0.008,
    "beta_wv": -1.45,
    "feature_order": ["kyoto", "liestal", "nyc", "vancouver", "washingtondc", "gdh", "cp", "wv"]
}
```

## 9.6 Additional Artifacts

- `data/processed/mean_bloom_doy.json`: per-site mean bloom DOY from full training data
- `data/processed/global_mean.json`: single float, overall mean bloom DOY

---

# SECTION 10 — Engineer Implementation Checklist

| Step | File | Action | Validation |
|------|------|--------|------------|
| 1 | `src/processing/bias_correction.py` | Implement `estimate_bias()` per Section 4.1 | `python -c "from src.processing.bias_correction import estimate_bias; print('ok')"` |
| 2 | `src/processing/warming_velocity.py` | Implement `compute_warming_velocity()` per Section 4.2 | Smoke test on DC 2020: should return finite slope |
| 3 | `src/modeling/__init__.py` | Create empty init | Module import works |
| 4 | `src/modeling/syh_cv.py` | Implement `SYHCrossValidator` per Section 4.3 | `python -m src.modeling.syh_cv` → cv_results.parquet exists |
| 5 | `src/modeling/empirical_bayes.py` | Implement `compute_shrinkage_weights()` and `apply_shrinkage()` per Section 4.4 | shrinkage_weights.json exists, NYC w < 0.5 |
| 6 | Add full-data refit | Either in syh_cv.py or separate module per Section 4.5 | model_coefficients.json + bias_coefficients.json exist |
| 7 | `src/validation/gates.py` | Add all 5 Phase 2 gates per Section 8 | All gates raise AssertionError with descriptive messages |
| 8 | `src/validation/run_all_gates.py` | Add phase "2" to PHASE_GATES map | `python -m src.validation.run_all_gates --phase 2` runs |
| 9 | Run SYH CV | Execute full pipeline | cv_results.parquet + mae_summary.json produced |
| 10 | Run Phase 2 gates | `python -m src.validation.run_all_gates --phase 2` | All PASS |
| 11 | Diagnostic review | Check coefficient signs, MAE plausibility, weight ranges | Flags documented in debrief |

---

# SECTION 11 — CLI Agent Prompt

Below is the prompt to paste into Claude Code.

---

```
# CLI_AGENT_PROMPT — Phase 2 Implementation

## 0) ROLE + NON-NEGOTIABLES

You are implementing Phase 2 of the Phenology Engine v1.7. Architecture is FROZEN.

Non-negotiables:
- ZERO data leakage. All statistics re-estimated per fold.
- ALL timestamps UTC-aware
- Deterministic: np.random.seed(42), sorted year iteration, n_jobs=1
- Gates raise AssertionError with descriptive messages
- No network access (Phase 2 is fully offline)

Read these files FIRST, in order:
1. context/ARCHITECTURE.md
2. context/agent.md
3. context/phase2.md

Then confirm by printing 5 key invariants from the specs.

---

## 1) PRE-FLIGHT

Run and report:
- source venv/bin/activate && python3 --version
- ls data/gold/features.parquet data/silver/weather/*/washingtondc_consolidated.parquet data/silver/asos/DCA.parquet
- python3 -c "import pandas as pd; df=pd.read_parquet('data/gold/features.parquet'); print('Gold:', df.shape); print(df.groupby('site_key')['year'].agg(['count']).to_string())"

Confirm: gold features exist with 239 rows, 5 sites.

---

## 2) CREATE: src/processing/bias_correction.py

Implement function:

    estimate_bias(asos_path, era5_path, exclude_year=None) -> dict

Logic:
- Load ASOS and ERA5 parquets
- Floor both timestamps to nearest hour
- Inner join on hour column
- If exclude_year is not None: filter OUT that year (this is the fold-safety mechanism)
- Drop rows where either temperature is NaN
- Fit OLS via np.linalg.lstsq: T_asos = β₀ + β₁ * T_era5
- Compute R²
- Return {"beta0": float, "beta1": float, "r2": float, "n_obs": int}

For sites without ASOS (kyoto, liestal): caller passes identity {"beta0": 0, "beta1": 1.0}

Type hints on all args. Google-style docstring. Import constants from config.settings.

Acceptance test:
    python3 -c "from src.processing.bias_correction import estimate_bias; from pathlib import Path; r = estimate_bias(Path('data/silver/asos/DCA.parquet'), Path('data/silver/weather/washingtondc/washingtondc_consolidated.parquet')); print(r); assert 0.9 < r['beta1'] < 1.1, f'beta1={r[\"beta1\"]}'"

---

## 3) CREATE: src/processing/warming_velocity.py

Implement function:

    compute_warming_velocity(weather_path, year, window_center_doy, window_days=14) -> dict

Logic:
- Load ERA5 consolidated parquet for the site
- Define window: [window_center_doy - window_days, window_center_doy]
- Filter to target year and DOY range
- Compute daily mean temperature (groupby DOY)
- If fewer than 3 days of data: return {"wv": NaN, "n_days": int, ...}
- Fit linear regression: T_daily_mean = a + b * DOY
- Return {"wv": b, "r2": float, "n_days": int, "uses_forecast": False}

CRITICAL: window_center_doy is passed IN by the caller. This function does NOT compute it. The caller (SYH CV) is responsible for providing the fold-safe value.

Acceptance test:
    python3 -c "from src.processing.warming_velocity import compute_warming_velocity; from pathlib import Path; r = compute_warming_velocity(Path('data/silver/weather/washingtondc/washingtondc_consolidated.parquet'), 2020, 87); print(r); assert r['n_days'] >= 10"

---

## 4) CREATE: src/modeling/__init__.py

Empty file. Just ensures the package is importable.

Also create src/modeling/ directory if not exists.

---

## 5) CREATE: src/modeling/syh_cv.py

Implement class SYHCrossValidator with method run() that returns (cv_results_df, fold_log).

This is the core of Phase 2. Follow this EXACT execution order per fold:

FOR each holdout year Y (sorted ascending):
  1. Split: train = gold[year != Y and year != 2026], test = gold[year == Y]
  2. If test empty, skip
  3. Bias correction: for each ASOS site, call estimate_bias(exclude_year=Y). Non-ASOS sites get identity.
  4. Mean bloom DOY: per site from training fold only. If site has 0 training rows, use global training mean.
  5. Warming velocity: for all (site, year) pairs in train+test, call compute_warming_velocity with fold-safe mean_bloom_doy as window_center.
  6. Build design matrix: site dummies (sorted alpha, no intercept) + gdh + cp + wv. NaN WV → fill with 0.
  7. OLS fit on training rows: np.linalg.lstsq(X_train, y_train, rcond=None)
  8. Predict test rows: y_pred = X_test @ beta
  9. Record: site_key, year, predicted_doy, actual_doy, residual, fold_holdout_year

IMPORTANT architectural decisions:
- Site names for dummy encoding: use site_key values from gold features, sorted alphabetically
- WV is computed for BOTH train and test rows in each fold (test rows need WV for prediction)
- The WV for test rows uses the FOLD-SAFE window center (not the test year's bloom_doy)
- fold_log must record: year, bias_exclude_year, mean_bloom_from_train_only=True

AFTER the CV loop:
- Compute per-site MAE and overall MAE → save data/processed/mae_summary.json
- Save CV results → data/processed/cv_results.parquet

PERFORMANCE NOTE: Pre-load all silver weather dataframes into a dict at __init__ time to avoid repeated I/O. ~35 MB total.

Also implement __main__ block:
    if __name__ == "__main__":
        # Load config, run CV, save artifacts, print summary

Acceptance:
    python3 -m src.modeling.syh_cv
    Expected: cv_results.parquet created with ~360 rows, mae_summary.json with per-site MAE

---

## 6) CREATE: src/modeling/empirical_bayes.py

Implement:

    compute_shrinkage_weights(cv_results_df, training_df, epsilon=1.0) -> dict

Formula:
    w_s = σ²_global / (σ²_global + (σ²_s + ε) / N_s)

Where:
- σ²_global = variance of ALL residuals in cv_results (ddof=1)
- σ²_s = variance of site-s residuals (ddof=1). If site has <2 residuals, use σ²_global.
- N_s = number of training rows for site s (NOT number of CV residuals)
- ε = epsilon floor (default 1.0)

Return dict keyed by site_key with: w, sigma2_s, sigma2_global, n_s, epsilon

Also implement:

    apply_shrinkage(model_pred, global_mean, w) -> float
        return w * model_pred + (1 - w) * global_mean

Acceptance:
    After CV completes, compute weights. Verify:
    - DC/Kyoto/Liestal weights > 0.8
    - Vancouver weight between 0.2 and 0.6
    - NYC weight < 0.5
    Save → data/processed/shrinkage_weights.json

---

## 7) ADD TO src/modeling/syh_cv.py: Full-data refit

After CV loop and shrinkage computation, refit on ALL training data (no holdout):
- Bias correction with exclude_year=None
- Mean bloom DOY from all training years
- WV with full-data mean bloom DOY
- OLS regression on all training rows
- Save:
  - data/processed/bias_coefficients.json
  - data/processed/model_coefficients.json (site intercepts + beta_gdh, beta_cp, beta_wv + feature_order)
  - data/processed/mean_bloom_doy.json (per-site means)
  - data/processed/global_mean.json (single float)

These are the artifacts Phase 3 consumes.

---

## 8) ADD TO src/validation/gates.py: Phase 2 gates

Implement these 5 gate functions:

    assert_bias_fold_safe(fold_log):
        For every fold, verify bias_exclude_year == fold year.

    assert_window_safe(fold_log):
        For every fold, verify mean_bloom_from_train_only is True.

    assert_precision_fold_safe(cv_results):
        Verify cv_results has no "shrunk_prediction" column (shrinkage not applied during CV).

    assert_vancouver_weight_stable(shrinkage_weights):
        Compute leave-one-out variants of vancouver weight. Assert std < 0.15.
        If vancouver has fewer than 3 CV residuals, skip with warning.

    assert_cv_no_leakage(cv_results, training_df):
        For each fold: verify no training row has year == holdout year.
        Verify cv_results has entries for multiple distinct holdout years.

All gates raise AssertionError with descriptive message on failure.

---

## 9) UPDATE: src/validation/run_all_gates.py

Add "2" to the PHASE_GATES map. Phase 2 gates should:
- Load cv_results.parquet
- Load shrinkage_weights.json
- Load fold_log (either from parquet or json — save it alongside cv_results)
- Load gold features (for training_df)
- Execute all 5 Phase 2 gates

Acceptance:
    python3 -m src.validation.run_all_gates --phase 2
    Should execute all Phase 2 gates.

---

## 10) RUN FULL PHASE 2 PIPELINE

Execute:
    python3 -m src.modeling.syh_cv

Then:
    python3 -m src.validation.run_all_gates --phase 2

Report:
- All gate results (PASS/FAIL)
- Per-site MAE from mae_summary.json
- Shrinkage weights from shrinkage_weights.json
- Model coefficient signs (β_gdh should be negative, β_wv should be negative)
- Any warnings

---

## 11) DIAGNOSTICS

After pipeline completes, run:

    python3 -c "
    import pandas as pd, json
    cv = pd.read_parquet('data/processed/cv_results.parquet')
    print('CV results shape:', cv.shape)
    print()
    print('Per-site MAE:')
    for s in sorted(cv['site_key'].unique()):
        site_cv = cv[cv['site_key'] == s]
        mae = site_cv['residual'].abs().mean()
        print(f'  {s}: MAE={mae:.2f} days ({len(site_cv)} folds)')
    print(f'  OVERALL: {cv[\"residual\"].abs().mean():.2f} days')
    print()
    with open('data/processed/shrinkage_weights.json') as f:
        w = json.load(f)
    print('Shrinkage weights:')
    for s in sorted(w.keys()):
        print(f'  {s}: w={w[s][\"w\"]:.3f} (N={w[s][\"n_s\"]})')
    print()
    with open('data/processed/model_coefficients.json') as f:
        mc = json.load(f)
    print('Model coefficients:')
    print(f'  β_gdh = {mc[\"beta_gdh\"]:.6f}')
    print(f'  β_cp  = {mc[\"beta_cp\"]:.6f}')
    print(f'  β_wv  = {mc[\"beta_wv\"]:.4f}')
    print('Site intercepts:')
    for s, v in sorted(mc['site_intercepts'].items()):
        print(f'  {s}: {v:.2f}')
    "

---

## 12) STOP CONDITIONS

- If any Phase 2 gate fails → STOP
- If β_gdh is positive → log WARNING (counterintuitive but not necessarily wrong)
- If overall MAE > 15 days → STOP (model is broken)
- If NYC weight > 0.8 → STOP (shrinkage not working)
- If cv_results has 0 rows → STOP

Do NOT proceed to Phase 3.

---

## 13) TECHNICAL DEBRIEF

Save as TECHNICAL_DEBRIEF_P2.md:

    # TECHNICAL DEBRIEF — Phase 2 Modeling & CV

    ## A) Executive Summary
    [COMPLETE/BLOCKED. Per-site MAE summary. Shrinkage weight summary.]

    ## B) Files Created
    | File | Purpose |
    |------|---------|
    | src/processing/bias_correction.py | ERA5→ASOS OLS bias estimator |
    | src/processing/warming_velocity.py | 14-day window slope estimator |
    | src/modeling/__init__.py | Package init |
    | src/modeling/syh_cv.py | SYH CV engine + full-data refit |
    | src/modeling/empirical_bayes.py | Shrinkage weight computation |

    ## C) Artifacts Produced
    | Artifact | Path | Size | Key Stats |
    |----------|------|------|-----------|
    | CV results | data/processed/cv_results.parquet | | rows: |
    | MAE summary | data/processed/mae_summary.json | | overall: |
    | Shrinkage weights | data/processed/shrinkage_weights.json | | |
    | Bias coefficients | data/processed/bias_coefficients.json | | |
    | Model coefficients | data/processed/model_coefficients.json | | |
    | Mean bloom DOY | data/processed/mean_bloom_doy.json | | |
    | Global mean | data/processed/global_mean.json | | |
    | Fold log | data/processed/fold_log.json | | |

    ## D) Gate Results
    | Gate | Result | Evidence |
    |------|--------|----------|
    | assert_bias_fold_safe | | |
    | assert_window_safe | | |
    | assert_precision_fold_safe | | |
    | assert_vancouver_weight_stable | | |
    | assert_cv_no_leakage | | |

    ## E) Model Diagnostics
    - Coefficient signs: β_gdh=, β_cp=, β_wv=
    - Per-site intercepts: [list]
    - Residual distribution: mean=, std=, skew=
    - Any unexpected patterns: [describe]

    ## F) Phase 3 Readiness
    [YES/NO. List any blockers.]
```

---

End of Phase 2 Implementation Blueprint.