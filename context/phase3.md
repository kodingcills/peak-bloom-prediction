# PHASE 3 BLUEPRINT — Inference & Bimodal Resolution

## Phenology Engine v1.7 — Execution-Safe Probabilistic Inference Specification

---

## SECTION 1 — Phase 3 Role in System Architecture

### 1.1 Why Phase 2 produces distributions, not answers

Phase 2 estimated the regression model:

    bloom_doy(s, y) = α_s + β_gdh · GDH + β_cp · CP + β_wv · WV + ε

with coefficients fitted on 234 training rows via site-year holdout CV. The full-data refit produced point estimates stored in `model_coefficients.json`. But this model maps **known features → predicted DOY**. For 2026, GDH and CP are observed (DOY 1–59), but **WV is unobservable** because its computation window (centered on mean_bloom_doy, typically DOY 73–95) extends past the inference cutoff (DOY 59).

Therefore Phase 2's regression model cannot produce a single 2026 prediction — it requires a forecast of post-cutoff temperatures. Different plausible temperature futures produce different WV values, which produce different bloom DOY predictions. The output is a **distribution over possible bloom dates**, not a point.

### 1.2 Why SEAS5 (or fallback) introduces multimodality

The SEAS5 seasonal forecast provides 50 ensemble members, each representing a physically consistent future temperature trajectory. In our FALLBACK mode (no SEAS5 available), we substitute 30 historical spring temperature trajectories (1996–2025) as pseudo-ensemble members.

Each member m produces:
- A distinct spring temperature trajectory T_m(t) for DOY 60–180
- A distinct WV_m computed from that trajectory
- A distinct predicted bloom DOY via f(GDH_2026, CP_2026, WV_m)

The resulting distribution of bloom DOYs can be **multimodal** when the historical record contains two distinct spring regimes:
- **Warm spring regime:** Rapid March warming → high WV → earlier bloom
- **Cold spring regime:** Delayed warming into April → low WV → later bloom

This bimodality is not noise — it reflects genuine climate variability in spring onset timing. A unimodal summary (e.g., ensemble mean) would split the difference between the two modes, potentially landing in a low-probability region between them.

### 1.3 How Phase 3 converts uncertainty → decision

Phase 3 is the **decision layer**:

    {WV_m : m = 1..M} → {DOY_m : m = 1..M} → GMM(k ∈ {1,2}) → BIC → cluster_select → shrinkage → round → submission

The pipeline is:
1. **Propagate** ensemble members through the regression model → M predicted bloom DOYs per site
2. **Detect** multimodality via GMM with k ∈ {1,2}, selected by BIC
3. **Select** the appropriate cluster (climatology-closest or dominant)
4. **Shrink** toward global mean via Empirical Bayes weights
5. **Round** to integer DOY
6. **Clip** to biologically feasible range [60, 140]
7. **Write** submission.csv

### 1.4 Dependency graph

```
Phase 1 (Online)
    ├── Silver weather parquets (ERA5-Land hourly)
    ├── Silver ASOS parquets
    ├── Gold features.parquet (GDH, CP observed through DOY 59)
    └── SEAS5_FETCH_FAILED flag (fallback mode trigger)
         │
         ▼
Phase 2 (Offline)
    ├── model_coefficients.json    {α_s, β_gdh, β_cp, β_wv, feature_order}
    ├── bias_coefficients.json     {per-site β₀, β₁}
    ├── shrinkage_weights.json     {per-site w_s}
    ├── mean_bloom_doy.json        {per-site climatological mean}
    ├── global_mean.json           {scalar}
    ├── cv_results.parquet         {234 rows, MAE=5.51}
    └── mae_summary.json           {per-site and overall MAE}
         │
         ▼
Phase 3 (Offline)  ← THIS SPECIFICATION
    ├── submission.csv                               (5 rows)
    ├── data/processed/diagnostics/ensemble_distributions.json
    ├── data/processed/diagnostics/gmm_results.json
    └── data/processed/diagnostics/prediction_summary.json
         │
         ▼
Phase 4 (Offline — Quarto render)
    ├── analysis.html (self-contained)
    └── submission.csv (unchanged, verified)
```

---

## SECTION 2 — Forecast Uncertainty Model

### 2.1 Temperature trajectory as random variable

Let T_e(t) denote the hourly temperature trajectory for ensemble member e at time t (hours since Jan 1, 2026). For the fallback ensemble, each "member" corresponds to a historical year Y_e:

    T_e(t) = T_2026(t)     for t ∈ [1, 59·24]     (observed)
    T_e(t) = T_{Y_e}(t')   for t > 59·24           (historical scenario)

where t' maps the 2026 hour-of-year to the corresponding hour in year Y_e.

### 2.2 Feature mapping under each member

For each site s and member e:

    GDH_e = GDH_2026(s)     (constant — fully observed DOY 1–59)
    CP_e  = CP_2026(s)       (constant — fully observed DOY 1–59)
    WV_e  = slope(T̄_d : d ∈ [d_center - 14, d_center])

where:
- d_center = round(mean_bloom_doy[s]) from Phase 2 full-data fit
- T̄_d = daily mean temperature on DOY d, computed from the spliced trajectory T_e
- slope is the OLS slope of T̄_d regressed on d

### 2.3 Ensemble-to-outcome mapping

    f_model(s, e) = α_s + β_gdh · GDH_2026(s) + β_cp · CP_2026(s) + β_wv · WV_e(s)

Since GDH and CP are constant across members:

    f_model(s, e) = C_s + β_wv · WV_e(s)

where C_s = α_s + β_gdh · GDH_2026(s) + β_cp · CP_2026(s) is a site-specific constant.

**This is a critical simplification:** The entire ensemble spread is driven solely by β_wv · WV_e. The distribution of predicted bloom DOYs is a **linear transformation of the distribution of WV values**.

### 2.4 Implications for multimodality

A GMM on {DOY_e} is bimodal if and only if {WV_e} is bimodal (since the mapping is affine). Bimodality in WV arises when historical springs cluster into two regimes:
- Years with early warmth (WV_e > WV_threshold)
- Years with delayed warmth (WV_e < WV_threshold)

The practical question is whether β_wv is large enough for this WV spread to produce meaningfully separated bloom date clusters. Given Phase 2's condition number (~55,774) and the standardized β_wv effect size (~0.28 DOY per 1σ of WV), the spread may be modest.

---

## SECTION 3 — Ensemble Propagation (Fallback Mode)

### 3.1 Fallback ensemble construction

Since SEAS5 data is unavailable, we use the last N_fallback years of observed post-cutoff temperatures as pseudo-ensemble members. Default N_fallback = 30, giving scenario years {1996, 1997, ..., 2025}.

For each site s, the silver weather file contains hourly ERA5-Land temperatures. We extract:

    DailyMeans[s][y][d] = mean(T_hour : hour ∈ day d, year y)

for all sites s, years y, and DOYs d ∈ [1, 180].

### 3.2 Bias correction consistency requirement

**CRITICAL:** Phase 2's SYH CV computed WV on **raw ERA5 temperatures** (bias was estimated per fold but not applied to WV inputs). This means the regression coefficients β_wv learned the relationship:

    bloom_doy ~ ... + β_wv · WV(raw ERA5)

Phase 3 MUST compute WV on raw ERA5 temperatures to be consistent. Applying bias correction in Phase 3 but not in Phase 2 would create a distribution shift:

    WV_corrected = slope(β₀ + β₁ · T̄_d)
                 = β₁ · slope(T̄_d) + (β₀ contribution cancels in slope)
                 = β₁ · WV_raw

For washingtondc (β₁ = 0.957), this would scale WV by ~4%, systematically shifting predictions.

**Decision:** Phase 3 uses raw ERA5 temperatures for WV computation, matching Phase 2 behavior.

**Evidence required from implementer:** Verify this by running:
```
rg -n "bias" src/modeling/syh_cv.py
```
and confirming bias is NOT applied to temperature data before WV computation.

### 3.3 WV computation for each ensemble member

For site s, ensemble member e (scenario year Y_e):

1. Build spliced daily mean temperatures:
   ```
   T̄_spliced[d] = DailyMeans[s][2026][d]   for d ∈ [1, 59]
   T̄_spliced[d] = DailyMeans[s][Y_e][d]    for d ∈ [60, 180]
   ```

2. Extract WV window:
   ```
   d_center = round(mean_bloom_doy[s])
   window = [d_center - 14, d_center]    # 15 days inclusive
   ```

3. Compute slope:
   ```
   days = [d for d in window if d in T̄_spliced]
   if len(days) < 7:
       WV_e = NaN  (skip this member)
   else:
       X = [1, d] for d in days   # intercept + DOY
       y = [T̄_spliced[d] for d in days]
       WV_e = OLS_slope(X, y)
   ```

4. Apply regression:
   ```
   DOY_e = α_s + β_gdh · GDH_2026(s) + β_cp · CP_2026(s) + β_wv · WV_e
   ```

### 3.4 Expected ensemble properties

For a typical site:
- N_members ≈ 28–30 (some years may lack data in the WV window)
- WV range: approximately [-0.3, +0.5] °C/day
- DOY range: approximately [C_s + β_wv · (-0.3), C_s + β_wv · (+0.5)]
- Given β_wv ≈ small positive (from Phase 2), the DOY spread is modest

### 3.5 Edge cases

- **Missing DOY data:** If scenario year Y_e has no temperature data for DOYs in the WV window, skip that member. Log warning.
- **Insufficient members:** If fewer than 5 valid members for any site, STOP. The ensemble is too sparse for meaningful GMM fitting.
- **2026 pre-cutoff data:** The first 59 DOYs of spliced data always come from 2026 observed temps. If the WV window falls entirely within DOY 1–59 (which would mean mean_bloom_doy < 73), the "ensemble" has zero spread — all members produce identical WV. In this case, skip GMM and use the single prediction directly.

---

## SECTION 4 — Multimodality Detection

### 4.1 Why bloom predictions become bimodal

The ensemble of predicted bloom DOYs {DOY_1, ..., DOY_M} may exhibit two modes when:

1. **Climate regime clustering:** Historical springs divide into warm-early and cold-late categories. Example: El Niño years tend toward warm springs, La Niña toward cold springs. If the last 30 years contain ~15 of each regime, the DOY distribution shows two peaks.

2. **Nonlinear amplification:** Even if WV is approximately Gaussian, the regression model's interaction with site intercepts and the floor/ceiling constraints can create apparent bimodality.

3. **Geographic specificity:** Some sites (e.g., Kyoto, with strong East Asian monsoon influence) show more regime separation than others (e.g., Liestal, where European spring transitions are smoother).

### 4.2 Gaussian Mixture Model formulation

The likelihood for a GMM with K components on data x = {x_1, ..., x_n}:

    p(x | θ) = Π_{i=1}^{n} Σ_{k=1}^{K} π_k · N(x_i | μ_k, σ²_k)

where:
- π_k = mixing weight for component k (Σ π_k = 1, π_k ≥ 0)
- μ_k = mean of component k
- σ²_k = variance of component k
- θ = {π_k, μ_k, σ²_k : k = 1..K}

### 4.3 EM algorithm (used internally by sklearn)

**E-step:** Compute responsibilities:
    r_{ik} = π_k · N(x_i | μ_k, σ²_k) / Σ_j π_j · N(x_i | μ_j, σ²_j)

**M-step:** Update parameters:
    N_k = Σ_i r_{ik}
    π_k = N_k / n
    μ_k = (1/N_k) Σ_i r_{ik} · x_i
    σ²_k = (1/N_k) Σ_i r_{ik} · (x_i - μ_k)²

**Convergence:** Iterate until log-likelihood change < tolerance (sklearn default: 1e-3) or max_iter reached.

**Initialization:** With `random_state=42`, sklearn uses a deterministic k-means++ initialization, ensuring reproducible results.

### 4.4 Parameter count

- k=1: 2 free parameters (μ₁, σ²₁) — π₁ = 1 is fixed
- k=2: 5 free parameters (π₁, μ₁, σ²₁, μ₂, σ²₂) — π₂ = 1 - π₁

---

## SECTION 5 — BIC Model Selection

### 5.1 Bayesian Information Criterion derivation

    BIC_k = -2 · log L̂_k + p_k · log(n)

where:
- L̂_k = maximized likelihood under model with k components
- p_k = number of free parameters (2 for k=1, 5 for k=2)
- n = number of ensemble members (~30)

### 5.2 Penalty term interpretation

The term p_k · log(n) penalizes model complexity. For n = 30:

    Penalty(k=1) = 2 · log(30) ≈ 6.8
    Penalty(k=2) = 5 · log(30) ≈ 17.0

The k=2 model incurs +10.2 additional penalty. It must improve the log-likelihood by at least +5.1 to overcome this penalty. This is a substantial bar for 30 data points, which inherently limits the power to detect bimodality in small ensembles.

### 5.3 Why k ∈ {1, 2} ONLY

ARCHITECTURE.md constrains k to {1, 2}. The justification:

1. **Physical basis:** Spring temperature regimes are at most bimodal (warm vs. cold). A third mode would require a physical mechanism not present in mid-latitude spring climate.

2. **Statistical power:** With n ≈ 30 ensemble members, fitting k=3 (8 parameters) consumes 8/30 = 27% of degrees of freedom. The estimates would be unreliable.

3. **Overfitting risk:** k=3 would almost always reduce BIC on small samples due to fitting noise, but would produce spurious modes that don't reflect real climate variability.

4. **Decision simplicity:** The downstream decision rule (cluster selection) is designed for binary choice. Three modes would require additional decision logic not specified in the architecture.

### 5.4 BIC selection with neutral threshold

Raw BIC comparison can be noisy. We add a conservative threshold:

    relative_improvement = (BIC_1 - BIC_2) / |BIC_1|

Decision:
- If relative_improvement > NEUTRAL_THRESHOLD (0.3): select k=2
- Otherwise: select k=1 (conservative default)

This prevents selecting k=2 when the bimodal signal is marginal.

---

## SECTION 6 — Cluster Selection Decision Rule

### 6.1 The selection problem

When k=2 is selected, we have two Gaussian components:
    Component A: (π_A, μ_A, σ_A)
    Component B: (π_B, μ_B, σ_B)

We must choose ONE component mean as our point prediction. Three candidate rules:

1. **Largest cluster:** argmax_k(π_k) — picks the mode with more probability mass
2. **Climatology-closest:** argmin_k(|μ_k - climatology_s|) — picks the mode nearest historical average
3. **Hybrid:** Dominant cluster if π_max > 0.70, otherwise climatology-closest

### 6.2 Why largest cluster fails

Consider: a warm El Niño spring puts 60% of ensemble members in an early-bloom cluster (μ = 82) and 40% in a normal cluster (μ = 92). The largest-cluster rule picks DOY 82. But if the site's climatological mean is DOY 90, this is a 8-day departure from baseline — confident directional prediction based on a 60/40 split.

The problem: a 60/40 split doesn't justify high confidence. The model should hedge toward climatology unless the signal is overwhelming. This is especially dangerous for cold-start sites (Vancouver, NYC) where the regression model's predictions are already uncertain.

### 6.3 Adopted rule (hybrid)

```
IF max(π_A, π_B) > 0.70:
    # One cluster is dominant — trust it
    selected = argmax_k(π_k)
ELSE:
    # Neither dominates — hedge toward climatology
    selected = argmin_k(|μ_k - climatology_s|)
```

This rule:
- Follows strong ensemble signals (>70% agreement)
- Hedges toward historical norms under ambiguity
- Is deterministic (no random tie-breaking needed)

### 6.4 Tie-breaking

If |μ_A - climatology| = |μ_B - climatology| exactly (probability zero for continuous data, but for numerical safety):
- Select the component with smaller index (lower μ)
- This is deterministic and arbitrary, but the case is astronomically unlikely

### 6.5 Decision-theoretic interpretation

Under quadratic loss L(y, ŷ) = (y - ŷ)², the optimal point prediction from a mixture is the **posterior mean**:

    ŷ_optimal = π_A · μ_A + π_B · μ_B

However, under absolute loss L(y, ŷ) = |y - ŷ| (MAE, our competition metric), the optimal prediction is the **posterior median**, which for a well-separated bimodal distribution lies near the larger mode.

Our hybrid rule approximates the MAE-optimal strategy: follow the dominant mode when it's clear, hedge toward prior when it's ambiguous.

---

## SECTION 7 — Thermodynamic Floor

### 7.1 Biological constraint

Cherry blossom bloom requires sufficient chill accumulation followed by sufficient heat accumulation. There is a minimum DOY below which bloom is physiologically impossible regardless of temperature trajectory.

### 7.2 Implementation via clipping

Rather than computing an explicit thermodynamic floor from first principles (which would require modeling the full chill-heat interaction), we use the empirical floor:

    DOY_min = 60  (March 1)
    DOY_max = 140 (May 20)

These bounds are derived from the historical training data: no site in the competition dataset has ever bloomed before March 1 or after May 20.

### 7.3 Clipping rule

    DOY_final = max(60, min(140, round(DOY_raw)))

This is applied AFTER shrinkage, as the final step before writing to submission.csv.

### 7.4 Fold-safety

The clipping bounds [60, 140] are constants derived from domain knowledge, not from the training data of any specific fold. Therefore they do not introduce data leakage.

---

## SECTION 8 — Deterministic Prediction Formation

### 8.1 Empirical Bayes shrinkage

After cluster selection produces a point estimate μ_selected for site s:

    ŷ_shrunk(s) = w_s · μ_selected + (1 - w_s) · global_mean

where:
- w_s = shrinkage weight from Phase 2 (range: ~0.15 for NYC to ~0.95 for DC/Kyoto/Liestal)
- global_mean = overall mean bloom DOY from Phase 2

For cold-start sites (Vancouver w ≈ 0.3–0.5, NYC w ≈ 0.15–0.3), shrinkage substantially pulls the prediction toward the global mean. For established sites (w > 0.90), the effect is minimal.

### 8.2 Rounding

    DOY_int = round(ŷ_shrunk)

Python's `round()` uses banker's rounding (round-half-to-even). For a typical prediction like 86.5, this rounds to 86 (even). The impact on MAE is at most 0.5 days per prediction.

### 8.3 Full prediction formula

    DOY_final(s) = clip(60, 140, round(w_s · μ_selected(s) + (1 - w_s) · global_mean))

### 8.4 Reproducibility guarantees

All sources of variation are controlled:
- Ensemble members: deterministic (fixed historical years)
- GMM fit: `random_state=42`
- Cluster selection: deterministic rule (no ties)
- Shrinkage: deterministic formula
- Rounding: deterministic
- Clipping: deterministic

Therefore: identical inputs → identical submission.csv.

---

## SECTION 9 — Full Execution Order (ABSOLUTE)

### Pre-loop setup:

```
1. Load Phase 2 artifacts:
   - model_coefficients.json → {α_s, β_gdh, β_cp, β_wv, site_intercepts, feature_order}
   - bias_coefficients.json → {per-site β₀, β₁}
   - shrinkage_weights.json → {per-site w_s}
   - mean_bloom_doy.json → {per-site climatological mean}
   - global_mean.json → scalar

2. Load gold features:
   - gold_2026 = features.parquet WHERE year == 2026
   - Extract per-site GDH_2026, CP_2026

3. Pre-load silver weather into memory:
   - For each site s: weather_cache[s] = pd.read_parquet(consolidated)
   - Total memory: ~35 MB

4. Pre-compute daily mean temperatures:
   - For each site s, for each year y in weather_cache[s]:
     daily_means[s][y] = groupby(DOY).mean() of temperature column
   - This is the performance optimization: O(1) lookup per (site, year, DOY)

5. Determine ensemble scenario years:
   - For each site s: last 30 years available in daily_means[s] excluding 2026
   - Typically: scenario_years[s] = [1996, 1997, ..., 2025]

6. Initialize results containers:
   - ensemble_results = {}
   - gmm_results = {}
   - prediction_chain = {}
```

### Per-site loop (sorted alphabetically: kyoto, liestal, nyc, vancouver, washingtondc):

```
FOR site s in site_order:

    7. Retrieve 2026 observed features:
       gdh = gold_2026[s]['gdh']
       cp = gold_2026[s]['cp']

    8. Retrieve model parameters:
       alpha_s = model_coefficients['site_intercepts'][s]
       beta_gdh = model_coefficients['beta_gdh']
       beta_cp = model_coefficients['beta_cp']
       beta_wv = model_coefficients['beta_wv']

    9. Compute constant term:
       C_s = alpha_s + beta_gdh * gdh + beta_cp * cp

   10. Retrieve WV window center:
       d_center = round(mean_bloom_doy[s])

   11. FOR each scenario year Y in scenario_years[s]:

       a. Build spliced daily means:
          spliced[d] = daily_means[s][2026][d]  for d in 1..59
          spliced[d] = daily_means[s][Y][d]     for d in 60..180

       b. Extract WV window:
          window_doys = [d for d in range(d_center - 14, d_center + 1)
                         if d in spliced]

       c. Check minimum data:
          if len(window_doys) < 7: skip, log warning

       d. Compute WV slope:
          X = np.array(window_doys)
          Y_vals = np.array([spliced[d] for d in window_doys])
          WV_m = OLS_slope(X, Y_vals)

       e. Compute predicted DOY:
          DOY_m = C_s + beta_wv * WV_m

       f. Append to member list

   12. Validate ensemble:
       if n_valid < 5: STOP ("Insufficient ensemble members for {s}")

   13. Fit GMM:
       predictions = np.array(member_doys)
       gmm_result = fit_bimodal(predictions, s, mean_bloom_doy[s])

   14. Record GMM diagnostics

   15. Get selected mean:
       mu_selected = gmm_result['selected_mean']

   16. Apply shrinkage:
       w = shrinkage_weights[s]['w']
       shrunk = w * mu_selected + (1 - w) * global_mean

   17. Round and clip:
       final_doy = max(60, min(140, round(shrunk)))

   18. Map site_key to location:
       location = SITES[s].loc_id
       # nyc → newyorkcity, others → same as site_key

   19. Record to prediction chain

END FOR
```

### Post-loop:

```
20. Build submission DataFrame:
    columns = ['location', 'year', 'bloom_doy']
    Sort alphabetically by location
    All year = 2026, all bloom_doy = int

21. Validate submission:
    assert len(df) == 5
    assert set(df['location']) == {'kyoto','liestal','newyorkcity','vancouver','washingtondc'}
    assert all(60 <= df['bloom_doy']) and all(df['bloom_doy'] <= 140)

22. Write submission.csv

23. Write diagnostics:
    - ensemble_distributions.json
    - gmm_results.json
    - prediction_summary.json

24. Print summary table
```

---

## SECTION 10 — Phase 3 Pseudocode

### 10.1 load_phase2_artifacts()

```python
def load_phase2_artifacts(processed_dir: Path) -> dict:
    """Load all Phase 2 artifacts into a single dict.
    
    Returns:
        {
            'model_coeff': dict,       # site_intercepts, beta_*, feature_order
            'bias_coeff': dict,        # per-site {beta0, beta1, r2, n_obs}
            'shrinkage': dict,         # per-site {w, sigma2_s, ...}
            'mean_bloom': dict,        # per-site float
            'global_mean': float,      # scalar
        }
    """
    artifacts = {}
    for name, filename in [
        ('model_coeff', 'model_coefficients.json'),
        ('bias_coeff', 'bias_coefficients.json'),
        ('shrinkage', 'shrinkage_weights.json'),
        ('mean_bloom', 'mean_bloom_doy.json'),
    ]:
        path = processed_dir / filename
        assert path.exists(), f"Missing Phase 2 artifact: {path}"
        artifacts[name] = json.loads(path.read_text())
    
    gm_path = processed_dir / 'global_mean.json'
    assert gm_path.exists()
    artifacts['global_mean'] = json.loads(gm_path.read_text())
    # global_mean may be stored as {"global_mean": 90.1} or as bare float
    if isinstance(artifacts['global_mean'], dict):
        artifacts['global_mean'] = artifacts['global_mean']['global_mean']
    
    return artifacts
```

### 10.2 precompute_daily_means()

```python
def precompute_daily_means(
    weather_cache: dict[str, pd.DataFrame],
    temp_column: str,   # discovered from actual parquet schema
) -> dict[str, dict[int, dict[int, float]]]:
    """Pre-compute daily mean temperatures.
    
    Returns:
        daily_means[site][year][doy] = float (mean temp in °C)
    
    Shape: ~5 sites × ~76 years × ~365 DOYs
    Memory: ~5 × 76 × 365 × 8 bytes ≈ 1 MB (negligible)
    """
    daily_means = {}
    for site, df in weather_cache.items():
        df = df.copy()
        df['year'] = df['timestamp'].dt.year
        df['doy'] = df['timestamp'].dt.dayofyear
        grouped = df.groupby(['year', 'doy'])[temp_column].mean()
        daily_means[site] = {}
        for (year, doy), val in grouped.items():
            if year not in daily_means[site]:
                daily_means[site][year] = {}
            daily_means[site][year][doy] = val
    return daily_means
```

### 10.3 build_fallback_ensemble()

```python
def build_fallback_ensemble(
    daily_means: dict,
    gold_2026: pd.DataFrame,
    phase2: dict,       # loaded artifacts
    sites_config: dict,
    n_fallback: int = 30,
) -> dict[str, dict]:
    """
    Returns: ensemble_results[site] = {
        'predictions': np.array,
        'wv_values': np.array,
        'scenario_years': list,
        'n_valid': int,
        'gdh_2026': float,
        'cp_2026': float,
    }
    """
    mc = phase2['model_coeff']
    site_order = sorted(gold_2026['site_key'].unique())
    results = {}
    
    for s in site_order:
        row = gold_2026[gold_2026['site_key'] == s].iloc[0]
        gdh = float(row['gdh'])
        cp = float(row['cp'])
        
        C_s = mc['site_intercepts'][s] + mc['beta_gdh'] * gdh + mc['beta_cp'] * cp
        d_center = round(phase2['mean_bloom'][s])
        
        # Scenario years: last n_fallback historical years
        available = sorted(y for y in daily_means[s] if y < 2026)
        scenarios = available[-n_fallback:]
        
        member_doys = []
        member_wvs = []
        valid_years = []
        
        for Y in scenarios:
            # Build spliced daily means
            spliced = {}
            for d in range(1, 60):
                if 2026 in daily_means[s] and d in daily_means[s][2026]:
                    spliced[d] = daily_means[s][2026][d]
            for d in range(60, 181):
                if d in daily_means[s].get(Y, {}):
                    spliced[d] = daily_means[s][Y][d]
            
            # WV window
            window_start = d_center - 14
            window_end = d_center  # inclusive
            window_doys = [d for d in range(window_start, window_end + 1)
                          if d in spliced]
            
            if len(window_doys) < 7:
                logger.warning(f"{s}/{Y}: only {len(window_doys)} days in WV window, skipping")
                continue
            
            # OLS slope
            X = np.array(window_doys, dtype=float)
            Y_vals = np.array([spliced[d] for d in window_doys])
            # slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
            x_mean = X.mean()
            y_mean = Y_vals.mean()
            wv = np.sum((X - x_mean) * (Y_vals - y_mean)) / np.sum((X - x_mean)**2)
            
            doy_pred = C_s + mc['beta_wv'] * wv
            
            member_doys.append(doy_pred)
            member_wvs.append(wv)
            valid_years.append(Y)
        
        n_valid = len(member_doys)
        assert n_valid >= 5, f"{s}: only {n_valid} valid ensemble members (need ≥5)"
        
        results[s] = {
            'predictions': np.array(member_doys),
            'wv_values': np.array(member_wvs),
            'scenario_years': valid_years,
            'n_valid': n_valid,
            'gdh_2026': gdh,
            'cp_2026': cp,
        }
    
    return results
```

### 10.4 fit_bimodal()

```python
def fit_bimodal(
    predictions: np.ndarray,
    site_key: str,
    climatological_mean: float,
    neutral_threshold: float = 0.3,
) -> dict:
    n = len(predictions)
    X = predictions.reshape(-1, 1)
    
    # Fit k=1
    gmm1 = GaussianMixture(n_components=1, random_state=42, max_iter=200)
    gmm1.fit(X)
    bic_1 = gmm1.bic(X)
    
    # Fit k=2
    gmm2 = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm2.fit(X)
    bic_2 = gmm2.bic(X)
    
    # BIC selection
    rel_improvement = (bic_1 - bic_2) / abs(bic_1) if bic_1 != 0 else 0
    select_k2 = rel_improvement > neutral_threshold
    
    if select_k2:
        means = gmm2.means_.flatten()
        weights = gmm2.weights_
        stds = np.sqrt(gmm2.covariances_.flatten())
        
        # Cluster selection
        if max(weights) > 0.70:
            idx = int(np.argmax(weights))
        else:
            dists = [abs(means[i] - climatological_mean) for i in range(2)]
            idx = int(np.argmin(dists))
        
        selected_mean = float(means[idx])
        selected_std = float(stds[idx])
        k = 2
    else:
        selected_mean = float(gmm1.means_[0, 0])
        selected_std = float(np.sqrt(gmm1.covariances_[0, 0, 0]))
        k = 1
    
    return {
        'k': k,
        'bic_1': float(bic_1),
        'bic_2': float(bic_2),
        'relative_improvement': float(rel_improvement),
        'selected_mean': selected_mean,
        'selected_std': selected_std,
        'all_predictions': predictions.tolist(),
        'gmm_details': { ... },  # full parameters for diagnostics
    }
```

### 10.5 generate_predictions() and write_submission()

```python
def generate_predictions(ensemble_results, phase2, sites_config):
    site_order = sorted(ensemble_results.keys())
    rows = []
    summaries = {}
    gmm_all = {}
    
    for s in site_order:
        preds = ensemble_results[s]['predictions']
        clim_mean = phase2['mean_bloom'][s]
        
        # GMM
        gmm_result = fit_bimodal(preds, s, clim_mean)
        gmm_all[s] = gmm_result
        mu_sel = gmm_result['selected_mean']
        
        # Shrinkage
        w = phase2['shrinkage'][s]['w']
        gm = phase2['global_mean']
        shrunk = w * mu_sel + (1 - w) * gm
        
        # Round + clip
        final = max(60, min(140, round(shrunk)))
        
        # Location mapping
        location = sites_config[s].loc_id
        
        rows.append({'location': location, 'year': 2026, 'bloom_doy': int(final)})
        summaries[s] = {
            'ensemble_mean': float(np.mean(preds)),
            'gmm_k': gmm_result['k'],
            'gmm_selected_mean': mu_sel,
            'shrinkage_weight': w,
            'global_mean': gm,
            'shrunk_prediction': shrunk,
            'final_doy': int(final),
            'final_date': (datetime(2026, 1, 1) + timedelta(days=final-1)).isoformat()[:10],
        }
    
    df = pd.DataFrame(rows).sort_values('location').reset_index(drop=True)
    return df, summaries, gmm_all


def write_submission(df, path='submission.csv'):
    # Validate
    assert len(df) == 5
    assert list(df.columns) == ['location', 'year', 'bloom_doy']
    assert all(df['year'] == 2026)
    expected = {'kyoto', 'liestal', 'newyorkcity', 'vancouver', 'washingtondc'}
    assert set(df['location']) == expected
    assert all(60 <= df['bloom_doy']) and all(df['bloom_doy'] <= 140)
    assert all(df['bloom_doy'].apply(lambda x: isinstance(x, (int, np.integer))))
    
    df.to_csv(path, index=False)
```

---

## SECTION 11 — Failure Mode Analysis

### 11.1 GMM collapse

**Failure:** k=2 GMM converges with one component having σ → 0 (degenerate). sklearn handles this via `reg_covar` (default 1e-6), which adds a small regularization to covariance estimates.

**Recovery:** No action needed — sklearn's default regularization prevents true collapse. Monitor: if any σ_k < 0.1 days, log warning (the component is fitting a single data point).

### 11.2 High-variance ensembles

**Failure:** Ensemble std > 20 days. This means the regression model amplifies WV uncertainty far beyond reasonable bloom date ranges.

**Recovery:** This is a STOP condition. If ensemble std > 20 for any site, the model is unreliable. Check β_wv magnitude and WV distribution.

**Expected range:** Given β_wv is small (standardized effect ~0.28), ensemble std should be 2–8 days.

### 11.3 Identical BIC scores

**Failure:** BIC_1 = BIC_2 exactly (floating point equality).

**Recovery:** The relative_improvement formula handles this: (BIC_1 - BIC_2)/|BIC_1| = 0, which is < neutral_threshold, so k=1 is selected. No special case needed.

### 11.4 SEAS5 fallback behavior

**Failure:** In fallback mode, we only have ~30 members vs. 50 for real SEAS5. The ensemble is smaller and may underrepresent tail scenarios.

**Mitigation:** n=30 is sufficient for GMM with k ∈ {1,2}. The conservative BIC threshold (0.3) compensates for reduced sample size.

### 11.5 Sparse-site instability (Vancouver/NYC)

**Failure:** For Vancouver (4 training labels) and NYC (2 training labels), the site intercept α_s and mean_bloom_doy are estimated from very few observations. The regression predictions may be unreliable.

**Mitigation:** Empirical Bayes shrinkage. With w_nyc ≈ 0.15–0.30, the final prediction is 70–85% determined by the global mean, heavily damping any instability in the site-specific prediction. This is the designed safety mechanism.

### 11.6 Zero ensemble spread

**Failure:** If mean_bloom_doy for a site is < 73 (DOY 59 + 14), the entire WV window falls within the observed period (DOY 1–59). All ensemble members see identical data → zero spread → GMM trivially selects k=1 with σ → 0.

**Recovery:** This is expected behavior, not a failure. The model is confident because the WV window is fully observed. The prediction is deterministic.

### 11.7 WV sign interaction

**Status from Phase 2:** β_wv is positive. Higher WV (faster warming) → higher predicted DOY (later bloom). This is counterintuitive but was analyzed and deemed non-blocking in Phase 2 (Section K of debrief): the high condition number (~55,774) and multicollinearity between GDH, CP, and WV means individual coefficient signs may not match univariate intuition.

**Phase 3 impact:** With positive β_wv, warmer-spring scenario years produce LATER predicted bloom dates. This inverts the usual expectation. The ensemble distribution will still be valid (it captures the model's learned relationship), but the direction of spread may surprise a reviewer.

---

## SECTION 12 — Determinism & Offline Audit

### 12.1 All randomness sources

| Source | Control | Location |
|--------|---------|----------|
| GMM initialization | `random_state=42` | `fit_bimodal()` |
| Site iteration order | `sorted(site_order)` | All loops |
| Ensemble member order | `sorted(scenario_years)` | `build_fallback_ensemble()` |
| Floating-point accumulation | Deterministic CPU ops, no threading | Throughout |
| DataFrame ordering | `.sort_values()` before writes | `write_submission()` |

### 12.2 Reproducibility contract

Given identical Phase 2 artifacts and silver weather data, two independent runs of Phase 3 MUST produce byte-identical `submission.csv` files. Verify:

```bash
python3 -m src.modeling.predictor
cp submission.csv /tmp/run1.csv
python3 -m src.modeling.predictor
diff submission.csv /tmp/run1.csv  # expect: no output
```

### 12.3 Offline verification

```bash
rg -n "requests|cdsapi|urllib|httpx|socket|http.client" \
    src/modeling/seas5_processor.py \
    src/modeling/gmm_selector.py \
    src/modeling/predictor.py
# Expected: no results
```

---

## SECTION 13 — Computational Complexity

### 13.1 Daily means precomputation

Time: O(Σ_s |hours_s|) ≈ O(5 × 76 × 8760) ≈ O(3.3M) rows, single-pass groupby
Memory: O(5 × 76 × 365 × 8) ≈ 1 MB
**Expected: < 10 seconds**

### 13.2 Ensemble propagation

Per site: O(N_fallback × W) where W = window size (15 days)
Total: O(5 × 30 × 15) = O(2,250) slope computations
Each slope: O(W) = O(15)
**Expected: < 1 second**

### 13.3 GMM fitting

Per site: O(N_members × k × max_iter) for EM
With n=30, k=2, max_iter=200: O(12,000) per site
Total: O(5 × 12,000) = O(60,000)
**Expected: < 1 second**

### 13.4 Total runtime

**Dominant cost:** Daily means precomputation from parquet I/O
**Expected total:** 10–30 seconds
**Memory peak:** ~50 MB (silver weather cache)

---

## SECTION 14 — Phase 3 Validation Gates

### 14.1 assert_gmm_k_range

**What:** Every GMM k ∈ {1, 2}
**Prevents:** Implementation bug where k=3+ is accidentally fitted
**Failure if removed:** k=3 overfits to noise in small ensembles, produces spurious trimodal predictions

### 14.2 assert_no_noise_injection

**What:** Ensemble member count ≤ 35 per site
**Prevents:** Artificial inflation of ensemble size (adding Gaussian noise to "smooth" distributions)
**Failure if removed:** Inflated ensemble masks true distribution shape, GMM fit becomes misleading

### 14.3 assert_submission_schema

**What:** Exactly 5 rows, correct columns, correct locations, year=2026, integer DOY
**Prevents:** Malformed submission that would be rejected by competition evaluator
**Failure if removed:** Missing site, wrong column names, float DOY → competition disqualification

### 14.4 assert_predictions_reasonable

**What:** All bloom_doy ∈ [60, 140]
**Prevents:** Biologically impossible predictions (e.g., January bloom)
**Failure if removed:** Extreme predictions from model instability pass through uncaught

### 14.5 assert_shrinkage_applied

**What:** For Vancouver and NYC, shrunk ≠ gmm_selected_mean
**Prevents:** Bypassing Empirical Bayes for cold-start sites
**Failure if removed:** NYC (2 labels) gets unshrunk prediction from an unreliable site intercept

---

## SECTION 15 — Expected Outputs

### 15.1 submission.csv

```
location,year,bloom_doy
kyoto,2026,<int>
liestal,2026,<int>
newyorkcity,2026,<int>
vancouver,2026,<int>
washingtondc,2026,<int>
```

Sorted alphabetically by location. Exactly 5 rows. All bloom_doy in [60, 140].

### 15.2 data/processed/diagnostics/ensemble_distributions.json

```json
{
  "<site_key>": {
    "n_members": <int>,
    "predictions": [<float>, ...],
    "wv_values": [<float>, ...],
    "scenario_years": [<int>, ...],
    "mean": <float>,
    "std": <float>,
    "gdh_2026": <float>,
    "cp_2026": <float>,
    "percentiles": {
      "p10": <float>, "p25": <float>, "p50": <float>,
      "p75": <float>, "p90": <float>
    }
  }
}
```

### 15.3 data/processed/diagnostics/gmm_results.json

```json
{
  "<site_key>": {
    "k": <int>,
    "bic_1": <float>,
    "bic_2": <float>,
    "relative_improvement": <float>,
    "selected_mean": <float>,
    "selected_std": <float>,
    "gmm_details": {
      "means": [<float>, ...],
      "weights": [<float>, ...],
      "covariances": [<float>, ...]
    }
  }
}
```

### 15.4 data/processed/diagnostics/prediction_summary.json

```json
{
  "<site_key>": {
    "ensemble_mean": <float>,
    "gmm_k": <int>,
    "gmm_selected_mean": <float>,
    "shrinkage_weight": <float>,
    "global_mean": <float>,
    "shrunk_prediction": <float>,
    "final_doy": <int>,
    "final_date": "<YYYY-MM-DD>"
  }
}
```

---

## SECTION 16 — Engineer Completion Checklist

```
□ 1. Verify Phase 2 complete:
     python3 -m src.validation.run_all_gates --phase 2  → all PASS

□ 2. Read context files:
     context/ARCHITECTURE.md, context/agent.md, context/phase3.md

□ 3. Inspect Phase 2 artifact schemas:
     cat model_coefficients.json, bias_coefficients.json, etc.

□ 4. Discover Phase 2 function signatures:
     inspect.signature(compute_warming_velocity), estimate_bias, apply_shrinkage

□ 5. Verify bias-WV consistency:
     rg -n "bias" src/modeling/syh_cv.py → confirm raw ERA5 used for WV

□ 6. Discover temperature column name:
     Inspect silver weather parquet schema

□ 7. Create src/modeling/seas5_processor.py:
     build_fallback_ensemble() + process_seas5_ensemble() stub

□ 8. Create src/modeling/gmm_selector.py:
     fit_bimodal() with k∈{1,2}, random_state=42, BIC threshold 0.3

□ 9. Create src/modeling/predictor.py:
     generate_predictions() + save_submission() + save_diagnostics() + __main__

□ 10. Acceptance-test each module:
      Import OK for all three new modules

□ 11. Add Phase 3 gates to src/validation/gates.py:
      5 new gates (GMM k range, no noise, schema, bounds, shrinkage)

□ 12. Update src/validation/run_all_gates.py:
      Add phase "3" to PHASE_GATES map

□ 13. Execute Phase 3:
      python3 -m src.modeling.predictor

□ 14. Run Phase 3 gates:
      python3 -m src.validation.run_all_gates --phase 3  → all PASS

□ 15. Verify determinism:
      Run twice, diff submission.csv → identical

□ 16. Verify offline:
      rg for network imports in Phase 3 files → none

□ 17. Artifact inventory:
      submission.csv + 3 diagnostic JSONs exist with non-zero size

□ 18. Post-pipeline diagnostics:
      Print full prediction chain, ensemble stats, shrinkage verification

□ 19. Write TECHNICAL_DEBRIEF_P3.md:
      Sections A–K with evidence anchors, hashes, gate results

□ 20. PHASE 3 COMPLETE. Do NOT proceed to Phase 4.
```