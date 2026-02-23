# RESIDUAL_MODEL_POLICY.md

## ROLE
Govern the residual (ML) correction layer to prevent it from learning to compensate for mechanistic failures.

## PRIMARY PURPOSE
Ensure strict division of labor:
- Mechanistic model captures the primary phenological structure.
- Residual model captures small systematic error corrections only.

This prevents the "Safety Floor Bias" where a mechanistic fallback (e.g., DOY 151) forces ML to become the main predictor.

## NON-GOALS
- Does not define mechanistic model structure (see MODELING_ASSUMPTIONS.md).
- Does not define CV splitting (see EXPERIMENTAL_DESIGN.md).
- Does not define biology priors (see BIOLOGICAL_BACKGROUND.md).

---

## DEFINITIONS
- t_mech: mechanistic predicted bloom DOY (or mechanistic implied estimate)
- bloom_doy: observed bloom DOY
- residual: bloom_doy - t_mech

- catastrophic mechanistic failure:
  - either unreachable sentinel (t_mech == 151)
  - or absolute error too large (|bloom_doy - t_mech| >= threshold)

---

## HARD TRAINING FILTER (NON-NEGOTIABLE)
Residual ML training MUST only use rows where:

1) Mechanistic reachability is valid:
   - t_mech != SENTINEL_DOY_UNREACHABLE (151)

AND

2) Mechanistic error is within tolerance:
   - abs(bloom_doy - t_mech) < 25

This is mandatory.

If there are too few rows after filtering for a given site:
- prefer a simpler residual model (ridge regression) or partial pooling,
- do NOT relax the catastrophic failure filter.

---

## MODEL CAPACITY CONSTRAINTS
Residual ML is intentionally low-capacity.

Allowed model families (preferred order):
1) Regularized linear regression (Ridge/Lasso)
2) GAM with strong smoothing penalties
3) Constrained gradient boosting (tight depth, strong regularization)

Prohibited (unless explicitly justified in EXPERIMENTAL_DESIGN.md):
- high-depth boosting
- unconstrained neural nets
- models with large feature sets relative to sample size

---

## FEATURE POLICY
Allowed feature types (examples):
- t_mech
- site indicator / one-hot
- low-dimensional climate anomalies computed without leakage
- stable seasonal summaries (winter mean, spring mean up to cutoff)
- mechanistic diagnostics (chill completion DOY, forcing rate proxy)

Prohibited features:
- any feature requiring temperatures after FORECAST_CUTOFF_DATE for TARGET_YEAR
- any feature directly encoding year as a proxy for trend unless justified
- any feature derived from the observed bloom date (obvious leakage)

---

## VALIDATION REQUIREMENTS
Residual model evaluation must follow the same time-aware splits as the mechanistic model:
- LOYO or rolling-origin as defined in EXPERIMENTAL_DESIGN.md

Metrics reported:
- MAE of mechanistic-only
- MAE of hybrid (mechanistic + residual)
- per-site MAE for both

Residual model must be rejected if:
- it improves overall MAE but causes catastrophic degradation in any single site
- it improves MAE only by exploiting years with mechanistic failures (should not be possible if filters are applied)

---

## OUTPUT REQUIREMENTS
Every training run must log:

1) residual_training_row_count (global and per site)
2) rows_excluded_due_to_sentinel
3) rows_excluded_due_to_large_error
4) final feature list used
5) residual model hyperparameters

This enables reproducibility and auditability.

---

## RATIONALE (WHY THIS EXISTS)
Training residual ML on mechanistic failures creates a feedback loop:
- mechanistic fails → sentinel bias → ML learns giant offsets → final prediction collapses

This policy prevents that loop by construction.