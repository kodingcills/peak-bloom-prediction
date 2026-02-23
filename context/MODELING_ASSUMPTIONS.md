---
ROLE: Chief Modeling Officer (CMO)
INPUTS: `BIOLOGICAL_BACKGROUND.md`, `LITERATURE_GAP.md`
OUTPUTS: Model architecture specs (Hybrid Mechanistic + Residual ML)
CONSTRAINTS: Assume temperature as dominant driver; assume additive residual structure.
USED_BY: All Python implementation scripts
---
# Hybrid Mechanistic + Residual ML Modeling Assumptions

1. Purpose of This Document

This document explicitly states all modeling assumptions underlying the hybrid forecasting system.

Its goals are to:

Ensure transparency

Prevent implicit assumption drift

Align implementation with biological and statistical discipline

Provide reproducibility and auditability

Support novelty and scientific justification

Any deviation from these assumptions must be documented in revision history.

2. High-Level Modeling Philosophy

The forecasting system follows a hybrid structure:

A biologically grounded mechanistic chill–forcing model

A statistically disciplined residual correction model

Strict time-aware validation

The system assumes:

Temperature is the dominant causal driver of bloom timing.

Biological structure should constrain statistical modeling.

Predictive performance must be evaluated under realistic temporal splits.

3. Biological Assumptions
3.1 Temperature Dominance

Assumption:

Bloom timing is primarily determined by accumulated temperature exposure (chill + heat).

Implication:

Non-temperature factors (precipitation, radiation, soil moisture) are treated as secondary.

Photoperiod is not explicitly modeled unless added later.

Risk:

If photoperiod or stress effects are significant, model may miss systematic shifts.

3.2 Sequential Chill → Forcing Structure

Assumption:

Chill accumulation must reach a threshold before forcing accumulation meaningfully contributes to bloom.

Formally:

Forcing accumulation begins only after chill requirement is satisfied.

Implication:

Chill and forcing are treated as sequential processes.

Overlapping accumulation is not modeled in v1.

Risk:

If chill and forcing overlap biologically, this assumption introduces bias.

3.3 Fixed Parameter Within Site

Assumption:

Each site has a stable chill requirement (C*) and forcing requirement (F*) across years.

Implication:

Parameters are estimated per site (or partially pooled).

Inter-annual plasticity is not explicitly modeled in v1.

Risk:

If phenotypic plasticity is strong, static parameters may misrepresent dynamics.

3.4 Homogeneous Tree Population Within Site

Assumption:

Bloom observations represent a consistent cultivar or population with stable physiology.

Implication:

No tree age modeling

No cultivar mixture modeling

No genetic adaptation modeling

Risk:

Site-level structural changes may affect long historical records.

3.5 No Explicit Photoperiod Constraint

Assumption:

Temperature alone sufficiently explains bloom timing.

Implication:

Day length is not included as a gating variable.

Risk:

Warm winters with short photoperiod may behave differently biologically.

4. Statistical Assumptions
4.1 Time-Aware Validation Required

Assumption:

Future years cannot inform past predictions.

Implementation:

Leave-One-Year-Out (LOYO) cross-validation

Training years strictly < test year

Implication:

Random CV is prohibited.

Leakage is treated as critical modeling failure.

4.2 MAE as Optimization Target

Assumption:

Mean Absolute Error across sites is the competition’s primary objective.

Implication:

Model selection is based on LOYO MAE.

Other metrics (RMSE, R²) are secondary diagnostics.

4.3 Additive Residual Structure

Assumption:

Residual deviations from mechanistic prediction can be modeled additively.

Formally:

Hybrid_Prediction = Mechanistic_Prediction + Residual_Model_Output

Implication:

Residual model corrects bias but does not replace mechanism.

Residual model is constrained in complexity.

Risk:

Non-additive interaction errors may remain unmodeled.

4.4 Limited Residual Model Complexity

Assumption:

Residual model must remain low-capacity to prevent overfitting.

Implementation:

Regularized linear model, GAM, or constrained boosting.

Feature count limited and biologically interpretable.

Risk:

Underfitting possible if residual structure is complex.

5. Parameter Estimation Assumptions
5.1 Grid Search Stability

Assumption:

Parameter optimization via grid search provides sufficient stability and interpretability.

Implication:

Continuous unconstrained optimization avoided in v1.

Parameter bounds biologically constrained.

5.2 Identifiability Is Imperfect

Assumption:

Multiple parameter combinations may yield similar MAE.

Implication:

Sensitivity analysis required.

Preference given to stable parameter regions over narrow minima.

5.3 No Explicit Bayesian Priors in v1

Assumption:

Parameter estimates are frequentist point estimates.

Future extension:

Hierarchical partial pooling may be added if instability detected.

6. Forecast-Time Assumptions
6.1 Temperature Data Availability

Assumption:

Only observed climate data up to the submission cutoff is used.

Future spring temperatures:

Approximated using climatology or historical analog distributions.

6.2 No Future Leakage

Explicit rule:

No feature may include temperature data from days after the forecast cutoff.

Violation constitutes invalid modeling.

7. Cross-Site Modeling Assumptions
7.1 Sites Modeled Separately (v1)

Assumption:

Each site is modeled independently.

Implication:

No hierarchical sharing in baseline hybrid model.

Allows maximum flexibility per site.

Future extension:

Partial pooling if data sparsity observed.

7.2 Equal Weighting Across Sites

Assumption:

Overall MAE computed as average across sites.

Implication:

No site-specific weighting scheme.

8. Plasticity & Nonstationarity
8.1 Static Sensitivity (v1)

Assumption:

Sensitivity of bloom to temperature remains stable across historical record.

Risk:

Climate change may induce shifting parameters.

Future test:

Decade-wise parameter drift analysis.

9. Data Assumptions

Bloom DOY measurements are accurate and consistent.

Station climate data represent bloom site conditions.

Missing climate days are imputed consistently.

No systematic measurement shifts unless documented.

10. Known Limitations

Photoperiod excluded.

Tree age effects excluded.

Drought stress excluded.

Genetic adaptation not modeled.

Urban microclimate heterogeneity not explicitly modeled.

Interaction between chill and forcing treated as thresholded, not continuous.

These are recognized but constrained for stability and competition feasibility.

11. Revision Policy

Any change to:

Parameter structure

Validation scheme

Residual model capacity

Forecast-time constraints

Must be reflected in this document.

12. Summary

The modeling system assumes:

Temperature-driven phenology

Sequential chill–forcing mechanism

Site-stable parameters

Additive residual correction

Strict time-aware validation

MAE-optimized calibration

This document defines the structural contract between biological theory and statistical implementation.