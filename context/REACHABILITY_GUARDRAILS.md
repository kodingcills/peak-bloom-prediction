# REACHABILITY_GUARDRAILS.md

## ROLE
Mechanistic reachability and safety specification for the phenology model.

## PRIMARY PURPOSE
Prevent "threshold unreachability" failures where the mechanistic model cannot reach its forcing requirement (F*),
causing fallback/sentinel DOY values (e.g., 151) that poison residual learning and produce nonsense forecasts
(e.g., 171 DOY outcomes).

## NON-GOALS
- Does not define biological theory (see BIOLOGICAL_BACKGROUND.md).
- Does not define validation splits (see EXPERIMENTAL_DESIGN.md).
- Does not define ML residual policy (see RESIDUAL_MODEL_POLICY.md).

---

## DEFINITIONS
- "Reachability": For a given site-year and parameter set θ, the mechanistic system reaches its bloom trigger
  (or equivalent smooth target) within the modeled season window.

- "Season window": For reachability checks, define a fixed evaluation horizon:
  - Start: Jan 1 of the target year (or model-specific start)
  - End: May 31 of the target year (DOY 151 baseline window)
  This is a safety check window, not necessarily the true bloom window.

- "Unreachable": The mechanistic trigger condition is never met by May 31.

- "Headroom": The number of days between the predicted trigger date and May 31:
  - headroom_days = 151 - predicted_doy
  If predicted_doy is 149, headroom_days = 2 (fragile).
  If predicted_doy is 120, headroom_days = 31 (robust).

---

## HARD GUARDRAILS (PASS/FAIL)
These must hold for the training set under time-aware evaluation:

### G1) Unreachable Rate Threshold
For each site:
- unreachable_rate(site) = (# years unreachable) / (# years evaluated)

Requirement:
- unreachable_rate(site) ≤ 0.05  (≤ 5%)

If any site exceeds 5%, the mechanistic parameterization is invalid and must be revised.

### G2) Headroom Robustness Threshold
For each site, compute mean headroom over reachable years:
- mean_headroom(site) = mean(151 - predicted_doy) over reachable years

Requirement:
- mean_headroom(site) ≥ 5 days

If mean headroom < 5, the model is considered fragile (too close to failing in slightly cooler springs).

### G3) Anchor Consistency (Empirical Anchors)
If the model uses site anchors (e.g., a_site priors, forcing requirements, etc.), anchors must be computed as:
- "empirical cumulative forcing at historical bloom DOY" (median across years)

Anchors must NOT be computed from:
- averages of pre-assembled training rows that may not match bloom DOY
- heuristics like "500/600 GDD guess"

### G4) Unit-Safe Reachability
Reachability checks must be run AFTER unit normalization.
Any reachability failure must trigger a unit audit first before adjusting biology priors.

---

## IMPLEMENTATION REQUIREMENTS
To enforce guardrails, the repo must contain:
- A reachability computation module (e.g., src/reachability.py)
- A test gate (e.g., tests/test_reachability.py) that fails CI if G1 or G2 is violated
- Reporting artifacts saved for debugging:
  - reachability_summary.csv (site, year, reachable, predicted_doy, headroom)
  - reachability_site_stats.csv (site, unreachable_rate, mean_headroom)

---

## REQUIRED DIAGNOSTICS OUTPUTS
Every mechanistic model training run must output:

1) Per-site unreachable rate
2) Per-site mean headroom
3) List of unreachable site-years (for inspection)
4) Confirmation that anchors were computed from daily paths at bloom DOY

---

## HOW TO RESPOND TO FAILURES (ORDERED PLAYBOOK)
If reachability fails:

1) Unit audit:
   - Verify TMAX/TMIN/TAVG are in correct units for the site.
   - Verify scaling logic uses Jul/Aug TMAX-based detection.

2) Anchor audit:
   - Verify anchors are computed from cumulative forcing at the true historical bloom DOY.

3) Simplify forcing function:
   - Replace overly complex forms (e.g., exponential coupling) with a simpler linear synergy
     if needed for identifiability: F* = a_site + b_site * CP

4) Tighten priors / shrinkage:
   - Use partial pooling or regularization to avoid extreme F* values.

---

## NOTE ON SMOOTH LIKELIHOOD (IMPORTANT FOR NUTS)
If PyMC/NUTS is used:
- Discrete "tripwire day-of-year" logic is non-differentiable and may cause divergences.
- Prefer a smooth likelihood target:
  - compare accumulated forcing at actual bloom DOY to F*
  - or use continuous surrogates for bloom timing

This is a modeling constraint for sampler stability, not optional.