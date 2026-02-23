# SYSTEM_CONTEXT_INDEX.md

## ROLE
Repository "operating system" for Gemini CLI reasoning and AI-assisted development.

## PRIMARY PURPOSE
Provide a single, canonical:
- Priority ordering of context files (precedence rules)
- Global constants (cutoff date, target year, objective metric)
- Non-negotiable constraints (no leakage, time-aware validation, MAE optimization)
- Conflict-resolution rules (what overrides what)

## NON-GOALS
- Does not contain implementation code.
- Does not restate full rules or biology (those live in their respective files).
- Does not include heuristics unless they are hard constraints.

---

## GLOBAL CONSTANTS (SINGLE SOURCE OF TRUTH)
- TARGET_YEAR = 2026
- FORECAST_CUTOFF_DATE = 2026-02-28
- FORECAST_CUTOFF_STANDARD = AOE (Anywhere on Earth)
- PRIMARY_METRIC = MAE (mean absolute error), averaged equally across 5 sites
- SITES = ["washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"]
- SENTINEL_DOY_UNREACHABLE = 151  (May 31 DOY in non-leap year context used by legacy tripwire)

If any file conflicts with GLOBAL CONSTANTS, GLOBAL CONSTANTS override.

---

## PRECEDENCE ORDER (HIGHEST → LOWEST)
When instructions conflict, follow the highest priority document below:

1) COMPETITION_BRIEF.md
   - Defines required outputs and compliance constraints.

2) EVALUATION_METRIC.md
   - Defines what "winning" means (MAE) and how to compute it.

3) EXPERIMENTAL_DESIGN.md
   - Defines how experiments must be run (time-aware validation rules).

4) FORECAST_CONSTRAINTS.md
   - Defines what information is allowed at forecast time (cutoff logic, no leakage).

5) MODELING_ASSUMPTIONS.md
   - Defines the structural contract of the model (hybrid mech + residual, etc).

6) BIOLOGICAL_BACKGROUND.md
   - Defines biological plausibility constraints and parameter realism.

7) BASELINE_MODEL_SUMMARY.md
   - Defines benchmark assumptions and known weaknesses.

8) LITERATURE_GAP.md
   - Defines the innovation space (what we may test next).

9) GEMINI.md / TECHNICAL_OVERVIEW.md (if present)
   - Treated as heuristics and convenience documentation only.
   - Must NOT override items 1–8.

---

## NON-NEGOTIABLE CONSTRAINTS (HARD RULES)
1) No temporal leakage:
   - No feature for TARGET_YEAR may use any observations after FORECAST_CUTOFF_DATE.
   - All CV splits must be time-aware (no random CV).

2) Objective discipline:
   - Model selection must be based on time-aware MAE (LOYO or rolling-origin),
     not in-sample fit, not random CV.

3) Hybrid division of labor:
   - Mechanistic layer must be biologically plausible and (mathematically) reachable.
   - Residual ML must NOT be trained on catastrophic mechanistic failures.

4) Units sanity:
   - Temperature units must be verified per site (no accidental 10x scaling).
   - Any scaling logic must be explicitly documented and testable.

---

## REQUIRED DEVELOPMENT WORKFLOW (TWO-STEP)
All significant refactors follow:

STEP 1 — Spec Compliance Diff:
- Identify where current code violates constraints in this index and priority files.
- Produce a minimal patch plan.

STEP 2 — Surgical Rewrite:
- Implement only the minimal changes required to satisfy the spec.
- Keep changes small and auditable.
- Add or update tests/gates to prevent regressions.

---

## CONFLICT EXAMPLES (HOW TO RESOLVE)
- If a heuristic suggests a shortcut that violates FORECAST_CONSTRAINTS.md → reject it.
- If TECHNICAL_OVERVIEW.md suggests photoperiod but MODELING_ASSUMPTIONS.md excludes it → exclude it.
- If a code change improves RMSE but worsens MAE → reject it.

---

## OUTPUT REQUIREMENT FOR AI CODING AGENTS
When asked to modify code:
- Cite the relevant controlling document(s) by filename.
- Explain how the change satisfies each relevant HARD RULE.
- Do not introduce new assumptions without updating MODELING_ASSUMPTIONS.md.