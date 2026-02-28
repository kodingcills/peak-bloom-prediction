# phase4.md — Quarto Render & Submission Assembly

## Objective
Produce the self-contained `analysis.html` report and finalize `submission.csv`. The render must work fully offline (no network requests). Include the "Beat-the-AI" baseline comparison as a non-blocking module.

## Prerequisites
- Phases 1-3 complete
- All artifacts in `data/processed/` and `data/gold/`
- `submission.csv` exists from Phase 3
- Quarto installed (`quarto --version` works)

---

## Step 1: `analysis.qmd` — Quarto Document

### YAML Header
```yaml
---
title: "Phenology Engine v1.7 — 2026 Peak Bloom Predictions"
author: "[Your Name]"
format:
  html:
    self-contained: true
    html-math-method: katex
    toc: true
    toc-depth: 3
    code-fold: true
    theme: cosmo
execute:
  echo: true
  warning: false
  freeze: auto
---
```

**CRITICAL:** `self-contained: true` and `html-math-method: katex`. No MathJax (requires CDN). No external CSS/JS dependencies.

### Required Sections

#### 1. Introduction & Problem Statement
- Brief competition description
- Sites and species
- Inference cutoff: Feb 28, 2026

#### 2. Data Sources & Provenance
- Table of all data sources (from ARCHITECTURE.md)
- Data quality notes (missing hours, coverage gaps)
- Vancouver/NYC cold-start acknowledgment

#### 3. Methodology
- Feature engineering: GDH, CP, Warming Velocity (with LaTeX equations)
- Bias correction approach
- SYH Cross-Validation protocol (explain fold-safety)
- Empirical Bayes shrinkage (with LaTeX formula)
- Bimodal resolution via GMM + BIC

#### 4. Cross-Validation Results
- Per-site MAE table
- Overall MAE
- Residual distribution plots (histogram per site)
- Time series plot: predicted vs actual bloom DOY per site

#### 5. 2026 Predictions
- SEAS5 ensemble distribution plots per site (histogram + GMM overlay)
- BIC selection rationale per site
- Shrinkage weight table
- **Final prediction table** with confidence intervals (from ensemble percentiles)

#### 6. Beat-the-AI Baseline Comparison
- Execute `src/baselines/o3_mini.py`
- Show ΔMAE (our model vs baseline) per site
- **NON-BLOCKING:** Wrap in try/except. If baseline fails, show "Baseline comparison unavailable" and continue.

#### 7. Reproducibility
- Full cold-start runbook
- Environment details (Python version, package versions)
- `requirements.txt` contents

### Plotting Requirements
- Use `matplotlib` + `seaborn`. No plotly (adds JS dependencies that break self-contained).
- All plots must have: title, axis labels, legend where applicable.
- Color palette: use a colorblind-friendly palette (e.g., `seaborn.color_palette("colorblind")`).
- Save plots inline (Quarto handles this automatically with `self-contained: true`).

---

## Step 2: `src/baselines/o3_mini.py`

### Function: `compute_baseline_predictions() -> pd.DataFrame`

A simple baseline predictor for comparison. Implements the approach an o3-mini level model would use.

**Logic (intentionally simple):**
1. For each site, compute the mean bloom DOY from the most recent 10 years of data
2. Adjust by the current year's GDH anomaly: `pred = mean_recent + β * (gdh_2026 - mean_gdh_recent)`
   where β is estimated from simple regression on recent data
3. Return predictions DataFrame

### Function: `compare_to_baseline(our_predictions, baseline_predictions, cv_results) -> dict`

Compute ΔMAE per site: `our_mae - baseline_mae` (negative = we're better).

**This entire module is non-blocking.** Wrap all calls in try/except. If anything fails, return None and log warning.

---

## Step 3: `requirements.txt`

```
pandas>=2.0
numpy>=1.24
pyarrow>=12.0
requests>=2.28
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12
xarray>=2023.1
netcdf4>=1.6
cdsapi>=0.6
jupyter>=1.0
```

Pin major versions but allow minor updates.

---

## Step 4: `.env.example`

```
# Set to 'true' if CDS credentials unavailable
SEAS5_FALLBACK_MODE=false

# Number of parallel jobs (0 = auto)
N_JOBS=0
```

---

## Step 5: Validation Gates (add to `src/validation/gates.py`)

```python
def assert_html_self_contained(html_path):
    """Verify analysis.html contains no external resource URLs (no http:// or https:// in link/script tags)."""

def assert_submission_final(submission_path):
    """Final check: 5 rows, correct columns, integer DOY, all sites present."""

def assert_offline_render():
    """Verify quarto render completes without network (check no download warnings in render log)."""
```

---

## Step 6: Final Assembly

After Quarto render completes:
1. Verify `analysis.html` exists and is self-contained
2. Verify `submission.csv` matches Phase 3 output (no accidental overwrites)
3. Run final validation suite (all gates from all phases)
4. Print final summary:

```
╔═══════════════════════════════════════════════════╗
║ PHENOLOGY ENGINE v1.7 — SUBMISSION READY          ║
╠═══════════════════════════════════════════════════╣
║ Predictions:                                      ║
║   Washington, D.C.  → DOY XX (Month DD)           ║
║   Kyoto             → DOY XX (Month DD)           ║
║   Liestal           → DOY XX (Month DD)           ║
║   Vancouver         → DOY XX (Month DD)           ║
║   New York City     → DOY XX (Month DD)           ║
║                                                   ║
║ CV MAE: X.XX days                                 ║
║ All validation gates: PASSED                      ║
║ Baseline ΔMAE: -X.XX days (we win)               ║
║                                                   ║
║ Files:                                            ║
║   submission.csv    ✓                             ║
║   analysis.html     ✓ (self-contained)            ║
╚═══════════════════════════════════════════════════╝
```

---

## Validation Checklist (Phase 4 Exit Criteria)

- [ ] `analysis.html` renders without network access
- [ ] `analysis.html` uses KaTeX (not MathJax)
- [ ] `self-contained: true` in YAML header
- [ ] All plots render inline (no broken images)
- [ ] Baseline comparison section present (even if showing "unavailable")
- [ ] `submission.csv` unchanged from Phase 3
- [ ] `requirements.txt` present with pinned versions
- [ ] Cold-start test passes: `rm -rf data/silver data/processed data/gold && python refresh_data.py && quarto render analysis.qmd`
- [ ] Final submission.csv has exactly 5 rows of integer DOY
