# CLI_PROMPTS.md — Exact Prompts for Claude Code

## How to Use This File
Each prompt below is designed to be copy-pasted directly into Claude Code (terminal). Execute them **in order**. After each phase, bring the outputs back to your review team for audit before proceeding.

---

## 0. Project Setup

### Prompt 0a: Clone and scaffold
```
Clone https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction into this directory. Then read docs/ARCHITECTURE.md and docs/agent.md to understand the project. Create the full directory structure described in ARCHITECTURE.md (config/, src/ingestion/, src/processing/, src/modeling/, src/validation/, src/baselines/, data/raw/gmu/, data/silver/weather/, data/silver/asos/, data/processed/, data/gold/, data/raw/vancouver_sources/). Create empty __init__.py files in all Python package directories. Do not write any implementation code yet — just the skeleton.
```

### Prompt 0b: Move GMU data
```
Copy all competition CSV files (washingtondc.csv, kyoto.csv, liestal.csv, vancouver.csv, nyc.csv, japan.csv, meteoswiss.csv, south_korea.csv) from the repo's data/ directory into data/raw/gmu/. Also copy the USA-NPN files if present.
```

### Prompt 0c: Install dependencies
```
Create requirements.txt with: pandas>=2.0, numpy>=1.24, pyarrow>=12.0, requests>=2.28, scipy>=1.10, scikit-learn>=1.2, matplotlib>=3.7, seaborn>=0.12, xarray>=2023.1, netcdf4>=1.6, cdsapi>=0.6, jupyter>=1.0. Then pip install -r requirements.txt.
```

---

## 1. Phase 1: Data Acquisition

### Prompt 1a: Config module
```
Read docs/phase1.md Step 1 and docs/ARCHITECTURE.md. Implement config/settings.py and config/__init__.py with all site definitions, API endpoints, temporal constants, feature engineering constants, and path constants exactly as specified. Use a dataclass for Site. All 5 competition sites must be defined with their exact coordinates, species, bloom thresholds, ERA5 start years, and ASOS station codes from the ARCHITECTURE.md site table.
```

### Prompt 1b: ERA5-Land fetcher
```
Read docs/phase1.md Step 2 and docs/agent.md. Implement src/ingestion/era5_fetcher.py with fetch_era5_site() and fetch_all_era5() functions exactly as specified. Key requirements: decadal chunking, UTC timestamps, 2026 truncated at Feb 28, retry with backoff, skip existing files unless --force, save as parquet per chunk plus consolidated file. Import all constants from config.settings.
```

### Prompt 1c: ASOS fetcher
```
Read docs/phase1.md Step 3. Implement src/ingestion/asos_fetcher.py with fetch_asos_station() and fetch_all_asos() functions. Key: Iowa Mesonet CGI endpoint, UTC timezone, convert °F to °C, handle missing data markers ("M"), save as parquet per station. Only fetch for sites that have ASOS stations defined in config.
```

### Prompt 1d: SEAS5 fetcher
```
Read docs/phase1.md Step 4. Implement src/ingestion/seas5_fetcher.py with fetch_seas5() function. Key: uses cdsapi.Client() with ~/.cdsapirc, requests seasonal-original-single-levels for 2m_temperature, Feb 2026 init, 50 ensemble members, 3 months lead. If SEAS5_FALLBACK_MODE is true, skip and return None. If fetch fails after 3 retries, write a flag file data/processed/SEAS5_FETCH_FAILED and continue. Save as NetCDF. Add a TODO: AUDIT comment about the exact CDS product name.
```

### Prompt 1e: Label processing
```
Read docs/phase1.md Step 5. Implement src/processing/labels.py with load_competition_labels() and load_supplementary_labels(). Load all 5 competition CSVs from data/raw/gmu/, concatenate, validate bloom_doy matches bloom_date, add site_key column mapping location strings to our site keys (note: GMU uses "newyorkcity" for NYC). Return clean DataFrames.
```

### Prompt 1f: Feature engineering
```
Read docs/phase1.md Step 6. Implement src/processing/features.py with compute_gdh(), compute_chill_portions(), and build_gold_features(). GDH uses base temp 4.5°C, simple hourly accumulation. Chill portions use the quadratic kernel in [-2, 14]°C range with optimum at 6°C. build_gold_features loops over all site-years in labels, computes both features from silver weather data, adds 2026 rows with null bloom_doy, saves to data/gold/features.parquet.
```

### Prompt 1g: Validation gates
```
Read docs/phase1.md Step 7. Implement src/validation/gates.py with ALL Phase 1 validation functions: assert_historical_window_end, assert_inference_cutoff_utc, assert_labels_complete, assert_seas5_members, assert_silver_utc, assert_gold_schema. Each raises AssertionError with descriptive message on failure.
```

### Prompt 1h: CLI entry point
```
Read docs/phase1.md Step 8 and docs/agent.md CLI Design section. Implement refresh_data.py as the Phase 1 CLI entry point with argparse. Support --step (era5/asos/seas5/labels/features), --force, --sites flags. Execute steps in order, run all validation gates at the end, print the summary table. Use logging throughout.
```

### Prompt 1-test: Dry run validation
```
Run: python refresh_data.py --help
Verify the help text shows all flags. Then run: python refresh_data.py --step labels
This should work immediately since GMU CSVs are local. Check that data/raw/gmu/ labels load correctly and report the expected site-year counts: DC=105, Kyoto=837, Liestal=132, Vancouver=4, NYC=2.
```

---

## 2. Phase 2: Modeling & CV

### Prompt 2a: Bias correction
```
Read docs/phase2.md Step 1. Implement src/processing/bias_correction.py with estimate_bias(). OLS regression of ASOS vs ERA5-Land temperatures with fold-safe exclude_year parameter. Identity transform for sites without ASOS. Return dict of coefficients.
```

### Prompt 2b: Warming velocity
```
Read docs/phase2.md Step 2. Implement src/processing/warming_velocity.py with compute_warming_velocity(). Linear slope of daily mean temp in 14-day window. Window center MUST come from training fold mean bloom DOY. Flag when window extends past DOY 59 for 2026.
```

### Prompt 2c: SYH cross-validator
```
Read docs/phase2.md Step 3. Implement src/modeling/syh_cv.py with SYHCrossValidator class. Full Site-Year Holdout loop: for each year Y, re-estimate bias, re-compute mean_bloom_doy, compute warming velocity, re-derive weights, fit regression (bloom_doy ~ gdh + cp + wv with site intercepts), predict, record residuals. Output cv_results.parquet.
```

### Prompt 2d: Empirical Bayes
```
Read docs/phase2.md Step 4. Implement src/modeling/empirical_bayes.py with compute_shrinkage_weights() and apply_shrinkage(). Formula: w_s = σ²_global / (σ²_global + (σ²_s + ε) / N_s). Save weights to data/processed/shrinkage_weights.json.
```

### Prompt 2e: Phase 2 validation
```
Read docs/phase2.md Step 5. Add all Phase 2 validation gates to src/validation/gates.py: assert_bias_fold_safe, assert_window_safe, assert_precision_fold_safe, assert_vancouver_weight_stable, assert_cv_no_leakage. Then run the full SYH CV pipeline and report per-site MAE and shrinkage weights.
```

---

## 3. Phase 3: Inference

### Prompt 3a: SEAS5 processor
```
Read docs/phase3.md Step 1. Implement src/modeling/seas5_processor.py with extract_site_forecasts() and ensemble_to_bloom_predictions(). Extract nearest grid point forecasts per site, compute GDH continuation for each ensemble member, produce 50 predicted bloom DOYs per site. Include fallback mode using historical temperature trajectories.
```

### Prompt 3b: GMM selector
```
Read docs/phase3.md Step 2. Implement src/modeling/gmm_selector.py with fit_bimodal(). Fit GMM k=1 and k=2 only (NEVER higher). Select via BIC with neutral threshold tie-breaking at 0.3. Cluster selection: closest to climatological mean, unless dominant cluster has >70% weight. No artificial noise injection.
```

### Prompt 3c: Final predictor + submission
```
Read docs/phase3.md Step 3. Implement src/modeling/predictor.py with generate_predictions() and save_submission(). Apply shrinkage, round to integer, clip to [60,140], write submission.csv with exactly 5 rows. Also save diagnostic JSONs to data/processed/diagnostics/.
```

### Prompt 3d: Phase 3 validation
```
Read docs/phase3.md Step 5. Add Phase 3 validation gates. Run the full inference pipeline. Print the 5 predictions and verify submission.csv exists with correct schema.
```

---

## 4. Phase 4: Report

### Prompt 4a: Baseline module
```
Read docs/phase4.md Step 2. Implement src/baselines/o3_mini.py. Simple baseline: recent 10-year mean + GDD anomaly adjustment. Must be fully non-blocking — wrap everything in try/except, return None on any failure.
```

### Prompt 4b: Quarto document
```
Read docs/phase4.md Step 1. Create analysis.qmd with the exact YAML header specified (self-contained: true, katex, toc, code-fold). Implement all 7 sections: Introduction, Data Sources, Methodology (with LaTeX equations), CV Results (with plots), 2026 Predictions (with ensemble plots), Beat-the-AI Baseline, Reproducibility. Use matplotlib+seaborn only. The baseline section must be wrapped in try/except.
```

### Prompt 4c: Final render and validation
```
Run: quarto render analysis.qmd
Verify analysis.html is produced, is self-contained (no external URLs in link/script tags), and all plots render. Verify submission.csv is unchanged. Run all validation gates from all phases. Print the final submission summary.
```

---

## 5. Final Cold-Start Test

### Prompt 5: Nuclear test
```
Run a full cold-start test:
rm -rf data/silver data/processed data/gold submission.csv analysis.html
python refresh_data.py
quarto render analysis.qmd

Verify submission.csv has 5 rows of integer DOY predictions and analysis.html renders correctly. Report any failures.
```

---

## Audit Checkpoints

After each phase, paste the following into your review session:

**Phase 1 audit:**
> Phase 1 is implemented. Here are the outputs: [paste summary table from refresh_data.py]. Here are the validation gate results: [paste]. Any issues?

**Phase 2 audit:**
> Phase 2 CV is done. Per-site MAE: [paste]. Shrinkage weights: [paste]. Any red flags?

**Phase 3 audit:**
> Phase 3 predictions: [paste submission.csv contents]. GMM BIC results: [paste]. Ensemble distributions: [describe]. Review?

**Phase 4 audit:**
> Phase 4 render complete. analysis.html is [X] KB, self-contained=[yes/no]. Baseline ΔMAE: [paste]. Cold-start test: [pass/fail]. Ready to submit?
