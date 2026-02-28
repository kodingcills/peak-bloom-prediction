IMPLEMENTATION_STATE.md

# 1) Current Phase Position
- Current phase: 1
- Phase status: BLOCKED
- If BLOCKED: blocking issue summary
  - `refresh_data.py --step labels` fails with OutOfBoundsDatetime for year 0812 in Kyoto labels, blocking Phase 1 Step 5 (labels). Evidence: Command `source venv/bin/activate && python3 refresh_data.py --step labels` output shows `OutOfBoundsDatetime: 0812-04-01`. (Command output below)
  - Phase 1 gates fail because silver weather parquet files do not exist. Evidence: `python3 -m src.validation.run_all_gates --phase 1` failed at `assert_inference_cutoff_utc` with `No silver weather parquet files found`. (Command output below)

Evidence Anchors:
- Command: `source venv/bin/activate && python3 refresh_data.py --step labels`
  - Key output: `Out of bounds nanosecond timestamp: 0812-04-01` (see full output in Section 5)
- Command: `source venv/bin/activate && python3 -m src.validation.run_all_gates --phase 1`
  - Key output: `AssertionError: No silver weather parquet files found` (see full output in Section 5)

# 2) Spec Digest (10–20 invariants)
1. Phase 2–4 must execute with zero network access; all data fetching in Phase 1. Evidence: `context/ARCHITECTURE.md` “Key Invariant” section. (Code not applicable; Spec anchor: ARCHITECTURE.md)
2. Inference cutoff is `2026-02-28 23:59:59 UTC` and no weather timestamps beyond this. Evidence: `context/ARCHITECTURE.md` Temporal Contracts. (Spec anchor)
3. Feature truncation is DOY 59 for all feature computation. Evidence: `context/ARCHITECTURE.md` Temporal Contracts. (Spec anchor)
4. ERA5-Land must be fetched in decadal chunks with `timezone=UTC`. Evidence: `context/ARCHITECTURE.md` ERA5-Land section. (Spec anchor)
5. ASOS fetch uses `tz=Etc/UTC` and converts °F→°C. Evidence: `context/ARCHITECTURE.md` ASOS section. (Spec anchor)
6. SEAS5 ensemble must have 50 members (or fallback mode). Evidence: `context/ARCHITECTURE.md` SEAS5 section. (Spec anchor)
7. SYH CV excludes all sites for held-out year (fold-safe). Evidence: `context/ARCHITECTURE.md` SYH protocol. (Spec anchor)
8. GMM constraint: k ∈ {1,2} only; no noise injection. Evidence: `context/ARCHITECTURE.md` Inference section. (Spec anchor)
9. Submission schema: `location,year,bloom_doy` exactly 5 rows, integer DOY. Evidence: `context/ARCHITECTURE.md` Output Contract. (Spec anchor)
10. Quarto report must be self-contained and use KaTeX. Evidence: `context/ARCHITECTURE.md` Output Contract, `context/phase4.md` YAML. (Spec anchor)
11. All timestamps must be UTC-aware. Evidence: `context/agent.md` Data Integrity. (Spec anchor)
12. Never overwrite existing data files unless `--force`. Evidence: `context/agent.md` Data Integrity. (Spec anchor)
13. CLI design must support `refresh_data.py --step ... --force --sites`. Evidence: `context/agent.md` CLI Design. (Spec anchor)
14. Requirements must include baseline deps (phase4.md). Evidence: `context/phase4.md` Step 3. (Spec anchor)
15. Validation gates are authoritative and raise `AssertionError`. Evidence: `context/agent.md` Validation Gates. (Spec anchor)

# 3) Repository Reality Snapshot (Evidence Anchored)
## 3.1 File Tree (Top 3 levels)
Command:
- `ls`
Output snippet:
- `README.md`, `config/`, `context/`, `data/`, `docs/`, `refresh_data.py`, `requirements.txt`, `src/`, `venv/`
Evidence Anchor:
- Command output: `ls` (see raw output in tool log; key lines above)

## 3.2 Environment Snapshot
Commands and key outputs:
- `source venv/bin/activate && python3 --version` → `Python 3.11.13`
- `source venv/bin/activate && python3 -m pip --version` → `pip 26.0.1 .../venv/...`
- `source venv/bin/activate && quarto --version` → `1.8.27`
- `source venv/bin/activate && python3 -m pip check` → `No broken requirements found.`
Evidence Anchors:
- Command outputs in section 3 tool logs.

## 3.3 Artifact Inventory (Phase by Phase)
Phase 1 required artifacts (from phase1.md/ARCHITECTURE.md):
- `data/silver/weather/{site}/...parquet` → Exists: **NO** (dir empty)
  - Evidence: `ls -la data/silver/weather` shows only `.` and `..`
- `data/silver/asos/{station}.parquet` → Exists: **NO** (dir empty)
  - Evidence: `ls -la data/silver/asos` shows only `.` and `..`
- `data/processed/seas5_2026.nc` → Exists: **NO**
  - Evidence: `ls -la data/processed` shows only `diagnostics/`
- `data/processed/SEAS5_FETCH_FAILED` → Exists: **NO** (not listed)
  - Evidence: `ls -la data/processed`
- `data/gold/features.parquet` → Exists: **NO**
  - Evidence: `ls -la data/gold` shows only `.` and `..`

Phase 2 required artifacts:
- `data/processed/cv_results.parquet` → Exists: **NO**
- `data/processed/mae_summary.json` → Exists: **NO**
- `data/processed/shrinkage_weights.json` → Exists: **NO**
- `data/processed/bias_coefficients.json` → Exists: **NO**
Evidence: `ls -la data/processed` (only `diagnostics/`)

Phase 3 required artifacts:
- `submission.csv` → Exists: **NO**
  - Evidence: `ls -la submission.csv` → `No such file or directory`
- `data/processed/diagnostics/*.json` → Exists: **NO** (diagnostics dir exists but empty except `bimodal_plots/`)
  - Evidence: `ls -la data/processed/diagnostics`

Phase 4 required artifacts:
- `analysis.html` → Exists: **NO**
  - Evidence: `ls -la analysis.html` → `No such file or directory`

Auxiliary artifacts:
- `mlruns/` exists (MLflow local store)
  - Evidence: `ls -la mlruns`

# 4) Implementation Coverage Matrix (Spec → Code → Gate → Artifact)
Current phase: Phase 1

| Requirement | Spec Ref | Implementation (file:symbol) | Validation Gate | Evidence Artifact/Log | Status |
|---|---|---|---|---|---|
| Central config with site defs/constants | `context/phase1.md` Step 1 | `config/settings.py:11-129 SiteConfig, SITES` | N/A | N/A | OK (code present) |
| ERA5 decadal fetcher | `context/phase1.md` Step 2 | `src/ingestion/era5_fetcher.py:33-134 fetch_era5_site` | `assert_inference_cutoff_utc`, `assert_silver_utc` | No silver files present | FAIL (no artifacts) |
| ASOS fetcher | `context/phase1.md` Step 3 | `src/ingestion/asos_fetcher.py:13-84 fetch_asos_station` | `assert_silver_utc` | No ASOS files present | FAIL (no artifacts) |
| SEAS5 fetcher + member check | `context/phase1.md` Step 4 | `src/ingestion/seas5_fetcher.py:15-60 fetch_seas5` | `assert_seas5_members` | `data/processed/seas5_2026.nc` missing | FAIL (no artifact) |
| Label loader | `context/phase1.md` Step 5 | `src/processing/labels.py:24-54 load_competition_labels` | `assert_labels_complete` | `refresh_data.py --step labels` fails | FAIL (runtime error) |
| Feature builder | `context/phase1.md` Step 6 | `src/processing/features.py:31-114 build_gold_features` | `assert_gold_schema`, `assert_historical_window_end` | `data/gold/features.parquet` missing | FAIL (no artifact) |
| Phase 1 gates | `context/phase1.md` Step 7 | `src/validation/gates.py:27-150` | All Phase 1 gates | `run_all_gates` failed at first gate | FAIL (blocked) |
| Phase 1 CLI | `context/phase1.md` Step 8 | `refresh_data.py:31-212 main` | N/A | `refresh_data.py --step labels` failed | FAIL (blocked) |

Evidence anchors:
- Config code: `config/settings.py:11-129 SiteConfig, SITES`
- Label loader code: `src/processing/labels.py:24-54 load_competition_labels`
- Gate code: `src/validation/gates.py:27-150`
- CLI code: `refresh_data.py:31-212 main`

# 5) Gates & Validation Status
Implemented gates (from `src/validation/gates.py`):
- `assert_inference_cutoff_utc` (Phase 1) — checks no timestamps after cutoff. Evidence: `src/validation/gates.py:27-41`
- `assert_historical_window_end` (Phase 1) — verifies DOY 59 truncation. Evidence: `src/validation/gates.py:44-63`
- `assert_labels_complete` (Phase 1) — checks 5 sites and max year. Evidence: `src/validation/gates.py:65-75`
- `assert_seas5_members` (Phase 1) — checks 50 ensemble members. Evidence: `src/validation/gates.py:78-93`
- `assert_silver_utc` (Phase 1) — checks UTC timestamps. Evidence: `src/validation/gates.py:96-107`
- `assert_gold_schema` (Phase 1) — checks gold schema and 2026 nulls. Evidence: `src/validation/gates.py:110-123`
- `assert_requirements_present` (Aux) — verifies requirements baseline. Evidence: `src/validation/gates.py:126-150`

Gate runner:
- CLI: `python3 -m src.validation.run_all_gates --phase 1` (see `src/validation/run_all_gates.py:17-86`).
- Last run result: FAIL at `assert_inference_cutoff_utc` due to no silver files. Evidence: command output below.

Command output evidence:
- `source venv/bin/activate && python3 -m src.validation.run_all_gates --phase 1` → `AssertionError: No silver weather parquet files found`.

# 6) Problems / Failures / TODO: AUDIT Items
1) **Labels parsing fails for year 0812**
- Symptom: `refresh_data.py --step labels` crashes with `OutOfBoundsDatetime: 0812-04-01`.
- Root cause hypothesis: pandas `to_datetime` cannot parse pre-1677 dates at ns resolution. (INFERENCE)
- Evidence: command output from `source venv/bin/activate && python3 refresh_data.py --step labels`.
- Minimal fix: update `load_competition_labels` to handle pre-1677 dates (e.g., parse year/day to DOY without using pandas datetime for ancient dates), keeping UTC invariants for valid ranges.
- Next command(s): rerun `python3 refresh_data.py --step labels` and verify counts.

2) **Phase 1 gates fail due to missing silver artifacts**
- Symptom: `assert_inference_cutoff_utc` fails with `No silver weather parquet files found`.
- Root cause: ERA5/ASOS fetch not run or blocked by label failure. (INFERENCE)
- Evidence: gate runner output; `ls -la data/silver/weather` empty.
- Minimal fix: run Phase 1 data fetch after labels issue resolved.
- Next command(s): `python3 refresh_data.py` then `python3 -m src.validation.run_all_gates --phase 1`.

3) **TODO: AUDIT items present**
- `config/settings.py` Open-Meteo delay unknown. Evidence: `config/settings.py:57`.
- SEAS5 ensemble dimension detection TODO: `src/ingestion/seas5_fetcher.py:20`, `src/validation/gates.py:93`.
- SEAS5 product name params TODO: `src/ingestion/seas5_fetcher.py:54`.

4) **requirements.txt mismatch with spec**
- Symptom: `requirements.txt` lists `pyarrow` without `>=12.0` floor.
- Spec requires `pyarrow>=12.0`. Evidence: `context/phase4.md` Step 3; `requirements.txt` content.
- Minimal fix: update `requirements.txt` to `pyarrow>=12.0`.
- Next command(s): `python3 -m pip install -r requirements.txt` (offline-safe if already installed).

# 7) What Happens Next (Ordered Execution Plan)
1) Phase 1 Step 5 (labels): fix pre-1677 date parsing in `src/processing/labels.py`.
   - Files: `src/processing/labels.py`
   - Acceptance: `python3 refresh_data.py --step labels` completes; logs expected counts.
   - Command: `source venv/bin/activate && python3 refresh_data.py --step labels`

2) Phase 1 Step 1 (config): verify `requirements.txt` matches spec floors.
   - Files: `requirements.txt`
   - Acceptance: `pyarrow>=12.0` present.
   - Command: `cat requirements.txt`.

3) Phase 1 Step 2/3/4 (fetchers): run full Phase 1 after labels fix.
   - Files: `refresh_data.py`
   - Acceptance: silver/gold artifacts exist.
   - Command: `source venv/bin/activate && python3 refresh_data.py`

4) Phase 1 gates: validate artifacts.
   - Files: `src/validation/gates.py`, `src/validation/run_all_gates.py`
   - Acceptance: `python3 -m src.validation.run_all_gates --phase 1` passes.

5) Phase 2 readiness check: ensure `data/gold/features.parquet` exists.
   - Command: `ls -la data/gold/features.parquet`

# 8) One-True Run Sequence (Current State)
- Phase 1 only: `python3 refresh_data.py` → **BLOCKED** (labels parsing error). Evidence: `refresh_data.py --step labels` failure.
- Phase 2 only: `python3 -m src.modeling.syh_cv` → **BLOCKED** (Phase 1 artifacts missing).
- Phase 3 only: `python3 -m src.modeling.predictor` → **BLOCKED** (Phase 2 artifacts missing).
- Phase 4 only: `quarto render analysis.qmd` → **BLOCKED** (analysis.qmd missing; Phase 3 artifacts missing).
- Full cold-start: `rm -rf data/silver data/processed data/gold submission.csv analysis.html && python refresh_data.py && quarto render analysis.qmd` → **BLOCKED** (Phase 1 failure at labels).

