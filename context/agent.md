# agent.md — Claude Code Orchestration Protocol

## Who You Are
You are implementing the **Phenology Engine v1.7**, a cherry blossom bloom date forecasting system for the 2026 GMU competition. You are the implementation agent. Architecture decisions have already been made — your job is to execute them precisely.

## How This Works
1. You implement one phase at a time, following the corresponding `phase{N}.md` spec.
2. After each phase, your work will be audited by a human+AI review team.
3. Do NOT skip ahead to later phases. Do NOT refactor earlier phases unless explicitly told to.
4. If a spec is ambiguous, implement the most conservative interpretation and leave a `# TODO: AUDIT` comment.

## Rules

### Code Style
- Python 3.11+. Type hints on all function signatures.
- Use `logging` (not print) for all output. Logger name = module name.
- Docstrings on every public function (Google style).
- Constants in `config/settings.py`. Never hardcode site coordinates, URLs, thresholds, or paths in implementation files.
- All file I/O uses `pathlib.Path`, not string concatenation.

### Data Integrity
- ALL timestamps must be UTC-aware (`pd.Timestamp` with `tz='UTC'` or `datetime` with `tzinfo=timezone.utc`). This is non-negotiable.
- ALL weather data truncated at 2026-02-28 23:59:59 UTC before any feature computation.
- Parquet for tabular data. NetCDF for ensemble data. CSV only for final submission and GMU labels.
- Never overwrite existing data files unless `--force` flag is passed.

### CLI Design
- Entry points use `argparse` with clear `--help` text.
- `refresh_data.py` is the Phase 1 CLI. It should support:
  - `python refresh_data.py` (run everything)
  - `python refresh_data.py --step era5` (run only ERA5 fetch)
  - `python refresh_data.py --step asos`
  - `python refresh_data.py --step seas5`
  - `python refresh_data.py --step labels`
  - `python refresh_data.py --step features`
  - `python refresh_data.py --force` (overwrite existing files)
  - `python refresh_data.py --sites washingtondc,kyoto` (subset of sites)
- Every step logs start/end and writes a validation summary to stdout.

### Validation Gates
- Every phase has explicit validation gates defined in its spec.
- Implement gates as functions in `src/validation/gates.py`.
- Gates that fail should raise `AssertionError` with a descriptive message.
- Gates are called at the END of each step, not inline.

### Error Handling
- Network fetches: retry 3x with exponential backoff (1s, 5s, 15s).
- CDS API: if credentials missing and `SEAS5_FALLBACK_MODE != true`, raise clear error explaining what to do.
- Never silently swallow exceptions. Log the error, then re-raise or exit with code 1.

### Dependencies
- Core: `pandas`, `numpy`, `pyarrow`, `requests`, `scipy`, `scikit-learn`
- SEAS5: `cdsapi`, `xarray`, `netcdf4`
- Quarto: `jupyter`, `matplotlib`, `seaborn`
- Pin versions in `requirements.txt`.

### What NOT To Do
- Do not install or use `dask`, `spark`, or any distributed framework.
- Do not use `pickle`. Use Parquet or JSON for serialization.
- Do not create Jupyter notebooks. All code is `.py` files except `analysis.qmd`.
- Do not add ML models in Phase 1. Phase 1 is data only.
- Do not fetch data for analog/supplementary sites unless the phase spec explicitly says to.
- Do not create tests in a separate test suite — validation gates serve as the test layer.

## File Read Order
Before starting any phase, read these files in order:
1. `docs/ARCHITECTURE.md` — understand the full system
2. `docs/agent.md` — (this file) understand your constraints
3. `docs/phase{N}.md` — the specific phase you're implementing
4. `config/settings.py` — all constants and site definitions

## Completion Signal
When you finish a phase, print this summary:
```
═══════════════════════════════════════
PHASE {N} COMPLETE
Files created: [list]
Validation gates: [PASS/FAIL for each]
Ready for audit: YES/NO
═══════════════════════════════════════
```
