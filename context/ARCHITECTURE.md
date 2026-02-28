# ARCHITECTURE.md — Phenology Engine v1.7

## Purpose
Predict peak cherry blossom bloom dates (as integer DOY) for 5 sites in the 2026 GMU International Cherry Blossom Prediction Competition. Minimize MAE. Full reproducibility required.

## System Design

```
Phase 1 (Online)              Phase 2-4 (Offline — Quarto Render)
┌─────────────────┐           ┌──────────────────────────────────────┐
│ refresh_data.py  │           │ analysis.qmd                         │
│                  │           │                                      │
│ ERA5-Land ──────►│──► Silver │  Gold Features ──► SYH CV ──► GMM   │
│ ASOS ───────────►│   Parquet │  (GDH, CP, WV)    (fold-safe)  ▼    │
│ SEAS5 ──────────►│──► NetCDF │                              BIC k  │
│ GMU Labels ─────►│──► CSV    │                               ▼     │
│ Vancouver ──────►│──► CSV    │                         submission   │
└─────────────────┘           │                           .csv       │
                              │                         (5 rows)     │
                              │                                      │
                              │  + analysis.html (self-contained)    │
                              └──────────────────────────────────────┘
```

**Key Invariant:** Phase 2-4 must execute with ZERO network access. All data fetching happens in Phase 1.

---

## 5 Competition Sites

| Key | location_id | Lat | Lon | Alt(m) | Species | Bloom % | Labels | Year Range | ASOS Stations |
|-----|------------|------|------|--------|---------|---------|--------|------------|---------------|
| `washingtondc` | `washingtondc` | 38.8853 | -77.0386 | 0 | P. x yedoensis | 70% | 105 | 1921–2025 | DCA, IAD |
| `kyoto` | `kyoto` | 35.0120 | 135.6761 | 44 | P. jamasakura | newspaper | 837 | 812–2025 | — |
| `liestal` | `liestal` | 47.4814 | 7.7305 | 350 | P. avium | 25% | 132 | 1894–2025 | — |
| `vancouver` | `vancouver` | 49.2237 | -123.1636 | 24 | Yoshino | ~70% | 4 | 2022–2025 | CYVR |
| `nyc` | `newyorkcity` | 40.7304 | -73.9981 | 8.5 | P. x yedoensis | ~70% | 2 | 2024–2025 | JFK, LGA |

**Cold-start sites:** Vancouver (4 labels) and NYC (2 labels) cannot support site-specific models. They require shrinkage toward a global prior via Empirical Bayes weighting.

---

## Data Sources

| Asset | Provider | API | Auth | Grain | Years | Output Path |
|-------|----------|-----|------|-------|-------|-------------|
| ERA5-Land | Open-Meteo Archive | REST | None | Hourly | 1950–2026 | `data/silver/weather/{site}/` |
| ASOS | Iowa Mesonet | CGI | None | Hourly | 2000–2026 | `data/silver/asos/{station}.parquet` |
| SEAS5 | Copernicus CDS | cdsapi | `~/.cdsapirc` | Ensemble (50 members) | 2026 | `data/processed/seas5_2026.nc` |
| GMU Labels | Competition repo | Local | None | Annual | varies | `data/raw/gmu/*.csv` |
| Vancouver | VCBF (manual) | Local | None | Annual | 2022–2025 | `data/raw/vancouver_labels.csv` |

### ERA5-Land via Open-Meteo
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Variables: `temperature_2m`, `relative_humidity_2m`, `soil_temperature_0_to_7cm`
- **Must fetch in decadal chunks** (e.g., 1950–1959, 1960–1969) to avoid timeouts (Problem 27)
- All timestamps forced to UTC via `timezone=UTC` parameter
- Current year (2026) truncated at Feb 28

### ASOS via Iowa Mesonet
- Endpoint: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- Variables: `tmpf` (°F temp), `dwpf` (dewpoint), `relh` (humidity)
- Only for sites with stations: DC (DCA, IAD), NYC (JFK, LGA), Vancouver (CYVR)
- All timestamps UTC via `tz=Etc/UTC`

### SEAS5 via CDS API
- Product: `seasonal-original-single-levels`
- System 51, Feb 2026 initialization
- 50 ensemble members, ~90 days of daily lead times
- Variables: `2m_temperature`
- Requires `~/.cdsapirc` credentials
- **Fallback:** If `SEAS5_FALLBACK_MODE=true`, skip fetch and use climatological temperature trajectories

---

## Temporal Contracts

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Inference cutoff | 2026-02-28 23:59:59 UTC | Competition deadline — no data after this |
| Feature truncation | DOY 59 | All features computed from data ≤ DOY 59 |
| Chill accumulation start | DOY 274 (Oct 1 prior year) | Biological dormancy onset |
| Warming accumulation start | DOY 1 (Jan 1) | GDH accumulation window |
| ERA5-Land availability | 1950–present | Pre-1950 not available |
| ASOS availability | ~2000–present | Station coverage varies |

---

## Feature Definitions

### Growing Degree Hours (GDH) — Anderson Model
Accumulated hourly heat units from Jan 1 through DOY 59:
- Base temp: 4.5°C
- For each hour: `gdh = max(0, T_hour - 4.5)`
- Sum over all hours in window

### Chill Portions (CP) — Simplified Dynamic Model
Accumulated chill units from Oct 1 (prior year) through DOY 59:
- Effective range: -2°C to 14°C
- Optimal: 6°C
- `cp_hour = max(0, 1 - ((T - 6) / 8)^2)` when -2 ≤ T ≤ 14, else 0

### Warming Velocity (WV)
Rate of temperature increase in a 14-day window anchored to fold-safe mean bloom DOY:
- Computed as linear slope of daily mean temp over window
- Window center = `mean_bloom_doy` estimated from training fold (not test year)

---

## Cross-Validation Protocol: Site-Year Holdout (SYH)

For each held-out year Y:
1. Remove ALL sites' data for year Y from training
2. Re-estimate bias correction β₁ on training years only
3. Re-calculate `mean_bloom_doy` on training years only
4. Re-derive shrinkage weights on training years only
5. Predict all 5 sites for year Y
6. Record residuals

This prevents temporal leakage across sites within the same year.

---

## Inference: Bimodal Resolution via SEAS5

1. Extract 50-member SEAS5 temperature trajectories for each site
2. Compute GDH accumulation under each ensemble member → 50 predicted bloom DOYs
3. Fit GMM with k ∈ {1, 2} only (Problem 22)
4. Select k via BIC
5. If k=2: choose cluster whose mean is closer to the site's climatological mean, unless SEAS5 anomaly signal exceeds `SEAS5_NEUTRAL_THRESHOLD = 0.3`
6. If k=1: use the single Gaussian mean
7. Final prediction = selected cluster mean, rounded to integer DOY

---

## Empirical Bayes Precision Weighting

For cold-start sites (Vancouver, NYC), shrink predictions toward global mean:

```
w_s = σ²_global_s / (σ²_global_s + (σ²_s + ε) / N_s)
```

- `σ²_global_s`: variance of bloom DOY across all sites in training
- `σ²_s`: variance of site-specific residuals
- `N_s`: number of labels for site s
- `ε`: floor to prevent division instability (calibrate via LOO)
- Final prediction: `pred_s = w_s * model_pred_s + (1 - w_s) * global_mean`

---

## Output Contract

### `submission.csv`
```
location,year,bloom_doy
washingtondc,2026,<int>
kyoto,2026,<int>
liestal,2026,<int>
vancouver,2026,<int>
newyorkcity,2026,<int>
```
Exactly 5 rows. Integer DOY only.

### `analysis.html`
- Self-contained (`self-contained: true` in Quarto YAML)
- Math rendered via KaTeX (`html-math-method: katex`)
- Must render without network access

---

## Directory Structure (Target)

```
peak-bloom-prediction/
├── refresh_data.py              # Phase 1 entry point (CLI)
├── analysis.qmd                 # Phase 2-4 (Quarto render)
├── config/
│   ├── __init__.py
│   └── settings.py              # All constants, site defs, paths
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── era5_fetcher.py      # Open-Meteo decadal chunked fetch
│   │   ├── asos_fetcher.py      # Iowa Mesonet station fetch
│   │   └── seas5_fetcher.py     # CDS API SEAS5 ensemble fetch
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── features.py          # GDH, CP, WV computation
│   │   ├── bias_correction.py   # ASOS↔ERA5 bias estimation
│   │   └── labels.py            # GMU CSV loader + validation
│   ├── validation/
│   │   ├── __init__.py
│   │   └── gates.py             # All assert-based validation gates
│   └── baselines/
│       └── o3_mini.py           # Beat-the-AI baseline (non-blocking)
├── data/
│   ├── raw/
│   │   ├── gmu/                 # Competition CSVs (committed)
│   │   └── vancouver_sources/   # Archived VCBF proof
│   ├── silver/
│   │   ├── weather/             # ERA5-Land parquet by site
│   │   └── asos/                # ASOS parquet by station
│   ├── processed/
│   │   └── seas5_2026.nc        # SEAS5 ensemble NetCDF
│   └── gold/
│       └── features.parquet     # Final feature matrix (truncated DOY 59)
├── submission.csv               # Final output
├── requirements.txt
├── .env.example                 # SEAS5_FALLBACK_MODE, N_JOBS
└── docs/
    ├── ARCHITECTURE.md
    ├── agent.md
    ├── phase1.md
    ├── phase2.md
    ├── phase3.md
    ├── phase4.md
    └── CLI_PROMPTS.md
```

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Open-Meteo timeout | Decadal chunking + retry with backoff |
| CDS credentials missing | `SEAS5_FALLBACK_MODE` env toggle |
| NYC/Vancouver cold-start | Empirical Bayes shrinkage to global prior |
| Kyoto non-stationarity | Limit training to post-1950 for regression |
| Quarto needs network | `self-contained: true` + KaTeX (no MathJax CDN) |
| Evaluator RAM limits | `n_jobs = min(N_JOBS, os.cpu_count() - 1)` |
| Baseline failure | `o3_mini.py` is non-blocking; `submission.csv` generates regardless |
