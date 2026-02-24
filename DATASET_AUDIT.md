# Dataset Audit Report: Peak Bloom Prediction

## 1. Dataset Inventory

| File | Rows | Columns | Observation Unit | Key Candidates |
| :--- | :--- | :--- | :--- | :--- |
| `features_train.csv` | 135,258 | 18 | Daily Site-Date | `site`, `date` |
| `features_2026_forecast.csv` | 870 | 18 | Daily Site-Date | `site`, `date` |
| `data/washingtondc.csv` | 105 | 7 | Site-Year Target | `year` |
| `data/kyoto.csv` | 837 | 7 | Site-Year Target | `year` |
| `data/liestal.csv` | 132 | 7 | Site-Year Target | `year` |
| `data/vancouver.csv` | 4 | 7 | Site-Year Target | `year` |
| `data/nyc.csv` | 2 | 7 | Site-Year Target | `year` |
| `data/external/ao_daily.csv` | 27,789 | 2 | Daily Index | `date` |
| `data/external/nao_daily.csv` | 27,788 | 2 | Daily Index | `date` |

### Join Cardinalities
- **Features ↔ Targets**: Many-to-One join on `(site, bio_year)`. Join succeeds for all sites, but targets are extremely sparse for Vancouver (4) and NYC (2).
- **Features ↔ Teleconnections**: Many-to-One join on `date`. Note: `nao_daily.csv` has 1 fewer row than `ao_daily.csv`.

---

## 2. Column Semantics Table

| File | Column | Dtype | Role | Missing% | Unique% | Min / Median / Max | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `features_train.csv` | `TMAX` | float64 | Feature | 0.0% | 2.2% | -1.43 / 1.75 / 26.6 | **Unit Discontinuity Detected** |
| `features_train.csv` | `cp` | float64 | Feature | 0.0% | 0.0007% | 0.0 / 0.0 / 0.0 | **DEAD FEATURE** |
| `features_train.csv` | `gdd` | float64 | Feature | 0.0% | 0.0007% | 0.0 / 0.0 / 0.0 | **DEAD FEATURE** |
| `features_2026_forecast.csv` | `oni_30d` | float64 | Feature | 0.0% | 6.9% | -99.9 / -99.9 / -0.39 | **CORRUPT FORECAST DATA** |
| `features_train.csv` | `PRCP` | float64 | Feature | 0.0% | 1.9% | 0.0 / 0.0 / 288.6 | Extreme outlier 288.6mm |

---

## 3. Coverage & Continuity

### Temporal Coverage per Site (Training)
- **washingtondc**: 1950-01-01 to 2025-08-31 (27,637 rows, 0 gaps)
- **kyoto**: 1951-01-01 to 2025-08-31 (27,272 rows, 0 gaps)
- **liestal**: 1950-01-01 to 2025-08-31 (27,637 rows, 0 gaps)
- **vancouver**: 1957-01-06 to 2025-08-31 (25,075 rows, 0 gaps)
- **newyorkcity**: 1950-01-01 to 2025-08-31 (27,637 rows, 0 gaps)

### Target Coverage (The Sparse Regime)
- **Kyoto**: 837 targets (812 to 2025). The gold standard for training.
- **Vancouver**: **4 targets** (2022 to 2025). Model is effectively zero-shot here.
- **NYC**: **2 targets** (2024 to 2025). Model is effectively zero-shot here.

---

## 4. Missingness Audit

- **Structural Missingness**: `ao_90d` and `nao_90d` have ~12% missing in forecast due to rolling window warmup at the start of the 2025-09-01 bio-year.
- **ONI Corruption**: In `features_2026_forecast.csv`, **66% of rows (575/870)** contain `-99.9` instead of valid anomalies. This originates from unparsed future-month placeholders in `oni_raw.txt`.
- **Dead Columns**: `cp` and `gdd` are constant `0.0` in all files. Biological signaling is currently absent from the provided feature sets.

---

## 5. Unit & Scale Audit

### **CONFIRMED ANOMALY: Unit Discontinuity (Vancouver)**
- **Evidence**: 
  - Vancouver TMAX on 2025-08-24: **2.37**
  - Vancouver TMAX on 2025-08-25: **26.60**
- **Impact**: A 10x jump. Historical data is in **Tenths of Degrees Celsius**, while recent patched data is in **Full Celsius**.
- **Model Risk**: `smart_rescale` in `model_stacker.py` will fail to fix the "tenths" part of 2025 because the max (26.6) is > 15.0. July 2025 will be interpreted as near-freezing (2.4 C).

### **CONFIRMED ANOMALY: Multi-Site Tenths Scaling**
- **Evidence**: Washington DC July Median TMAX is **3.17**. Kyoto is **3.23**. Liestal is **2.54**. 
- **Status**: These are all scaled by 1/10. Any model not using `smart_rescale` will fail.

### **SUSPECTED ANOMALY: PRCP Outlier**
- **Evidence**: `features_train.csv` PRCP max is **288.6**.
- **Next Command**: `df[df.PRCP > 200]` to check if this is a single site-day error or a specific hurricane event.

---

## 6. Leakage Risk Audit

- **Rolling Windows**: `ao_30d`, `nao_30d`, etc., use `rolling(30).mean()`. In the forecast file, these must only use data up to `FORECAST_CUTOFF_DATE`.
- **Forecast Cutoff**: `features_2026_forecast.csv` max date is **2026-02-21**. This is safe (before the 2026-02-28 deadline).
- **Target Leakage**: UNKNOWN. Target files (`data/vancouver.csv`) contain 2025 dates. If training includes 2025 and the model uses future features for 2025 targets, leakage occurs.

---

## 7. Gap Backlog

| Gap | Evidence | Impact | Fix | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Dead Bio Features** | `cp`, `gdd` max=0.0 | No mechanistic signal | Rerun `feature_engineer.py` with fixed DCM | **P0** |
| **Vancouver Unit Jump** | 2.37 -> 26.60 jump | Corrupts 2025/2026 forcing | Patch `data_qa_cleaner.py` to harmonize units | **P0** |
| **ONI Sentinel Values** | `-99.9` in forecast | Breaks ENSO features | Update `load_teleconnections` to filter -99.9 | **P1** |
| **Snow Data Absent** | PRCP only | Cannot detect insulating snow cover | Fetch SNOW/SNWD from GHCND | **P2** |
| **Sparse Targets** | NYC=2, Vancouver=4 | High variance for expansion sites | Hierarchical partial pooling (implemented in V5) | **P1** |

---

## 8. Action Plan

1. **Harmonize Units**:
   - Run: `python3 -c "import pandas as pd; df=pd.read_csv('data/vancouver_historical_climate.csv'); ..."`
   - Acceptance: July Median TMAX ~ 25-30 for all sites.
2. **Revive Mechanistic Features**:
   - Inspect `feature_engineer.py` for why `total_cp` never hits threshold.
   - Acceptance: `features_train.csv['cp'].max() > 40`.
3. **Fix ONI Parsing**:
   - Filter `-99.9` in `teleconnections.py` or `feature_engineer.py`.
   - Acceptance: `features_2026_forecast.csv['oni_30d'].min() > -5.0`.
