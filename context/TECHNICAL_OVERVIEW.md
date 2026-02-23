---
ROLE: Senior Staff Research Engineer / Lead Phenology Architect
INPUTS: /data (Daily Weather + Normals), /data/external (AO/NAO/ENSO/AMO), /context (Technical Architecture + Biological Background)
OUTPUTS: feature_engineer.py (Vectorized Biological Transformer), features_train.csv, features_2026_forecast.csv
CONSTRAINTS: Vectorized NumPy/Pandas logic; Fishman & Erez (1987) Dynamic Chill Model; Sept-1 Bio-Year Indexing; 
Site-specific species thresholds. 
USED_BY: Hierarchical Bayesian Stacker / XGBoost Residual Model
---

# Technical Architecture: Hybrid Bayesian Phenology System

## 1. System Philosophy
This system combines a **Mechanistic Core** (biological state modeling) with a **Statistical Residual Layer** (atmospheric noise). It prioritizes biological causality over "Black Box" curve-fitting.

## 2. Competition Metadata (Source of Truth)
| Location | Station ID | Species/Cultivar | Bloom Definition |
| :--- | :--- | :--- | :--- |
| **Washington D.C.** | GHCND:USW00013743 | Yoshino (*P. x yedoensis*) | 70% Bloom |
| **Kyoto** | GHCND:JA000047759 | *P. jamasakura* | Full Bloom |
| **Liestal** | GHCND:SZ000001940 | Wild Cherry (*P. avium*) | 25% Bloom |
| **Vancouver** | GHCND:CA001108395 | Akebono | 70% Bloom |
| **NYC** | GHCND:USW00014732 | Yoshino (*P. x yedoensis*) | 70% Bloom |

## 3. The Four-Stage Pipeline

### Stage 1: Data Fusion
- **Micro-Climate:** Daily TMIN/TMAX (1950–Feb 21, 2026) using official Station IDs.
- **NYC Proxy:** Integrate USA-NPN data (Site: 32789, Species: 228) with a calculated offset for peak bloom.
- **Macro Drivers:** AO, NAO, ONI, and AMO indices for global atmospheric context.
- **Biological Gates:** Calculated Day Length (Photoperiod) based on site latitude.

### Stage 2: Feature Engineering (The Bio-Transformer)
- **Dynamic Chill Model:** Calculates "Chill Portions" (CP) using sine-interpolation for hourly kinetics. Essential for handling "chill negation" in warm winters.
- **Growing Degree Days (GDD):** Heat accumulation starting only after site-specific CP thresholds are met.
- **Bio-Year Indexing:** All data re-aligned to start on September 1st.
- **Lagged Teleconnections:** 30-day and 90-day rolling means of AO/NAO.

### Stage 3: Modeling Engine (Hierarchical Stacker)
- **Level 1 (Mechanistic):** Non-linear estimation of bloom dates based on CP and GDD.
- **Level 2 (Hierarchical):** Site-specific parameters drawn from a shared distribution to "borrow strength" for data-sparse sites (NYC and Vancouver).
- **Level 3 (Residual ML):** XGBoost trained on the errors of the mechanistic model using teleconnections as features.

### Stage 4: 2026 Stochastic Forecast
- **Bridge Simulation:** Observed data (to Feb 21, 2026) + Modern Normals (Feb 22 onwards).
- **Monte Carlo Output:** Generates a Probability Density Function (PDF) of predicted dates.