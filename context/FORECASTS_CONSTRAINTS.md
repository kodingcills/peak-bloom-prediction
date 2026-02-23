---
ROLE: Real-time Systems Engineer
INPUTS: NOAA GHCN-D (Observed), 30-year Climatology (Predicted)
OUTPUTS: Blended 2026 climate stream, Stochastic bloom dates
CONSTRAINTS: Cutoff date is Feb 21, 2026. Use weight-decay blending for future dates.
USED_BY: `forecast_2026.py`
---
# 2026 Forecast & Real-Time Constraints

## 1. The "Forecast Horizon" Rule
**Constraint:** The model must distinguish between 'Observed' data and 'Predicted' data.
- **Before Feb 21, 2026:** Use ground-truth NOAA GHCN-D observations.
- **After Feb 21, 2026:** Use a blended forecast.
- **Days 1–10:** High-weight on deterministic forecasts (e.g., AccuWeather/GFS).
- **Days 11–45:** Transition to **Historical Climatology** (30-year daily means).

## 2. Biological State Initialization
**Constraint:** The 2026 forecast cannot start from zero on Jan 1st. 
- The model must ingest weather data starting from **September 1, 2025**, to calculate the "Initial State" of winter dormancy (Chill Portions already accumulated).

## 3. Stochastic Simulation (The PDF Requirement)
**Constraint:** The final 2026 output must not be a single date.
- Run the forecast through a **Monte Carlo simulation** (N=1000) using the variance from the Bayesian Posterior.
- Output the **Expected Date** (Mean) and the **90% Prediction Interval**.

## 4. Input Sanitization
**Constraint:** Handle missing values in the 2026 forecast stream using linear interpolation for gaps < 3 days and climatological replacement for gaps > 3 days.