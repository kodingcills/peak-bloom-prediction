---
ROLE: Novelty Strategist & Research Lead
INPUTS: Traditional process-based models (Utah, GDD)
OUTPUTS: Justification for Hybrid Bayesian Stacking
CONSTRAINTS: Must address climate non-stationarity and extrapolation failure in standard ML.
USED_BY: `MODELING_ASSUMPTIONS.md`, Competition Abstract
---
# Research Gaps & Model Requirements

## 1. The "Rigid Biology" Gap (Process-Based Models)
**The Problem:** Traditional models (e.g., Spring Warming, Sequential, or Parallel models) rely on static thermal thresholds calibrated on historical data. 
**The Gap:** Climate non-stationarity. In rapidly warming winters, these models fail to accumulate sufficient "chill portions" ($C^*$) required to break dormancy. This causes the mathematical formulas to "break," resulting in extreme late-bloom predictions that contradict actual observations of early blooms.
**Implementation Requirement:** The Biological Core must use the **Dynamic Chill Model** (Fishman & Erez, 1987) to calculate "Chill Portions" ($P$) rather than simple "Chill Hours." This accounts for the reversible nature of chilling when interrupted by warm winter afternoons.

## 2. The "Extrapolation" Gap (Pure Machine Learning)
**The Problem:** Tree-based models (XGBoost, Random Forests) are inherently **interpolative**. They cannot predict values outside the range of their training data.
**The Gap:** If Spring 2026 presents record-breaking heat, a standard AI model will default to the earliest date in its history, failing to account for the accelerated biological forcing.
**Implementation Requirement:** Features must be **Mechanistically Informed**. Instead of raw daily temperatures, the model must ingest **Accumulated Chill Portions** and **Growing Degree Days (GDD)**. This constrains the model to physical reality, allowing it to extrapolate based on biological heat-accumulation limits.

## 3. The "Data Scarcity" Gap (Site-Specific Imbalance)
**The Problem:** Data volume is highly asymmetric. 
- Kyoto, Japan: ~1,200 years of records.
- Washington D.C.: ~100 years.
- NYC / Vancouver: < 10 years.
**The Gap:** Standard ML overfits on NYC/Vancouver, treating them as independent entities. This ignores the "universal" biological constants shared by *Prunus* species across the Northern Hemisphere.
**Implementation Requirement:** Use **Bayesian Hierarchical Modeling (Partial Pooling)**. The model must treat locations as "groups" that share a common "Global Prior." This allows the model to "borrow strength" from Kyoto's deep history to stabilize the parameters for NYC and Vancouver.

## 4. The "Uncertainty" Gap (Point Estimates vs. Distributions)
**The Problem:** Competition baselines provide a single "Point Estimate" (e.g., April 2nd).
**The Gap:** Weather is stochastic. A single date hides the risk of a late-season frost or an anomalous heatwave.
**Implementation Requirement:** The model must output a **Probability Density Function (PDF)**. We require **80% and 95% Credible Intervals** to quantify the risk profile of the 2026 forecast.

---

## Technical Specifications for Gemini CLI Guidance:
1. **Bio-Year:** All accumulation logic (Chill/Heat) must start on **September 1st** of the preceding year to capture the full dormancy cycle.
2. **Hourly Interpolation:** Use a sine-log-linear model to estimate 24-hour temperature curves from TMIN/TMAX for the Dynamic Chill equations.
3. **Hybrid Architecture:** - Base Level: Biological Mechanistic Model.
    - Meta Level: Bayesian Hierarchical Stacker to combine Biological and ML (XGBoost) residuals.
4. **Forecast Blending:** For the 2026 horizon, use a weight-decay blend of **AccuWeather (1-10 days)** and **Historical Climatology (11-60 days)**.