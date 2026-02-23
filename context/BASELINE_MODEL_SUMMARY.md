---
ROLE: Benchmarking & Adversarial Specialist
INPUTS: Previous competition winners, GMU 2025 Baseline code
OUTPUTS: Benchmark MAE/RMSE targets, baseline feature set
CONSTRAINTS: Do not replicate baseline logic; use only for competitive comparison.
USED_BY: `experimental_design.md`, `validation_harness.py`
---
# AI-Generated Baseline Model (2025 Reference Implementation)


This document summarizes the fully AI-generated 2025 model used as a benchmark in the GMU Cherry Blossom Prediction Competition.

This model serves as:

A performance baseline

A methodological reference

An adversarial target to outperform

A structural example of a purely data-driven approach

It should NOT be copied directly. It should be improved upon.

1. High-Level Overview

The baseline model predicts peak bloom day-of-year (DOY) using:

Daily historical climate data

Engineered phenological features

XGBoost regression

5-fold cross-validation

It integrates daily temperature data from NOAA and archived weather sources, engineers biologically inspired predictors, and fits a gradient-boosted tree model.

2. Data Sources Used
Historical Bloom Data

Washington DC

Kyoto

Liestal

Vancouver

New York City

Climate Data

NOAA GHCND API (daily TAVG, TMIN, TMAX)

Archived AccuWeather pages for recent year prediction

Units converted to Celsius.

3. Feature Engineering

The baseline constructs the following predictors:

3.1 Daily Temperature Variables

For each location and day:

TAVG = (TMIN + TMAX) / 2
GDD = max(TAVG - 0°C, 0)

Base temperature used: 0°C

3.2 7-Day Rolling Average of Temperature
roll7_TAVG = rolling_mean(TAVG, window = 7)

Purpose:

Smooth short-term temperature noise

Detect sustained warming periods

3.3 Dormancy Release Date

Defined as:

First date where 7-day rolling average ≥ 10°C

DormancyRelease = min(date where roll7_TAVG ≥ 10°C)
Dormancy_DOY = day_of_year(DormancyRelease)

Fixed threshold: 10°C

3.4 Cumulative Growing Degree Days Until Dormancy Release
cumGDD_dormancy = sum(GDD from Jan 1 until DormancyRelease)
3.5 Winter Chill Accumulation

Defined as:

Number of days between:

October 1 (previous year)

February 28 (current year)

Where:

TAVG < 7°C
WinterChill = count(TAVG < 7°C)

Fixed chill threshold: 7°C

3.6 Spring Warmth After Dormancy Release
SpringWarmth = sum(GDD from DormancyRelease to March 31)
4. Model Specification
Model Type:

XGBoost regression

Predictors:

Dormancy_DOY

cumGDD_dormancy

WinterChill

SpringWarmth

Location (one-hot encoded)

Preprocessing:

Centering and scaling numeric predictors

Dummy encoding categorical location

5. Hyperparameter Tuning

Random search over:

Number of trees

Learning rate

mtry

Minimum node size

Cross-validation:

5-fold CV

Not time-aware

Random fold assignment

Best RMSE ≈ 6.7 days

6. 2025 Predictions Produced
Location	Predicted Date
Kyoto	April 4
Liestal	March 30
New York City	April 8
Vancouver	March 25
Washington DC	March 27
7. Structural Strengths

✔ Incorporates biologically meaningful variables
✔ Uses daily temperature data
✔ Accounts for dormancy + forcing
✔ Multi-location joint model
✔ Nonlinear modeling capability
✔ Reproducible workflow

8. Structural Weaknesses

This is critical.

The baseline has the following weaknesses:

8.1 Fixed Dormancy Threshold

Dormancy threshold = 10°C

This is:

Arbitrary

Not species-specific

Not location-calibrated

Not optimized

True dormancy break likely varies by:

Species

Latitude

Climate regime

8.2 Fixed Chill Threshold

Chill = TAVG < 7°C

Again:

Arbitrary

Does not use dynamic chill models

Does not consider chill effectiveness weighting

More advanced chill models exist (e.g., Utah Model, Dynamic Model).

8.3 No Chill–Heat Interaction Modeling

True phenology requires:

Chill requirement to be satisfied

Then forcing accumulation

The baseline treats chill and forcing as independent features.

It does not model:

Conditional forcing

Nonlinear chill × heat interactions

8.4 Time-Agnostic Cross-Validation

5-fold random CV was used.

Problem:

Temporal leakage possible

Future data may influence training folds

Not realistic forecasting scenario

Correct method:

Leave-one-year-out

Block CV by year

8.5 No Uncertainty Modeling

The baseline predicts only point estimates.

It does not:

Produce prediction intervals

Model residual variance

Evaluate calibration

Narrative category favors uncertainty awareness.

8.6 Limited Species Differentiation

Locations use different species:

Prunus jamasakura

Prunus avium

Prunus × yedoensis

The baseline does not model species-specific biological parameters.

8.7 No Explicit Climate Trend Modeling

Long-term warming trend not modeled explicitly.

Potential issue:

Trend leakage vs climate anomaly signal.

9. Strategic Goal Relative to Baseline

To outperform the baseline, improvements should include:

Calibrated chill and heat thresholds

Mechanistic chill-forcing modeling

Time-aware validation

Species-specific parameterization

Hybrid mechanistic + ML ensemble

Prediction intervals

Better regularization for small datasets

10. Codex Instructions

When modifying this repository:

Do not replicate this baseline blindly.

Preserve reproducibility.

Avoid temporal leakage.

Consider biological realism.

Document all assumptions.

Justify threshold choices.

The objective is not to add complexity.
The objective is to add biological validity and forecasting robustness.

11. Strategic Positioning

The AI baseline demonstrates competent feature engineering but lacks:

Parameter calibration

Mechanistic grounding

Time-aware evaluation

Explicit uncertainty modeling

A biologically disciplined chill-forcing model with statistical calibration is expected to outperform it both in:

Accuracy

Narrative quality

End of baseline summary.