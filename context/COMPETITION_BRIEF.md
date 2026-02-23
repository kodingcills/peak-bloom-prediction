---
ROLE: Compliance Officer & Submission Manager
INPUTS: Official GMU 2026 Competition Rules
OUTPUTS: `cherry-predictions.csv`, formatted Quarto document
CONSTRAINTS: Strictly zero temporal leakage; no manual post-hoc adjustments.
USED_BY: `submission_generator.py`, `GEMINI.md`
---
# GMU International Cherry Blossom Prediction Competition (2026)


This repository is a formal entry into the
George Mason University International Cherry Blossom Prediction Competition (2026).

This document defines the objective, constraints, evaluation structure, and architectural expectations for this project.

All modeling decisions must comply with the requirements described here.

1. Core Objective

Predict the 2026 peak bloom date for five international locations:

Washington, D.C., USA

Kyoto, Japan

Liestal-Weideli, Switzerland

Vancouver, BC, Canada

New York City, NY, USA

Predictions must be reproducible and scientifically justified.

2. Submission Requirements

A complete competition submission must include:

Five peak bloom predictions for 2026

A blinded abstract (≤ 500 words)

A public Git repository containing:

All data used

All code required

A fully reproducible Quarto document

A final prediction file in the required format

Failure to meet reproducibility requirements invalidates the submission.

3. Evaluation Criteria

There are two primary judging categories:

3.1 Most Accurate Prediction

Scoring metric:

Mean Absolute Error (MAE) averaged across the five sites.

For each site:

MAE_site = | predicted_DOY - actual_DOY |

Overall score:

Final_MAE = (MAE_DC + MAE_Kyoto + MAE_Liestal + MAE_Vancouver + MAE_NYC) / 5

Lower MAE is better.

Official bloom dates are determined by:

National Park Service (DC)

Japan Meteorological Agency (Kyoto)

MeteoSwiss (Liestal)

Vancouver Cherry Blossom Festival

Washington Square Park Eco Projects (NYC)

3.2 Best / Most Novel Idea

Judged based on:

Biological plausibility

Statistical rigor

Interpretability

Creativity of modeling approach

Clarity of narrative

Predictions must outperform the AI-generated baseline model at each site to be eligible for this category.

4. Constraints

All modeling must satisfy the following:

No data leakage from future years

No unrealistic cross-validation (time-aware validation required)

All code must execute without manual intervention

All dependencies must be documented

All feature engineering must be transparent

Predictions must be reproducible from repository alone

5. Target Variable Definition

Peak bloom is defined differently by site:

Location	Bloom Definition
Kyoto	80% bloom
Liestal	25% bloom
Washington DC	70% bloom
Vancouver	70% bloom
NYC	70% bloom

Models must account for differences in bloom threshold definitions where relevant.

6. Data Usage Policy

Allowed:

Publicly available climate data

Phenology datasets

Meteorological APIs

Historical bloom records

Not allowed:

Private datasets

Manual adjustment of predictions post hoc

Use of future climate data beyond submission date

7. Reproducibility Standard

The repository must:

Render a Quarto document end-to-end

Reproduce feature engineering steps

Reproduce model training

Reproduce final predictions

Generate the submission CSV

The repository should function as a standalone forecasting pipeline.

8. Baseline Benchmark

The competition provides an AI-generated XGBoost baseline model.

This repository’s objective is to:

Outperform the baseline in MAE

Improve biological realism

Improve time-aware forecasting robustness

Improve interpretability

The baseline summary is documented in:
context/BASELINE_MODEL_SUMMARY.md

9. Modeling Philosophy

This repository prioritizes:

✔ Biological realism
✔ Time-aware validation
✔ Species-aware modeling
✔ Forecasting discipline
✔ Mechanistic interpretability

Over:

✘ Purely black-box boosting
✘ Random cross-validation
✘ Arbitrary thresholds without justification
✘ Post-hoc tuning to minimize error artificially

10. Forecasting Discipline

2026 predictions must:

Use only climate data available up to submission

Not assume knowledge of future spring temperatures

Maintain realistic forecasting assumptions

Forecasts must reflect real-world uncertainty.

11. Output Format

The final submission file:

cherry-predictions.csv

Required structure:

location, year, predicted_date
washingtondc, 2026, YYYY-MM-DD
kyoto, 2026, YYYY-MM-DD
liestal, 2026, YYYY-MM-DD
vancouver, 2026, YYYY-MM-DD
newyorkcity, 2026, YYYY-MM-DD

Dates must correspond to valid calendar dates.

12. Architectural Intent

The repository should reflect layered structure:

Strategic context (this folder)

Data ingestion

Feature engineering

Model estimation

Time-aware validation

Forecasting pipeline

Submission generation

Each layer must be clearly separated.

13. Strategic Goal

The objective is not merely to predict bloom.

The objective is to:

Demonstrate mechanistic understanding of phenology

Construct a statistically disciplined forecasting system

Beat a strong AI baseline model

Produce a reproducible scientific entry

14. Codex Instructions

When modifying this repository:

Respect competition constraints

Do not introduce leakage

Explain modeling assumptions

Justify thresholds biologically

Maintain Quarto reproducibility

Preserve clean architecture

All changes must align with the competition objective defined here.

End of competition brief.