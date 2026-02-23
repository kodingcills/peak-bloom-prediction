---
ROLE: Senior Statistician
INPUTS: Model predictions, Historical bloom DOY
OUTPUTS: Site-level MAE, Global Mean MAE
CONSTRAINTS: Optimize strictly for Mean Absolute Error (MAE), not R-squared or RMSE.
USED_BY: `EXPERIMENTAL_DESIGN.md`, `test_harness.py`
---
# Evaluation Framework for 2026 Cherry Blossom Prediction


This document defines the official scoring metric, internal validation strategy, and performance objectives for this repository.

All modeling decisions must optimize for this evaluation structure.

1. Official Competition Metric

The primary scoring metric for the competition is:

Mean Absolute Error (MAE)

For each site:

MAE_site = | predicted_DOY - actual_DOY |

Where:

predicted_DOY = model forecasted bloom day-of-year

actual_DOY = officially declared bloom day-of-year

Final Competition Score

The final score is the average MAE across the five sites:

Final_MAE =
  ( MAE_DC
  + MAE_Kyoto
  + MAE_Liestal
  + MAE_Vancouver
  + MAE_NYC ) / 5

Lower score = better performance.

2. Key Implications of MAE

MAE has specific properties:

✔ Penalizes errors linearly
✔ Does not disproportionately penalize outliers (unlike RMSE)
✔ Directly interpretable in days

Because MAE is linear:

Reducing large errors is valuable

Consistency across sites matters

One catastrophic miss can dominate score

3. Site-Level Equality

Each site contributes equally to final score.

This has strategic implications:

Large historical datasets (Kyoto) cannot dominate optimization

Small datasets (Vancouver, NYC) matter equally

Model must generalize well across climates

Do not overweight sites by sample size during optimization.

4. Internal Validation Strategy

Random cross-validation is NOT acceptable.

Cherry blossom prediction is a time-series forecasting problem.

Internal validation must simulate forward forecasting.

Acceptable methods:

4.1 Leave-One-Year-Out Cross-Validation (LOYO-CV)

For each year Y:

Train on all years < Y

Predict bloom for year Y

Compute absolute error

Aggregate errors across years.

This most closely mimics real forecasting.

4.2 Rolling Origin Evaluation

Example:

Train: 1921–1980 → Predict 1981
Train: 1921–1981 → Predict 1982
...

This preserves temporal structure.

4.3 Block Cross-Validation

Time blocks must not mix future and past data.

Random fold assignment is prohibited.

5. What NOT to Optimize

Do NOT optimize:

✘ RMSE as primary metric
✘ R²
✘ Training error
✘ In-sample fit
✘ Random CV score

These are secondary diagnostics only.

The objective is out-of-sample MAE under time-aware validation.

6. Baseline Performance Benchmark

The AI-generated XGBoost model achieved:

Approximate RMSE ≈ 6.7 days

However:

RMSE ≠ MAE

CV was not time-aware

May underestimate true forecasting error

Internal benchmark target:

Target_MAE ≤ 6 days
Stretch_Target ≤ 5 days

These are heuristic goals based on historical competition performance.

7. Forecasting Error Interpretation

Error magnitudes:

MAE	Interpretation
≤ 3 days	Exceptional
4–5 days	Competitive
6–7 days	Baseline-level
> 8 days	Likely uncompetitive

Small improvements (1–2 days) can determine winners.

8. Multi-Site Stability Requirement

The objective is not to minimize average MAE at the expense of one site.

Avoid:

Extremely low error at Kyoto

Catastrophic error at Vancouver

Balanced performance across all five sites is essential.

9. Secondary Evaluation Considerations

While MAE determines accuracy ranking, judges also consider:

Biological plausibility

Uncertainty modeling

Interpretability

Robustness under climate anomalies

Prediction intervals are not scored directly under MAE but strengthen narrative category.

10. Uncertainty Consideration (Optional but Recommended)

Although MAE uses point forecasts:

Well-calibrated uncertainty estimates improve narrative strength.

Recommended:

Residual-based prediction intervals

Bootstrapped intervals

Bayesian credible intervals (if mechanistic model used)

11. Optimization Discipline

All model selection must be based on:

Time-aware cross-validated MAE

NOT training error.

Hyperparameter tuning must:

Use rolling or LOYO CV

Prevent temporal leakage

Avoid using 2026 data

12. Codex Instructions

When proposing modeling changes:

Evaluate using LOYO-CV MAE

Compare against current best MAE

Report average and site-level MAE

Flag any site with large error

Do not accept model improvements based solely on in-sample performance.

13. Strategic Reminder

This competition is won by:

Forecast realism

Consistent cross-site accuracy

Mechanistic grounding

Strict validation discipline

Overfitting one climate regime will likely lose.

End of evaluation metric specification.