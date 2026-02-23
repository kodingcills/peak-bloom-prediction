---
ROLE: Quality Assurance & Lead Researcher
INPUTS: `master_training_matrix.csv`
OUTPUTS: LOYO-CV MAE logs, Ablation study results
CONSTRAINTS: No random K-fold CV. Use strictly Leave-One-Year-Out validation.
USED_BY: `test_harness.py`, `model_selection.py`
---
# Experimental Design & Rigorous Validation Protocol

## 1. Time-Aware Validation (The Gold Standard)
**Protocol:** Strictly forbid random K-Fold Cross-Validation.
- **Method:** **Leave-One-Year-Out (LOYO)**. 
- **Execution:** For each city, iterate through all historical years. Train the model on all years *except* year $Y$, then predict year $Y$. 
- **Reasoning:** This simulates the actual competition environment where the future is unknown.

## 2. Model Comparison Structure (The Benchmarks)
**Protocol:** Every version of the model must be compared against three baselines:
1. **The Naive Mean:** The simple average bloom date for that city.
2. **The AI Baseline:** A standard, non-hierarchical XGBoost model.
3. **The Biological Baseline:** A pure GDD-threshold model.
**Metric:** **Mean Absolute Error (MAE)**. Our hybrid stack must outperform all three.

## 3. Parameter Search Discipline
**Protocol:** When optimizing biological constants (T_base, Chill_threshold):
- Use a **Constrained Search Space** based on existing phenology literature (e.g., $T_{base}$ should be between $0^{\circ}C$ and $7^{\circ}C$).
- Prevent "parameter drifting" where the model chooses biologically impossible values to fit noise.

## 4. Ablation Studies
**Protocol:** Test the value-add of each component:
- **Test A:** Bayesian Stacker (Hybrid)
- **Test B:** Biological Features Only
- **Test C:** ML Features Only
- **Goal:** Quantify exactly how many days of MAE the "Bayesian Hierarchical" layer is saving us.

## 5. Overfitting Safeguards (The "Red Team")
- **Regularization:** Apply heavy L2 regularization on the XGBoost residual layer.
- **Prior Tightness:** In the Bayesian layer, use "Informative Priors" for the global mean to prevent the small datasets (NYC/Vancouver) from pulling the model into unrealistic results.