---
ROLE: Plant Physiologist & Phenology Expert
INPUTS: Botanical research, GHCN-D Daily climate data
OUTPUTS: Kinetic accumulation formulas (Chill/Forcing), threshold ranges
CONSTRAINTS: Must respect Sequential Dormancy (Chill before Heat); accumulation starts Sept 1st.
USED_BY: `feature_engineer.py`, `MODELING_ASSUMPTIONS.md`
---
# Biological Foundations of Cherry Blossom Phenology

This document outlines the biological principles governing cherry blossom peak bloom timing.
All modeling decisions in this repository should remain consistent with these mechanistic constraints.

Cherry blossom timing is not arbitrary. It is governed by well-studied physiological processes that regulate dormancy, cold exposure, and heat accumulation.

Understanding these processes is essential for constructing biologically valid predictive models.

1. Overview of Dormancy and Bloom Development

Temperate deciduous trees, including cherry species, undergo an annual dormancy cycle consisting of:

Endodormancy (true dormancy)

Ecodormancy (quiescence)

Budburst and flowering

Peak bloom occurs only after the tree progresses through these stages.

1.1 Endodormancy (Chilling Phase)

During autumn, shortening daylight and cooling temperatures trigger dormancy.

In this phase:

Bud growth is inhibited internally.

The tree requires exposure to sufficient cold temperatures to break dormancy.

This is known as the chilling requirement.

Without adequate chill accumulation:

Budburst may be delayed.

Flowering may become irregular or incomplete.

1.2 Ecodormancy (Forcing Phase)

Once the chilling requirement is satisfied:

Buds enter ecodormancy.

Growth resumes when sufficient warmth accumulates.

Heat accumulation drives bud development toward flowering.

This stage is governed by growing degree days (GDD).

2. Core Phenological Mechanisms

Peak bloom depends on two primary accumulations:

Chill accumulation (winter cold exposure)

Heat accumulation (spring warmth / forcing)

These processes interact nonlinearly.

Bloom does NOT occur based on heat alone.
Heat becomes effective only after chill requirements are met.

3. Chill Accumulation Models

Chilling models quantify exposure to cold temperatures.

3.1 Simple Chill Days Model

A basic formulation:

ChillUnits = sum( 1 if T_avg < T_chill_threshold else 0 )

Common threshold:

7°C

Limitations:

Treats all cold days equally

Does not weight temperature effectiveness

3.2 Weighted Chill Models (Advanced)

More biologically realistic models include:

Utah Model

Assigns positive or negative chill effectiveness based on temperature range.

Example:

2.5–9°C → high chill effectiveness

15°C → may negate chill accumulation

Dynamic Model

Models chill accumulation via biochemical processes.

More robust under climate change conditions.

4. Heat Accumulation (Growing Degree Days)

After chill requirement is satisfied:

Heat accumulation drives bud development.

Standard formulation:

GDD = max(T_avg - T_base, 0)

Where:

T_base typically 0–5°C depending on species

Units accumulated daily

Bloom occurs when cumulative heat reaches a species-specific threshold.

5. Chill–Heat Interaction

The relationship between chill and heat is NOT independent.

Important properties:

Insufficient chill → higher heat requirement

High chill → lower heat requirement

Warm winters can delay bloom despite warm springs

This interaction is often modeled as:

Bloom when:
  Chill ≥ ChillThreshold
  AND
  Heat ≥ f(Chill)

Where f(Chill) may decrease with increasing chill.

This nonlinear dependency is critical for accurate forecasting.

6. Species-Specific Differences

The competition includes multiple species:

Location	Species
Kyoto	Prunus jamasakura
Liestal	Prunus avium
Washington DC	Prunus × yedoensis ‘Somei-yoshino’
Vancouver	Prunus × yedoensis ‘Akebono’
NYC	Prunus × yedoensis

Different species have:

Different chill requirements

Different base temperatures

Different heat thresholds

Different sensitivity to anomalous winter warming

Modeling should account for species-level heterogeneity.

7. Latitude and Climate Regime Effects

Locations vary in:

Latitude

Altitude

Maritime vs continental climate

Interannual variability

Implications:

Chill accumulation windows differ

Spring warming rate differs

Temperature variance differs

Models should consider:

Site-specific calibration

Climate anomaly features

Potential random effects

8. Climate Change Considerations

Long-term warming affects phenology in complex ways:

Earlier spring warming trends

Reduced winter chill in some regions

Increased interannual variability

Earlier bloom over decades

Models should:

Avoid overfitting to historical trend

Consider detrending or anomaly-based features

Validate using time-aware cross-validation

9. Mechanistic Modeling Framework

A biologically grounded bloom model follows this structure:

Step 1: Accumulate Chill
ChillAccumulation(year, location)

Until chill requirement satisfied.

Step 2: Accumulate Heat

After chill threshold reached:

HeatAccumulation(year, location)

Until bloom threshold reached.

Step 3: Predict Bloom DOY

Bloom_DOY = first day where:

Chill ≥ ChillRequirement
AND
Heat ≥ HeatRequirement

Parameters may be estimated statistically.

10. Modeling Implications for This Repository

All predictive models should:

✔ Reflect chill-first, heat-second structure
✔ Avoid using spring heat before chill completion
✔ Consider species-specific thresholds
✔ Avoid unrealistic independence assumptions
✔ Use time-aware validation

11. Forecasting Constraints

Because 2026 prediction occurs before bloom:

Models must rely only on:

Observed winter chill up to submission

Spring warming trajectory up to prediction date

Historical relationships

No future temperature data beyond available observation should be used.

12. Key Modeling Philosophy

The goal is not to build the most complex model.

The goal is to:

Capture biological causality

Maintain interpretability

Preserve forecasting realism

Outperform purely data-driven boosting models

Mechanistic plausibility increases both:

Predictive robustness

Narrative strength

13. Strategic Advantage Over Baseline

The AI baseline:

Uses fixed thresholds

Treats chill and heat independently

Does not calibrate parameters

Uses non-time-aware CV

A calibrated chill–forcing model with parameter estimation and time-aware validation is expected to:

Improve generalization

Improve narrative clarity

Improve robustness under anomalous winters

14. Codex Instructions

When implementing modeling logic:

Respect dormancy sequence

Avoid using heat accumulation before chill threshold

Document chosen thresholds

Justify parameter values biologically

Avoid arbitrary constants unless calibrated

All modeling changes should align with biological constraints defined here.

End of biological background.