"""Empirical Bayes shrinkage utilities for cold-start sites."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sample_variance(values: np.ndarray) -> float:
    """Return sample variance (ddof=1) with safe fallback."""

    if values.size < 2:
        return float("nan")
    return float(np.var(values, ddof=1))


def compute_shrinkage_weights(
    cv_results_df: pd.DataFrame,
    training_df: pd.DataFrame,
    epsilon: float = 1.0,
) -> dict[str, Any]:
    """Compute empirical Bayes shrinkage weights per site.

    Args:
        cv_results_df: CV result rows containing residuals and site_key.
        training_df: Gold training frame used to derive N_s counts.
        epsilon: Stability floor added to site residual variance.

    Returns:
        Mapping with global variance, epsilon, and per-site weight diagnostics.
    """

    required_cv = {"site_key", "residual"}
    missing_cv = required_cv - set(cv_results_df.columns)
    if missing_cv:
        raise AssertionError(f"cv_results_df missing required columns: {sorted(missing_cv)}")

    if "site_key" not in training_df.columns:
        raise AssertionError("training_df missing required column: site_key")

    residual_all = pd.to_numeric(cv_results_df["residual"], errors="coerce").dropna().to_numpy()
    sigma2_global = _sample_variance(residual_all)
    if not np.isfinite(sigma2_global):
        sigma2_global = 1.0

    site_counts = training_df.groupby("site_key").size().to_dict()
    site_payload: dict[str, dict[str, float | int]] = {}

    for site_key in sorted(site_counts.keys()):
        site_resid = pd.to_numeric(
            cv_results_df.loc[cv_results_df["site_key"] == site_key, "residual"],
            errors="coerce",
        ).dropna().to_numpy()

        sigma2_s = _sample_variance(site_resid)
        if not np.isfinite(sigma2_s):
            sigma2_s = sigma2_global

        n_s = int(site_counts.get(site_key, 0))
        if n_s <= 0:
            raise AssertionError(f"Invalid training count for site {site_key}: {n_s}")

        denom = sigma2_global + ((sigma2_s + float(epsilon)) / n_s)
        if denom <= 0:
            raise AssertionError(
                f"Non-positive shrinkage denominator for site {site_key}: {denom}"
            )
        w = float(sigma2_global / denom)

        if sigma2_s < 0.1:
            logger.warning(
                "Low sigma2_s for %s (%.6f). epsilon=%.3f prevents overconfident weights.",
                site_key,
                sigma2_s,
                epsilon,
            )

        site_payload[site_key] = {
            "w": w,
            "sigma2_s": float(sigma2_s),
            "sigma2_global": float(sigma2_global),
            "n_s": n_s,
            "epsilon": float(epsilon),
        }

    return {
        "epsilon": float(epsilon),
        "sigma2_global": float(sigma2_global),
        "sites": site_payload,
    }


def apply_shrinkage(model_pred: float, global_mean: float, w: float) -> float:
    """Apply precision-based shrinkage toward global mean.

    Args:
        model_pred: Site-level model prediction.
        global_mean: Global training mean bloom DOY.
        w: Site-specific empirical Bayes precision weight.

    Returns:
        Shrunk prediction.
    """

    return float((w * model_pred) + ((1.0 - w) * global_mean))
