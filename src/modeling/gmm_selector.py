"""Bimodal selection utilities for Phase 3 inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def fit_bimodal(
    bloom_predictions: np.ndarray,
    site_key: str,
    climatological_mean: float,
    neutral_threshold: float = 0.3,
) -> dict[str, Any]:
    """Fit GMM k=1 and k=2, select via BIC with deterministic behavior.

    HARD CONSTRAINT: k in {1, 2} only.

    Args:
        bloom_predictions: Ensemble predicted bloom DOYs.
        site_key: Site key for diagnostics.
        climatological_mean: Site climatological mean bloom DOY.
        neutral_threshold: Relative BIC improvement threshold to select k=2.

    Returns:
        GMM selection payload and diagnostics.
    """

    preds = np.asarray(bloom_predictions, dtype=float)
    preds = preds[np.isfinite(preds)]
    if preds.size == 0:
        raise AssertionError(f"No valid bloom predictions for site {site_key}")

    if preds.size < 5:
        mean = float(np.mean(preds))
        std = float(np.std(preds, ddof=0))
        return {
            "k": 1,
            "bic_1": float("nan"),
            "bic_2": float("nan"),
            "selected_mean": mean,
            "selected_std": std,
            "all_predictions": preds.tolist(),
            "gmm_details": {
                "mode": "fallback_small_n",
                "n_members": int(preds.size),
            },
        }

    X = preds.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=42, max_iter=200)
    gmm1.fit(X)
    bic_1 = float(gmm1.bic(X))

    gmm2 = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm2.fit(X)
    bic_2 = float(gmm2.bic(X))

    rel_improve = 0.0 if bic_1 == 0 else float((bic_1 - bic_2) / abs(bic_1))
    select_k2 = rel_improve > float(neutral_threshold)

    if not select_k2:
        selected_mean = float(gmm1.means_[0, 0])
        selected_std = float(np.sqrt(gmm1.covariances_.reshape(-1)[0]))
        gmm_details = {
            "mode": "k1",
            "bic_relative_improvement": rel_improve,
            "k1": {
                "mean": selected_mean,
                "std": selected_std,
                "weight": 1.0,
            },
            "k2": {
                "means": gmm2.means_.flatten().astype(float).tolist(),
                "weights": gmm2.weights_.astype(float).tolist(),
                "stds": np.sqrt(gmm2.covariances_.flatten()).astype(float).tolist(),
            },
        }
        return {
            "k": 1,
            "bic_1": bic_1,
            "bic_2": bic_2,
            "selected_mean": selected_mean,
            "selected_std": selected_std,
            "all_predictions": preds.tolist(),
            "gmm_details": gmm_details,
        }

    means = gmm2.means_.flatten().astype(float)
    weights = gmm2.weights_.astype(float)
    variances = gmm2.covariances_.flatten().astype(float)
    stds = np.sqrt(variances)

    if float(np.max(weights)) > 0.70:
        selected_idx = int(np.argmax(weights))
        rule = "dominance"
    else:
        dists = np.abs(means - float(climatological_mean))
        selected_idx = int(np.argmin(dists))
        rule = "climatology"

    if abs(float(means[0]) - float(means[1])) <= 2.0:
        logger.warning(
            "%s: k=2 selected but component means are within 2 DOY (%.3f vs %.3f)",
            site_key,
            float(means[0]),
            float(means[1]),
        )

    selected_mean = float(means[selected_idx])
    selected_std = float(stds[selected_idx])

    gmm_details = {
        "mode": "k2",
        "bic_relative_improvement": rel_improve,
        "selection_rule": rule,
        "selected_component": selected_idx,
        "means": means.tolist(),
        "weights": weights.tolist(),
        "stds": stds.tolist(),
    }

    return {
        "k": 2,
        "bic_1": bic_1,
        "bic_2": bic_2,
        "selected_mean": selected_mean,
        "selected_std": selected_std,
        "all_predictions": preds.tolist(),
        "gmm_details": gmm_details,
    }
