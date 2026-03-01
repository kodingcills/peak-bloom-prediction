"""Analog shrinkage utilities for Phase 3.5 experimental mode."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar value to [lo, hi].

    Args:
        x: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """

    return max(lo, min(hi, x))


def compute_delta_from_climatology(
    mean_bloom_doy: dict[str, float],
    site_key: str,
    analog_site: str,
    lo: float = -7.0,
    hi: float = 7.0,
) -> float:
    """Compute bounded site-vs-analog climatology delta in days.

    Args:
        mean_bloom_doy: Site climatology means.
        site_key: Sparse site key.
        analog_site: Analog site key.
        lo: Minimum allowed delta.
        hi: Maximum allowed delta.

    Returns:
        Clamped delta in days.
    """

    if site_key not in mean_bloom_doy:
        raise AssertionError(f"Missing climatology mean for site {site_key}")
    if analog_site not in mean_bloom_doy:
        raise AssertionError(f"Missing climatology mean for analog site {analog_site}")

    raw_delta = float(mean_bloom_doy[site_key]) - float(mean_bloom_doy[analog_site])
    return float(clamp(raw_delta, lo=lo, hi=hi))


def compute_prior_mean(
    *,
    site_key: str,
    shrinkage_mode: str,
    global_mean: float,
    mean_bloom_doy: dict[str, float],
    mu_selected_by_site: dict[str, float],
) -> dict[str, Any]:
    """Resolve shrinkage prior mean payload for a site.

    Args:
        site_key: Site being shrunk.
        shrinkage_mode: Shrinkage mode (`global` or `analog`).
        global_mean: Global mean prior center.
        mean_bloom_doy: Site climatology means.
        mu_selected_by_site: GMM-selected means for all sites.

    Returns:
        Prior payload containing prior mean and provenance metadata.
    """

    if shrinkage_mode not in {"global", "analog"}:
        raise AssertionError(f"Unknown shrinkage_mode: {shrinkage_mode}")

    if shrinkage_mode == "global":
        return {
            "prior_mean": float(global_mean),
            "prior_source": "global_mean",
            "analog_site": None,
            "delta_days": None,
        }

    analog_map = {
        "nyc": "washingtondc",
        "vancouver": "kyoto",
    }
    analog_site = analog_map.get(site_key)
    if analog_site is None:
        return {
            "prior_mean": float(global_mean),
            "prior_source": "global_mean",
            "analog_site": None,
            "delta_days": None,
        }

    if analog_site not in mu_selected_by_site:
        raise AssertionError(
            f"Analog prior unavailable: site={site_key}, analog_site={analog_site}, "
            "missing selected mean for analog site."
        )

    delta = compute_delta_from_climatology(
        mean_bloom_doy=mean_bloom_doy,
        site_key=site_key,
        analog_site=analog_site,
        lo=-7.0,
        hi=7.0,
    )
    prior_mean = float(mu_selected_by_site[analog_site]) + float(delta)
    logger.info(
        "Analog prior for %s from %s: mu_analog=%.4f delta=%.4f prior=%.4f",
        site_key,
        analog_site,
        float(mu_selected_by_site[analog_site]),
        delta,
        prior_mean,
    )
    return {
        "prior_mean": prior_mean,
        "prior_source": "analog",
        "analog_site": analog_site,
        "delta_days": float(delta),
    }
