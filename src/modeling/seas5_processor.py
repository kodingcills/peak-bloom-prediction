"""SEAS5/fallback ensemble processing for Phase 3 inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import COMPETITION_YEAR
from src.processing.warming_velocity import compute_warming_velocity

logger = logging.getLogger(__name__)


def _daily_means_by_year(weather_df: pd.DataFrame) -> dict[int, pd.Series]:
    """Build daily mean temperature series keyed by year.

    Args:
        weather_df: Hourly weather frame with `timestamp` and `temperature_2m`.

    Returns:
        Mapping year -> Series indexed by DOY with daily mean temperatures.
    """

    df = weather_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = df["timestamp"].dt.year.astype(int)
    df["doy"] = df["timestamp"].dt.dayofyear.astype(int)
    grouped = df.groupby(["year", "doy"], sort=True)["temperature_2m"].mean()

    out: dict[int, pd.Series] = {}
    for year in sorted(grouped.index.get_level_values(0).unique().tolist()):
        s = grouped.loc[year]
        series = pd.Series(s.values, index=s.index.astype(int)).sort_index()
        out[int(year)] = series
    return out


def build_fallback_ensemble(
    weather_cache: dict[str, pd.DataFrame],
    gold_2026: pd.DataFrame,
    model_coefficients: dict,
    bias_coefficients: dict,
    mean_bloom_doy: dict,
    sites_config: dict,
    fallback_years: int = 30,
) -> dict[str, dict[str, Any]]:
    """Build pseudo-ensemble using historical spring temperature scenarios.

    For each site, uses the last `fallback_years` years of post-DOY-59 temperatures
    as ensemble members. Each member produces one predicted bloom DOY.

    Args:
        weather_cache: Pre-loaded ERA5 DataFrames keyed by site_key.
        gold_2026: Gold features rows where year == 2026.
        model_coefficients: Loaded model coefficients artifact.
        bias_coefficients: Loaded bias coefficients artifact.
        mean_bloom_doy: Per-site mean bloom DOY values.
        sites_config: Site configuration dict from config.settings.
        fallback_years: Number of historical years to use as scenarios.

    Returns:
        Per-site fallback ensemble diagnostics and predictions.
    """

    site_order = sorted(sites_config.keys())
    beta_gdh = float(model_coefficients["beta_gdh"])
    beta_cp = float(model_coefficients["beta_cp"])
    beta_wv = float(model_coefficients["beta_wv"])
    site_intercepts = model_coefficients["site_intercepts"]

    output: dict[str, dict[str, Any]] = {}

    for site_key in site_order:
        if site_key not in weather_cache:
            raise FileNotFoundError(f"Missing weather cache for site: {site_key}")

        site_row = gold_2026.loc[gold_2026["site_key"] == site_key]
        if site_row.empty:
            raise AssertionError(f"Missing 2026 gold features for site: {site_key}")
        row_2026 = site_row.iloc[0]
        gdh_2026 = float(row_2026["gdh"])
        cp_2026 = float(row_2026["cp"])

        weather_df = weather_cache[site_key]
        daily_by_year = _daily_means_by_year(weather_df)

        all_years = sorted(daily_by_year.keys())
        historical_years = [y for y in all_years if y < COMPETITION_YEAR]
        scenario_years = historical_years[-fallback_years:]

        if COMPETITION_YEAR not in daily_by_year:
            raise AssertionError(f"Missing {COMPETITION_YEAR} daily means for site {site_key}")

        daily_2026 = daily_by_year[COMPETITION_YEAR]
        base_2026 = daily_2026.loc[(daily_2026.index >= 1) & (daily_2026.index <= 59)]

        alpha_s = float(site_intercepts[site_key])
        center_doy = int(round(float(mean_bloom_doy[site_key])))
        bc = bias_coefficients.get(site_key, {"beta0": 0.0, "beta1": 1.0})
        beta0 = float(bc.get("beta0", 0.0))
        beta1 = float(bc.get("beta1", 1.0))

        preds: list[float] = []
        wv_vals: list[float] = []
        valid_years: list[int] = []
        skipped: list[dict[str, Any]] = []

        for year in scenario_years:
            scenario = daily_by_year.get(int(year))
            if scenario is None or scenario.empty:
                skipped.append({"scenario_year": int(year), "reason": "missing_scenario_year"})
                continue
            scenario_slice = scenario.loc[(scenario.index >= 60) & (scenario.index <= 180)]
            if scenario_slice.empty:
                skipped.append({"scenario_year": int(year), "reason": "missing_post59"})
                continue

            combined = pd.concat([base_2026, scenario_slice]).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            combined_year_map = {COMPETITION_YEAR: combined}

            wv_result = compute_warming_velocity(
                daily_means_by_year=combined_year_map,
                year=COMPETITION_YEAR,
                window_center_doy=center_doy,
                beta0=beta0,
                beta1=beta1,
                window_days=14,
            )
            wv = float(wv_result["wv"])
            if not np.isfinite(wv):
                skipped.append(
                    {
                        "scenario_year": int(year),
                        "reason": "wv_nan",
                        "n_days": int(wv_result.get("n_days", 0)),
                    }
                )
                continue

            pred = alpha_s + (beta_gdh * gdh_2026) + (beta_cp * cp_2026) + (beta_wv * wv)
            preds.append(float(pred))
            wv_vals.append(float(wv))
            valid_years.append(int(year))

        output[site_key] = {
            "predictions": np.array(preds, dtype=float),
            "wv_values": np.array(wv_vals, dtype=float),
            "scenario_years": valid_years,
            "skipped_members": skipped,
            "n_valid": int(len(preds)),
            "gdh_2026": gdh_2026,
            "cp_2026": cp_2026,
        }

    return output


def process_seas5_ensemble(nc_path: Path, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """Process real SEAS5 ensemble data.

    Raises:
        NotImplementedError: Real SEAS5 path is not implemented in this fallback run.
    """

    raise NotImplementedError("SEAS5 real ensemble not available. Use fallback.")
