"""Fold-safe ERA5->ASOS bias estimation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP_CANDIDATES = ("timestamp", "time", "valid")


def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    """Resolve the timestamp column name from known candidates.

    Args:
        df: Input DataFrame containing a timestamp-like column.

    Returns:
        Name of the timestamp column.

    Raises:
        AssertionError: If no supported timestamp column is found.
    """

    for col in _TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    raise AssertionError(
        "Unknown timestamp column. Expected one of: "
        f"{', '.join(_TIMESTAMP_CANDIDATES)}"
    )


def prepare_bias_merge(asos_df: pd.DataFrame, era5_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare fold-safe hourly merge for bias estimation.

    Args:
        asos_df: ASOS DataFrame with timestamp and temperature columns.
        era5_df: ERA5 DataFrame with timestamp and temperature columns.

    Returns:
        Merged DataFrame with columns: hour, year, t_asos, t_era5.

    Raises:
        AssertionError: If required columns are missing.
    """

    asos_ts_col = _resolve_timestamp_column(asos_df)
    era5_ts_col = _resolve_timestamp_column(era5_df)

    if "temperature_2m" not in asos_df.columns:
        raise AssertionError("ASOS dataframe missing required column: temperature_2m")
    if "temperature_2m" not in era5_df.columns:
        raise AssertionError("ERA5 dataframe missing required column: temperature_2m")

    asos = asos_df[[asos_ts_col, "temperature_2m"]].copy()
    era5 = era5_df[[era5_ts_col, "temperature_2m"]].copy()

    asos["hour"] = pd.to_datetime(asos[asos_ts_col], utc=True).dt.floor("h")
    era5["hour"] = pd.to_datetime(era5[era5_ts_col], utc=True).dt.floor("h")

    asos = asos.rename(columns={"temperature_2m": "t_asos"})[["hour", "t_asos"]]
    era5 = era5.rename(columns={"temperature_2m": "t_era5"})[["hour", "t_era5"]]

    merged = pd.merge(asos, era5, on="hour", how="inner")
    merged["t_asos"] = pd.to_numeric(merged["t_asos"], errors="coerce")
    merged["t_era5"] = pd.to_numeric(merged["t_era5"], errors="coerce")
    merged = merged.dropna(subset=["t_asos", "t_era5"]).copy()
    merged["year"] = merged["hour"].dt.year.astype(int)

    return merged[["hour", "year", "t_asos", "t_era5"]]


def estimate_bias_from_merged(
    merged_df: pd.DataFrame, exclude_year: int | None = None
) -> dict[str, Any]:
    """Estimate linear bias correction from pre-merged hourly data.

    Args:
        merged_df: Output of prepare_bias_merge.
        exclude_year: Optional held-out year to exclude for fold safety.

    Returns:
        Dict with keys: beta0, beta1, r2, n_obs.
    """

    data = merged_df
    if exclude_year is not None:
        data = data.loc[data["year"] != int(exclude_year)].copy()

    n_obs = int(len(data))
    if n_obs < 3:
        logger.warning(
            "Insufficient overlap for bias estimation (n_obs=%s, exclude_year=%s); "
            "falling back to identity transform.",
            n_obs,
            exclude_year,
        )
        return {"beta0": 0.0, "beta1": 1.0, "r2": 0.0, "n_obs": n_obs}

    x = data["t_era5"].to_numpy(dtype=float)
    y = data["t_asos"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta0 = float(coef[0])
    beta1 = float(coef[1])

    y_hat = X @ coef
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    return {"beta0": beta0, "beta1": beta1, "r2": r2, "n_obs": n_obs}


def estimate_bias(
    asos_path: Path, era5_path: Path, exclude_year: int | None = None
) -> dict[str, Any]:
    """Estimate ERA5->ASOS bias correction from parquet inputs.

    Args:
        asos_path: Path to ASOS parquet.
        era5_path: Path to ERA5 parquet.
        exclude_year: Optional held-out year excluded from fitting.

    Returns:
        Dict with keys: beta0, beta1, r2, n_obs.
    """

    asos_df = pd.read_parquet(asos_path)
    era5_df = pd.read_parquet(era5_path)
    merged = prepare_bias_merge(asos_df, era5_df)
    return estimate_bias_from_merged(merged, exclude_year=exclude_year)
