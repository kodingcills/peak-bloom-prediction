"""Validation gates for Phenology Engine v1.7."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from config.settings import (
    COMPETITION_YEAR,
    GOLD_DIR,
    INFERENCE_CUTOFF_UTC,
    PROCESSED_DIR,
    RAW_GMU_DIR,
    SEAS5_FALLBACK_MODE,
    SILVER_WEATHER_DIR,
    SITES,
)
from src.processing.features import compute_chill_portions, compute_gdh
from src.processing.labels import load_competition_labels

logger = logging.getLogger(__name__)


def assert_inference_cutoff_utc(weather_dir: Path | None = None) -> None:
    """No weather timestamps after 2026-02-28 23:59:59 UTC in any silver file."""

    root = weather_dir or SILVER_WEATHER_DIR
    parquet_files = list(root.rglob("*.parquet"))
    if not parquet_files:
        raise AssertionError("No silver weather parquet files found")

    for path in parquet_files:
        df = pd.read_parquet(path, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], utc=True)
        if ts.max() > INFERENCE_CUTOFF_UTC:
            raise AssertionError(
                f"Found timestamp after cutoff in {path}: {ts.max()} > {INFERENCE_CUTOFF_UTC}"
            )


def assert_historical_window_end(features_df: pd.DataFrame) -> None:
    """All feature years use data only up to DOY 59."""

    weather_root = SILVER_WEATHER_DIR
    for _, row in features_df.iterrows():
        year = int(row["year"])
        if year >= COMPETITION_YEAR:
            continue
        site_key = row["site_key"]
        weather_path = weather_root / site_key / f"{site_key}_consolidated.parquet"
        hourly = pd.read_parquet(weather_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        gdh = compute_gdh(hourly, year)
        cp = compute_chill_portions(hourly, year)
        if abs(gdh - float(row["gdh"])) > 1e-6 or abs(cp - float(row["cp"])) > 1e-6:
            raise AssertionError(
                "Feature mismatch suggests data beyond DOY 59 was used "
                f"for {site_key} {year}"
            )


def assert_labels_complete(labels_df: pd.DataFrame | None = None) -> None:
    """All 5 competition sites present. Max year >= 2024 for each."""

    labels = labels_df if labels_df is not None else load_competition_labels(RAW_GMU_DIR)
    sites = set(labels["site_key"].unique())
    if sites != set(SITES.keys()):
        raise AssertionError(f"Missing sites in labels: {set(SITES.keys()) - sites}")
    for site_key in SITES.keys():
        max_year = labels.loc[labels["site_key"] == site_key, "year"].max()
        if max_year < 2024:
            raise AssertionError(f"Labels for {site_key} do not reach 2024")


def assert_seas5_members(nc_path: Path | None = None) -> None:
    """SEAS5 NetCDF contains exactly 50 unique ensemble members."""

    flag_path = PROCESSED_DIR / "SEAS5_FETCH_FAILED"
    path = nc_path or (PROCESSED_DIR / "seas5_2026.nc")
    if path.exists():
        dataset = xr.open_dataset(path)
        for dim in ("number", "member", "ensemble"):
            if dim in dataset.dims:
                if int(dataset.dims[dim]) != 50:
                    raise AssertionError(f"SEAS5 members expected 50, got {dataset.dims[dim]}")
                return
        raise AssertionError("TODO: AUDIT — Unable to find SEAS5 ensemble dimension")
    if flag_path.exists():
        logger.warning("Skipping SEAS5 member check due to SEAS5_FETCH_FAILED flag")
        return
    raise AssertionError("SEAS5 NetCDF missing and fallback flag not set")


def assert_silver_utc(weather_dir: Path | None = None) -> None:
    """All parquet files in silver layer have UTC-aware timestamps."""

    root = weather_dir or SILVER_WEATHER_DIR
    parquet_files = list(root.rglob("*.parquet"))
    if not parquet_files:
        raise AssertionError("No silver weather parquet files found")
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            raise AssertionError(f"Timestamp missing UTC tzinfo in {path}")


def assert_gold_schema(features_df: pd.DataFrame | None = None) -> None:
    """Gold features have expected schema."""

    if features_df is None:
        path = GOLD_DIR / "features.parquet"
        if not path.exists():
            raise AssertionError("Gold features parquet missing")
        features_df = pd.read_parquet(path)
    required = {"site_key", "year", "gdh", "cp", "bloom_doy"}
    if set(features_df.columns) != required:
        raise AssertionError(f"Gold features schema mismatch: {features_df.columns}")
    missing_2026 = features_df.loc[features_df["year"] == COMPETITION_YEAR, "bloom_doy"]
    if missing_2026.notna().any():
        raise AssertionError("2026 rows must have null bloom_doy")


def assert_requirements_present(requirements_path: Path | None = None) -> None:
    """requirements.txt exists and contains baseline dependencies."""

    baseline = [
        "pandas>=2.0",
        "numpy>=1.24",
        "pyarrow>=12.0",
        "requests>=2.28",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "xarray>=2023.1",
        "netcdf4>=1.6",
        "cdsapi>=0.6",
        "jupyter>=1.0",
    ]
    path = requirements_path or Path("requirements.txt")
    if not path.exists():
        raise AssertionError("requirements.txt is missing")
    contents = path.read_text().splitlines()
    missing = [dep for dep in baseline if dep not in contents]
    if missing:
        raise AssertionError(f"requirements.txt missing deps: {missing}")
