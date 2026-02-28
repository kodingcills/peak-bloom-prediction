"""Feature computation for GDH and Chill Portions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import (
    CHILL_START_DOY,
    COMPETITION_YEAR,
    CP_MAX_C,
    CP_MIN_C,
    CP_OPT_C,
    GDH_BASE_C,
    GOLD_DIR,
    INFERENCE_CUTOFF_DOY,
    SILVER_WEATHER_DIR,
    SITES,
)

logger = logging.getLogger(__name__)


def _ensure_utc(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        raise AssertionError("Timestamps must be UTC-aware")
    return ts


def compute_gdh(hourly_df: pd.DataFrame, year: int, cutoff_doy: int = INFERENCE_CUTOFF_DOY) -> float:
    """Compute Growing Degree Hours for a single site-year."""

    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
    df = hourly_df.loc[
        (hourly_df["timestamp"] >= start) & (hourly_df["timestamp"] <= end)
    ].copy()
    _ensure_utc(df["timestamp"])
    df["doy"] = df["timestamp"].dt.dayofyear
    df = df.loc[df["doy"] <= cutoff_doy]
    missing_hours = (cutoff_doy * 24) - len(df)
    if missing_hours > 48:
        logger.warning("GDH missing %s hours for year %s", missing_hours, year)
    gdh = (df["temperature_2m"] - GDH_BASE_C).clip(lower=0).sum()
    return float(gdh)


def compute_chill_portions(
    hourly_df: pd.DataFrame, year: int, cutoff_doy: int = INFERENCE_CUTOFF_DOY
) -> float:
    """Compute Chill Portions for a single site-year."""

    start = datetime(year - 1, 10, 1, tzinfo=timezone.utc)
    end = datetime(year, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
    df = hourly_df.loc[
        (hourly_df["timestamp"] >= start) & (hourly_df["timestamp"] <= end)
    ].copy()
    _ensure_utc(df["timestamp"])
    df["doy"] = df["timestamp"].dt.dayofyear
    mask_year = (df["timestamp"].dt.year == year) & (df["doy"] > cutoff_doy)
    df = df.loc[~mask_year]

    within = (df["temperature_2m"] >= CP_MIN_C) & (df["temperature_2m"] <= CP_MAX_C)
    scaled = 1 - ((df["temperature_2m"] - CP_OPT_C) / (CP_MAX_C - CP_OPT_C)) ** 2
    df["cp_hour"] = scaled.where(within, 0.0).clip(lower=0.0)
    return float(df["cp_hour"].sum())


def build_gold_features(
    silver_weather_dir: Path | None,
    labels_df: pd.DataFrame,
    output_path: Path | None,
) -> pd.DataFrame:
    """Build gold feature matrix for all site-years."""

    weather_root = silver_weather_dir or SILVER_WEATHER_DIR
    output_path = output_path or (GOLD_DIR / "features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for (site_key, year), _ in labels_df.groupby(["site_key", "year"]):
        weather_path = weather_root / site_key / f"{site_key}_consolidated.parquet"
        if not weather_path.exists():
            raise FileNotFoundError(f"Missing weather data for {site_key}")
        hourly = pd.read_parquet(weather_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        gdh = compute_gdh(hourly, int(year))
        cp = compute_chill_portions(hourly, int(year))
        bloom_doy = float(
            labels_df.loc[
                (labels_df["site_key"] == site_key) & (labels_df["year"] == year),
                "bloom_doy",
            ].iloc[0]
        )
        records.append(
            {
                "site_key": site_key,
                "year": int(year),
                "gdh": gdh,
                "cp": cp,
                "bloom_doy": bloom_doy,
            }
        )

    for site_key in SITES.keys():
        weather_path = weather_root / site_key / f"{site_key}_consolidated.parquet"
        hourly = pd.read_parquet(weather_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        gdh = compute_gdh(hourly, COMPETITION_YEAR)
        cp = compute_chill_portions(hourly, COMPETITION_YEAR)
        records.append(
            {
                "site_key": site_key,
                "year": COMPETITION_YEAR,
                "gdh": gdh,
                "cp": cp,
                "bloom_doy": None,
            }
        )

    features = pd.DataFrame.from_records(records)
    features = features.sort_values(["site_key", "year"])
    features.to_parquet(output_path, index=False)
    return features
