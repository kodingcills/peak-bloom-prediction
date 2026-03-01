"""Feature computation for GDH, Chill Portions, and enriched winter aggregates."""

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


def compute_winter_aggregates(
    daily_df: pd.DataFrame,
    year: int,
    cutoff_doy: int = INFERENCE_CUTOFF_DOY,
) -> dict[str, float]:
    """Compute winter aggregate features from daily ERA5-Land values.

    Args:
        daily_df: Daily weather DataFrame with UTC `time` column.
        year: Target bloom year.
        cutoff_doy: Inference truncation DOY (kept for contract clarity).

    Returns:
        Dictionary of aggregate winter features.
    """

    del cutoff_doy  # Window boundaries are date-based and fixed to Feb 28.
    start_date = pd.Timestamp(f"{year - 1}-10-01", tz="UTC")
    end_date = pd.Timestamp(f"{year}-02-28", tz="UTC")
    window_df = daily_df.loc[(daily_df["time"] >= start_date) & (daily_df["time"] <= end_date)]
    if window_df.empty:
        raise AssertionError(f"No daily winter rows available for year {year}")

    return {
        "precip_sum": float(pd.to_numeric(window_df["precip"], errors="coerce").sum()),
        "rain_sum": float(pd.to_numeric(window_df["rain"], errors="coerce").sum()),
        "snow_sum": float(pd.to_numeric(window_df["snow"], errors="coerce").sum()),
        "tmax_mean": float(pd.to_numeric(window_df["tmax"], errors="coerce").mean()),
        "tmin_mean": float(pd.to_numeric(window_df["tmin"], errors="coerce").mean()),
        "tmean_mean": float(pd.to_numeric(window_df["tmean"], errors="coerce").mean()),
        "daylight_mean": float(pd.to_numeric(window_df["daylight"], errors="coerce").mean()),
    }


def build_gold_features(
    silver_weather_dir: Path | None,
    labels_df: pd.DataFrame,
    output_path: Path | None,
) -> pd.DataFrame:
    """Build gold feature matrix for all site-years."""

    weather_root = silver_weather_dir or SILVER_WEATHER_DIR
    output_path = output_path or (GOLD_DIR / "features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df = labels_df.sort_values(by=["site_key", "year"]).copy()
    labels_df["bloom_doy_lag1"] = labels_df.groupby("site_key")["bloom_doy"].shift(1)

    records: list[dict] = []
    total_pairs = len(labels_df.groupby(["site_key", "year"]))
    skipped = 0

    for (site_key, year), _ in labels_df.groupby(["site_key", "year"], sort=True):
        hourly_path = weather_root / site_key / f"{site_key}_hourly_consolidated.parquet"
        daily_path = weather_root / site_key / f"{site_key}_daily_consolidated.parquet"
        if not hourly_path.exists() or not daily_path.exists():
            raise FileNotFoundError(f"Missing weather data for {site_key}")
        hourly = pd.read_parquet(hourly_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        daily = pd.read_parquet(daily_path)
        daily["time"] = pd.to_datetime(daily["time"], utc=True)
        available_years = set(hourly["timestamp"].dt.year.unique().tolist())
        # ERA5-Land starts in 1950; pre-1950 label years are skipped due to no weather data.
        if int(year) not in available_years:
            logger.warning(
                "Skipping %s %s: weather data not available for year",
                site_key,
                int(year),
            )
            skipped += 1
            continue
        gdh = compute_gdh(hourly, int(year))
        cp = compute_chill_portions(hourly, int(year))
        winter = compute_winter_aggregates(daily, int(year))
        bloom_doy = float(
            labels_df.loc[
                (labels_df["site_key"] == site_key) & (labels_df["year"] == year),
                "bloom_doy",
            ].iloc[0]
        )
        bloom_doy_lag1 = labels_df.loc[
            (labels_df["site_key"] == site_key) & (labels_df["year"] == year),
            "bloom_doy_lag1",
        ].iloc[0]
        records.append(
            {
                "site_key": site_key,
                "year": int(year),
                "gdh": gdh,
                "cp": cp,
                "precip_sum": winter["precip_sum"],
                "rain_sum": winter["rain_sum"],
                "snow_sum": winter["snow_sum"],
                "tmax_mean": winter["tmax_mean"],
                "tmin_mean": winter["tmin_mean"],
                "tmean_mean": winter["tmean_mean"],
                "daylight_mean": winter["daylight_mean"],
                "bloom_doy_lag1": (
                    float(bloom_doy_lag1) if pd.notna(bloom_doy_lag1) else None
                ),
                "bloom_doy": bloom_doy,
            }
        )

    for site_key in SITES.keys():
        hourly_path = weather_root / site_key / f"{site_key}_hourly_consolidated.parquet"
        daily_path = weather_root / site_key / f"{site_key}_daily_consolidated.parquet"
        if not hourly_path.exists() or not daily_path.exists():
            raise FileNotFoundError(f"Missing weather data for {site_key}")
        hourly = pd.read_parquet(hourly_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        daily = pd.read_parquet(daily_path)
        daily["time"] = pd.to_datetime(daily["time"], utc=True)
        available_years = set(hourly["timestamp"].dt.year.unique().tolist())
        if COMPETITION_YEAR not in available_years:
            raise FileNotFoundError(f"Missing {COMPETITION_YEAR} weather data for {site_key}")
        gdh = compute_gdh(hourly, COMPETITION_YEAR)
        cp = compute_chill_portions(hourly, COMPETITION_YEAR)
        winter = compute_winter_aggregates(daily, COMPETITION_YEAR)
        site_labels = labels_df.loc[labels_df["site_key"] == site_key].sort_values("year")
        lag_val = site_labels["bloom_doy"].iloc[-1] if not site_labels.empty else None
        records.append(
            {
                "site_key": site_key,
                "year": COMPETITION_YEAR,
                "gdh": gdh,
                "cp": cp,
                "precip_sum": winter["precip_sum"],
                "rain_sum": winter["rain_sum"],
                "snow_sum": winter["snow_sum"],
                "tmax_mean": winter["tmax_mean"],
                "tmin_mean": winter["tmin_mean"],
                "tmean_mean": winter["tmean_mean"],
                "daylight_mean": winter["daylight_mean"],
                "bloom_doy_lag1": (float(lag_val) if pd.notna(lag_val) else None),
                "bloom_doy": None,
            }
        )

    logger.info(
        "Built features for %s of %s site-years (%s skipped due to missing weather data)",
        len(records),
        total_pairs + len(SITES),
        skipped,
    )

    features = pd.DataFrame.from_records(records)
    features = features.sort_values(["site_key", "year"])
    features.to_parquet(output_path, index=False)
    return features
