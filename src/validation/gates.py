"""Validation gates for Phenology Engine v1.7."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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


def assert_bias_fold_safe(fold_log: dict) -> None:
    """Verify each fold re-estimates bias with held-out year excluded."""

    folds = fold_log.get("folds", [])
    if not folds:
        raise AssertionError("fold_log has no folds for bias fold-safety validation")
    for fold in folds:
        year = int(fold["year"])
        exclude_year = int(fold["bias_exclude_year"])
        if exclude_year != year:
            raise AssertionError(
                f"Bias fold safety violation: fold year={year}, bias_exclude_year={exclude_year}"
            )


def assert_window_safe(fold_log: dict) -> None:
    """Verify warming velocity window center uses training-only mean bloom."""

    folds = fold_log.get("folds", [])
    if not folds:
        raise AssertionError("fold_log has no folds for window fold-safety validation")

    gold = pd.read_parquet(GOLD_DIR / "features.parquet")
    training = gold.loc[(gold["year"] != COMPETITION_YEAR) & gold["bloom_doy"].notna()].copy()

    for fold in folds:
        year = int(fold["year"])
        if not bool(fold.get("mean_bloom_from_train_only", False)):
            raise AssertionError(f"Fold {year} mean_bloom_from_train_only is False")
        recorded = fold.get("mean_bloom_by_site", {})
        fold_train = training.loc[training["year"] != year]
        global_mean = float(fold_train["bloom_doy"].mean())
        for site_key in sorted(training["site_key"].unique().tolist()):
            subset = fold_train.loc[fold_train["site_key"] == site_key, "bloom_doy"]
            expected = float(subset.mean()) if not subset.empty else global_mean
            observed = float(recorded[site_key])
            if abs(expected - observed) > 1e-6:
                raise AssertionError(
                    f"Fold {year} window center leakage for {site_key}: "
                    f"expected {expected}, observed {observed}"
                )


def assert_precision_fold_safe(cv_results_df: pd.DataFrame) -> None:
    """Verify shrinkage was not applied inside CV residual generation."""

    if "shrunk_prediction" in cv_results_df.columns:
        raise AssertionError(
            "Precision fold safety violation: cv_results contains shrunk_prediction column"
        )


def assert_vancouver_weight_stable(
    shrinkage_weights: dict,
    cv_results_df: pd.DataFrame,
    training_df: pd.DataFrame,
    epsilon: float,
) -> None:
    """Validate leave-one-out Vancouver shrinkage weight stability."""

    vancouver = cv_results_df.loc[cv_results_df["site_key"] == "vancouver", "residual"]
    vancouver = pd.to_numeric(vancouver, errors="coerce").dropna().to_numpy(dtype=float)
    if vancouver.size < 3:
        logger.warning(
            "Skipping Vancouver weight stability: only %s residuals available", vancouver.size
        )
        return

    residual_all = pd.to_numeric(cv_results_df["residual"], errors="coerce").dropna().to_numpy(
        dtype=float
    )
    n_s = int((training_df["site_key"] == "vancouver").sum())
    if n_s <= 0:
        raise AssertionError("Training data missing Vancouver rows for stability gate")

    weights: list[float] = []
    for idx in range(vancouver.size):
        mask = np.ones(vancouver.size, dtype=bool)
        mask[idx] = False
        v_res = vancouver[mask]

        all_copy = residual_all.copy()
        removed = 0
        for pos, val in enumerate(all_copy):
            if np.isclose(val, vancouver[idx]) and removed == 0:
                all_copy = np.delete(all_copy, pos)
                removed = 1
                break
        sigma2_global = float(np.var(all_copy, ddof=1)) if all_copy.size >= 2 else 1.0
        sigma2_v = float(np.var(v_res, ddof=1)) if v_res.size >= 2 else sigma2_global
        denom = sigma2_global + ((sigma2_v + float(epsilon)) / n_s)
        if denom <= 0:
            raise AssertionError("Non-positive denominator in Vancouver LOO stability check")
        weights.append(float(sigma2_global / denom))

    std = float(np.std(weights, ddof=0))
    if std >= 0.15:
        raise AssertionError(
            f"Vancouver weight instability: std={std:.6f} >= 0.15 across leave-one-out variants"
        )


def assert_cv_no_leakage(cv_results_df: pd.DataFrame, fold_log: dict) -> None:
    """Verify held-out year isolation and sufficient fold coverage."""

    required = {"year", "fold_holdout_year"}
    missing = required - set(cv_results_df.columns)
    if missing:
        raise AssertionError(f"cv_results missing columns for leakage check: {sorted(missing)}")

    if not (cv_results_df["year"].astype(int) == cv_results_df["fold_holdout_year"].astype(int)).all():
        raise AssertionError("CV leakage: found predictions where year != fold_holdout_year")

    folds = fold_log.get("folds", [])
    if not folds:
        raise AssertionError("fold_log has no folds for leakage validation")
    for fold in folds:
        year = int(fold["year"])
        if bool(fold.get("train_has_holdout_year", True)):
            raise AssertionError(f"Fold {year} leakage: train_has_holdout_year=True")
        if not bool(fold.get("test_all_holdout_year", False)):
            raise AssertionError(f"Fold {year} leakage: test_all_holdout_year=False")

    if int(cv_results_df["fold_holdout_year"].nunique()) <= 10:
        raise AssertionError(
            "Insufficient SYH year coverage: fold_holdout_year unique count must exceed 10"
        )


def assert_gmm_k_range(gmm_results: dict) -> None:
    """Validate all selected GMM component counts are in {1, 2}."""

    for site, result in gmm_results.items():
        k = int(result["k"])
        if k not in {1, 2}:
            raise AssertionError(
                f"GMM k={k} for {site} — only k in {{1,2}} allowed"
            )


def assert_no_noise_injection(ensemble_distributions: dict) -> None:
    """Validate ensemble size is consistent with fallback scenario generation."""

    for site, dist in ensemble_distributions.items():
        n_members = int(dist["n_members"])
        if n_members > 35:
            raise AssertionError(
                f"{site}: n_members={n_members} exceeds fallback upper bound (~30), "
                "suggesting artificial ensemble inflation"
            )


def assert_submission_schema(submission_path: str | Path) -> None:
    """Validate Phase 3 submission schema and required rows."""

    df = pd.read_csv(submission_path)
    expected_cols = ["location", "year", "bloom_doy"]
    if list(df.columns) != expected_cols:
        raise AssertionError(f"Wrong columns: {list(df.columns)}")
    if len(df) != 5:
        raise AssertionError(f"Expected 5 rows, got {len(df)}")
    if not (df["year"] == COMPETITION_YEAR).all():
        raise AssertionError("Not all rows are year 2026")
    if not all(isinstance(x, (int, np.integer)) for x in df["bloom_doy"].tolist()):
        raise AssertionError(f"Non-integer bloom_doy: {df['bloom_doy'].tolist()}")

    expected = {"washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"}
    actual = set(df["location"].tolist())
    if actual != expected:
        raise AssertionError(
            f"Location mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def assert_predictions_reasonable(submission_path: str | Path) -> None:
    """Validate all predicted bloom DOYs fall in the constrained range [60, 140]."""

    df = pd.read_csv(submission_path)
    for _, row in df.iterrows():
        doy = int(row["bloom_doy"])
        if not (60 <= doy <= 140):
            raise AssertionError(
                f"{row['location']}: bloom_doy={doy} outside allowed range [60, 140]"
            )


def assert_shrinkage_applied(prediction_summary: dict) -> None:
    """Validate shrinkage changed cold-start site predictions."""

    for site in ["vancouver", "nyc"]:
        if site not in prediction_summary:
            continue
        ps = prediction_summary[site]
        gmm_mean = float(ps["gmm_selected_mean"])
        shrunk = float(ps["shrunk_prediction"])
        if abs(gmm_mean - shrunk) <= 0.01:
            raise AssertionError(
                f"{site}: shrinkage had no effect (gmm={gmm_mean:.2f}, shrunk={shrunk:.2f})"
            )
