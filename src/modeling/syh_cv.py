"""Site-Year Holdout (SYH) cross-validation pipeline for Phase 2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    COMPETITION_YEAR,
    GOLD_DIR,
    PROCESSED_DIR,
    SILVER_ASOS_DIR,
    SILVER_WEATHER_DIR,
    SITES,
)
from src.modeling.empirical_bayes import compute_shrinkage_weights
from src.processing.bias_correction import estimate_bias_from_merged, prepare_bias_merge
from src.processing.warming_velocity import compute_warming_velocity

logger = logging.getLogger(__name__)


class SYHCrossValidator:
    """Execute fold-safe Site-Year Holdout CV and full-data refit artifacts."""

    def __init__(
        self,
        features_path: Path | None = None,
        weather_root: Path | None = None,
        asos_root: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        """Initialize CV pipeline and caches.

        Args:
            features_path: Path to gold features parquet.
            weather_root: Root directory for consolidated weather files.
            asos_root: Root directory for ASOS station parquet files.
            processed_dir: Output directory for Phase 2 artifacts.
        """

        np.random.seed(42)
        self.features_path = features_path or (GOLD_DIR / "features.parquet")
        self.weather_root = weather_root or SILVER_WEATHER_DIR
        self.asos_root = asos_root or SILVER_ASOS_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        features_df = pd.read_parquet(self.features_path)
        self.training_df = (
            features_df.loc[
                (features_df["year"] != COMPETITION_YEAR) & features_df["bloom_doy"].notna()
            ]
            .copy()
            .sort_values(["year", "site_key"])
        )
        self.training_df["year"] = self.training_df["year"].astype(int)
        self.training_df["bloom_doy"] = pd.to_numeric(
            self.training_df["bloom_doy"], errors="coerce"
        )

        self.site_order = sorted(self.training_df["site_key"].unique().tolist())
        self.weather_cache = self._build_weather_cache()
        self.daily_means_cache = self._build_daily_means_cache()
        self.bias_merge_cache = self._build_bias_merge_cache()

    def _build_weather_cache(self) -> dict[str, pd.DataFrame]:
        cache: dict[str, pd.DataFrame] = {}
        for site_key in self.site_order:
            path = self.weather_root / site_key / f"{site_key}_consolidated.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing consolidated weather parquet: {path}")
            df = pd.read_parquet(path)
            if "timestamp" not in df.columns:
                raise AssertionError(f"Weather file missing timestamp column: {path}")
            if "temperature_2m" not in df.columns:
                raise AssertionError(f"Weather file missing temperature_2m column: {path}")
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cache[site_key] = df
        return cache

    def _build_daily_means_cache(self) -> dict[str, dict[int, pd.Series]]:
        output: dict[str, dict[int, pd.Series]] = {}
        for site_key, hourly in self.weather_cache.items():
            daily = hourly.copy()
            daily["year"] = daily["timestamp"].dt.year.astype(int)
            daily["doy"] = daily["timestamp"].dt.dayofyear.astype(int)
            grouped = daily.groupby(["year", "doy"], sort=True)["temperature_2m"].mean()
            per_year: dict[int, pd.Series] = {}
            for year in sorted(grouped.index.get_level_values(0).unique().tolist()):
                series = grouped.loc[year]
                series = pd.Series(series.values, index=series.index.astype(int))
                series = series.sort_index()
                per_year[int(year)] = series
            output[site_key] = per_year
        return output

    def _build_bias_merge_cache(self) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for site_key in self.site_order:
            stations = SITES[site_key].asos_stations
            if not stations:
                continue
            merged_frames: list[pd.DataFrame] = []
            era5_df = self.weather_cache[site_key]
            for station in sorted(stations):
                asos_path = self.asos_root / f"{station}.parquet"
                if not asos_path.exists():
                    logger.warning(
                        "ASOS parquet missing for %s station %s; skipping station in bias cache.",
                        site_key,
                        station,
                    )
                    continue
                asos_df = pd.read_parquet(asos_path)
                merged = prepare_bias_merge(asos_df, era5_df)
                if not merged.empty:
                    merged_frames.append(merged)
            if merged_frames:
                output[site_key] = pd.concat(merged_frames, ignore_index=True)
        return output

    def _compute_fold_bias(self, holdout_year: int) -> dict[str, dict[str, float | int]]:
        bias: dict[str, dict[str, float | int]] = {}
        for site_key in self.site_order:
            if site_key in self.bias_merge_cache:
                bias[site_key] = estimate_bias_from_merged(
                    self.bias_merge_cache[site_key], exclude_year=holdout_year
                )
            else:
                bias[site_key] = {"beta0": 0.0, "beta1": 1.0, "r2": 0.0, "n_obs": 0}
        return bias

    def _compute_mean_bloom(self, fold_train: pd.DataFrame) -> dict[str, float]:
        global_mean = float(fold_train["bloom_doy"].mean())
        mean_by_site: dict[str, float] = {}
        for site_key in self.site_order:
            subset = fold_train.loc[fold_train["site_key"] == site_key, "bloom_doy"]
            if subset.empty:
                mean_by_site[site_key] = global_mean
            else:
                mean_by_site[site_key] = float(subset.mean())
        return mean_by_site

    def _compute_wv_map(
        self,
        rows: pd.DataFrame,
        mean_bloom: dict[str, float],
        bias: dict[str, dict[str, float | int]],
    ) -> dict[tuple[str, int], float]:
        output: dict[tuple[str, int], float] = {}
        for row in rows.itertuples(index=False):
            site_key = str(row.site_key)
            year = int(row.year)
            center = int(round(mean_bloom[site_key]))
            beta0 = float(bias[site_key]["beta0"])
            beta1 = float(bias[site_key]["beta1"])
            wv_payload = compute_warming_velocity(
                self.daily_means_cache[site_key],
                year,
                center,
                beta0=beta0,
                beta1=beta1,
                window_days=14,
            )
            output[(site_key, year)] = float(wv_payload["wv"])
        return output

    @staticmethod
    def _build_design_matrix(frame: pd.DataFrame, site_order: list[str]) -> np.ndarray:
        n = len(frame)
        X = np.zeros((n, len(site_order) + 3), dtype=float)
        site_vals = frame["site_key"].astype(str).to_numpy()
        for idx, site_key in enumerate(site_order):
            X[:, idx] = (site_vals == site_key).astype(float)
        X[:, len(site_order)] = pd.to_numeric(frame["gdh"], errors="coerce").to_numpy(dtype=float)
        X[:, len(site_order) + 1] = pd.to_numeric(frame["cp"], errors="coerce").to_numpy(dtype=float)
        X[:, len(site_order) + 2] = pd.to_numeric(frame["wv"], errors="coerce").to_numpy(dtype=float)
        return X

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run deterministic SYH CV, save artifacts, and return results.

        Returns:
            Tuple of CV results DataFrame and fold log dictionary.
        """

        years = sorted(self.training_df["year"].unique().tolist())
        cv_records: list[dict[str, Any]] = []
        fold_log_entries: list[dict[str, Any]] = []

        for holdout_year in years:
            fold_train = self.training_df.loc[self.training_df["year"] != holdout_year].copy()
            fold_test = self.training_df.loc[self.training_df["year"] == holdout_year].copy()
            if fold_test.empty:
                continue

            bias = self._compute_fold_bias(holdout_year)
            mean_bloom = self._compute_mean_bloom(fold_train)

            fold_all = pd.concat([fold_train, fold_test], ignore_index=True)
            wv_map = self._compute_wv_map(fold_all, mean_bloom, bias)

            train_df = fold_train.copy()
            test_df = fold_test.copy()
            train_df["wv"] = [wv_map[(s, int(y))] for s, y in zip(train_df["site_key"], train_df["year"])]
            test_df["wv"] = [wv_map[(s, int(y))] for s, y in zip(test_df["site_key"], test_df["year"])]

            gdh_nan_rate = float(pd.to_numeric(train_df["gdh"], errors="coerce").isna().mean())
            cp_nan_rate = float(pd.to_numeric(train_df["cp"], errors="coerce").isna().mean())
            if gdh_nan_rate > 0.05:
                raise RuntimeError(
                    f"Fold {holdout_year}: GDH NaN rate {gdh_nan_rate:.3f} exceeds 5% threshold"
                )
            if cp_nan_rate > 0.05:
                raise RuntimeError(
                    f"Fold {holdout_year}: CP NaN rate {cp_nan_rate:.3f} exceeds 5% threshold"
                )

            gdh_median = float(pd.to_numeric(train_df["gdh"], errors="coerce").median())
            cp_median = float(pd.to_numeric(train_df["cp"], errors="coerce").median())

            train_gdh_nan = int(pd.to_numeric(train_df["gdh"], errors="coerce").isna().sum())
            train_cp_nan = int(pd.to_numeric(train_df["cp"], errors="coerce").isna().sum())
            test_gdh_nan = int(pd.to_numeric(test_df["gdh"], errors="coerce").isna().sum())
            test_cp_nan = int(pd.to_numeric(test_df["cp"], errors="coerce").isna().sum())
            train_wv_nan = int(pd.to_numeric(train_df["wv"], errors="coerce").isna().sum())
            test_wv_nan = int(pd.to_numeric(test_df["wv"], errors="coerce").isna().sum())

            train_df["gdh"] = pd.to_numeric(train_df["gdh"], errors="coerce").fillna(gdh_median)
            test_df["gdh"] = pd.to_numeric(test_df["gdh"], errors="coerce").fillna(gdh_median)
            train_df["cp"] = pd.to_numeric(train_df["cp"], errors="coerce").fillna(cp_median)
            test_df["cp"] = pd.to_numeric(test_df["cp"], errors="coerce").fillna(cp_median)
            train_df["wv"] = pd.to_numeric(train_df["wv"], errors="coerce").fillna(0.0)
            test_df["wv"] = pd.to_numeric(test_df["wv"], errors="coerce").fillna(0.0)

            X_train = self._build_design_matrix(train_df, self.site_order)
            y_train = pd.to_numeric(train_df["bloom_doy"], errors="coerce").to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

            X_test = self._build_design_matrix(test_df, self.site_order)
            y_pred = X_test @ coef
            y_true = pd.to_numeric(test_df["bloom_doy"], errors="coerce").to_numpy(dtype=float)
            residual = y_true - y_pred

            for site_key, year, pred, actual, res in zip(
                test_df["site_key"].tolist(),
                test_df["year"].astype(int).tolist(),
                y_pred.tolist(),
                y_true.tolist(),
                residual.tolist(),
            ):
                cv_records.append(
                    {
                        "site_key": site_key,
                        "year": int(year),
                        "predicted_doy": float(pred),
                        "actual_doy": float(actual),
                        "residual": float(res),
                        "fold_holdout_year": int(holdout_year),
                    }
                )

            fold_log_entries.append(
                {
                    "year": int(holdout_year),
                    "bias_exclude_year": int(holdout_year),
                    "mean_bloom_from_train_only": True,
                    "mean_bloom_by_site": {
                        k: float(v) for k, v in sorted(mean_bloom.items())
                    },
                    "site_order": list(self.site_order),
                    "train_has_holdout_year": bool((fold_train["year"] == holdout_year).any()),
                    "test_all_holdout_year": bool((fold_test["year"] == holdout_year).all()),
                    "imputation": {
                        "train_gdh_nan": train_gdh_nan,
                        "test_gdh_nan": test_gdh_nan,
                        "train_cp_nan": train_cp_nan,
                        "test_cp_nan": test_cp_nan,
                        "train_wv_nan": train_wv_nan,
                        "test_wv_nan": test_wv_nan,
                        "gdh_median": gdh_median,
                        "cp_median": cp_median,
                    },
                }
            )

        if not cv_records:
            raise AssertionError("SYH CV produced no prediction rows")

        cv_results = pd.DataFrame.from_records(cv_records).sort_values(
            ["year", "site_key"]
        )
        cv_path = self.processed_dir / "cv_results.parquet"
        cv_results.to_parquet(cv_path, index=False)

        fold_log = {
            "site_order": list(self.site_order),
            "folds": fold_log_entries,
            "n_folds": len(fold_log_entries),
        }
        fold_log_path = self.processed_dir / "fold_log.json"
        fold_log_path.write_text(json.dumps(fold_log, indent=2, sort_keys=True))

        mae_summary = self.compute_mae(cv_results)
        mae_path = self.processed_dir / "mae_summary.json"
        mae_path.write_text(json.dumps(mae_summary, indent=2, sort_keys=True))

        epsilon = 1.0
        shrinkage = compute_shrinkage_weights(
            cv_results_df=cv_results,
            training_df=self.training_df,
            epsilon=epsilon,
        )
        shrinkage_path = self.processed_dir / "shrinkage_weights.json"
        shrinkage_path.write_text(json.dumps(shrinkage, indent=2, sort_keys=True))

        full_bias = self._compute_full_bias()
        (self.processed_dir / "bias_coefficients.json").write_text(
            json.dumps(full_bias, indent=2, sort_keys=True)
        )

        mean_bloom_full = self._compute_mean_bloom(self.training_df)
        (self.processed_dir / "mean_bloom_doy.json").write_text(
            json.dumps(mean_bloom_full, indent=2, sort_keys=True)
        )

        full_model = self._fit_full_model(mean_bloom_full, full_bias)
        (self.processed_dir / "model_coefficients.json").write_text(
            json.dumps(full_model, indent=2, sort_keys=True)
        )

        global_mean = float(self.training_df["bloom_doy"].mean())
        (self.processed_dir / "global_mean.json").write_text(
            json.dumps({"global_mean": global_mean}, indent=2, sort_keys=True)
        )

        return cv_results, fold_log

    @staticmethod
    def compute_mae(cv_results: pd.DataFrame) -> dict[str, Any]:
        """Compute per-site and overall MAE from CV results.

        Args:
            cv_results: CV prediction rows.

        Returns:
            MAE summary payload.
        """

        work = cv_results.copy()
        work["ae"] = (work["actual_doy"] - work["predicted_doy"]).abs()
        per_site = (
            work.groupby("site_key", sort=True)["ae"].mean().astype(float).to_dict()
        )
        overall = float(work["ae"].mean())
        return {"overall_mae": overall, "per_site_mae": per_site}

    def _compute_full_bias(self) -> dict[str, dict[str, float | int]]:
        payload: dict[str, dict[str, float | int]] = {}
        for site_key in self.site_order:
            if site_key in self.bias_merge_cache:
                payload[site_key] = estimate_bias_from_merged(
                    self.bias_merge_cache[site_key], exclude_year=None
                )
            else:
                payload[site_key] = {"beta0": 0.0, "beta1": 1.0, "r2": 0.0, "n_obs": 0}
        return payload

    def _fit_full_model(
        self,
        mean_bloom_full: dict[str, float],
        full_bias: dict[str, dict[str, float | int]],
    ) -> dict[str, Any]:
        frame = self.training_df.copy()
        wv_map = self._compute_wv_map(frame, mean_bloom_full, full_bias)
        frame["wv"] = [wv_map[(s, int(y))] for s, y in zip(frame["site_key"], frame["year"])]

        gdh_median = float(pd.to_numeric(frame["gdh"], errors="coerce").median())
        cp_median = float(pd.to_numeric(frame["cp"], errors="coerce").median())
        frame["gdh"] = pd.to_numeric(frame["gdh"], errors="coerce").fillna(gdh_median)
        frame["cp"] = pd.to_numeric(frame["cp"], errors="coerce").fillna(cp_median)
        frame["wv"] = pd.to_numeric(frame["wv"], errors="coerce").fillna(0.0)

        X = self._build_design_matrix(frame, self.site_order)
        y = pd.to_numeric(frame["bloom_doy"], errors="coerce").to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        x_condition_number_full = float(np.linalg.cond(X))

        feature_order = list(self.site_order) + ["gdh", "cp", "wv"]
        site_intercepts = {
            site_key: float(coef[idx]) for idx, site_key in enumerate(self.site_order)
        }
        beta_gdh = float(coef[len(self.site_order)])
        beta_cp = float(coef[len(self.site_order) + 1])
        beta_wv = float(coef[len(self.site_order) + 2])

        gdh_std = float(frame["gdh"].astype(float).std(ddof=1))
        cp_std = float(frame["cp"].astype(float).std(ddof=1))
        wv_std = float(frame["wv"].astype(float).std(ddof=1))

        feature_scales_full = {
            "gdh_std": gdh_std,
            "cp_std": cp_std,
            "wv_std": wv_std,
        }
        standardized_betas_full = {
            "beta_gdh_std_effect_doy_per_1sd": beta_gdh * gdh_std,
            "beta_cp_std_effect_doy_per_1sd": beta_cp * cp_std,
            "beta_wv_std_effect_doy_per_1sd": beta_wv * wv_std,
        }

        def _sign_label(value: float) -> str:
            if abs(value) < 1e-12:
                return "zero"
            if value > 0:
                return "positive"
            return "negative"

        beta_signs_full = {
            "beta_gdh_sign": _sign_label(beta_gdh),
            "beta_cp_sign": _sign_label(beta_cp),
            "beta_wv_sign": _sign_label(beta_wv),
        }

        return {
            "site_intercepts": site_intercepts,
            "beta_gdh": beta_gdh,
            "beta_cp": beta_cp,
            "beta_wv": beta_wv,
            "feature_order": feature_order,
            "imputation": {"gdh_median": gdh_median, "cp_median": cp_median},
            "x_condition_number_full": x_condition_number_full,
            "feature_scales_full": feature_scales_full,
            "standardized_betas_full": standardized_betas_full,
            "beta_signs_full": beta_signs_full,
            "note_wv_interpretation": (
                "beta_wv sign can be counterintuitive due to conditioning/ill-conditioning "
                "with GDH/CP and site dummies; interpret standardized effect size and "
                "condition number alongside raw sign."
            ),
        }


def main() -> None:
    """Run deterministic Phase 2 SYH CV pipeline end-to-end."""

    parser = argparse.ArgumentParser(description="Run Phase 2 SYH cross-validation pipeline.")
    parser.parse_args()

    runner = SYHCrossValidator()
    cv_results, fold_log = runner.run()
    mae = runner.compute_mae(cv_results)
    logger.info(
        "Phase 2 complete: %s CV rows, %s folds, overall MAE=%.4f",
        len(cv_results),
        fold_log["n_folds"],
        mae["overall_mae"],
    )
    logger.info("site_order=%s", runner.site_order)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
