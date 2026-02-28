"""Phase 1: Data Acquisition & Silver Layer."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import (
    GOLD_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SILVER_ASOS_DIR,
    SILVER_WEATHER_DIR,
    SITES,
)
from src.ingestion.asos_fetcher import fetch_all_asos
from src.ingestion.era5_fetcher import fetch_all_era5
from src.ingestion.seas5_fetcher import fetch_seas5
from src.monitoring import mlflow_utils
from src.processing.features import build_gold_features
from src.processing.labels import load_competition_labels
from src.validation import gates

logger = logging.getLogger(__name__)


def _parse_sites(sites_arg: str | None) -> list[str]:
    if not sites_arg or sites_arg == "all":
        return list(SITES.keys())
    sites = [s.strip() for s in sites_arg.split(",") if s.strip()]
    unknown = [s for s in sites if s not in SITES]
    if unknown:
        raise ValueError(f"Unknown site keys: {unknown}")
    return sites


def _ensure_dirs() -> None:
    for path in [RAW_DIR, SILVER_WEATHER_DIR, SILVER_ASOS_DIR, PROCESSED_DIR, GOLD_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Data Acquisition & Silver Layer")
    parser.add_argument("--step", help="era5/asos/seas5/labels/features")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--sites", help="Comma-separated site keys or 'all'")
    args = parser.parse_args()

    _ensure_dirs()
    sites = _parse_sites(args.sites)

    steps = ["era5", "asos", "seas5", "labels", "features"]
    if args.step:
        if args.step not in steps:
            raise ValueError(f"Unknown step: {args.step}")
        steps = [args.step]

    summary = {}

    with mlflow_utils.init_mlflow(
        run_name="phase1_refresh_data",
        tags={"phase": "1"},
    ):
        mlflow_utils.log_params(
            {
                "force": args.force,
                "sites": ",".join(sites),
                "step": args.step or "all",
                "chunk_years": 10,
            }
        )

        labels_df: pd.DataFrame | None = None

        if "era5" in steps:
            with mlflow_utils._require_mlflow().start_run(
                run_name="phase1_era5", nested=True
            ):
                start = time.perf_counter()
                logger.info("Starting ERA5-Land fetch")
                result = fetch_all_era5(
                    output_dir=SILVER_WEATHER_DIR, force=args.force, site_keys=sites
                )
                duration = time.perf_counter() - start
                ok_count = sum(1 for v in result.values() if v.get("ok"))
                fail_count = sum(1 for v in result.values() if not v.get("ok"))
                summary["era5"] = result
                mlflow_utils.log_metrics(
                    {
                        "duration_seconds": duration,
                        "sites_ok": ok_count,
                        "sites_failed": fail_count,
                    }
                )
                logger.info("ERA5-Land fetch complete")

        if "asos" in steps:
            with mlflow_utils._require_mlflow().start_run(
                run_name="phase1_asos", nested=True
            ):
                start = time.perf_counter()
                logger.info("Starting ASOS fetch")
                result = fetch_all_asos(
                    output_dir=SILVER_ASOS_DIR, force=args.force, site_keys=sites
                )
                duration = time.perf_counter() - start
                ok_count = sum(1 for v in result.values() if v.get("ok"))
                fail_count = sum(1 for v in result.values() if not v.get("ok"))
                summary["asos"] = result
                mlflow_utils.log_metrics(
                    {
                        "duration_seconds": duration,
                        "stations_ok": ok_count,
                        "stations_failed": fail_count,
                    }
                )
                logger.info("ASOS fetch complete")

        if "seas5" in steps:
            with mlflow_utils._require_mlflow().start_run(
                run_name="phase1_seas5", nested=True
            ):
                start = time.perf_counter()
                logger.info("Starting SEAS5 fetch")
                result = fetch_seas5(
                    output_path=PROCESSED_DIR / "seas5_2026.nc", force=args.force
                )
                duration = time.perf_counter() - start
                summary["seas5"] = {"path": str(result) if result else None}
                mlflow_utils.log_metrics({"duration_seconds": duration})
                if result:
                    mlflow_utils.log_artifact(result)
                else:
                    flag = PROCESSED_DIR / "SEAS5_FETCH_FAILED"
                    if flag.exists():
                        mlflow_utils.log_artifact(flag)
                logger.info("SEAS5 fetch complete")

        if "labels" in steps:
            with mlflow_utils._require_mlflow().start_run(
                run_name="phase1_labels", nested=True
            ):
                start = time.perf_counter()
                logger.info("Loading competition labels")
                labels_df = load_competition_labels()
                duration = time.perf_counter() - start
                summary["labels"] = {"rows": len(labels_df)}
                mlflow_utils.log_metrics({"duration_seconds": duration, "rows": len(labels_df)})
                logger.info("Labels loaded: %s rows", len(labels_df))

        if "features" in steps:
            with mlflow_utils._require_mlflow().start_run(
                run_name="phase1_features", nested=True
            ):
                start = time.perf_counter()
                logger.info("Building gold features")
                features_path = GOLD_DIR / "features.parquet"
                if features_path.exists() and not args.force:
                    logger.info("Skipping existing gold features: %s", features_path)
                else:
                    if labels_df is None:
                        labels_df = load_competition_labels()
                    features_df = build_gold_features(
                        SILVER_WEATHER_DIR, labels_df, features_path
                    )
                    summary["features"] = {"rows": len(features_df)}
                    mlflow_utils.log_metrics({"rows": len(features_df)})
                    mlflow_utils.log_artifact(features_path)
                    logger.info("Gold features built: %s rows", len(features_df))
                duration = time.perf_counter() - start
                mlflow_utils.log_metrics({"duration_seconds": duration})

        logger.info("Running validation gates")
        gate_results: list[tuple[str, str]] = []
        try:
            if labels_df is None and (GOLD_DIR / "features.parquet").exists():
                labels_df = load_competition_labels()
            gates.assert_inference_cutoff_utc(SILVER_WEATHER_DIR)
            gate_results.append(("assert_inference_cutoff_utc", "PASS"))
            if (GOLD_DIR / "features.parquet").exists():
                features_df = pd.read_parquet(GOLD_DIR / "features.parquet")
                gates.assert_historical_window_end(features_df)
                gate_results.append(("assert_historical_window_end", "PASS"))
                gates.assert_gold_schema(features_df)
                gate_results.append(("assert_gold_schema", "PASS"))
            if labels_df is not None:
                gates.assert_labels_complete(labels_df)
                gate_results.append(("assert_labels_complete", "PASS"))
            gates.assert_silver_utc(SILVER_WEATHER_DIR)
            gate_results.append(("assert_silver_utc", "PASS"))
            gates.assert_seas5_members(PROCESSED_DIR / "seas5_2026.nc")
            gate_results.append(("assert_seas5_members", "PASS"))
        except Exception as exc:
            gate_results.append((type(exc).__name__, "FAIL"))
            table = "\n".join(f"{name}: {status}" for name, status in gate_results)
            mlflow_utils.log_text(table, "phase1_gate_results.txt")
            raise

        table = "\n".join(f"{name}: {status}" for name, status in gate_results)
        mlflow_utils.log_text(table, "phase1_gate_results.txt")
        logger.info("Validation gates complete")

        logger.info("Phase 1 summary: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
