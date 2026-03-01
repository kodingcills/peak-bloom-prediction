"""Phase 1: Data Acquisition & Silver Layer."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_sites(sites_arg: str | None, site_keys: list[str]) -> list[str]:
    if not sites_arg or sites_arg == "all":
        return list(site_keys)
    sites = [s.strip() for s in sites_arg.split(",") if s.strip()]
    unknown = [s for s in sites if s not in site_keys]
    if unknown:
        raise ValueError(f"Unknown site keys: {unknown}")
    return sites


def _ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _validate_era5_inputs(
    required_sites: list[str],
    weather_root: Path,
    status: dict[str, dict],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for site_key in required_sites:
        site_dir = weather_root / site_key
        hourly_consolidated = site_dir / f"{site_key}_hourly_consolidated.parquet"
        daily_consolidated = site_dir / f"{site_key}_daily_consolidated.parquet"
        incomplete = site_dir / "_INCOMPLETE.txt"
        site_status = status.get(site_key, {})
        if (
            not site_status.get("ok", False)
            or not hourly_consolidated.exists()
            or not daily_consolidated.exists()
            or incomplete.exists()
        ):
            failures.append(
                {
                    "site": site_key,
                    "status_error": str(site_status.get("error", "")),
                    "hourly_consolidated_exists": str(hourly_consolidated.exists()),
                    "daily_consolidated_exists": str(daily_consolidated.exists()),
                    "incomplete_marker": str(incomplete.exists()),
                }
            )
    return failures


def _required_asos_stations(sites: list[str], site_map: dict) -> list[str]:
    stations: set[str] = set()
    for site_key in sites:
        stations.update(site_map[site_key].asos_stations)
    return sorted(stations)


def _validate_asos_inputs(
    required_stations: list[str],
    asos_root: Path,
    status: dict[str, dict],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for station_id in required_stations:
        station_file = asos_root / f"{station_id}.parquet"
        station_status = status.get(station_id, {})
        if not station_status.get("ok", False) or not station_file.exists():
            failures.append(
                {
                    "station": station_id,
                    "status_error": str(station_status.get("error", "")),
                    "file_exists": str(station_file.exists()),
                }
            )
    return failures


def main() -> None:
    _load_dotenv(Path(".env"))

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
    from src.processing.features import build_gold_features
    from src.processing.labels import load_competition_labels
    from src.validation import gates

    parser = argparse.ArgumentParser(description="Phase 1: Data Acquisition & Silver Layer")
    parser.add_argument("--step", help="era5/asos/seas5/labels/features")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--sites", help="Comma-separated site keys or 'all'")
    args = parser.parse_args()

    _ensure_dirs([RAW_DIR, SILVER_WEATHER_DIR, SILVER_ASOS_DIR, PROCESSED_DIR, GOLD_DIR])
    sites = _parse_sites(args.sites, list(SITES.keys()))

    steps = ["era5", "asos", "seas5", "labels", "features"]
    if args.step:
        if args.step not in steps:
            raise ValueError(f"Unknown step: {args.step}")
        steps = [args.step]

    summary = {}

    labels_df: pd.DataFrame | None = None

    if "era5" in steps:
        start = time.perf_counter()
        logger.info("Starting ERA5-Land fetch")
        try:
            result = fetch_all_era5(
                output_dir=SILVER_WEATHER_DIR, force=args.force, site_keys=sites
            )
        except Exception:
            logger.error("ERA5 ingestion failed. Aborting Phase 1.")
            sys.exit(1)
        era5_failures = _validate_era5_inputs(sites, SILVER_WEATHER_DIR, result)
        if era5_failures:
            logger.error("ERA5 completeness check failed for required sites:")
            for failure in era5_failures:
                logger.error(
                    "site=%s error=%s hourly_consolidated_exists=%s daily_consolidated_exists=%s incomplete_marker=%s",
                    failure["site"],
                    failure["status_error"],
                    failure["hourly_consolidated_exists"],
                    failure["daily_consolidated_exists"],
                    failure["incomplete_marker"],
                )
            logger.error("ERA5 ingestion incomplete. Aborting Phase 1 before downstream steps.")
            sys.exit(1)
        duration = time.perf_counter() - start
        summary["era5"] = result
        logger.info("ERA5-Land fetch complete in %.1fs", duration)

    if "asos" in steps:
        start = time.perf_counter()
        logger.info("Starting ASOS fetch")
        try:
            result = fetch_all_asos(
                output_dir=SILVER_ASOS_DIR, force=args.force, site_keys=sites
            )
        except Exception:
            logger.error("ASOS ingestion failed. Aborting Phase 1.")
            sys.exit(1)
        required_stations = _required_asos_stations(sites, SITES)
        asos_failures = _validate_asos_inputs(required_stations, SILVER_ASOS_DIR, result)
        if asos_failures:
            logger.error("ASOS completeness check failed for required stations:")
            for failure in asos_failures:
                logger.error(
                    "station=%s error=%s file_exists=%s",
                    failure["station"],
                    failure["status_error"],
                    failure["file_exists"],
                )
            logger.error("ASOS ingestion incomplete. Aborting Phase 1 before downstream steps.")
            sys.exit(1)
        duration = time.perf_counter() - start
        summary["asos"] = result
        logger.info("ASOS fetch complete in %.1fs", duration)

    if "seas5" in steps:
        start = time.perf_counter()
        logger.info("Starting SEAS5 fetch")
        try:
            result = fetch_seas5(
                output_path=PROCESSED_DIR / "seas5_2026.nc", force=args.force
            )
        except Exception:
            logger.error("SEAS5 ingestion failed. Aborting Phase 1.")
            sys.exit(1)
        duration = time.perf_counter() - start
        summary["seas5"] = {"path": str(result) if result else None}
        logger.info("SEAS5 fetch complete in %.1fs", duration)

    if "labels" in steps:
        start = time.perf_counter()
        logger.info("Loading competition labels")
        labels_df = load_competition_labels()
        duration = time.perf_counter() - start
        summary["labels"] = {"rows": len(labels_df)}
        logger.info("Labels loaded: %s rows (%.1fs)", len(labels_df), duration)

    if "features" in steps:
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
            logger.info("Gold features built: %s rows", len(features_df))
        duration = time.perf_counter() - start
        logger.info("Gold features step complete in %.1fs", duration)

    logger.info("Running validation gates")
    if labels_df is None and (GOLD_DIR / "features.parquet").exists():
        labels_df = load_competition_labels()

    weather_files = list(SILVER_WEATHER_DIR.rglob("*.parquet"))
    gold_path = GOLD_DIR / "features.parquet"
    seas5_path = PROCESSED_DIR / "seas5_2026.nc"

    if weather_files:
        gates.assert_inference_cutoff_utc(SILVER_WEATHER_DIR)
        gates.assert_silver_utc(SILVER_WEATHER_DIR)
    else:
        logger.info("Skipping weather gates; no silver weather files present")

    if gold_path.exists():
        features_df = pd.read_parquet(gold_path)
        gates.assert_historical_window_end(features_df)
        gates.assert_gold_schema(features_df)
    else:
        logger.info("Skipping gold feature gates; no gold features present")

    if labels_df is not None:
        gates.assert_labels_complete(labels_df)
    else:
        logger.info("Skipping labels gate; labels not loaded")

    if seas5_path.exists():
        gates.assert_seas5_members(seas5_path)
    else:
        logger.info("Skipping SEAS5 gate; NetCDF missing")

    logger.info("Validation gates complete")
    logger.info("Phase 1 summary: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
