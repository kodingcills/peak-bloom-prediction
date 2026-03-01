"""Unified validation gate runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


PHASE_GATES = {
    "1": [
        "assert_inference_cutoff_utc",
        "assert_historical_window_end",
        "assert_labels_complete",
        "assert_seas5_members",
        "assert_silver_utc",
        "assert_gold_schema",
    ],
    "2": [
        "assert_bias_fold_safe",
        "assert_window_safe",
        "assert_precision_fold_safe",
        "assert_cv_no_leakage",
    ],
    "3": [
        "assert_gmm_k_range",
        "assert_no_noise_injection",
        "assert_submission_schema",
        "assert_predictions_reasonable",
    ],
}


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


def _run_gate(name: str) -> None:
    logger.info("Running gate: %s", name)
    from config.settings import COMPETITION_YEAR, GOLD_DIR, PROCESSED_DIR, SILVER_WEATHER_DIR
    from src.processing.labels import load_competition_labels
    from src.validation import gates
    fn = getattr(gates, name)
    if name == "assert_labels_complete":
        labels = load_competition_labels()
        fn(labels)
    elif name == "assert_gold_schema":
        features = pd.read_parquet(GOLD_DIR / "features.parquet")
        fn(features)
    elif name == "assert_historical_window_end":
        features = pd.read_parquet(GOLD_DIR / "features.parquet")
        fn(features)
    elif name == "assert_inference_cutoff_utc":
        fn(SILVER_WEATHER_DIR)
    elif name == "assert_silver_utc":
        fn(SILVER_WEATHER_DIR)
    elif name == "assert_seas5_members":
        fn(PROCESSED_DIR / "seas5_2026.nc")
    elif name == "assert_bias_fold_safe":
        fold_log = json.loads((PROCESSED_DIR / "fold_log.json").read_text())
        fn(fold_log)
    elif name == "assert_window_safe":
        fold_log = json.loads((PROCESSED_DIR / "fold_log.json").read_text())
        fn(fold_log)
    elif name == "assert_precision_fold_safe":
        cv_results = pd.read_parquet(PROCESSED_DIR / "cv_results.parquet")
        fn(cv_results)
    elif name == "assert_cv_no_leakage":
        cv_results = pd.read_parquet(PROCESSED_DIR / "cv_results.parquet")
        fold_log = json.loads((PROCESSED_DIR / "fold_log.json").read_text())
        fn(cv_results, fold_log)
    elif name == "assert_gmm_k_range":
        gmm = json.loads((PROCESSED_DIR / "diagnostics" / "gmm_results.json").read_text())
        fn(gmm)
    elif name == "assert_no_noise_injection":
        ens = json.loads(
            (PROCESSED_DIR / "diagnostics" / "ensemble_distributions.json").read_text()
        )
        fn(ens)
    elif name == "assert_submission_schema":
        fn(Path("submission.csv"))
    elif name == "assert_predictions_reasonable":
        fn(Path("submission.csv"))
    else:
        fn()


def main() -> None:
    _load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(description="Run Phenology Engine validation gates.")
    parser.add_argument("--phase", help="Optional phase number to run gates for.")
    args = parser.parse_args()

    if args.phase:
        phase = str(args.phase)
        if phase not in PHASE_GATES:
            raise ValueError(f"Unknown phase: {phase}")
        gate_names = PHASE_GATES[phase]
    else:
        gate_names = [name for names in PHASE_GATES.values() for name in names]

    results: list[tuple[str, str]] = []
    for gate_name in gate_names:
        try:
            _run_gate(gate_name)
            results.append((gate_name, "PASS"))
        except Exception:
            results.append((gate_name, "FAIL"))
            break
    table = "\n".join(f"{name}: {status}" for name, status in results)
    print(table)

    if any(status == "FAIL" for _, status in results):
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
