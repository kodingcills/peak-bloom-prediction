"""Unified validation gate runner."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config.settings import GOLD_DIR, PROCESSED_DIR, SILVER_WEATHER_DIR
from src.processing.labels import load_competition_labels
from src.validation import gates

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
}


def _run_gate(name: str) -> None:
    logger.info("Running gate: %s", name)
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
    else:
        fn()


def main() -> None:
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
