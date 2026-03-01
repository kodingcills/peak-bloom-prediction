"""Offline backtest comparing global vs analog shrinkage on CV outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR
from src.modeling.analog_shrinkage import compute_delta_from_climatology

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _resolve_weight(payload: dict[str, Any], site_key: str) -> float:
    if "sites" in payload:
        return float(payload["sites"][site_key]["w"])
    return float(payload[site_key]["w"])


def run_backtest() -> dict[str, Any]:
    """Run deterministic per-fold shrinkage backtest and return summary payload."""

    cv = pd.read_parquet(PROCESSED_DIR / "cv_results.parquet")
    mean_bloom = _load_json(PROCESSED_DIR / "mean_bloom_doy.json")
    global_mean = float(_load_json(PROCESSED_DIR / "global_mean.json")["global_mean"])
    weights = _load_json(PROCESSED_DIR / "shrinkage_weights.json")

    required_sites = ["washingtondc", "kyoto", "liestal", "nyc", "vancouver"]
    fold_errors_global: dict[str, list[float]] = {s: [] for s in required_sites}
    fold_errors_analog: dict[str, list[float]] = {s: [] for s in required_sites}

    delta_nyc = compute_delta_from_climatology(
        mean_bloom_doy=mean_bloom, site_key="nyc", analog_site="washingtondc"
    )
    delta_van = compute_delta_from_climatology(
        mean_bloom_doy=mean_bloom, site_key="vancouver", analog_site="kyoto"
    )

    for year in sorted(cv["fold_holdout_year"].astype(int).unique().tolist()):
        fold = cv.loc[cv["fold_holdout_year"].astype(int) == int(year)].copy()
        by_site = fold.set_index("site_key")
        if not all(site in by_site.index for site in required_sites):
            continue

        pred = {s: float(by_site.loc[s, "predicted_doy"]) for s in required_sites}
        actual = {s: float(by_site.loc[s, "actual_doy"]) for s in required_sites}

        for site in required_sites:
            w = _resolve_weight(weights, site)
            shr_global = float((w * pred[site]) + ((1.0 - w) * global_mean))

            if site == "nyc":
                prior_analog = float(pred["washingtondc"] + delta_nyc)
            elif site == "vancouver":
                prior_analog = float(pred["kyoto"] + delta_van)
            else:
                prior_analog = float(global_mean)
            shr_analog = float((w * pred[site]) + ((1.0 - w) * prior_analog))

            fold_errors_global[site].append(abs(shr_global - actual[site]))
            fold_errors_analog[site].append(abs(shr_analog - actual[site]))

    def _mae(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    per_site_global = {site: _mae(vals) for site, vals in fold_errors_global.items()}
    per_site_analog = {site: _mae(vals) for site, vals in fold_errors_analog.items()}
    overall_global = _mae([v for vals in fold_errors_global.values() for v in vals])
    overall_analog = _mae([v for vals in fold_errors_analog.values() for v in vals])

    payload = {
        "overall_mae_global": overall_global,
        "overall_mae_analog": overall_analog,
        "overall_delta_analog_minus_global": float(overall_analog - overall_global),
        "per_site_mae_global": per_site_global,
        "per_site_mae_analog": per_site_analog,
        "delta_by_site_analog_minus_global": {
            site: float(per_site_analog[site] - per_site_global[site])
            for site in required_sites
        },
        "analog_offsets_days": {
            "nyc_minus_washingtondc": float(delta_nyc),
            "vancouver_minus_kyoto": float(delta_van),
        },
    }
    return payload


def main() -> None:
    """Entrypoint for shrinkage backtest diagnostics."""

    payload = run_backtest()
    out_dir = PROCESSED_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shrinkage_backtest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    logger.info("Wrote %s", out_path)
    logger.info(
        "Overall MAE global=%.4f analog=%.4f delta=%.4f",
        payload["overall_mae_global"],
        payload["overall_mae_analog"],
        payload["overall_delta_analog_minus_global"],
    )
    logger.info(
        "NYC MAE global=%.4f analog=%.4f",
        payload["per_site_mae_global"]["nyc"],
        payload["per_site_mae_analog"]["nyc"],
    )
    logger.info(
        "Vancouver MAE global=%.4f analog=%.4f",
        payload["per_site_mae_global"]["vancouver"],
        payload["per_site_mae_analog"]["vancouver"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
