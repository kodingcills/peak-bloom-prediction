"""Phase 3 predictor: ensemble -> GMM -> shrinkage -> submission."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import COMPETITION_YEAR, GOLD_DIR, PROCESSED_DIR, SEAS5_NEUTRAL_THRESHOLD, SITES
from src.modeling.analog_shrinkage import compute_prior_mean
from src.modeling.empirical_bayes import apply_shrinkage
from src.modeling.gmm_selector import fit_bimodal
from src.modeling.seas5_processor import build_fallback_ensemble, process_seas5_ensemble

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _load_phase2_artifacts(processed_dir: Path) -> dict[str, Any]:
    return {
        "model_coefficients": _load_json(processed_dir / "model_coefficients.json"),
        "bias_coefficients": _load_json(processed_dir / "bias_coefficients.json"),
        "shrinkage_weights": _load_json(processed_dir / "shrinkage_weights.json"),
        "mean_bloom_doy": _load_json(processed_dir / "mean_bloom_doy.json"),
        "global_mean": _load_json(processed_dir / "global_mean.json"),
    }


def _load_weather_cache() -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for site_key in sorted(SITES.keys()):
        path = Path("data/silver/weather") / site_key / f"{site_key}_consolidated.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing weather file for {site_key}: {path}")
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cache[site_key] = df
    return cache


def _resolve_weight(shrinkage_weights: dict[str, Any], site_key: str) -> float:
    # Phase 2 artifact schema currently nests site payload under "sites".
    if "sites" in shrinkage_weights:
        site_payload = shrinkage_weights["sites"].get(site_key)
        if site_payload is None:
            raise KeyError(f"Missing shrinkage weight for site {site_key}")
        return float(site_payload["w"])
    site_payload = shrinkage_weights.get(site_key)
    if site_payload is None:
        raise KeyError(f"Missing shrinkage weight for site {site_key}")
    return float(site_payload["w"])


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _build_prediction_chain(
    ensemble_results: dict[str, dict[str, Any]],
    shrinkage_weights: dict[str, Any],
    mean_bloom_doy: dict[str, float],
    global_mean: float,
    shrinkage_mode: str = "global",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    gmm_results: dict[str, Any] = {}
    prediction_summary: dict[str, Any] = {}
    mu_selected_by_site: dict[str, float] = {}
    weights_by_site: dict[str, float] = {}

    for site_key in sorted(SITES.keys()):
        ens = ensemble_results[site_key]
        preds = np.asarray(ens["predictions"], dtype=float)
        if preds.size < 5:
            raise RuntimeError(
                f"{site_key}: only {preds.size} valid fallback members (<5)"
            )

        gmm_result = fit_bimodal(
            bloom_predictions=preds,
            site_key=site_key,
            climatological_mean=float(mean_bloom_doy[site_key]),
            neutral_threshold=float(SEAS5_NEUTRAL_THRESHOLD),
        )
        gmm_results[site_key] = gmm_result
        mu_selected_by_site[site_key] = float(gmm_result["selected_mean"])
        weights_by_site[site_key] = _resolve_weight(shrinkage_weights, site_key)

    for site_key in sorted(SITES.keys()):
        ens = ensemble_results[site_key]
        preds = np.asarray(ens["predictions"], dtype=float)
        model_pred = float(mu_selected_by_site[site_key])
        w = float(weights_by_site[site_key])
        prior_payload = compute_prior_mean(
            site_key=site_key,
            shrinkage_mode=shrinkage_mode,
            global_mean=float(global_mean),
            mean_bloom_doy=mean_bloom_doy,
            mu_selected_by_site=mu_selected_by_site,
        )
        prior_mean = float(prior_payload["prior_mean"])
        shrunk = apply_shrinkage(model_pred, prior_mean, w)

        final_doy = int(round(shrunk))
        final_doy = max(60, min(140, final_doy))
        location = SITES[site_key].loc_id

        records.append({"location": location, "year": COMPETITION_YEAR, "bloom_doy": final_doy})

        final_date = datetime(COMPETITION_YEAR, 1, 1) + timedelta(days=final_doy - 1)
        prediction_summary[site_key] = {
            "ensemble_mean": float(np.mean(preds)),
            "gmm_k": int(gmm_results[site_key]["k"]),
            "gmm_selected_mean": model_pred,
            "shrinkage_weight": w,
            "global_mean": float(global_mean),
            "shrunk_prediction": float(shrunk),
            "shrinkage_mode": shrinkage_mode,
            "prior_mean": prior_mean,
            "prior_source": prior_payload["prior_source"],
            "analog_site": prior_payload["analog_site"],
            "delta_days": prior_payload["delta_days"],
            "pre_shrinkage": model_pred,
            "post_shrinkage": float(shrunk),
            "final_doy": final_doy,
            "final_date": final_date.strftime("%Y-%m-%d"),
        }

    predictions_df = pd.DataFrame.from_records(records)
    predictions_df = predictions_df.sort_values("location").reset_index(drop=True)
    return predictions_df, gmm_results, prediction_summary


def generate_predictions(
    ensemble_results: dict[str, dict[str, Any]],
    shrinkage_weights: dict[str, Any],
    mean_bloom_doy: dict[str, float],
    global_mean: float,
    shrinkage_mode: str = "global",
) -> pd.DataFrame:
    """Generate final 5 bloom DOY predictions from ensemble results.

    Args:
        ensemble_results: Ensemble prediction payload per site.
        shrinkage_weights: Shrinkage weights artifact.
        mean_bloom_doy: Site climatological means.
        global_mean: Global mean bloom DOY.

    Returns:
        Submission-shaped prediction DataFrame.
    """

    predictions_df, _, _ = _build_prediction_chain(
        ensemble_results=ensemble_results,
        shrinkage_weights=shrinkage_weights,
        mean_bloom_doy=mean_bloom_doy,
        global_mean=global_mean,
        shrinkage_mode=shrinkage_mode,
    )
    return predictions_df


def save_submission(predictions_df: pd.DataFrame, path: str = "submission.csv") -> Path:
    """Validate and write submission.csv.

    Args:
        predictions_df: Predictions DataFrame.
        path: Output path.

    Returns:
        Path to written submission.
    """

    df = predictions_df.copy()
    expected_cols = ["location", "year", "bloom_doy"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Wrong columns: {list(df.columns)}; expected {expected_cols}")
    if len(df) != 5:
        raise ValueError(f"Expected 5 rows, got {len(df)}")
    if not (df["year"] == COMPETITION_YEAR).all():
        raise ValueError("Not all rows are year 2026")

    if not all(isinstance(x, (int, np.integer)) for x in df["bloom_doy"].tolist()):
        raise ValueError(f"Non-integer bloom_doy values: {df['bloom_doy'].tolist()}")

    if not ((df["bloom_doy"] >= 60) & (df["bloom_doy"] <= 140)).all():
        raise ValueError("Found bloom_doy outside [60, 140]")

    expected_locations = {"washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"}
    actual_locations = set(df["location"].tolist())
    if actual_locations != expected_locations:
        raise ValueError(
            "Location mismatch: "
            f"missing={sorted(expected_locations - actual_locations)}, "
            f"extra={sorted(actual_locations - expected_locations)}"
        )

    df = df.sort_values("location").reset_index(drop=True)
    out = Path(path)
    df.to_csv(out, index=False)
    return out


def save_diagnostics(
    ensemble_results: dict[str, dict[str, Any]],
    gmm_results: dict[str, Any],
    prediction_summary: dict[str, Any],
    diagnostics_dir: Path,
) -> None:
    """Save Phase 3 diagnostics artifacts.

    Args:
        ensemble_results: Per-site ensemble output.
        gmm_results: Per-site GMM selection output.
        prediction_summary: Per-site prediction chain summary.
        diagnostics_dir: Target diagnostics directory.
    """

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    ensemble_payload: dict[str, Any] = {}
    for site_key in sorted(ensemble_results.keys()):
        preds = np.asarray(ensemble_results[site_key]["predictions"], dtype=float)
        wv_vals = np.asarray(ensemble_results[site_key]["wv_values"], dtype=float)
        ensemble_payload[site_key] = {
            "n_members": int(preds.size),
            "predictions": preds.astype(float).tolist(),
            "wv_values": wv_vals.astype(float).tolist(),
            "scenario_years": [int(y) for y in ensemble_results[site_key]["scenario_years"]],
            "skipped_members": ensemble_results[site_key].get("skipped_members", []),
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds, ddof=0)),
            "gdh_2026": float(ensemble_results[site_key]["gdh_2026"]),
            "cp_2026": float(ensemble_results[site_key]["cp_2026"]),
            "percentiles": _percentiles(preds),
        }

    (diagnostics_dir / "ensemble_distributions.json").write_text(
        json.dumps(ensemble_payload, indent=2, sort_keys=True)
    )
    (diagnostics_dir / "gmm_results.json").write_text(
        json.dumps(gmm_results, indent=2, sort_keys=True)
    )
    (diagnostics_dir / "prediction_summary.json").write_text(
        json.dumps(prediction_summary, indent=2, sort_keys=True)
    )


def _resolve_shrinkage_mode(cli_mode: str) -> str:
    """Resolve shrinkage mode from CLI arg with env override."""

    env_mode = os.getenv("SHRINKAGE_MODE")
    if env_mode:
        mode = env_mode.strip().lower()
    else:
        mode = cli_mode.strip().lower()

    if mode not in {"global", "analog"}:
        raise ValueError(
            f"Invalid shrinkage mode: {mode}. Expected one of ['global', 'analog']."
        )
    return mode


def main() -> None:
    """Phase 3 entrypoint for fallback ensemble inference and submission."""

    parser = argparse.ArgumentParser(description="Run Phase 3 predictor.")
    parser.add_argument(
        "--shrinkage-mode",
        choices=["global", "analog"],
        default="global",
        help="Shrinkage prior mode (default: global).",
    )
    args = parser.parse_args()
    shrinkage_mode = _resolve_shrinkage_mode(args.shrinkage_mode)
    logger.info("shrinkage_mode=%s", shrinkage_mode)

    artifacts = _load_phase2_artifacts(PROCESSED_DIR)
    global_mean = float(artifacts["global_mean"]["global_mean"])

    gold = pd.read_parquet(GOLD_DIR / "features.parquet")
    gold_2026 = gold.loc[gold["year"] == COMPETITION_YEAR].copy()

    weather_cache = _load_weather_cache()

    seas5_path = PROCESSED_DIR / "seas5_2026.nc"
    fallback_flag = PROCESSED_DIR / "SEAS5_FETCH_FAILED"

    if seas5_path.exists() and not fallback_flag.exists():
        logger.info("SEAS5 NetCDF found; attempting real ensemble processing")
        ensemble_results = process_seas5_ensemble(seas5_path)
    else:
        logger.info("Using fallback ensemble mode")
        ensemble_results = build_fallback_ensemble(
            weather_cache=weather_cache,
            gold_2026=gold_2026,
            model_coefficients=artifacts["model_coefficients"],
            bias_coefficients=artifacts["bias_coefficients"],
            mean_bloom_doy=artifacts["mean_bloom_doy"],
            sites_config=SITES,
            fallback_years=30,
        )

    predictions_df, gmm_results, prediction_summary = _build_prediction_chain(
        ensemble_results=ensemble_results,
        shrinkage_weights=artifacts["shrinkage_weights"],
        mean_bloom_doy=artifacts["mean_bloom_doy"],
        global_mean=global_mean,
        shrinkage_mode=shrinkage_mode,
    )

    submission_path = save_submission(predictions_df, path="submission.csv")
    save_diagnostics(
        ensemble_results=ensemble_results,
        gmm_results=gmm_results,
        prediction_summary=prediction_summary,
        diagnostics_dir=PROCESSED_DIR / "diagnostics",
    )

    if all(int(v["k"]) == 2 for v in gmm_results.values()):
        logger.warning("All 5 sites selected k=2; this is unusual and should be reviewed")

    for site_key in ["vancouver", "nyc"]:
        if site_key in prediction_summary:
            delta = abs(
                float(prediction_summary[site_key]["gmm_selected_mean"])
                - float(prediction_summary[site_key]["shrunk_prediction"])
            )
            if delta <= 0.01:
                logger.warning("%s shrinkage effect is near zero (Δ=%.4f)", site_key, delta)

    logger.info("Submission written: %s", submission_path)
    for site_key in sorted(SITES.keys()):
        loc = SITES[site_key].name
        loc_id = SITES[site_key].loc_id
        row = predictions_df.loc[predictions_df["location"] == loc_id].iloc[0]
        doy = int(row["bloom_doy"])
        date = datetime(COMPETITION_YEAR, 1, 1) + timedelta(days=doy - 1)
        logger.info("%s -> DOY %3d (%s)", f"{loc:25s}", doy, date.strftime("%b %d"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
