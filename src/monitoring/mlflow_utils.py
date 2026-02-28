"""MLflow utilities (local file-store only, always enabled)."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _require_mlflow():
    try:
        import mlflow  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("MLflow is required but failed to import") from exc
    return mlflow


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _configure_mlflow() -> None:
    mlflow = _require_mlflow()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "phenology_engine_v1_7")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextlib.contextmanager
def init_mlflow(run_name: str, tags: dict | None = None) -> Iterator[None]:
    """Context manager that starts and ends an MLflow run."""

    mlflow = _require_mlflow()
    _configure_mlflow()
    base_tags = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
    }
    commit = _git_commit()
    if commit:
        base_tags["git_commit"] = commit
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=run_name) as _run:
        mlflow.set_tags(base_tags)
        yield


def log_params(params: dict) -> None:
    """Log MLflow parameters."""

    mlflow = _require_mlflow()
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log MLflow metrics."""

    mlflow = _require_mlflow()
    if metrics:
        if step is None:
            mlflow.log_metrics(metrics)
        else:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)


def log_text(text: str, artifact_path: str) -> None:
    """Log text content as an MLflow artifact."""

    mlflow = _require_mlflow()
    mlflow.log_text(text, artifact_file=artifact_path)


def log_artifact(path: str | Path) -> None:
    """Log a file artifact."""

    mlflow = _require_mlflow()
    mlflow.log_artifact(str(path))
