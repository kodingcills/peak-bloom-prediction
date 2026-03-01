"""Fold-safe warming velocity feature utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_warming_velocity(
    daily_means_by_year: dict[int, pd.Series],
    year: int,
    window_center_doy: int,
    beta0: float = 0.0,
    beta1: float = 1.0,
    window_days: int = 14,
) -> dict[str, Any]:
    """Compute warming velocity (daily temperature slope) for a site-year.

    Args:
        daily_means_by_year: Mapping of year to daily mean temperature series indexed by DOY.
        year: Target year.
        window_center_doy: Fold-safe center day-of-year.
        beta0: Bias correction intercept.
        beta1: Bias correction slope.
        window_days: Lookback window size in days.

    Returns:
        Dictionary with keys: wv, n_days, r2, uses_forecast, window_start_doy, window_end_doy.
    """

    series = daily_means_by_year.get(int(year))
    window_end = int(window_center_doy)
    window_start = int(window_center_doy) - int(window_days)

    if series is None or series.empty:
        return {
            "wv": float("nan"),
            "n_days": 0,
            "r2": 0.0,
            "uses_forecast": False,
            "window_start_doy": window_start,
            "window_end_doy": window_end,
        }

    values = series.copy()
    values.index = pd.Index(values.index.astype(int), name="doy")
    window = values.loc[(values.index >= window_start) & (values.index <= window_end)]

    if window.empty:
        return {
            "wv": float("nan"),
            "n_days": 0,
            "r2": 0.0,
            "uses_forecast": False,
            "window_start_doy": window_start,
            "window_end_doy": window_end,
        }

    window = pd.to_numeric(window, errors="coerce").dropna()
    window = beta0 + (beta1 * window)

    n_days = int(len(window))
    if n_days < 3:
        return {
            "wv": float("nan"),
            "n_days": n_days,
            "r2": 0.0,
            "uses_forecast": False,
            "window_start_doy": window_start,
            "window_end_doy": window_end,
        }

    x = window.index.to_numpy(dtype=float)
    y = window.to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ coef
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    return {
        "wv": float(coef[1]),
        "n_days": n_days,
        "r2": r2,
        "uses_forecast": False,
        "window_start_doy": window_start,
        "window_end_doy": window_end,
    }
