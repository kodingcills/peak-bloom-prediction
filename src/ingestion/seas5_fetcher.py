"""SEAS5 ensemble ingestion via CDS API."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import xarray as xr

from config.settings import PROCESSED_DIR, SEAS5_FALLBACK_MODE

logger = logging.getLogger(__name__)


def _count_members(dataset: xr.Dataset) -> int:
    for dim in ("number", "member", "ensemble"):
        if dim in dataset.dims:
            return int(dataset.dims[dim])
    raise AssertionError("TODO: AUDIT — Unable to determine SEAS5 ensemble dimension")


def fetch_seas5(output_path: Path | None = None, force: bool = False) -> Path | None:
    """Fetch SEAS5 2026 ensemble forecast via CDS API."""

    output_path = output_path or (PROCESSED_DIR / "seas5_2026.nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Skipping existing SEAS5 file: %s", output_path)
        return output_path

    if SEAS5_FALLBACK_MODE:
        logger.warning("SEAS5_FALLBACK_MODE is true; skipping SEAS5 download")
        return None

    import cdsapi  # local import to avoid dependency when offline

    client = cdsapi.Client()
    request = {
        "product_type": "monthly_mean",
        "variable": "2m_temperature",
        "year": "2026",
        "month": "02",
        "leadtime_month": ["1", "2", "3"],
        "system": "51",
        "area": [50, -125, 24, 140],
        "format": "netcdf",
    }

    for attempt in range(1, 4):
        try:
            logger.info("Submitting SEAS5 request (attempt %s/3)", attempt)
            # TODO: AUDIT — verify SEAS5 product name and parameters
            client.retrieve(
                "seasonal-original-single-levels",
                request,
                str(output_path),
            )
            break
        except Exception as exc:  # pragma: no cover - external dependency
            logger.exception("SEAS5 fetch failed: %s", exc)
            if attempt == 3:
                flag_path = output_path.parent / "SEAS5_FETCH_FAILED"
                flag_path.write_text("SEAS5 fetch failed after 3 attempts\n")
                return None
            time.sleep([1, 5, 15][attempt - 1])

    dataset = xr.open_dataset(output_path)
    members = _count_members(dataset)
    if members != 50:
        raise AssertionError(f"SEAS5 ensemble members expected 50, got {members}")
    return output_path
