"""ASOS ingestion utilities (Iowa Mesonet)."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config.settings import ASOS_ENDPOINT, COMPETITION_YEAR, SILVER_ASOS_DIR, SITES

logger = logging.getLogger(__name__)


def fetch_asos_station(
    station_id: str,
    start_year: int,
    end_year: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch ASOS hourly data for a single station."""

    output_root = output_dir or SILVER_ASOS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{station_id}.parquet"
    if output_path.exists() and not force:
        logger.info("Skipping existing ASOS file: %s", output_path)
        return output_path

    params = {
        "station": station_id,
        "data": "tmpf,dwpf,relh",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "yes",
        "elev": "yes",
        "year1": start_year,
        "month1": 1,
        "day1": 1,
        "year2": end_year,
        "month2": 2,
        "day2": 28,
        "missing": "M",
        "trace": "T",
        "report_type": "3",
    }

    logger.info("Fetching ASOS station %s", station_id)
    response = requests.get(ASOS_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))

    if "valid" not in df.columns:
        raise ValueError(f"ASOS response missing 'valid' column for {station_id}")

    df = df.replace("M", pd.NA)
    df["temperature_2m"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["temperature_2m"])
    df["temperature_2m"] = (df["temperature_2m"] - 32.0) * 5.0 / 9.0
    df = df.rename(columns={"valid": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df.to_parquet(output_path, index=False)
    return output_path


def fetch_all_asos(
    output_dir: Path | None = None,
    force: bool = False,
    site_keys: list[str] | None = None,
) -> dict[str, dict]:
    """Fetch ASOS data for all stations referenced by sites."""

    status: dict[str, dict] = {}
    sites = (
        {key: SITES[key] for key in site_keys} if site_keys is not None else SITES
    )
    stations = sorted({st for site in sites.values() for st in site.asos_stations})
    for station in stations:
        try:
            fetch_asos_station(
                station_id=station,
                start_year=2000,
                end_year=COMPETITION_YEAR,
                output_dir=output_dir,
                force=force,
            )
            status[station] = {"ok": True}
        except Exception as exc:  # pragma: no cover - logging error
            logger.exception("ASOS fetch failed for %s: %s", station, exc)
            status[station] = {"ok": False, "error": str(exc)}
    return status
