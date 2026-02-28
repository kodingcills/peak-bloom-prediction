"""ERA5-Land (Open-Meteo) ingestion utilities."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from config.settings import (
    COMPETITION_YEAR,
    ERA5_CHUNK_YEARS,
    INFERENCE_CUTOFF_DOY,
    OPEN_METEO_ENDPOINT,
    OPENMETEO_DELAY_SECONDS,
    SILVER_WEATHER_DIR,
    SITES,
    SiteConfig,
)

logger = logging.getLogger(__name__)


def _iter_year_chunks(start_year: int) -> Iterable[tuple[int, int]]:
    year = start_year
    while year < COMPETITION_YEAR:
        end_year = min(year + ERA5_CHUNK_YEARS - 1, COMPETITION_YEAR - 1)
        yield year, end_year
        year = end_year + 1
    yield COMPETITION_YEAR, COMPETITION_YEAR


def _chunk_end_date(end_year: int) -> date:
    if end_year == COMPETITION_YEAR:
        return date(COMPETITION_YEAR, 2, 28)
    return date(end_year, 12, 31)


def _fetch_json(url: str, params: dict) -> dict:
    for attempt in range(1, 4):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            wait = [1, 5, 15][attempt - 1]
            logger.warning(
                "Open-Meteo request failed (attempt %s/3): %s", attempt, exc
            )
            if attempt == 3:
                raise
            time.sleep(wait)
    raise RuntimeError("Open-Meteo retries exhausted")


def fetch_era5_site(
    site_key: str, site_config: SiteConfig, output_dir: Path | None = None, force: bool = False
) -> Path:
    """Fetch ERA5-Land hourly data for a single site.

    Args:
        site_key: Site key identifier.
        site_config: SiteConfig definition.
        output_dir: Output root directory.
        force: Overwrite existing files if True.

    Returns:
        Path to the site directory containing chunk and consolidated files.
    """

    output_root = output_dir or SILVER_WEATHER_DIR
    site_dir = output_root / site_key
    site_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    for start_year, end_year in _iter_year_chunks(site_config.era5_start):
        end_date = _chunk_end_date(end_year)
        start_date = date(start_year, 1, 1)
        chunk_path = site_dir / f"{site_key}_{start_year}_{end_year}.parquet"
        chunk_paths.append(chunk_path)

        if chunk_path.exists() and not force:
            logger.info("Skipping existing ERA5 chunk: %s", chunk_path)
            continue

        params = {
            "latitude": site_config.lat,
            "longitude": site_config.lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,soil_temperature_0_to_7cm",
            "timezone": "UTC",
        }
        logger.info(
            "Fetching ERA5-Land %s %s-%s",
            site_key,
            start_year,
            end_year,
        )
        payload = _fetch_json(OPEN_METEO_ENDPOINT, params)
        if "hourly" not in payload:
            raise ValueError(f"Missing hourly data in response for {site_key}")
        hourly = payload["hourly"]
        df = pd.DataFrame(hourly)
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df.to_parquet(chunk_path, index=False)
        time.sleep(OPENMETEO_DELAY_SECONDS)

    logger.info("Consolidating ERA5-Land chunks for %s", site_key)
    frames = []
    for path in chunk_paths:
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No ERA5 chunks found for {site_key}")

    consolidated_path = site_dir / f"{site_key}_consolidated.parquet"
    if consolidated_path.exists() and not force:
        logger.info("Skipping existing consolidated ERA5 file: %s", consolidated_path)
        return site_dir

    consolidated = pd.concat(frames, ignore_index=True)
    consolidated = consolidated.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    consolidated.to_parquet(consolidated_path, index=False)
    return site_dir


def fetch_all_era5(
    output_dir: Path | None = None,
    force: bool = False,
    site_keys: Iterable[str] | None = None,
) -> dict[str, dict]:
    """Fetch ERA5-Land data for all sites."""

    status: dict[str, dict] = {}
    keys = list(site_keys) if site_keys is not None else list(SITES.keys())
    for site_key in keys:
        site = SITES[site_key]
        try:
            fetch_era5_site(site_key, site, output_dir=output_dir, force=force)
            status[site_key] = {"ok": True}
        except Exception as exc:  # pragma: no cover - logging error
            logger.exception("ERA5-Land fetch failed for %s: %s", site_key, exc)
            status[site_key] = {"ok": False, "error": str(exc)}
    return status
