"""ERA5-Land (Open-Meteo) ingestion utilities."""

from __future__ import annotations

import logging
import random
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
    OPENMETEO_429_MAX_SLEEP_SECONDS,
    OPENMETEO_429_MIN_SLEEP_SECONDS,
    OPENMETEO_DELAY_SECONDS,
    OPENMETEO_MAX_ATTEMPTS,
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


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _fetch_json(url: str, params: dict) -> tuple[dict, bool]:
    last_exception: Exception | None = None
    had_429 = False
    chunk_label = f"{params.get('start_date')}..{params.get('end_date')}"
    for attempt in range(1, OPENMETEO_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 429:
                had_429 = True
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_for = _clamp(
                            int(retry_after) + random.uniform(0, 3),
                            OPENMETEO_429_MIN_SLEEP_SECONDS,
                            OPENMETEO_429_MAX_SLEEP_SECONDS,
                        )
                        logger.warning(
                            "429 rate-limited; sleeping %.2fs; attempt %s/%s; chunk %s. "
                            "Retry-After header honored (%ss).",
                            sleep_for,
                            attempt,
                            OPENMETEO_MAX_ATTEMPTS,
                            chunk_label,
                            retry_after,
                        )
                    except ValueError:
                        sleep_for = _clamp(
                            (2**attempt) * 5 + random.uniform(0, 3),
                            OPENMETEO_429_MIN_SLEEP_SECONDS,
                            OPENMETEO_429_MAX_SLEEP_SECONDS,
                        )
                        logger.warning(
                            "429 rate-limited; sleeping %.2fs; attempt %s/%s; chunk %s. "
                            "Retry-After header invalid (%s).",
                            sleep_for,
                            attempt,
                            OPENMETEO_MAX_ATTEMPTS,
                            chunk_label,
                            retry_after,
                        )
                else:
                    sleep_for = _clamp(
                        (2**attempt) * 5 + random.uniform(0, 3),
                        OPENMETEO_429_MIN_SLEEP_SECONDS,
                        OPENMETEO_429_MAX_SLEEP_SECONDS,
                    )
                    logger.warning(
                        "429 rate-limited; sleeping %.2fs; attempt %s/%s; chunk %s. "
                        "Retry-After header not present.",
                        sleep_for,
                        attempt,
                        OPENMETEO_MAX_ATTEMPTS,
                        chunk_label,
                    )
                if attempt == OPENMETEO_MAX_ATTEMPTS:
                    response.raise_for_status()
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            return response.json(), had_429
        except requests.RequestException as exc:
            last_exception = exc
            sleep_for = _clamp((2**attempt) + random.uniform(0, 0.5), 0.0, 60.0)
            logger.warning(
                "Open-Meteo request failed (attempt %s/%s): %s. "
                "Sleeping %.2fs before retry; chunk %s.",
                attempt,
                OPENMETEO_MAX_ATTEMPTS,
                exc,
                sleep_for,
                chunk_label,
            )
            if attempt == OPENMETEO_MAX_ATTEMPTS:
                raise
            time.sleep(sleep_for)
    if last_exception is not None:
        raise last_exception
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
    incomplete_path = site_dir / "_INCOMPLETE.txt"

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
        try:
            payload, had_429 = _fetch_json(OPEN_METEO_ENDPOINT, params)
        except Exception as exc:
            incomplete_path.write_text(
                f"site={site_key}\nchunk={start_year}-{end_year}\nerror={exc}\n"
            )
            raise
        if "hourly" not in payload:
            raise ValueError(f"Missing hourly data in response for {site_key}")
        hourly = payload["hourly"]
        df = pd.DataFrame(hourly)
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df.to_parquet(chunk_path, index=False)
        time.sleep(OPENMETEO_DELAY_SECONDS)
        if had_429:
            cooldown = OPENMETEO_DELAY_SECONDS * 2
            logger.info(
                "Applying post-429 cool-down for %s %.0f-%.0f: %.2fs",
                site_key,
                start_year,
                end_year,
                cooldown,
            )
            time.sleep(cooldown)

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
        if all(path.exists() for path in chunk_paths) and incomplete_path.exists():
            incomplete_path.unlink()
        return site_dir

    consolidated = pd.concat(frames, ignore_index=True)
    consolidated = consolidated.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    consolidated.to_parquet(consolidated_path, index=False)
    if all(path.exists() for path in chunk_paths) and incomplete_path.exists():
        incomplete_path.unlink()
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
        except Exception as exc:
            logger.exception("ERA5-Land fetch failed for %s: %s", site_key, exc)
            status[site_key] = {"ok": False, "error": str(exc)}
    return status
