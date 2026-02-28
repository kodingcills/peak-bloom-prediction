"""Central configuration for Phenology Engine v1.7."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SiteConfig:
    """Configuration for a single competition site."""

    name: str
    loc_id: str
    lat: float
    lon: float
    alt_m: float
    species: str
    bloom_pct: str
    era5_start: int
    asos_stations: list[str]


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_GMU_DIR = RAW_DIR / "gmu"
SILVER_WEATHER_DIR = DATA_DIR / "silver" / "weather"
SILVER_ASOS_DIR = DATA_DIR / "silver" / "asos"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"

# Temporal constants
COMPETITION_YEAR = 2026
INFERENCE_CUTOFF_DOY = 59
INFERENCE_CUTOFF_UTC = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
ERA5_CHUNK_YEARS = 10
CHILL_START_DOY = 274
WARM_START_DOY = 1

# Feature constants
GDH_BASE_C = 4.5
CP_MIN_C = -2.0
CP_OPT_C = 6.0
CP_MAX_C = 14.0

# Inference constants
SEAS5_NEUTRAL_THRESHOLD = 0.3

# API endpoints
OPEN_METEO_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
ASOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Operational settings
# TODO: AUDIT — confirm desired Open-Meteo inter-chunk delay.
OPENMETEO_DELAY_SECONDS = float(os.getenv("OPENMETEO_DELAY_SECONDS", "1.0"))


def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y"}


SEAS5_FALLBACK_MODE = _env_flag("SEAS5_FALLBACK_MODE", "false")
try:
    N_JOBS = int(os.getenv("N_JOBS", "0"))
except ValueError:
    N_JOBS = 0


SITES: dict[str, SiteConfig] = {
    "washingtondc": SiteConfig(
        name="Washington, D.C.",
        loc_id="washingtondc",
        lat=38.8853,
        lon=-77.0386,
        alt_m=0.0,
        species="P. x yedoensis",
        bloom_pct="70%",
        era5_start=1950,
        asos_stations=["DCA", "IAD"],
    ),
    "kyoto": SiteConfig(
        name="Kyoto",
        loc_id="kyoto",
        lat=35.0120,
        lon=135.6761,
        alt_m=44.0,
        species="P. jamasakura",
        bloom_pct="newspaper",
        era5_start=1950,
        asos_stations=[],
    ),
    "liestal": SiteConfig(
        name="Liestal",
        loc_id="liestal",
        lat=47.4814,
        lon=7.7305,
        alt_m=350.0,
        species="P. avium",
        bloom_pct="25%",
        era5_start=1950,
        asos_stations=[],
    ),
    "vancouver": SiteConfig(
        name="Vancouver",
        loc_id="vancouver",
        lat=49.2237,
        lon=-123.1636,
        alt_m=24.0,
        species="Yoshino",
        bloom_pct="~70%",
        era5_start=1950,
        asos_stations=["CYVR"],
    ),
    "nyc": SiteConfig(
        name="New York City",
        loc_id="newyorkcity",
        lat=40.7304,
        lon=-73.9981,
        alt_m=8.5,
        species="P. x yedoensis",
        bloom_pct="~70%",
        era5_start=1950,
        asos_stations=["JFK", "LGA"],
    ),
}
