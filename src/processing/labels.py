"""Label loaders for competition and supplementary datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import RAW_GMU_DIR

logger = logging.getLogger(__name__)


LOCATION_TO_SITE = {
    "washingtondc": "washingtondc",
    "kyoto": "kyoto",
    "liestal": "liestal",
    "vancouver": "vancouver",
    "newyorkcity": "nyc",
}


def load_competition_labels(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate GMU competition labels."""

    raw_root = raw_dir or RAW_GMU_DIR
    files = [
        "washingtondc.csv",
        "kyoto.csv",
        "liestal.csv",
        "vancouver.csv",
        "nyc.csv",
    ]
    frames = []
    for name in files:
        path = raw_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing label file: {path}")
        frames.append(pd.read_csv(path))

    labels = pd.concat(frames, ignore_index=True)
    labels["bloom_date"] = pd.to_datetime(labels["bloom_date"], utc=True)
    labels["bloom_doy_check"] = labels["bloom_date"].dt.dayofyear
    mismatch = labels["bloom_doy_check"] != labels["bloom_doy"]
    if mismatch.any():
        bad = labels.loc[mismatch, ["location", "year", "bloom_date", "bloom_doy"]]
        raise AssertionError(f"Bloom DOY mismatch in labels:\n{bad.head(10)}")
    labels = labels.drop(columns=["bloom_doy_check"])
    labels["site_key"] = labels["location"].map(LOCATION_TO_SITE)
    if labels["site_key"].isna().any():
        missing = labels.loc[labels["site_key"].isna(), "location"].unique()
        raise AssertionError(f"Unknown location values in labels: {missing}")
    return labels


def load_supplementary_labels(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load supplementary labels for transfer learning (not used in Phase 1)."""

    raw_root = raw_dir or RAW_GMU_DIR
    files = ["japan.csv", "meteoswiss.csv", "south_korea.csv"]
    frames = []
    for name in files:
        path = raw_root / name
        if not path.exists():
            logger.warning("Supplementary label file missing: %s", path)
            continue
        frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    labels = pd.concat(frames, ignore_index=True)
    labels["bloom_date"] = pd.to_datetime(labels["bloom_date"], utc=True, errors="coerce")
    labels["is_supplementary"] = True
    return labels
