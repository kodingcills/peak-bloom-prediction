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
    # Pre-1677 dates exceed pandas ns timestamp range. bloom_doy is authoritative. See IMPLEMENTATION_STATE blocker 1.
    labels["bloom_date"] = labels["bloom_date"].astype(str)
    labels["bloom_doy"] = pd.to_numeric(labels["bloom_doy"], errors="raise")
    if (labels["bloom_doy"] % 1 != 0).any():
        bad = labels.loc[labels["bloom_doy"] % 1 != 0, ["location", "year", "bloom_date", "bloom_doy"]]
        raise AssertionError(f"Non-integer bloom_doy values:\n{bad.head(10)}")
    labels["bloom_doy"] = labels["bloom_doy"].astype(int)
    if ((labels["bloom_doy"] < 1) | (labels["bloom_doy"] > 366)).any():
        bad = labels.loc[(labels["bloom_doy"] < 1) | (labels["bloom_doy"] > 366),
                         ["location", "year", "bloom_date", "bloom_doy"]]
        raise AssertionError(f"bloom_doy out of range [1, 366]:\n{bad.head(10)}")

    year_numeric = pd.to_numeric(labels["year"], errors="raise").astype(int)
    modern_mask = year_numeric >= 1677
    if modern_mask.any():
        parsed = pd.to_datetime(labels.loc[modern_mask, "bloom_date"], utc=True, errors="coerce")
        valid_mask = parsed.notna()
        if (~valid_mask).any():
            logger.warning(
                "Skipping bloom_date validation for %s rows outside pandas range",
                int((~valid_mask).sum()),
            )
        if valid_mask.any():
            doy_check = parsed.loc[valid_mask].dt.dayofyear
            mismatch = doy_check.to_numpy() != labels.loc[modern_mask].loc[valid_mask, "bloom_doy"].to_numpy()
            if mismatch.any():
                bad = labels.loc[modern_mask].loc[valid_mask].loc[mismatch, ["location", "year", "bloom_date", "bloom_doy"]]
                logger.warning(
                    "Bloom DOY mismatch for %s rows; bloom_doy remains authoritative. Sample:\n%s",
                    int(mismatch.sum()),
                    bad.head(10),
                )
    labels["year"] = year_numeric
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
