import pandas as pd
import os
import logging

# --- Configuration ---
ANOMALY_CSV_PATH = "data/external/seasonal_anomalies_2026.csv"

def load_local_anomalies():
    """Loads monthly anomalies from the local CSV file."""
    if not os.path.exists(ANOMALY_CSV_PATH):
        return None
    try:
        df = pd.read_csv(ANOMALY_CSV_PATH)
        # Required columns check
        required = ['site', 'year', 'month', 'tavg_anom_c', 'tmax_anom_c', 'tmin_anom_c']
        for col in required:
            if col not in df.columns:
                logging.warning(f"Missing required column {col} in {ANOMALY_CSV_PATH}")
                return None
        return df
    except Exception as e:
        logging.error(f"Error reading {ANOMALY_CSV_PATH}: {e}")
        return None

class AnomalyEngine:
    def __init__(self, allow_fallback=False):
        self.anomalies_df = load_local_anomalies()
        self.allow_fallback = allow_fallback
        self.fallback_triggered = False

        if self.anomalies_df is None:
            if not self.allow_fallback:
                raise FileNotFoundError(
                    f"Seasonal anomaly file {ANOMALY_CSV_PATH} not found. "
                    "Provide this file or set ALLOW_CLIMATOLOGY_FALLBACK=True in config."
                )
            else:
                logging.warning("Seasonal anomaly file not found. Falling back to deterministic climatology.")
                self.fallback_triggered = True

    def get_monthly_anomaly(self, site, year, month, variable):
        """
        Returns the anomaly value for a given site, year, month, and variable.
        Variable must be one of: 'TAVG', 'TMAX', 'TMIN'.
        """
        if self.anomalies_df is None:
            return 0.0

        # Map variable names to CSV columns
        var_map = {
            'TAVG': 'tavg_anom_c',
            'TMAX': 'tmax_anom_c',
            'TMIN': 'tmin_anom_c'
        }
        col = var_map.get(variable)
        if not col:
            return 0.0

        mask = (
            (self.anomalies_df['site'] == site) &
            (self.anomalies_df['year'] == year) &
            (self.anomalies_df['month'] == month)
        )
        match = self.anomalies_df[mask]
        if match.empty:
            # If no match for specific site/year/month, return 0.0 but don't fail yet
            return 0.0
        
        return float(match[col].iloc[0])

def download_cds_seasonal_anomalies(sites_metadata, api_key=None):
    """
    Optional: Implementation placeholder for Copernicus CDS seasonal anomalies.
    Requires 'cdsapi' and valid credentials.
    """
    # This would use sites_metadata (lat/lon) to extract nearest grid points.
    # For now, we prioritize the local CSV as per P0-B.1.
    pass
