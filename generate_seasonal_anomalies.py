"""
generate_seasonal_anomalies.py

Generates data/external/seasonal_anomalies_2026.csv by:
  1. Computing 1991-2020 site-specific TAVG/TMAX/TMIN normals from features_train.csv
  2. Fetching ECMWF SEAS5 ensemble mean forecast (March-May 2026) from Open-Meteo seasonal API
  3. Computing anomaly = forecast_mean - normal
  4. Writing the result in the exact format expected by seasonal_anomaly_engine.py

Usage:
    python3 generate_seasonal_anomalies.py

Output:
    data/external/seasonal_anomalies_2026.csv
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ── Site definitions ──────────────────────────────────────────────────────────
# Keys must match the 'site' column in features_train.csv
SITE_COORDS = {
    'washingtondc': {'lat': 38.9072,  'lon': -77.0369},
    'kyoto':        {'lat': 35.0116,  'lon': 135.7681},
    'liestal':      {'lat': 47.4814,  'lon':   7.7344},
    'vancouver':    {'lat': 49.2827,  'lon': -123.1207},
    'newyorkcity':  {'lat': 40.7128,  'lon':  -74.0060},
}

FORECAST_MONTHS = [3, 4, 5]   # March, April, May
FORECAST_YEAR   = 2026
NORMAL_START    = 1991
NORMAL_END      = 2020
FEATURES_PATH   = 'features_train.csv'
OUTPUT_PATH     = 'data/external/seasonal_anomalies_2026.csv'


# ── Step 1: Compute normals from training data ────────────────────────────────
def compute_normals(features_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        site, month, tavg_normal, tmax_normal, tmin_normal
    computed from the 1991-2020 climatological baseline.
    """
    print(f"Loading {features_path} ...")
    df = pd.read_csv(features_path)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    mask = (
        (df['year']  >= NORMAL_START) &
        (df['year']  <= NORMAL_END)   &
        (df['month'].isin(FORECAST_MONTHS))
    )
    normals = (
        df[mask]
        .groupby(['site', 'month'])[['TAVG', 'TMAX', 'TMIN']]
        .mean()
        .rename(columns={'TAVG': 'tavg_normal',
                         'TMAX': 'tmax_normal',
                         'TMIN': 'tmin_normal'})
        .reset_index()
    )
    print("1991-2020 normals computed:")
    print(normals.to_string(index=False))
    print()
    return normals


# ── Step 2: Fetch SEAS5 ensemble mean from Open-Meteo ────────────────────────
def fetch_seas5_forecast(site: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches daily ensemble mean temperature for March-May 2026
    from Open-Meteo seasonal API and returns monthly means as a DataFrame:
        month, tavg_forecast, tmax_forecast, tmin_forecast
    """
    # URL without models param works reliably
    url = (
        "https://seasonal-api.open-meteo.com/v1/seasonal"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        "&timezone=auto"
    )

    print(f"  Fetching forecast for {site} ({lat}, {lon}) ...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  WARNING: Fetch failed for {site}: {e}")
        return None

    daily = data.get('daily', {})
    if not daily or 'time' not in daily:
        print(f"  WARNING: No daily data returned for {site}")
        return None

    df = pd.DataFrame({
        'date': pd.to_datetime(daily['time']),
        'tavg_forecast': daily.get('temperature_2m_mean'),
        'tmax_forecast': daily.get('temperature_2m_max'),
        'tmin_forecast': daily.get('temperature_2m_min'),
    })
    
    # Filter for March-May 2026
    df = df[(df['date'] >= f'{FORECAST_YEAR}-03-01') & 
            (df['date'] <= f'{FORECAST_YEAR}-05-31')]
    
    if df.empty or df['tavg_forecast'].isna().all():
        print(f"  WARNING: No valid MAM {FORECAST_YEAR} data for {site}")
        return None

    df['month'] = df['date'].dt.month

    monthly = (
        df.groupby('month')[['tavg_forecast', 'tmax_forecast', 'tmin_forecast']]
        .mean()
        .reset_index()
    )
    return monthly


# ── Step 3: Compute anomalies ─────────────────────────────────────────────────
def build_anomaly_csv(normals: pd.DataFrame) -> pd.DataFrame:
    """
    For each site, fetches the SEAS5 forecast and computes:
        anomaly = forecast_mean - 1991-2020 normal
    Returns a DataFrame ready to write as seasonal_anomalies_2026.csv.
    """
    rows = []

    for site, coords in SITE_COORDS.items():
        site_normals = normals[normals['site'] == site].set_index('month')
        forecast_df  = fetch_seas5_forecast(site, coords['lat'], coords['lon'])

        for month in FORECAST_MONTHS:
            if month not in site_normals.index:
                print(f"  WARNING: No normal for {site} month {month}, skipping.")
                continue

            norm_row = site_normals.loc[month]

            if forecast_df is not None and month in forecast_df['month'].values:
                fc_row = forecast_df[forecast_df['month'] == month].iloc[0]
                tavg_anom = round(fc_row['tavg_forecast'] - norm_row['tavg_normal'], 2)
                tmax_anom = round(fc_row['tmax_forecast'] - norm_row['tmax_normal'], 2)
                tmin_anom = round(fc_row['tmin_forecast'] - norm_row['tmin_normal'], 2)
                source = 'SEAS5'
            else:
                print(f"  FALLBACK: Using 0.0 anomaly for {site} month {month}")
                tavg_anom = tmax_anom = tmin_anom = 0.0
                source = 'fallback_zero'

            rows.append({
                'site':          site,
                'year':          FORECAST_YEAR,
                'month':         month,
                'tavg_anom_c':   tavg_anom,
                'tmax_anom_c':   tmax_anom,
                'tmin_anom_c':   tmin_anom,
                'source':        source,
            })
            print(f"    {site} month={month}: "
                  f"TAVG_anom={tavg_anom:+.2f}  "
                  f"TMAX_anom={tmax_anom:+.2f}  "
                  f"TMIN_anom={tmin_anom:+.2f}  [{source}]")

    return pd.DataFrame(rows)


# ── Step 4: Write output ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("generate_seasonal_anomalies.py")
    print(f"Generating {OUTPUT_PATH}")
    print(f"Normals baseline: {NORMAL_START}-{NORMAL_END}")
    print(f"Forecast: ECMWF SEAS5 via Open-Meteo seasonal API")
    print("=" * 60)
    print()

    # Step 1
    normals = compute_normals(FEATURES_PATH)

    # Steps 2 + 3
    print("Fetching SEAS5 forecasts and computing anomalies ...")
    print()
    anomaly_df = build_anomaly_csv(normals)

    # Step 4
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Drop the 'source' column before writing — engine doesn't need it
    out_df = anomaly_df[['site', 'year', 'month',
                          'tavg_anom_c', 'tmax_anom_c', 'tmin_anom_c']]
    out_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 60)
    print(f"Written: {OUTPUT_PATH}")
    print()
    print("Final anomaly table:")
    print(anomaly_df[['site', 'month', 'tavg_anom_c',
                       'tmax_anom_c', 'tmin_anom_c', 'source']].to_string(index=False))
    print()

    # Sanity check: flag any anomaly > 3°C or < -3°C as suspicious
    suspicious = anomaly_df[anomaly_df['tavg_anom_c'].abs() > 3.0]
    if not suspicious.empty:
        print("WARNING: The following anomalies exceed ±3°C — verify these:")
        print(suspicious[['site', 'month', 'tavg_anom_c']].to_string(index=False))
    else:
        print("Sanity check passed: all anomalies within ±3°C.")
    print("=" * 60)


if __name__ == "__main__":
    main()