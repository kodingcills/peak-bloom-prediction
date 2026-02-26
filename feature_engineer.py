import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants for Dynamic Chill Model (Fishman & Erez 1987)
# IMPORTANT: E0 and E1 are in cal/mol; the gas constant R must be included
# in the exponent so the units cancel correctly (cal/mol ÷ cal/(mol·K) = K).
# Original code was missing R, making exp(-E0/T) ≈ exp(-44.8) ≈ 0 always.
# ─────────────────────────────────────────────────────────────────────────────
A0 = 1.395e5          # pre-exponential factor
A1 = 2.567e18         # pre-exponential factor
E0 = 12400            # activation energy, cal/mol
E1 = 41400            # activation energy, cal/mol
R  = 1.987            # gas constant, cal/(mol·K)  ← THE MISSING CONSTANT

# Toggle to True to print per-site per-bio_year DCM diagnostics
DEBUG_DCM = False


def calculate_nyc_offset(npn_path):
    """
    Calculates the median delta between first flower (501) and peak bloom (70%+)
    for Yoshino cherries (Species 228).
    """
    if not os.path.exists(npn_path):
        return 7.0
    df = pd.read_csv(npn_path)
    df = df[df['Species_ID'] == 228]
    df['Observation_Date'] = pd.to_datetime(df['Observation_Date'])
    df['Year'] = df['Observation_Date'].dt.year

    deltas = []
    peak_categories = ['75-94%', '95% or more', 'More than 10']

    for (ind_id, year), group in df.groupby(['Individual_ID', 'Year']):
        ff_group = group[
            (group['Phenophase_ID'] == 501) & (group['Phenophase_Status'] == 1)
        ]
        if ff_group.empty:
            continue
        first_flower_date = ff_group['Observation_Date'].min()

        peak_group = group[
            (group['Phenophase_ID'] == 501) &
            (group['Intensity_Value'].isin(peak_categories))
        ]
        if not peak_group.empty:
            peak_bloom_date = peak_group['Observation_Date'].min()
            delta = (peak_bloom_date - first_flower_date).days
            if 0 <= delta <= 30:
                deltas.append(delta)

    return np.median(deltas) if deltas else 7.0


def calculate_photoperiod(lat, doy):
    """Civil twilight photoperiod using Cooper (1969)."""
    phi   = np.radians(lat)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    # Civil twilight at −6°
    cos_h = (
        (np.sin(np.radians(-6.0)) - np.sin(phi) * np.sin(delta))
        / (np.cos(phi) * np.cos(delta))
    )
    cos_h = np.clip(cos_h, -1, 1)
    return 2 * np.degrees(np.arccos(cos_h)) / 15.0


def dynamic_chill_model(tmin, tmax):
    """
    Vectorized DCM for a single day using hourly sine interpolation.

    Fixes applied vs. original:
    1. Gas constant R added to exponents: exp(-E/(R*T)) not exp(-E/T).
       Without R the exponents were ~-45 and ~-150, making ft and gt ≈ 0
       for all biologically relevant temperatures, so no chill portions
       ever accumulated.
    2. Sine interpolation replaced with the standard phenology form that
       places the daily minimum at 06:00 and maximum at 14:00, giving a
       full 24-hour cycle rather than the compressed 18-hour cycle of the
       original formula.
    """
    hours = np.arange(24)

    # ── Standard sine interpolation (min at 06:00, max at 14:00) ─────────────
    # Formula: T(h) = Tmean - A * cos(π * (h - h_min) / 12)
    # where A = (Tmax - Tmin) / 2 and h_min = 6 (hour of daily minimum).
    # Converted to Kelvin for use in Fishman & Erez equations.
    t_mean  = (tmax + tmin) / 2.0
    amp     = (tmax - tmin) / 2.0
    hourly_temps_c = t_mean - amp * np.cos(np.pi * (hours - 6) / 12)
    hourly_temps   = hourly_temps_c + 273.15   # → Kelvin

    x        = 0.0
    portions = 0.0

    for t in hourly_temps:
        # ── Fishman & Erez (1987) with correct R in denominator ───────────────
        ft = A0 * np.exp(-E0 / (R * t))
        gt = A1 * np.exp(-E1 / (R * t))

        if gt > 1e-300:                        # guard against true underflow
            x = x * np.exp(-gt) + (ft / gt) * (1.0 - np.exp(-gt))
        else:
            x += ft                            # linear approximation when gt→0

        if x >= 1.0:
            portions += 1.0
            x -= 1.0                           # equivalent to x*(1 - 1/x) but clearer

    return portions


def process_site(site_name, climate_df, lat, cp_threshold, tbase=0):
    """Full Bio-Year processing for a site."""
    climate_df         = climate_df.copy()
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df         = climate_df.sort_values('date')
    climate_df['bio_year'] = climate_df['date'].dt.year
    climate_df.loc[climate_df['date'].dt.month < 9, 'bio_year'] -= 1

    results = []

    for year, group in climate_df.groupby('bio_year'):
        group          = group.copy().reset_index(drop=True)
        group['doy']   = group['date'].dt.dayofyear
        group['photoperiod'] = group['doy'].apply(
            lambda d: calculate_photoperiod(lat, d)
        )

        cp_accum  = []
        gdd_accum = []
        total_cp  = 0.0
        total_gdd = 0.0
        cp_met    = False

        for _, row in group.iterrows():
            total_cp += dynamic_chill_model(row['TMIN'], row['TMAX'])
            if total_cp >= cp_threshold:
                cp_met = True
            if cp_met:
                total_gdd += max(0.0, (row['TMIN'] + row['TMAX']) / 2.0 - tbase)
            cp_accum.append(total_cp)
            gdd_accum.append(total_gdd)

        group['cp']   = cp_accum
        group['gdd']  = gdd_accum
        group['site'] = site_name

        # ── DCM diagnostics ───────────────────────────────────────────────────
        if DEBUG_DCM:
            print(
                f"[DCM DEBUG] site={site_name:15s}  bio_year={year}  "
                f"max_cp={max(cp_accum):7.2f}  "
                f"max_gdd={max(gdd_accum):7.2f}  "
                f"cp_threshold={cp_threshold}"
            )

        results.append(group)

    return pd.concat(results) if results else pd.DataFrame()


def load_teleconnections():
    """Load AO, NAO, ONI and calculate rolling anomalies."""
    indices = {}

    # AO and NAO (daily CSV)
    for name in ['ao', 'nao']:
        path = f'data/external/{name}_daily.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        indices[f'{name}_30d'] = df['value'].rolling(30, min_periods=1).mean()
        indices[f'{name}_90d'] = df['value'].rolling(90, min_periods=1).mean()

    # ONI (monthly fixed-width text → daily interpolation)
    oni_path = 'data/external/oni_raw.txt'
    if os.path.exists(oni_path):
        with open(oni_path, 'r') as f:
            lines = f.readlines()[1:]   # skip header

        oni_data = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                year = int(parts[0])
                for month_idx, value_str in enumerate(parts[1:], 1):
                    oni_data.append({
                        'YEAR':  year,
                        'MONTH': month_idx,
                        'ANOM':  float(value_str)
                    })
            except ValueError:
                print(f"Skipping malformed ONI line: {line.strip()}")

        oni_df         = pd.DataFrame(oni_data)
        oni_df['date'] = pd.to_datetime(
            oni_df['YEAR'].astype(str) + '-' + oni_df['MONTH'].astype(str) + '-01'
        )
        oni_df = oni_df[['date', 'ANOM']].set_index('date')

        # ── Filter sentinel values BEFORE interpolation ───────────────────────
        # -99.9 placeholders for future months must be removed here;
        # if they survive into resample/interpolate they corrupt the entire
        # rolling window downstream (you had 635 rows of oni_30d ≤ -90).
        oni_df = oni_df[oni_df['ANOM'] > -90].copy()

        oni_daily         = oni_df.resample('D').asfreq()
        oni_daily['ANOM'] = oni_daily['ANOM'].interpolate(
            method='linear', limit_direction='both'
        )

        indices['oni_30d'] = oni_daily['ANOM'].rolling(30, min_periods=1).mean()
        indices['oni_90d'] = oni_daily['ANOM'].rolling(90, min_periods=1).mean()

    return pd.DataFrame(indices)


def main():
    print("Feature Engineering — Biological Feature Extraction")

    nyc_delta = calculate_nyc_offset(
        'data/USA-NPN_status_intensity_observations_data.csv'
    )
    print(f"NYC Bloom Delta (First-to-Peak): {nyc_delta:.2f} days")

    # ── Site definitions ──────────────────────────────────────────────────────
    # cp_threshold: chill portions required for dormancy release (Fishman & Erez)
    # t_base:       GDD base temperature (°C) after dormancy release
    SITE_DEFS = {
        'washingtondc': {'lat': 38.85, 'cp_threshold': 45, 't_base': 5},
        'kyoto':        {'lat': 35.01, 'cp_threshold': 38, 't_base': 0},
        'liestal':      {'lat': 47.48, 'cp_threshold': 55, 't_base': 0},
        'vancouver':    {'lat': 49.25, 'cp_threshold': 48, 't_base': 5},
        'newyorkcity':  {'lat': 40.77, 'cp_threshold': 45, 't_base': 5},
    }

    tele = load_teleconnections()
    all_historical = []
    all_forecast   = []

    for site, meta in SITE_DEFS.items():
        print(f"Processing {site}...")
        hist_file = f'data/{site}_historical_climate.csv'

        if not os.path.exists(hist_file):
            print(f"  WARNING: {hist_file} not found, skipping.")
            continue

        df_hist     = pd.read_csv(hist_file)
        df_processed = process_site(
            site, df_hist, meta['lat'], meta['cp_threshold'], meta['t_base']
        )

        # Merge teleconnections on date
        df_processed = df_processed.merge(
            tele, left_on='date', right_index=True, how='left'
        )

        bio_years = sorted(df_processed['bio_year'].unique())
        print(f"  Bio_years present: {bio_years[0]} – {bio_years[-1]} "
              f"({len(bio_years)} years)")

        # Training: all complete bio-years through 2024
        all_historical.append(df_processed[df_processed['bio_year'] < 2025])

        # Forecast 2026: bio-year 2025 (Sep 2025 – Feb 28 2026)
        df_2026 = df_processed[df_processed['bio_year'] == 2025].copy()
        print(f"  Forecast rows (bio_year=2025): {len(df_2026)}")
        all_forecast.append(df_2026)

    if all_historical:
        train_df = pd.concat(all_historical, ignore_index=True)
        train_df.to_csv('features_train.csv', index=False)
        print(f"\nfeatures_train.csv written: {train_df.shape[0]:,} rows, "
              f"{train_df.shape[1]} cols")

        # Quick DCM sanity check
        cp_max  = train_df['cp'].max()
        gdd_max = train_df['gdd'].max()
        print(f"DCM sanity — max(cp)={cp_max:.2f}, max(gdd)={gdd_max:.2f}")
        if cp_max == 0.0:
            print("  ⚠ WARNING: cp is still all zeros — DCM did not accumulate.")
        else:
            print("  ✔ cp is non-zero — DCM is accumulating chill portions.")
        if gdd_max == 0.0:
            print("  ⚠ WARNING: gdd is still all zeros — "
                  "either cp_threshold never reached or t_base too high.")
        else:
            print("  ✔ gdd is non-zero — forcing accumulation is working.")
    else:
        print("ERROR: No historical data processed.")

    if all_forecast:
        forecast_df = pd.concat(all_forecast, ignore_index=True)
        forecast_df.to_csv('features_2026_forecast.csv', index=False)
        print(f"features_2026_forecast.csv written: {forecast_df.shape[0]:,} rows")
    else:
        print("ERROR: No forecast data processed.")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()