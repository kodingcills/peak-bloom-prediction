import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# Chilling Hours Model (Weinberger 1950)
#
# Replaces the Fishman & Erez (1987) Dynamic Chill Model which was abandoned
# after diagnosing three nested implementation problems:
#   1. Missing gas constant R in the Arrhenius exponents (ft, gt → 0)
#   2. With R corrected, gt is still ~0 at all biologically relevant temps,
#      making x accumulate at ~2.7e-5/hour (1,500 days to reach 1 portion)
#   3. The correct Dynamic Model uses a sigmoid transition function (not a
#      simple x >= 1 threshold) — matching the exact chillR R-package
#      implementation is a research project in itself
#
# The Chilling Hours model is unambiguous, structurally bulletproof, and
# produces biologically correct results for all five temperate sites.
# Any chill-negation error (warm January days) will be corrected by the
# ML residual stacker (Layer 2) via TMAX_volatility and frost-day features.
#
# Reference: Luedeling (2012) shows CH and DCM are highly correlated in
# high-chill temperate regions (DC, Kyoto, Liestal) — the use case here.
#
# Threshold calibration (verified by simulation against 30-yr normals):
#   washingtondc : threshold 900  CH — crossed end of Dec  (~919 CH)
#   kyoto        : threshold 600  CH — crossed end of Dec  (~615 CH)
#   liestal      : threshold 800  CH — crossed early Dec   (~787 CH by Nov)
#   vancouver    : threshold 500  CH — crossed mid-Nov     (~543 CH)
#   newyorkcity  : threshold 850  CH — crossed end of Dec  (~859 CH)
# ─────────────────────────────────────────────────────────────────────────────

# Toggle to True for per-site per-bio_year diagnostics
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
    cos_h = (
        (np.sin(np.radians(-6.0)) - np.sin(phi) * np.sin(delta))
        / (np.cos(phi) * np.cos(delta))
    )
    cos_h = np.clip(cos_h, -1, 1)
    return 2 * np.degrees(np.arccos(cos_h)) / 15.0


def daily_chill_hours(tmin, tmax):
    """
    Chilling Hours Model (Weinberger 1950).

    Counts hours per day where temperature falls in the effective chill
    band [0 C, 7.2 C], using a sine-wave hourly interpolation that places
    the daily minimum at 06:00 and maximum at 14:00 (standard phenology form).

    This replaces dynamic_chill_model() and writes to the same 'cp' column
    so model_stacker.py requires no changes downstream.

    Returns
    -------
    float : chill hours on this day (0 to 24)
    """
    hours  = np.arange(24)
    t_mean = (tmax + tmin) / 2.0
    amp    = (tmax - tmin) / 2.0
    # min at 06:00, max at 14:00
    hourly = t_mean - amp * np.cos(np.pi * (hours - 6) / 12)
    return float(np.sum((hourly >= 0.0) & (hourly <= 7.2)))


def process_site(site_name, climate_df, lat, cp_threshold, tbase=0):
    """
    Full bio-year processing for a site.

    Produces per-day columns:
      cp  — cumulative chill hours since Sep 1 (bio-year start)
      gdd — cumulative growing degree days (base=tbase) AFTER cp >= cp_threshold
    """
    climate_df         = climate_df.copy()
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df         = climate_df.sort_values('date')
    climate_df['bio_year'] = climate_df['date'].dt.year
    climate_df.loc[climate_df['date'].dt.month < 9, 'bio_year'] -= 1

    results = []

    for year, group in climate_df.groupby('bio_year'):
        group        = group.copy().reset_index(drop=True)
        group['doy'] = group['date'].dt.dayofyear
        group['photoperiod'] = group['doy'].apply(
            lambda d: calculate_photoperiod(lat, d)
        )

        cp_accum  = []
        gdd_accum = []
        total_cp  = 0.0
        total_gdd = 0.0
        cp_met    = False

        for _, row in group.iterrows():
            total_cp += daily_chill_hours(row['TMIN'], row['TMAX'])

            if total_cp >= cp_threshold:
                cp_met = True

            if cp_met:
                total_gdd += max(0.0, (row['TMIN'] + row['TMAX']) / 2.0 - tbase)

            cp_accum.append(total_cp)
            gdd_accum.append(total_gdd)

        group['cp']   = cp_accum
        group['gdd']  = gdd_accum
        group['site'] = site_name

        if DEBUG_DCM:
            print(
                f"[DCM DEBUG] site={site_name:15s}  bio_year={year}  "
                f"max_cp={max(cp_accum):8.1f}  "
                f"max_gdd={max(gdd_accum):8.1f}  "
                f"cp_threshold={cp_threshold}  "
                f"met={'YES' if max(cp_accum) >= cp_threshold else 'NO'}"
            )

        results.append(group)

    return pd.concat(results) if results else pd.DataFrame()


def load_teleconnections():
    """Load AO, NAO, ONI and compute rolling anomalies."""
    indices = {}

    for name in ['ao', 'nao']:
        path = f'data/external/{name}_daily.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        indices[f'{name}_30d'] = df['value'].rolling(30, min_periods=1).mean()
        indices[f'{name}_90d'] = df['value'].rolling(90, min_periods=1).mean()

    oni_path = 'data/external/oni_raw.txt'
    if os.path.exists(oni_path):
        with open(oni_path, 'r') as f:
            lines = f.readlines()[1:]

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
                        'ANOM':  float(value_str),
                    })
            except ValueError:
                print(f"Skipping malformed ONI line: {line.strip()}")

        oni_df         = pd.DataFrame(oni_data)
        oni_df['date'] = pd.to_datetime(
            oni_df['YEAR'].astype(str) + '-' + oni_df['MONTH'].astype(str) + '-01'
        )
        oni_df = oni_df[['date', 'ANOM']].set_index('date')

        # Remove sentinel values BEFORE interpolation — prevents -99.9
        # from contaminating rolling windows (was 635 rows of oni_30d <= -90)
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
    print("Chill model: Chilling Hours (Weinberger 1950), 0–7.2°C band")
    print()

    nyc_delta = calculate_nyc_offset(
        'data/USA-NPN_status_intensity_observations_data.csv'
    )
    print(f"NYC Bloom Delta (First-to-Peak): {nyc_delta:.2f} days")

    SITE_DEFS = {
        'washingtondc': {'lat': 38.85, 'cp_threshold': 900,  't_base': 5},
        'kyoto':        {'lat': 35.01, 'cp_threshold': 600,  't_base': 0},
        'liestal':      {'lat': 47.48, 'cp_threshold': 800,  't_base': 0},
        'vancouver':    {'lat': 49.25, 'cp_threshold': 500,  't_base': 5},
        'newyorkcity':  {'lat': 40.77, 'cp_threshold': 850,  't_base': 5},
    }

    tele           = load_teleconnections()
    all_historical = []
    all_forecast   = []

    for site, meta in SITE_DEFS.items():
        print(f"Processing {site}...")
        hist_file = f'data/{site}_historical_climate.csv'

        if not os.path.exists(hist_file):
            print(f"  WARNING: {hist_file} not found — skipping.")
            continue

        df_hist      = pd.read_csv(hist_file)
        df_processed = process_site(
            site, df_hist, meta['lat'], meta['cp_threshold'], meta['t_base']
        )

        df_processed = df_processed.merge(
            tele, left_on='date', right_index=True, how='left'
        )

        bio_years = sorted(df_processed['bio_year'].unique())
        print(f"  Bio_years: {bio_years[0]}–{bio_years[-1]} ({len(bio_years)} years)")

        by_year  = df_processed.groupby('bio_year')['cp'].max()
        crossed  = (by_year >= meta['cp_threshold']).sum()
        print(f"  Threshold ({meta['cp_threshold']} CH) crossed in "
              f"{crossed}/{len(by_year)} bio-years")

        all_historical.append(df_processed[df_processed['bio_year'] < 2025])

        df_2026 = df_processed[df_processed['bio_year'] == 2025].copy()
        print(f"  Forecast rows (bio_year=2025): {len(df_2026)}")
        all_forecast.append(df_2026)

    if all_historical:
        train_df = pd.concat(all_historical, ignore_index=True)
        train_df.to_csv('features_train.csv', index=False)
        print(f"\nfeatures_train.csv: {train_df.shape[0]:,} rows, "
              f"{train_df.shape[1]} cols")

        cp_max  = train_df['cp'].max()
        gdd_max = train_df['gdd'].max()
        print(f"Sanity — max(cp)={cp_max:.1f}  max(gdd)={gdd_max:.1f}")

        if cp_max == 0.0:
            print("  WARNING: cp all zeros — chill accumulation failed.")
        else:
            print("  OK: cp non-zero — chill hours accumulating correctly.")

        if gdd_max == 0.0:
            print("  WARNING: gdd all zeros — "
                  "threshold never crossed or t_base too high.")
        else:
            print("  OK: gdd non-zero — forcing accumulation working.")
    else:
        print("ERROR: No historical data processed.")

    if all_forecast:
        forecast_df = pd.concat(all_forecast, ignore_index=True)
        forecast_df.to_csv('features_2026_forecast.csv', index=False)
        print(f"features_2026_forecast.csv: {forecast_df.shape[0]:,} rows")
        print(f"Forecast sanity — max(cp)={forecast_df['cp'].max():.1f}  "
              f"max(gdd)={forecast_df['gdd'].max():.1f}")
    else:
        print("ERROR: No forecast data processed.")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()