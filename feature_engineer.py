import pandas as pd
import numpy as np
import os
from datetime import datetime

# Constants for Dynamic Chill Model (Fishman & Erez 1987)
A0 = 1.395e5
A1 = 2.567e18
E0 = 12400
E1 = 41400

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
        ff_group = group[(group['Phenophase_ID'] == 501) & (group['Phenophase_Status'] == 1)]
        if ff_group.empty: continue
        first_flower_date = ff_group['Observation_Date'].min()
        
        peak_group = group[(group['Phenophase_ID'] == 501) & (group['Intensity_Value'].isin(peak_categories))]
        if not peak_group.empty:
            peak_bloom_date = peak_group['Observation_Date'].min()
            delta = (peak_bloom_date - first_flower_date).days
            if 0 <= delta <= 30: deltas.append(delta)
                
    return np.median(deltas) if deltas else 7.0

def calculate_photoperiod(lat, doy):
    """ Civil twilight photoperiod using Cooper (1969). """
    phi = np.radians(lat)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    # Civil twilight (-6 degrees)
    cos_h = (np.sin(np.radians(-6.0)) - np.sin(phi) * np.sin(delta)) / (np.cos(phi) * np.cos(delta))
    cos_h = np.clip(cos_h, -1, 1)
    return 2 * np.degrees(np.arccos(cos_h)) / 15.0

def dynamic_chill_model(tmin, tmax):
    """ Vectorized DCM for a single day using hourly interpolation. """
    hours = np.arange(24)
    # Hourly interpolation: sine wave approximation
    hourly_temps = (tmax + tmin) / 2 + (tmax - tmin) / 2 * np.sin(np.pi * (hours - 8) / 9) + 273.15
    
    x = 0.0
    portions = 0.0
    for t in hourly_temps:
        ft = A0 * np.exp(-E0 / t)
        gt = A1 * np.exp(-E1 / t)
        if gt > 0:
            x = x * np.exp(-gt) + (ft / gt) * (1 - np.exp(-gt))
        else:
            x += ft
        if x >= 1.0:
            portions += 1.0
            x = x * (1 - 1.0/x)
    return portions

def process_site(site_name, climate_df, lat, cp_threshold, tbase=0):
    """ Full Bio-Year processing for a site. """
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df = climate_df.sort_values('date').copy()
    climate_df['bio_year'] = climate_df['date'].dt.year
    climate_df.loc[climate_df['date'].dt.month < 9, 'bio_year'] -= 1
    
    results = []
    for year, group in climate_df.groupby('bio_year'):
        if len(group) < 250: continue
        group = group.copy().reset_index(drop=True)
        group['doy'] = group['date'].dt.dayofyear
        group['photoperiod'] = group['doy'].apply(lambda d: calculate_photoperiod(lat, d))
        
        # Chill & GDD
        cp_accum = []
        gdd_accum = []
        total_cp = 0.0
        total_gdd = 0.0
        cp_met = False
        
        for _, row in group.iterrows():
            total_cp += dynamic_chill_model(row['TMIN'], row['TMAX'])
            if total_cp >= cp_threshold:
                cp_met = True
            if cp_met:
                total_gdd += max(0, (row['TMIN'] + row['TMAX'])/2 - tbase)
            cp_accum.append(total_cp)
            gdd_accum.append(total_gdd)
            
        group['cp'] = cp_accum
        group['gdd'] = gdd_accum
        group['site'] = site_name
        results.append(group)
    return pd.concat(results) if results else pd.DataFrame()

def load_teleconnections():
    """ Load AO, NAO, ONI, AMO and calculate rolling anomalies. """
    indices = {}
    for name in ['ao', 'nao']:
        df = pd.read_csv(f'data/external/{name}_daily.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        indices[f'{name}_30d'] = df['value'].rolling(30).mean()
        indices[f'{name}_90d'] = df['value'].rolling(90).mean()
    # Add more indices as needed (AMO/ONI usually monthly, skip for Sprint 1 complexity)
    return pd.DataFrame(indices)

def main():
    print("Sprint 1 - Biological Feature Engineering Execution")
    nyc_delta = calculate_nyc_offset('data/USA-NPN_status_intensity_observations_data.csv')
    print(f"NYC Bloom Delta (First-to-Peak): {nyc_delta:.2f} days")
    
    sites = {
        'washingtondc': {'lat': 38.85, 'cp_threshold': 45},
        'kyoto': {'lat': 35.01, 'cp_threshold': 38},
        'liestal': {'lat': 47.48, 'cp_threshold': 55},
        'vancouver': {'lat': 49.25, 'cp_threshold': 48},
        'newyorkcity': {'lat': 40.77, 'cp_threshold': 45}
    }
    
    tele = load_teleconnections()
    all_historical = []
    all_forecast = []
    
    for site, meta in sites.items():
        print(f"Processing {site}...")
        hist_file = f'data/{site}_historical_climate.csv'
        norm_file = f'data/{site}_climatology_normals.csv'
        
        if not os.path.exists(hist_file): continue
        df_hist = pd.read_csv(hist_file)
        df_processed = process_site(site, df_hist, meta['lat'], meta['cp_threshold'])
        
        # Merge teleconnections
        df_processed = df_processed.merge(tele, left_on='date', right_index=True, how='left')
        
        # Train: All bio-years up to 2024
        all_historical.append(df_processed[df_processed['bio_year'] < 2025])
        
        # Forecast 2026: Bio-year 2025
        # Current data up to Feb 21, 2026 (Bio-year 2025)
        df_2026 = df_processed[df_processed['bio_year'] == 2025].copy()
        
        # Bridge with Normals if missing dates up to May 31
        last_date = df_2026['date'].max()
        if last_date < pd.Timestamp('2026-05-31') and os.path.exists(norm_file):
            df_norm = pd.read_csv(norm_file)
            df_norm['date'] = pd.to_datetime(df_norm['date'])
            # Shift normals to 2026
            df_norm['date'] = df_norm['date'].apply(lambda d: d.replace(year=2026))
            bridge = df_norm[df_norm['date'] > last_date].copy()
            bridge['bio_year'] = 2025
            bridge = process_site(site, pd.concat([df_2026[['date','TMIN','TMAX']], bridge[['date','TMIN','TMAX']]]), meta['lat'], meta['cp_threshold'])
            df_2026 = bridge[bridge['bio_year'] == 2025]
        
        all_forecast.append(df_2026)

    pd.concat(all_historical).to_csv('features_train.csv', index=False)
    pd.concat(all_forecast).to_csv('features_2026_forecast.csv', index=False)
    print("Sprint 1 Complete: features_train.csv and features_2026_forecast.csv generated.")

if __name__ == "__main__":
    main()
