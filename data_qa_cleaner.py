import pandas as pd
import numpy as np
import os

def clean_station_data(city):
    filepath = f"data/{city}_historical_climate.csv"
    if not os.path.exists(filepath):
        print(f"File missing for {city}.")
        return

    print(f"\nQA Audit & Cleaning: {city.upper()}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Enforce strict daily continuity (Find missing dates)
    start_date = df['date'].min()
    end_date = pd.to_datetime("2026-02-21")
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    
    missing_dates = len(full_idx) - len(df)
    if missing_dates > 0:
        print(f"Found {missing_dates} completely missing days. Re-indexing")
        df = df.set_index('date').reindex(full_idx).rename_axis('date').reset_index()
    else:
        print("Temporal continuity verified. No missing days.")

    # Check for NaNs
    nan_counts = df[['TMAX', 'TMIN', 'PRCP']].isna().sum()
    print(f" 🔍 Missing Values detected:\n{nan_counts.to_string()}")

    # Imputation Strategy
    # First, calculate TAVG if missing (some stations don't report it)
    if 'TAVG' not in df.columns or df['TAVG'].isna().all():
         df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0

    # For short gaps (<= 3 days), use linear interpolation
    for col in ['TMAX', 'TMIN', 'TAVG', 'PRCP']:
        df[col] = df[col].interpolate(method='linear', limit=3)

    # For long gaps, we use the historical average for that specific Day of Year
    df['doy'] = df['date'].dt.dayofyear
    for col in ['TMAX', 'TMIN', 'TAVG', 'PRCP']:
        if df[col].isna().any():
            # Calculate historical mean for each day of year
            doy_means = df.groupby('doy')[col].transform('mean')
            df[col] = df[col].fillna(doy_means)

    # Fill any remaining PRCP NaNs with 0 (safe assumption for precipitation)
    df['PRCP'] = df['PRCP'].fillna(0)

    # Physical Integrity Checks
    # Fix instances where TMIN > TMAX due to sensor error
    invalid_temps = df[df['TMIN'] > df['TMAX']]
    if not invalid_temps.empty:
        print(f"Found {len(invalid_temps)} days where TMIN > TMAX. Swapping values.")
        # Swap logic
        temp_min = df.loc[df['TMIN'] > df['TMAX'], 'TMIN']
        df.loc[df['TMIN'] > df['TMAX'], 'TMIN'] = df.loc[df['TMIN'] > df['TMAX'], 'TMAX']
        df.loc[df['TMIN'] > df['TMAX'], 'TMAX'] = temp_min

    # Recalculate TAVG just to be strictly consistent after any swaps
    df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0

    # Final Verification
    final_nans = df[['TMAX', 'TMIN', 'TAVG']].isna().sum().sum()
    if final_nans == 0:
        print("Final Sanity Check: PASSED (0 NaNs remaining).")
    else:
        print(f"ERROR: {final_nans} NaNs survived the cleaning process!")

    # Overwrite with the pristine dataset
    df = df.drop(columns=['doy']) # Drop helper column
    df.to_csv(filepath, index=False)
    print(f"Saved pristine dataset for {city}.")

if __name__ == "__main__":
    cities = ["washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"]
    for city in cities:
        clean_station_data(city)