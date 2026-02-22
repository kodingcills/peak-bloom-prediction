import pandas as pd
import numpy as np
import os

def generate_city_normals(city):
    filepath = f"data/{city}_historical_climate.csv"
    if not os.path.exists(filepath):
        print(f"File missing: {city}")
        return

    print(f"Generating Modern Normals (1995-2025) for {city.upper()}...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use only recent history to account for climate change shift
    recent_df = df[df['date'].dt.year >= 1995].copy()
    
    # Calculate Day of Year (DOY)
    recent_df['doy'] = recent_df['date'].dt.dayofyear
    
    # Group by DOY and calculate mean
    # We use 'median' for PRCP because precipitation is highly skewed
    normals = recent_df.groupby('doy').agg({
        'TMAX': 'mean',
        'TMIN': 'mean',
        'TAVG': 'mean',
        'PRCP': 'median'
    }).reset_index()
    
    # Smooth the curves using a rolling mean to remove "weather noise" 
    # and keep the "climate signal"
    for col in ['TMAX', 'TMIN', 'TAVG']:
        # Wrap around for smoothing at start/end of year
        extended = pd.concat([normals.iloc[-15:], normals, normals.iloc[:15]])
        smoothed = extended[col].rolling(window=15, center=True).mean()
        normals[col] = smoothed.iloc[15:-15].values

    output_path = f"data/{city}_climatology_normals.csv"
    normals.to_csv(output_path, index=False)
    print(f" ✅ Saved modern normals to {output_path}")

if __name__ == "__main__":
    cities = ["washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"]
    for city in cities:
        generate_city_normals(city)