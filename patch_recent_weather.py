import pandas as pd
import requests
import os
from datetime import timedelta

# Exact coordinates for the 5 bloom sites
COORDS = {
    "washingtondc": {"lat": 38.8853, "lon": -77.0369},
    "kyoto": {"lat": 35.0116, "lon": 135.6776},
    "liestal": {"lat": 47.4814, "lon": 7.7305},
    "vancouver": {"lat": 49.2827, "lon": -123.1207},
    "newyorkcity": {"lat": 40.7308, "lon": -73.9973}
}

def patch_city_data(city):
    filepath = f"data/{city}_historical_climate.csv"
    if not os.path.exists(filepath):
        print(f"File missing for {city}. Skipping.")
        return

    # Load existing data and find the gap
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    target_date = pd.to_datetime("2026-02-21")
    
    if last_date >= target_date:
        print(f"[{city.upper()}] Already up to date.")
        return

    start_patch = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_patch = target_date.strftime('%Y-%m-%d')
    
    print(f"[{city.upper()}] Patching missing data from {start_patch} to {end_patch}...")

    # Fetch from Open-Meteo Archive API (No API key required)
    lat, lon = COORDS[city]["lat"], COORDS[city]["lon"]
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&"
           f"start_date={start_patch}&end_date={end_patch}&"
           f"daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum&"
           f"timezone=auto")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Format to match NOAA schema
        patch_df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'TMAX': data['daily']['temperature_2m_max'],
            'TMIN': data['daily']['temperature_2m_min'],
            'TAVG': data['daily']['temperature_2m_mean'],
            'PRCP': data['daily']['precipitation_sum'] # in mm, matches NOAA metric
        })
        
        # Save
        combined_df = pd.concat([df, patch_df], ignore_index=True)
        combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        combined_df.to_csv(filepath, index=False)
        print(f"  -> Successfully patched {len(patch_df)} days for {city}.")
        
    except Exception as e:
        print(f"  -> Failed to patch {city}: {e}")

if __name__ == "__main__":
    for city in COORDS.keys():
        patch_city_data(city)
    print("\nPATCH COMPLETE")