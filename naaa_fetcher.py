import requests
import pandas as pd
import time
from datetime import date
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
NOAA_API_KEY = os.getenv("NOAA")

if not NOAA_API_KEY:
    raise ValueError("NOAA_API_KEY not found in environment variables")
BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'

STATIONS = {
    "washingtondc": "GHCND:USW00013743",
    "vancouver": "GHCND:CA001108395",
    "newyorkcity": "GHCND:USW00014732",
    "liestal": "GHCND:SZ000001940",
    "kyoto": "GHCND:JA000047759"
}

def fetch_data_chunk(station_id, start_date, end_date):
    """Fetches a chunk of data from NOAA API with exponential backoff."""
    headers = {'token': NOAA_API_KEY}
    params = {
        'datasetid': 'GHCND',
        'stationid': station_id,
        'datatypeid': 'TMAX,TMIN,TAVG,PRCP', # Adding precipitation for extra signal
        'units': 'metric',
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return pd.DataFrame(data['results'])
                else:
                    return pd.DataFrame() # No data for this period
            elif response.status_code == 429: # Rate limit
                time.sleep((2 ** attempt) + 1)
            else:
                print(f"Error {response.status_code}: {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
            
    return pd.DataFrame()

def process_station(location_name, station_id, start_year, end_year):
    """Downloads and formats temperature data for a specific station."""
    print(f"Fetching data for {location_name} ({station_id})...")
    all_chunks = []
    
    for year in range(start_year, end_year + 1):
        # Splitting the year into two 6-month chunks to avoid the 1000 limit
        chunks = [
            (f"{year}-01-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-12-31")
        ]
        
        for start_date, end_date in chunks:
            df_chunk = fetch_data_chunk(station_id, start_date, end_date)
            if not df_chunk.empty:
                all_chunks.append(df_chunk)
            time.sleep(0.25) # Polite API delay
            
    if not all_chunks:
        print(f"No data found for {location_name}.")
        return None

    # Combine all chunks
    raw_df = pd.concat(all_chunks, ignore_index=True)
    
    # Pivot the data so TMAX, TMIN, TAVG, PRCP are columns
    df = raw_df.pivot_table(
        index='date', 
        columns='datatype', 
        values='value', 
        aggfunc='first'
    ).reset_index()
    
    # Clean up the date column format
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Handle Missing TAVG: Calculate it from TMAX and TMIN if necessary
    for col in ['TMAX', 'TMIN', 'TAVG', 'PRCP']:
        if col not in df.columns:
            df[col] = float('nan')
            
    # Convert from tenths of degrees to standard degrees Celsius (NOAA standard quirk)
    df['TMAX'] = df['TMAX'] / 10.0
    df['TMIN'] = df['TMIN'] / 10.0
    if df['TAVG'].notna().any():
        df['TAVG'] = df['TAVG'] / 10.0
        
    # Impute missing TAVG using the mathematical average
    df['TAVG'] = df['TAVG'].fillna((df['TMAX'] + df['TMIN']) / 2.0)
    
    return df

def main():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # We want data from 1950 (or earliest available) to current year
    current_year = date.today().year
    start_year = 1950
    
    for location, station_id in STATIONS.items():
        df = process_station(location, station_id, start_year, current_year)
        
        if df is not None:
            # Save directly to the data folder
            filepath = f"data/{location}_historical_climate.csv"
            df.to_csv(filepath, index=False)
            print(f"Successfully saved {len(df)} days of data to {filepath}\n")

if __name__ == "__main__":
    main()