import pandas as pd
import requests
import io
import os

def fetch_cpc_teleconnections_final():
    os.makedirs('data/external', exist_ok=True)
    
    urls = {
        "ao": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii",
        "nao": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii"
    }
    
    headers = {'User-Agent': 'Mozilla/5.0'}

    for name, url in urls.items():
        print(f"Parsing {name.upper()} from CPC (Flexible Mode)...")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            
            # Use sep=r'\s+' which is a regex for "one or more whitespaces"
            # This is more robust than fixed-width for these specific ASCII files
            df = pd.read_csv(
                io.StringIO(r.text), 
                sep=r'\s+', 
                header=None, 
                names=['year', 'month', 'day', 'value'],
                engine='python' # More stable for regex separators
            )
            
            # Clean up: Force year/month/day to integers to drop metadata/empty lines
            df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['year', 'month', 'day'])
            df[['year', 'month', 'day']] = df[['year', 'month', 'day']].astype(int)
            
            # Convert to standard datetime
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            
            # Filter for 1950 onwards and save the essential columns
            df = df[df['date'] >= '1950-01-01'][['date', 'value']]
            
            df.to_csv(f"data/external/{name}_daily.csv", index=False)
            print(f"{name.upper()} verified and saved.")
            
        except Exception as e:
            print(f"Failed to parse {name}: {e}")

if __name__ == "__main__":
    fetch_cpc_teleconnections_final()