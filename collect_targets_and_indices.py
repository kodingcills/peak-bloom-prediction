import pandas as pd
import requests
import os

def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {local_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    os.makedirs('data/external', exist_ok=True)
    
    # 1. Collection: Historical Bloom Targets (GitHub)
    bloom_urls = {
        "washingtondc": "https://raw.githubusercontent.com/jandot/cherry-blossoms/main/data/washingtondc.csv",
        "kyoto": "https://raw.githubusercontent.com/jandot/cherry-blossoms/main/data/kyoto.csv",
        "liestal": "https://raw.githubusercontent.com/jandot/cherry-blossoms/main/data/liestal.csv",
        "vancouver": "https://raw.githubusercontent.com/jandot/cherry-blossoms/main/data/vancouver.csv",
        "nyc": "https://raw.githubusercontent.com/jandot/cherry-blossoms/main/data/nyc.csv"
    }
    
    print("--- Collecting Historical Bloom Targets ---")
    for city, url in bloom_urls.items():
        download_file(url, f"data/{city}.csv")

    # 2. Collection: Macro-Climate Indices (NOAA PSL)
    # ONI (El Niño) and AMO (Atlantic Oscillation)
    indices_urls = {
        "oni": "https://psl.noaa.gov/data/correlation/oni.data",
        "amo": "https://psl.noaa.gov/data/correlation/amon.us.long.data"
    }
    
    print("\n--- Collecting Macro-Climate Indices ---")
    for index, url in indices_urls.items():
        download_file(url, f"data/external/{index}_raw.txt")

    print("\n--- STAGE 1 COMPLETE ---")

if __name__ == "__main__":
    main()