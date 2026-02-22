import pandas as pd
import os
from datetime import date

def audit_data_completeness():
    cities = ["washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"]
    print("--- DATA INTEGRITY AUDIT ---")
    
    for city in cities:
        file = f"data/{city}_historical_climate.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            last_date = pd.to_datetime(df['date']).max()
            days_missing = (pd.to_datetime('2026-02-21') - last_date).days
            
            print(f"[{city.upper()}]")
            print(f"  - Records: {len(df)}")
            print(f"  - Last Observation: {last_date.date()}")
            if days_missing > 1:
                print(f"  - ⚠️ MISSING: {days_missing} days (Winter 2025-26 signal is incomplete)")
            else:
                print(f"  - ✅ UP TO DATE")
        else:
            print(f"[{city.upper()}] ❌ FILE MISSING")

if __name__ == "__main__":
    audit_data_completeness()