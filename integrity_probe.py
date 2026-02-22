import pandas as pd
import numpy as np

def run_integrity_probe(train_path, forecast_path):
    files = {'Train': train_path, 'Forecast': forecast_path}
    
    for label, path in files.items():
        print(f"\n--- {label} Data Audit ({path}) ---")
        df = pd.read_csv(path)
        
        # 1. Teleconnection Check
        atmos_cols = ['ao_30d', 'nao_30d', 'oni_30d']
        missing_atmos = df[atmos_cols].isnull().sum()
        print(f"  Missing Teleconnections:\n{missing_atmos}")
        
        # 2. GDD Accumulation Check (The "Tripwire" Audit)
        # GDD should increase monotonically within a bio_year
        if 'bio_year' in df.columns and 'gdd' in df.columns:
            sample_site = df['site'].iloc[0]
            site_df = df[df['site'] == sample_site].sort_values(['bio_year', 'doy'])
            # Check if GDD ever decreases within the same bio_year
            gdd_diffs = site_df.groupby('bio_year')['gdd'].diff().dropna()
            decreases = (gdd_diffs < 0).sum()
            if decreases > 0:
                print(f"  CRITICAL: GDD is NOT cumulative (found {decreases} decreases).")
            else:
                print("  GDD Integrity: Verified cumulative.")

        # 3. Bio-Year Gap Check (Forecast Only)
        if label == 'Forecast':
            current_bio_year = df['bio_year'].max()
            count_2025 = len(df[df['bio_year'] == current_bio_year])
            print(f"  Current Forecast Bio-Year ({current_bio_year}) Row Count: {count_2025}")
            if count_2025 < 270:
                print("  Note: Forecast is partial (Bridge logic required).")

if __name__ == "__main__":
    run_integrity_probe('features_train.csv', 'features_2026_forecast.csv')