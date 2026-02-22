import pandas as pd
import numpy as np

def sanitize_features(train_path, forecast_path):
    # 1. Sanitize Training Data (Simple Imputation)
    df_train = pd.read_csv(train_path)
    # Forward fill then backward fill for the small gaps in training
    df_train[['ao_30d', 'nao_30d', 'oni_30d']] = df_train.groupby('site')[['ao_30d', 'nao_30d', 'oni_30d']].ffill().bfill().fillna(0)
    df_train.to_csv(train_path, index=False)
    print(f"✅ Sanitized {train_path}: Missing values removed.")

    # 2. Sanitize Forecast Data (Persistence + Decay)
    df_forecast = pd.read_csv(forecast_path)
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    
    sanitized_frames = []
    for site, group in df_forecast.groupby('site'):
        group = group.sort_values('date').copy()
        
        # Identify the last date with actual Teleconnection data (Feb 21)
        last_known_idx = group[['ao_30d', 'nao_30d']].last_valid_index()
        if last_known_idx is not None:
            last_known_date = group.loc[last_known_idx, 'date']
            
            # AO/NAO Persistence: Carry forward for 14 days, then decay to 0
            for col in ['ao_30d', 'nao_30d']:
                val = group.loc[last_known_idx, col]
                # Dates > last_known_date
                future_mask = group['date'] > last_known_date
                # Persistence window (Next 14 days)
                persist_mask = (group['date'] > last_known_date) & (group['date'] <= last_known_date + pd.Timedelta(days=14))
                
                group.loc[persist_mask, col] = val
                group.loc[group['date'] > last_known_date + pd.Timedelta(days=14), col] = 0.0
            
            # ONI (Oceanic Index): Very slow-moving. Forward fill for the entire spring window.
            group['oni_30d'] = group['oni_30d'].ffill().fillna(0.0)
            
        sanitized_frames.append(group)
    
    df_sanitized = pd.concat(sanitized_frames)
    df_sanitized.to_csv(forecast_path, index=False)
    print(f"✅ Sanitized {forecast_path}: Persistence Padding & Climatological Decay applied.")

if __name__ == "__main__":
    sanitize_features('features_train.csv', 'features_2026_forecast.csv')