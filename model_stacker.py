import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
import xgboost as xgb
import os
import warnings
from sklearn.metrics import mean_absolute_error

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration and Constants ---
SITE_DEFS = {
    'washingtondc': {'lat': 38.85, 't_base': 5, 'cp_threshold': 45, 'bloom_def': 70, 'med_doy': 90},
    'kyoto': {'lat': 35.01, 't_base': 0, 'cp_threshold': 38, 'bloom_def': 100, 'med_doy': 95},
    'liestal': {'lat': 47.48, 't_base': 0, 'cp_threshold': 55, 'bloom_def': 25, 'med_doy': 92},
    'vancouver': {'lat': 49.25, 't_base': 5, 'cp_threshold': 48, 'bloom_def': 70, 'med_doy': 100},
    'newyorkcity': {'lat': 40.77, 't_base': 5, 'cp_threshold': 45, 'bloom_def': 70, 'med_doy': 105}
}
NYC_OFFSET_DAYS = 0.5
VANCOUVER_PRECISION_WEIGHT = 0.2
MAY_31_DOY = 151 
TEMP_SCALE_FACTOR = 10.0
MODEL_T_BASE_HEAT = 0.0 
MODEL_T_BASE_CHILL = 7.0
# Increased to ensure vernalization is complete in warm winters
MIN_CHILL_REQUIRED = 450.0 

def smart_rescale(df):
    cols = ['TAVG', 'TMAX', 'TMIN']
    for site in df['site'].unique():
        mask = df['site'] == site
        site_vals = df.loc[mask, 'TAVG']
        if site_vals.mean() < 5.0 and site_vals.max() < 15.0:
            for col in cols:
                if col in df.columns: df.loc[mask, col] *= 10.0
    return df

def compute_bio_thermal_features(df, site):
    df = df.sort_values('date').copy()
    df['gdd'] = np.maximum(0, df['TAVG'] - MODEL_T_BASE_HEAT).cumsum()
    df['chill'] = np.maximum(0, MODEL_T_BASE_CHILL - df['TAVG']).cumsum()
    return df

def load_data_orchestrator(features_path, targets_dir, nyc_offset_days):
    df_raw = pd.read_csv(features_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = smart_rescale(df_raw)
    processed_feats = []
    for site in df_raw['site'].unique():
        if site not in SITE_DEFS: continue
        site_df = df_raw[df_raw['site'] == site].copy()
        if 'bio_year' not in site_df.columns: continue
        site_res = []
        for byear, group in site_df.groupby('bio_year'):
            res = compute_bio_thermal_features(group, site)
            res['bio_year'] = byear 
            site_res.append(res)
        if site_res: processed_feats.append(pd.concat(site_res))
    df_features = pd.concat(processed_feats)
    all_targets = []
    for site_name in SITE_DEFS.keys():
        fname = site_name.replace('newyorkcity', 'nyc')
        target_file = f"{targets_dir}/{fname}.csv"
        if os.path.exists(target_file):
            df_t = pd.read_csv(target_file)
            if 'location' in df_t.columns: df_t.rename(columns={'location': 'site'}, inplace=True)
            if 'year' in df_t.columns: df_t.rename(columns={'year': 'bio_year'}, inplace=True)
            df_t = df_t[['bio_year', 'bloom_doy', 'site']]
            df_t['site'] = site_name 
            all_targets.append(df_t)
    df_targets = pd.concat(all_targets, ignore_index=True).drop_duplicates(subset=['bio_year', 'site'])
    df_targets['bloom_doy'] = df_targets['bloom_doy'].astype(float)
    df_targets.loc[df_targets['site'] == 'newyorkcity', 'bloom_doy'] += nyc_offset_days
    training_rows = []
    for _, row in df_targets.iterrows():
        site_year_daily = df_features[(df_features['bio_year'] == row['bio_year']) & (df_features['site'] == row['site'])]
        if site_year_daily.empty: continue
        target_doy = int(round(row['bloom_doy']))
        day_feat = site_year_daily[site_year_daily['doy'] == target_doy]
        if day_feat.empty: day_feat = site_year_daily.iloc[(site_year_daily['doy'] - target_doy).abs().argsort()[:1]]
        if day_feat.empty: continue
        feat = day_feat.iloc[0].copy()
        feat['bloom_doy'] = row['bloom_doy']
        training_rows.append(feat)
    return pd.DataFrame(training_rows), df_features

def sanitize_forecast_features(df_forecast, df_train):
    df = df_forecast.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = smart_rescale(df)
    df_train['month_day'] = df_train['date'].dt.strftime('%m-%d')
    normals = df_train.groupby(['site', 'month_day'])['TAVG'].mean().reset_index()
    processed = []
    for site in df['site'].unique():
        if site not in SITE_DEFS: continue
        sdf = df[df['site'] == site].sort_values('date').copy()
        last_date = sdf['date'].max()
        last_temp = sdf.loc[sdf.index[-1], 'TAVG']
        pad_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end='2026-05-31')
        pad_df = pd.DataFrame({'date': pad_dates, 'site': site})
        pad_df['month_day'] = pad_df['date'].dt.strftime('%m-%d')
        pad_df = pad_df.merge(normals[normals['site'] == site], on='month_day', how='left')
        bridge_len = min(7, len(pad_df))
        for i in range(bridge_len):
            alpha = (i + 1) / (bridge_len + 1)
            pad_df.loc[i, 'TAVG'] = (1 - alpha) * last_temp + alpha * pad_df.loc[i, 'TAVG']
        pad_df['doy'] = pad_df['date'].dt.dayofyear
        full_df = pd.concat([sdf, pad_df], ignore_index=True)
        cols = ['ao_30d', 'nao_30d', 'oni_30d']
        full_df[cols] = full_df[cols].ffill(limit=14).fillna(0.0)
        full_df['photoperiod'] = full_df['photoperiod'].ffill() 
        full_df = compute_bio_thermal_features(full_df, site)
        processed.append(full_df)
    return pd.concat(processed)

def build_hierarchical_model(data, site_coords, sites_map):
    site_gdd_means = data.groupby('site')['gdd'].mean()
    mu_priors = np.array([site_gdd_means.get(s, 2500) for s in site_coords['site']])
    with pm.Model(coords=site_coords) as model:
        site_idx = data['site'].map(sites_map).values
        intercept_global = pm.Normal('intercept_global', mu=0, sigma=100) 
        tau_intercept = pm.HalfNormal('tau_intercept', sigma=50)
        intercept_offset = pm.Normal('intercept_offset', mu=0, sigma=1, dims='site')
        a_site = pm.Deterministic('a_site', mu_priors + intercept_global + intercept_offset * tau_intercept, dims='site')
        b_chill_global = pm.Normal('b_chill_global', mu=-0.1, sigma=0.5)
        tau_chill = pm.HalfNormal('tau_chill', sigma=0.1)
        b_chill_offset = pm.Normal('b_chill_offset', mu=0, sigma=1, dims='site')
        b_chill = pm.Deterministic('b_chill', b_chill_global + b_chill_offset * tau_chill, dims='site')
        b_photo_global = pm.Normal('b_photo_global', mu=-1.0, sigma=2.0)
        tau_photo = pm.HalfNormal('tau_photo', sigma=0.5)
        b_photo_offset = pm.Normal('b_photo_offset', mu=0, sigma=1, dims='site')
        b_photo = pm.Deterministic('b_photo', b_photo_global + b_photo_offset * tau_photo, dims='site')
        mu_gdd = a_site[site_idx] + b_chill[site_idx] * data['chill'].values + b_photo[site_idx] * data['photoperiod'].values
        weights = np.ones(len(data))
        weights[data['site'] == 'vancouver'] = VANCOUVER_PRECISION_WEIGHT
        sigma = pm.HalfNormal('sigma', sigma=50, dims='site')
        pm.Normal('obs', mu=mu_gdd, sigma=sigma[site_idx] / np.sqrt(weights), observed=data['gdd'].values)
    return model

def get_tripwire_doy(site_path, intercept, b_photo, b_chill, site):
    df = site_path.reset_index(drop=True)
    threshold_path = float(intercept) + float(b_photo) * df['photoperiod'] + float(b_chill) * df['chill']
    valid_mask = (df.index >= 122) & (df['chill'] >= MIN_CHILL_REQUIRED)
    trip_indices = df.index[valid_mask & (df['gdd'] >= threshold_path)].tolist()
    if trip_indices: return min(df.loc[trip_indices[0], 'doy'], MAY_31_DOY)
    return SITE_DEFS[site]['med_doy'] 

def run_2026_stochastic_forecast(idata, features_2026, xgbr_model):
    post = idata.posterior
    forecasts = []
    trained_sites = post.coords['site'].values
    for site in SITE_DEFS.keys():
        if site not in trained_sites: continue
        site_data = features_2026[features_2026['site'] == site].sort_values('date').reset_index(drop=True)
        sim_dates = []
        for _ in range(5000): # High-stability ensemble
            d, c = np.random.randint(0, post.draw.size), np.random.randint(0, post.chain.size)
            a, bc, bp = post['a_site'].sel(site=site, draw=d, chain=c).values, post['b_chill'].sel(site=site, draw=d, chain=c).values, post['b_photo'].sel(site=site, draw=d, chain=c).values
            d_mech = get_tripwire_doy(site_data, a, bp, bc, site)
            day_feat = site_data[site_data['doy'] == int(d_mech)]
            if day_feat.empty: day_feat = site_data.iloc[(site_data['doy'] - d_mech).abs().argsort()[:1]]
            correction = np.clip(xgbr_model.predict(day_feat[['ao_30d', 'nao_30d', 'oni_30d']].values.reshape(1, -1))[0], -10, 10)
            sim_dates.append(max(60, min(151, d_mech + correction)))
        forecasts.append({'site': site, 'year': 2026, 'prediction': int(round(np.median(sim_dates)))})
    return pd.DataFrame(forecasts)

def main():
    print("Starting Sprint 2: Final Ensemble Run (Smart-Scale-Bridge)")
    df_agg, df_daily = load_data_orchestrator('features_train.csv', 'data', NYC_OFFSET_DAYS)
    df_agg = df_agg.dropna(subset=['gdd', 'chill', 'photoperiod', 'bloom_doy', 'ao_30d', 'nao_30d', 'oni_30d'])
    df_loyo = df_agg[df_agg['bio_year'] >= 1990].copy()
    sites = sorted(df_loyo['site'].unique())
    sites_map = {site: i for i, site in enumerate(sites)}
    site_coords = {'site': sites}
    
    print("\nTraining Final Bayesian Hierarchical Model...")
    with build_hierarchical_model(df_loyo, site_coords, sites_map) as model:
        idata_f = pm.sample(1000, tune=1000, cores=1, target_accept=0.99, progressbar=False, random_seed=2026)
    
    post_f = idata_f.posterior.mean(dim=['chain', 'draw'])
    t_m = [get_tripwire_doy(df_daily[(df_daily['bio_year']==r['bio_year']) & (df_daily['site']==r['site'])], post_f['a_site'].sel(site=r['site']).values, post_f['b_photo'].sel(site=r['site']).values, post_f['b_chill'].sel(site=r['site']).values, r['site']) for _, r in df_loyo.iterrows()]
    xgbr_f = xgb.XGBRegressor(max_depth=2, reg_lambda=10.0, n_estimators=50)
    xgbr_f.fit(df_loyo[['ao_30d', 'nao_30d', 'oni_30d']], df_loyo['bloom_doy'].values - np.array(t_m))
    
    df_f = sanitize_forecast_features(pd.read_csv('features_2026_forecast.csv'), df_daily)
    df_sub = run_2026_stochastic_forecast(idata_f, df_f, xgbr_f)
    
    # Format for Submission
    location_map = {'washingtondc': 'washingtondc', 'kyoto': 'kyoto', 'liestal': 'liestal', 'vancouver': 'vancouver', 'newyorkcity': 'nyc'}
    df_sub['location'] = df_sub['site'].map(location_map)
    df_sub = df_sub[['location', 'year', 'prediction']]
    df_sub.to_csv('submission_2026.csv', index=False)
    print("\nFinal 2026 Submission Generated:\n", df_sub)

if __name__ == "__main__": main()
