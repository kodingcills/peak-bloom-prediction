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
MODEL_T_BASE_HEAT = 0.0 
MODEL_T_BASE_CHILL = 7.0
MIN_CHILL_REQUIRED = 200.0 # Adjusted after scale audit

# Global State for Reporting
SCALING_REPORT = {}

def check_and_scale_site(df, site):
    """
    Performs data integrity audit per site.
    Returns scaled dataframe and logs decision.
    """
    cols = ['TAVG', 'TMAX', 'TMIN']
    # Check stats on original data
    tavg_max = df['TAVG'].max()
    tavg_mean = df['TAVG'].mean()
    
    # Heuristic: If max < 25C and mean < 8C (conservative for yearly data including summer), it's likely Tenths
    # Kyoto/DC summer is ~30C. If data is 3.0C, it's tenths.
    is_tenths = (tavg_max < 25.0) and (tavg_mean < 10.0)
    
    scale_factor = 1.0
    if is_tenths:
        scale_factor = 10.0
        for col in cols:
            if col in df.columns: df[col] *= scale_factor
            
    SCALING_REPORT[site] = "Tenths (x10)" if is_tenths else "Celsius (x1)"
    return df

def calculate_vpd(tmax, tmin):
    """ Calculates Daily Max Vapor Pressure Deficit (kPa). """
    svp_max = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    svp_min = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    return np.maximum(0, svp_max - svp_min)

def compute_bio_thermal_features(df):
    """ Accumulates GDD, Chill, and 14-day Rolling VPD. """
    df = df.sort_values('date').copy()
    
    # 1. Heat & Chill (Base 0/7)
    df['gdd'] = np.maximum(0, df['TAVG'] - MODEL_T_BASE_HEAT).cumsum()
    df['chill'] = np.maximum(0, MODEL_T_BASE_CHILL - df['TAVG']).cumsum()
    
    # 2. VPD & Rolling Brake
    df['vpd_raw'] = calculate_vpd(df['TMAX'], df['TMIN'])
    # Rolling 14-day mean, min_periods=1 to avoid NaNs at start
    df['vpd'] = df['vpd_raw'].rolling(window=14, min_periods=1).mean()
    
    return df

# --- Data Loading ---
def load_data_orchestrator(features_path, targets_dir, nyc_offset_days):
    df_raw = pd.read_csv(features_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    processed_feats = []
    for site in df_raw['site'].unique():
        if site not in SITE_DEFS: continue
        site_df = df_raw[df_raw['site'] == site].copy()
        
        # 1. Data Integrity & Scaling
        site_df = check_and_scale_site(site_df, site)
        
        # 2. Bio-Thermal Calculation per Year
        if 'bio_year' not in site_df.columns: continue
        site_res = []
        for byear, group in site_df.groupby('bio_year'):
            res = compute_bio_thermal_features(group)
            res['bio_year'] = byear 
            site_res.append(res)
        if site_res: processed_feats.append(pd.concat(site_res))
            
    df_features = pd.concat(processed_feats)

    # Targets
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
    
    # Normals from Training
    df_train['month_day'] = df_train['date'].dt.strftime('%m-%d')
    normals = df_train.groupby(['site', 'month_day'])[['TAVG', 'TMAX', 'TMIN']].mean().reset_index()
    
    processed = []
    for site in df['site'].unique():
        if site not in SITE_DEFS: continue
        sdf = df[df['site'] == site].sort_values('date').copy()
        
        # Integrity Scale Check (re-use logic or force consistency?)
        # We must assume the forecast file follows the same convention as raw training data
        # But we can check stats again.
        sdf = check_and_scale_site(sdf, site) # This will overwrite the report, which is fine
        
        last_date = sdf['date'].max()
        last_vals = sdf.iloc[-1][['TAVG', 'TMAX', 'TMIN']]
        
        pad_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end='2026-05-31')
        pad_df = pd.DataFrame({'date': pad_dates, 'site': site})
        pad_df['month_day'] = pad_df['date'].dt.strftime('%m-%d')
        pad_df = pad_df.merge(normals[normals['site'] == site], on='month_day', how='left')
        
        # Linear Bridge
        bridge_len = min(7, len(pad_df))
        for i in range(bridge_len):
            alpha = (i + 1) / (bridge_len + 1)
            for col in ['TAVG', 'TMAX', 'TMIN']:
                pad_df.loc[i, col] = (1 - alpha) * last_vals[col] + alpha * pad_df.loc[i, col]
        
        pad_df['doy'] = pad_df['date'].dt.dayofyear
        full_df = pd.concat([sdf, pad_df], ignore_index=True)
        
        cols = ['ao_30d', 'nao_30d', 'oni_30d']
        full_df[cols] = full_df[cols].ffill(limit=14).fillna(0.0)
        full_df['photoperiod'] = full_df['photoperiod'].ffill() 
        
        full_df = compute_bio_thermal_features(full_df)
        processed.append(full_df)
    return pd.concat(processed)

# --- Hierarchical Bayesian Model v4.1 (Student-T Pheno-Flex) ---
def build_god_tier_model(data, site_coords, sites_map):
    # Historical Anchors
    site_gdd_means = data.groupby('site')['gdd'].mean()
    mu_anchors = np.array([site_gdd_means.get(s, 500) for s in site_coords['site']])
    
    with pm.Model(coords=site_coords) as model:
        site_idx = data['site'].map(sites_map).values
        
        # 1. Non-Centered Intercept with Student-T Offsets
        # This allows Vancouver/NYC to escape the "Kyoto Gravity Well"
        mu_global = pm.Normal('mu_global', mu=0, sigma=100)
        sigma_site = pm.HalfNormal('sigma_site', sigma=50)
        # Student-T with nu=3 for heavy tails
        offset_raw = pm.StudentT('offset_raw', nu=3, mu=0, sigma=1, dims='site')
        
        a_site = pm.Deterministic('a_site', mu_anchors + mu_global + offset_raw * sigma_site, dims='site')
        
        # 2. Adaptive Site Slopes (Chill Sensitivity)
        # Non-Centered Hierarchical Slopes
        b_chill_global = pm.Normal('b_chill_global', mu=-0.1, sigma=0.5)
        sigma_b_chill = pm.HalfNormal('sigma_b_chill', sigma=0.2)
        offset_b_chill = pm.StudentT('offset_b_chill', nu=3, mu=0, sigma=1, dims='site')
        b_chill = pm.Deterministic('b_chill', b_chill_global + offset_b_chill * sigma_b_chill, dims='site')

        # 3. Pheno-Flex Decay Rate (Log-Normal)
        lam = pm.LogNormal('lam', mu=np.log(0.005), sigma=0.5)
        
        # 4. Photoperiod (Non-Centered)
        b_photo_global = pm.Normal('b_photo_global', mu=-1.0, sigma=2.0)
        sigma_b_photo = pm.HalfNormal('sigma_b_photo', sigma=1.0)
        offset_b_photo = pm.Normal('offset_b_photo', mu=0, sigma=1, dims='site')
        b_photo = pm.Deterministic('b_photo', b_photo_global + offset_b_photo * sigma_b_photo, dims='site')

        # Model Structure
        # F* = a_site - b_chill * (1 - exp(-lam * CP))
        # Note: b_chill is positive magnitude of reduction in this formulation?
        # Or standard slope?
        # Let's use: Threshold = a_site + b_chill * (1 - exp(-lam * CP))
        # If b_chill is negative (standard), then high chill reduces threshold.
        
        cp_norm = data['chill'].values # Keep raw scale
        decay_factor = 1.0 - pm.math.exp(-lam * cp_norm)
        
        # Threshold calculation
        # a_site is the "Low Chill" base requirement (High GDD)
        # As chill increases, threshold drops by b_chill amount
        # So b_chill should be negative.
        
        F_star = a_site[site_idx] + b_chill[site_idx] * cp_norm # Linear approximation for stability in v4.1?
        # User requested Log-Normal prior for lambda, implying exponential model.
        # F* = a_site + b_chill * (1 - exp(-lam * CP)) -> This assumes saturation
        # Let's stick to the requested structure.
        
        # Refined Pheno-Flex:
        # F* = a_site + b_chill * (1 - exp(-lam * CP))
        # If b_chill is large negative, threshold drops.
        
        mu_model = a_site[site_idx] + b_chill[site_idx] * (1 - pm.math.exp(-lam * cp_norm)) + b_photo[site_idx] * data['photoperiod'].values
        
        weights = np.ones(len(data))
        weights[data['site'] == 'vancouver'] = VANCOUVER_PRECISION_WEIGHT
        sigma = pm.HalfNormal('sigma', sigma=50, dims='site')
        
        pm.Normal('obs', mu=mu_model, sigma=sigma[site_idx] / np.sqrt(weights), observed=data['gdd'].values)
        
    return model

def get_tripwire_doy(site_path, a_site, b_chill, lam, b_photo):
    df = site_path.reset_index(drop=True)
    
    decay_factor = 1.0 - np.exp(-lam * df['chill'])
    threshold_path = float(a_site) + float(b_chill) * decay_factor + float(b_photo) * df['photoperiod']
    
    valid_mask = (df.index >= 181) & (df['chill'] >= MIN_CHILL_REQUIRED)
    trip = df.index[valid_mask & (df['gdd'] >= threshold_path)].tolist()
    
    if trip: return df.loc[trip[0], 'doy']
    return MAY_31_DOY # No clamp, just safety ceiling

def run_stochastic_forecast(idata, features, xgbr):
    post = idata.posterior
    forecasts = []
    
    for site in SITE_DEFS.keys():
        if site not in post.coords['site'].values: continue
        sdf = features[features['site'] == site].sort_values('date').reset_index(drop=True)
        
        sims = []
        for _ in range(1000):
            d, c = np.random.randint(0, post.draw.size), np.random.randint(0, post.chain.size)
            
            # Extract scalars
            a = post['a_site'].sel(site=site, draw=d, chain=c).values
            bc = post['b_chill'].sel(site=site, draw=d, chain=c).values
            bp = post['b_photo'].sel(site=site, draw=d, chain=c).values
            l = post['lam'].sel(draw=d, chain=c).values
            
            d_mech = get_tripwire_doy(sdf, a, bc, l, bp)
            
            day_feat = sdf[sdf['doy'] == int(d_mech)]
            if day_feat.empty: day_feat = sdf.iloc[(sdf['doy'] - d_mech).abs().argsort()[:1]]
            
            # XGB features: [AO, NAO, ONI, VPD]
            resid_feats = day_feat[['ao_30d', 'nao_30d', 'oni_30d', 'vpd']].values.reshape(1, -1)
            resid = xgbr.predict(resid_feats)[0]
            
            sims.append(d_mech + resid)
            
        forecasts.append({'site': site, 'prediction': int(np.median(sims))})
        
    return pd.DataFrame(forecasts)

def main():
    print("Starting v4.1 God-Tier Stabilization")
    
    # 1. Load
    df_agg, df_daily = load_data_orchestrator('features_train.csv', 'data', NYC_OFFSET_DAYS)
    df_agg = df_agg.dropna(subset=['gdd', 'chill', 'vpd', 'bloom_doy'])
    
    sites = sorted(df_agg['site'].unique())
    sites_map = {s: i for i, s in enumerate(sites)}
    coords = {'site': sites}
    
    # 2. Audit 2015-2024
    print("\nExecuting Data Integrity Audit...")
    for s, decision in SCALING_REPORT.items():
        print(f"  {s}: {decision}")
        
    print("\nStarting LOYO Audit (2015-2024)...")
    audit_years = range(2015, 2025)
    loyo_res = []
    
    # Pre-tune model? No, loop needs clean state.
    # To save time, we run a simplified loop or just final training as requested "Immediate Action" implied refactor and proceed.
    # But instructions say "Audit Window: LOYO-CV for 2015-2024". I must do it.
    
    for year in audit_years:
        train = df_agg[df_agg['bio_year'] != year]
        test = df_agg[df_agg['bio_year'] == year]
        if test.empty: continue
        
        with build_god_tier_model(train, coords, sites_map) as model:
            # 2 chains, 2000 tune, 2000 draw
            idata = pm.sample(2000, tune=2000, cores=1, target_accept=0.999, progressbar=False, random_seed=42)
            
        post = idata.posterior.mean(dim=['chain', 'draw'])
        
        # Mech Preds
        t_mech = []
        for _, r in train.iterrows():
            sdf = df_daily[(df_daily['site']==r['site']) & (df_daily['bio_year']==r['bio_year'])]
            dm = get_tripwire_doy(sdf, 
                                  post['a_site'].sel(site=r['site']).values, 
                                  post['b_chill'].sel(site=r['site']).values, 
                                  post['lam'].values, 
                                  post['b_photo'].sel(site=r['site']).values)
            t_mech.append(dm)
            
        # Hybrid
        resid = train['bloom_doy'].values - np.array(t_mech)
        xgbr = xgb.XGBRegressor(max_depth=2, n_estimators=50)
        xgbr.fit(train[['ao_30d', 'nao_30d', 'oni_30d', 'vpd']], resid)
        
        for _, row in test.iterrows():
            sdf = df_daily[(df_daily['site']==row['site']) & (df_daily['bio_year']==row['bio_year'])]
            d_m = get_tripwire_doy(sdf, 
                                   post['a_site'].sel(site=row['site']).values, 
                                   post['b_chill'].sel(site=row['site']).values, 
                                   post['lam'].values, 
                                   post['b_photo'].sel(site=row['site']).values)
            
            day_feat = sdf[sdf['doy'] == int(d_m)]
            if day_feat.empty: day_feat = sdf.iloc[(sdf['doy']-d_m).abs().argsort()[:1]]
            d_h = d_m + xgbr.predict(day_feat[['ao_30d', 'nao_30d', 'oni_30d', 'vpd']].values.reshape(1, -1))[0]
            
            loyo_res.append({'year': year, 'site': row['site'], 'mech_err': abs(row['bloom_doy']-d_m), 'hyb_err': abs(row['bloom_doy']-d_h)})
            
    audit_df = pd.DataFrame(loyo_res)
    print("\nValidation Matrix (2015-2024):")
    print(audit_df.groupby('site')[['mech_err', 'hyb_err']].mean())
    
    # 3. Final Training
    print("\nTraining Final God-Tier Model...")
    df_train = df_agg[df_agg['bio_year'] >= 1990].copy()
    with build_god_tier_model(df_train, coords, sites_map) as model:
        idata_f = pm.sample(2000, tune=2000, cores=1, target_accept=0.999, progressbar=False, random_seed=2026)
    
    # Diagnostics
    print("\nMCMC Diagnostics:")
    divs = idata_f.sample_stats.diverging.sum().item()
    rhat = az.rhat(idata_f).max().to_array().max().item()
    ess = az.ess(idata_f).min().to_array().min().item()
    print(f"Divergences: {divs}")
    print(f"Max R-hat: {rhat:.4f}")
    print(f"Min ESS: {ess:.1f}")
    
    # Sensitivity Analysis
    post_f = idata_f.posterior.mean(dim=['chain', 'draw'])
    offsets = post_f['offset_raw'].to_dataframe()
    max_off_idx = offsets.abs().idxmax()
    if isinstance(max_off_idx, pd.Series):
        max_off = max_off_idx.iloc[0]
    else:
        max_off = max_off_idx # Fallback
        
    print(f"\nSensitivity: Site with Max Offset: {max_off}")
    
    # Hybrid Train
    t_mech = []
    for _, r in df_train.iterrows():
        sdf = df_daily[(df_daily['site']==r['site']) & (df_daily['bio_year']==r['bio_year'])]
        dm = get_tripwire_doy(sdf, 
                              post_f['a_site'].sel(site=r['site']).values, 
                              post_f['b_chill'].sel(site=r['site']).values, 
                              post_f['lam'].values, 
                              post_f['b_photo'].sel(site=r['site']).values)
        t_mech.append(dm)
        
    xgbr_f = xgb.XGBRegressor(max_depth=2, n_estimators=100)
    xgbr_f.fit(df_train[['ao_30d', 'nao_30d', 'oni_30d', 'vpd']], df_train['bloom_doy'].values - np.array(t_mech))
    
    # Forecast
    df_fcast = sanitize_forecast_features(pd.read_csv('features_2026_forecast.csv'), df_daily)
    sub = run_stochastic_forecast(idata_f, df_fcast, xgbr_f)
    
    # Check Confidence
    medians = df_train.groupby('site')['bloom_doy'].median()
    stds = df_train.groupby('site')['bloom_doy'].std()
    
    print("\nFinal Forecast Summary:")
    for _, row in sub.iterrows():
        s = row['site']
        p = row['prediction']
        if abs(p - medians[s]) > 2 * stds[s]:
            print(f"WARNING: {s} prediction {p} is >2 sigma from median {medians[s]:.1f}")
            
    sub['year'] = 2026
    sub['location'] = sub['site'].replace({'newyorkcity': 'nyc'})
    sub = sub[['location', 'year', 'prediction']]
    print(sub)
    sub.to_csv('submission_2026.csv', index=False)

if __name__ == "__main__":
    main()
