import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
import xgboost as xgb
import os
import warnings
import logging
from sklearn.metrics import mean_absolute_error
from seasonal_anomaly_engine import AnomalyEngine

# --- Policy & Global Constants (V5.0) ---
# Aligned with context/SYSTEM_CONTEXT_INDEX.md
FORECAST_CUTOFF_DATE = pd.to_datetime("2026-02-28")
MAY_31_DOY = 151
MODEL_T_BASE_HEAT = 0.0      # GDD0
MODEL_T_BASE_CHILL = 7.0     # Reference base for chill units

# Patch P0-B Config
ALLOW_CLIMATOLOGY_FALLBACK = True
ENSEMBLE_SIZE = 100
RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

# --- Submission Controls ---
# Keep residual stacker OFF until rebuilt with winner-style Oct–Mar meteorological features.
USE_RESIDUAL_STACKER = False

# Sparse-site climatology anchoring (NYC/Vancouver): blend mechanistic prediction toward
# historical median based on training support.
USE_CLIMATOLOGY_ANCHOR = True
CLIMATOLOGY_ANCHOR_DENOM_YEARS = 10.0  # w = min(n / denom, 1.0)

# Simplified SITE_DEFS (Task 1)
SITE_DEFS = {
    'washingtondc': {'lat': 38.85, 'bloom_def': 70},
    'kyoto': {'lat': 35.01, 'bloom_def': 100},
    'liestal': {'lat': 47.48, 'bloom_def': 25},
    'vancouver': {'lat': 49.25, 'bloom_def': 70},
    'newyorkcity': {'lat': 40.77, 'bloom_def': 70}
}

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Task 3: Unit-Safe Scaling ---
def smart_rescale(df, site):
    """
    Detects Tenths-of-Celsius encoding using month-aware heuristics.
    Returns (df, is_scaled).
    """
    cols = ['TAVG', 'TMAX', 'TMIN']
    scaled = False
    
    # Isolate seasonal windows
    df['month'] = df['date'].dt.month
    jul_aug = df[df['month'].isin([7, 8])]
    jun_sep = df[df['month'].isin([6, 7, 8, 9])]
    mar_may = df[df['month'].isin([3, 4, 5])]
    
    # Heuristics based on typical max temps
    if not jul_aug.empty:
        if jul_aug['TMAX'].max() < 15.0: scaled = True
    elif not jun_sep.empty:
        if jun_sep['TMAX'].max() < 15.0: scaled = True
    elif not mar_may.empty:
        # Conservative check for spring-only data
        if mar_may['TMAX'].max() < 12.0: scaled = True
    else:
        # No warm season data
        pass

    if scaled:
        for col in cols:
            if col in df.columns: df[col] *= 10.0
            
    return df, scaled

# --- Task 2: Runtime Data Contract ---
def validate_data_contract(df, site):
    """
    Enforces integrity: TMAX>=TMIN, Fahrenheit conversion, GDD semantics.
    """
    # A) Basic Integrity
    if not (df["TMAX"] >= df["TMIN"]).all():
        # Auto-correct minor inversions
        inv = df["TMAX"] < df["TMIN"]
        df.loc[inv, ["TMAX", "TMIN"]] = df.loc[inv, ["TMIN", "TMAX"]].values

    # B) Fahrenheit Detection
    df['month'] = df['date'].dt.month
    warm_window = df[df['month'].isin([6, 7, 8, 9])] # Jun-Sep
    if warm_window.empty: warm_window = df[df['month'].isin([3, 4, 5])] # Mar-May fallback
    
    if not warm_window.empty and warm_window['TMAX'].max() > 60.0:
        print(f"  [Info] {site}: Fahrenheit detected (Max > 60). Converting.")
        for col in ['TAVG', 'TMAX', 'TMIN']:
            df[col] = (df[col] - 32) * 5/9
        # Reslice to check the newly converted values
        warm_window = df[df['month'].isin([6, 7, 8, 9])]
        if warm_window.empty: warm_window = df[df['month'].isin([3, 4, 5])]
            
    # Post-conversion check
    if not warm_window.empty:
        if warm_window['TMAX'].max() > 50.0:
             raise ValueError(f"Critical: {site} TMAX > 50C post-validation.")

    return df

# --- Task 2C / Task 4: Bio-Thermal Path Engine ---
def compute_bio_thermal_path(df):
    """
    Recomputes proxy thermal paths from temps:
      - gdd0_daily/gdd0_cum
      - chill7_daily/chill7_cum
      - vpd_14d
    AND ensures cp/gdd are usable for padded forecast rows:
      - If cp exists: fill forward into padded region; optionally add new chill-hours increments if cold days occur.
      - If gdd exists: extend forcing accumulation into padded region from last known gdd.
      - If cp/gdd do not exist: leaves them absent (mechanistic layer will fallback to chill7/gdd0).
    """
    df = df.sort_values('date').copy()

    # --- Proxy GDD0 semantics (always computed) ---
    df["gdd0_daily"] = np.maximum(df["TAVG"] - MODEL_T_BASE_HEAT, 0)
    df["gdd0_cum"] = df["gdd0_daily"].cumsum()

    # --- Proxy Chill7 semantics (always computed) ---
    df["chill7_daily"] = np.maximum(MODEL_T_BASE_CHILL - df["TAVG"], 0)
    df["chill7_cum"] = df["chill7_daily"].cumsum()

    # --- VPD (always computed) ---
    svp_max = 0.6108 * np.exp((17.27 * df['TMAX']) / (df['TMAX'] + 237.3))
    svp_min = 0.6108 * np.exp((17.27 * df['TMIN']) / (df['TMIN'] + 237.3))
    df['vpd_raw'] = np.maximum(0, svp_max - svp_min)
    df['vpd_14d'] = df['vpd_raw'].rolling(window=14, min_periods=1).mean()

    # --- Monotonicity check (proxy path) ---
    if not (df["gdd0_cum"].diff().dropna() >= 0).all():
        raise ValueError("GDD non-monotonicity detected")

    # --- Ensure cp/gdd extend through padded rows if present ---
    # NOTE: This does NOT create cp/gdd from scratch; it only extends if they exist.
    # This keeps feature_engineer as the source of truth.
    if "cp" in df.columns:
        # If cp has gaps (padded days), extend cp forward with a simple daily chill-hours proxy:
        # +24 hours when TAVG in [0, 7.2], else +0. This matches the Chilling Hours "band" concept.
        if df["cp"].isna().any():
            last_valid = df["cp"].last_valid_index()
            if last_valid is not None:
                base_cp = float(df.loc[last_valid, "cp"])
                after = df.index[df.index > last_valid]
                if len(after) > 0:
                    # daily chill hours proxy
                    chill_hours_daily = 24.0 * ((df.loc[after, "TAVG"] >= 0.0) & (df.loc[after, "TAVG"] <= 7.2)).astype(float)
                    df.loc[after, "cp"] = base_cp + np.cumsum(chill_hours_daily.values)

    if "gdd" in df.columns:
        # If gdd has gaps (padded days), extend forcing accumulation from last known gdd.
        if df["gdd"].isna().any():
            last_valid = df["gdd"].last_valid_index()
            if last_valid is not None:
                base_gdd = float(df.loc[last_valid, "gdd"])
                after = df.index[df.index > last_valid]
                if len(after) > 0:
                    gdd_daily = np.maximum(df.loc[after, "TAVG"] - MODEL_T_BASE_HEAT, 0.0)
                    df.loc[after, "gdd"] = base_gdd + np.cumsum(gdd_daily.values)

    return df

# --- Task 4: Empirical Anchors ---
def extract_empirical_anchors(df_daily, df_targets):
    """
    Computes site-specific anchor medians in forcing space at bloom.
    Prefers 'gdd' (post-chill forcing) if available; falls back to 'gdd0_cum'.
    """
    site_anchors = {}

    HEAT_COL = "gdd" if "gdd" in df_daily.columns else "gdd0_cum"

    for site in SITE_DEFS.keys():
        s_targets = df_targets[df_targets['site'] == site]
        heat_at_bloom = []

        for _, row in s_targets.iterrows():
            path = df_daily[(df_daily['site'] == site) & (df_daily['bio_year'] == row['bio_year'])]
            if path.empty:
                continue

            target_doy = int(round(row['bloom_doy']))
            bloom_row = path[path['doy'] == target_doy]

            if bloom_row.empty:
                offsets = [1, -1, 2, -2, 3, -3]
                for off in offsets:
                    fallback_row = path[path['doy'] == target_doy + off]
                    if not fallback_row.empty:
                        bloom_row = fallback_row
                        break

            if not bloom_row.empty and HEAT_COL in bloom_row.columns:
                heat_at_bloom.append(float(bloom_row[HEAT_COL].values[0]))

        site_anchors[site] = np.median(heat_at_bloom) if heat_at_bloom else None

    valid = [v for v in site_anchors.values() if v is not None]
    global_median = np.median(valid) if valid else 1000.0

    for site in site_anchors:
        if site_anchors[site] is None:
            site_anchors[site] = global_median

    return site_anchors

# --- Task 5: PyMC Model (Smooth Likelihood) ---
def build_hierarchical_model(train_data, site_anchors, sites_map, coords):
    """
    Hierarchical mechanistic model in forcing space:
      Observed: heat_at_bloom  (prefers gdd_at_bloom, fallback gdd0_cum_at_bloom)
      Chill covariate: chill_at_bloom (prefers cp_at_bloom, fallback chill7_cum_at_bloom)
    """
    mu_a_anchors = np.array([site_anchors[s] for s in coords['site']])

    # Choose biology columns
    CHILL_COL = "cp_at_bloom" if "cp_at_bloom" in train_data.columns else "chill7_cum_at_bloom"
    HEAT_COL  = "gdd_at_bloom" if "gdd_at_bloom" in train_data.columns else "gdd0_cum_at_bloom"

    if CHILL_COL not in train_data.columns:
        raise ValueError(f"Missing chill column for model: expected cp_at_bloom or chill7_cum_at_bloom")
    if HEAT_COL not in train_data.columns:
        raise ValueError(f"Missing heat column for model: expected gdd_at_bloom or gdd0_cum_at_bloom")

    chill_stats = train_data.groupby('site')[CHILL_COL].agg(['mean', 'std']).to_dict('index')

    with pm.Model(coords=coords) as model:
        site_idx = train_data['site'].map(sites_map).values

        # ---- a_site: base heat requirement centered at empirical anchors ----
        offset_raw_a = pm.StudentT('offset_raw_a', nu=3, mu=0, sigma=1, dims='site')
        sigma_a = pm.HalfNormal('sigma_a', sigma=50)
        mu_global_a = pm.Normal('mu_global_a', mu=0, sigma=50)
        a_site = pm.Deterministic('a_site', mu_a_anchors + mu_global_a + offset_raw_a * sigma_a, dims='site')

        # ---- chill standardization (z-score per site) ----
        chill_means = np.array([chill_stats[s]['mean'] for s in train_data['site']])
        chill_stds  = np.array([chill_stats[s]['std'] if chill_stats[s]['std'] > 0 else 1.0 for s in train_data['site']])
        chill_stdized = (train_data[CHILL_COL].values - chill_means) / chill_stds

        # ---- b_site: sensitivity of heat requirement to chill (per 1-SD chill) ----
        mu_global_b = pm.Normal('mu_global_b', mu=-100.0, sigma=100.0)
        sigma_b = pm.HalfNormal('sigma_b', sigma=50.0)
        offset_raw_b = pm.StudentT('offset_raw_b', nu=3, mu=0, sigma=1, dims='site')
        b_site = pm.Deterministic('b_site', mu_global_b + offset_raw_b * sigma_b, dims='site')

        # ---- threshold in heat space ----
        F_star = a_site[site_idx] + b_site[site_idx] * chill_stdized

        sigma_obs = pm.HalfNormal('sigma_obs', sigma=50, dims='site')
        pm.Normal('obs', mu=F_star, sigma=sigma_obs[site_idx], observed=train_data[HEAT_COL].values)

    return model

# --- Task 6: Mechanistic Simulation ---
def simulate_bloom_doy(daily_path, a_hat, b_hat, mu_c, sd_c):
    """
    Simulates bloom DOY in standardized chill space.
    Prefers cp/gdd if available; falls back to chill7/gdd0.
    """
    path = daily_path.reset_index(drop=True)

    CHILL_PATH_COL = "cp" if "cp" in path.columns else "chill7_cum"
    HEAT_PATH_COL  = "gdd" if "gdd" in path.columns else "gdd0_cum"

    # Standardize chill using training stats passed in (mu_c/sd_c)
    chill_stdized = (path[CHILL_PATH_COL] - mu_c) / (sd_c if sd_c > 0 else 1.0)
    threshold = a_hat + b_hat * chill_stdized

    # Must occur after March 1 (DOY 60) for plausibility
    mask = (path['doy'] >= 60) & (path[HEAT_PATH_COL] >= threshold)

    crossings = path.index[mask]
    if len(crossings) > 0:
        return min(int(path.loc[crossings[0], 'doy']), MAY_31_DOY)

    return MAY_31_DOY

# --- Task 7: Residual Firewall ---
def train_residual_stacker(df_train):
    """
    Trains XGBoost only on valid mechanistic predictions.
    Uses MAD-based outlier filtering.
    Sparse sites (< MIN_SITE_ROWS) are excluded from stacker training
    and will receive zero residual correction at prediction time,
    trusting the mechanistic model alone.
    """
    MIN_SITE_ROWS = 10

    # 1. Exclude Sentinels
    df_valid = df_train[df_train['t_mech'] != MAY_31_DOY].copy()

    features = ['ao_30d', 'nao_30d', 'oni_30d', 'vpd_14d']
    keep_indices = []
    sparse_sites = []

    for site in SITE_DEFS.keys():
        site_data = df_valid[df_valid['site'] == site]
        if site_data.empty:
            continue

        # Skip sparse sites - stacker cannot learn reliably from too few points
        if len(site_data) < MIN_SITE_ROWS:
            sparse_sites.append(site)
            print(f"[Stacker] Skipping {site}: only {len(site_data)} rows (min={MIN_SITE_ROWS}). "
                  f"Will use mechanistic prediction only.")
            continue

        resid_abs = np.abs(site_data['resid'])
        med = resid_abs.median()
        mad = (resid_abs - med).abs().median() + 1e-6

        # Threshold: median + 2.5 * MAD
        mask = resid_abs < (med + 2.5 * mad)
        keep_indices.extend(site_data[mask].index.tolist())

    df_firewalled = df_valid.loc[keep_indices]

    # Check for empty set
    if df_firewalled.empty:
        print("[Stacker] Warning: Firewall removed all data. Falling back to unfiltered valid.")
        df_firewalled = df_valid

    # Ensure features exist
    for f in features:
        if f not in df_firewalled.columns:
            df_firewalled[f] = 0.0

    xgbr = xgb.XGBRegressor(
        max_depth=2,
        n_estimators=100,
        learning_rate=0.05,
        reg_lambda=10,
        random_state=42
    )
    xgbr.fit(df_firewalled[features], df_firewalled['resid'])

    return xgbr, sparse_sites

# --- Task 8: Forecast Sanitization ---
def sanitize_forecast_features(df_forecast, df_train_daily, anomaly_engine=None):
    """
    Enforces the canonical Feb 28th cutoff and fills temporal gaps.
    Fills teleconnections via climatology post-cutoff.
    Integrates Patch P0-B: Seasonal Anomaly Perturbations.
    """
    # Normals from scaled training paths
    df_train_daily['month_day'] = df_train_daily['date'].dt.strftime('%m-%d')
    normals = df_train_daily.groupby(['site', 'month_day'])[['TAVG', 'TMAX', 'TMIN', 'ao_30d', 'nao_30d', 'oni_30d']].mean().reset_index()
    
    processed = []
    for site in df_forecast['site'].unique():
        if site not in SITE_DEFS: continue
        
        # 1. Slicing to Cutoff
        sdf = df_forecast[df_forecast['site'] == site].sort_values('date').copy()
        sdf, _ = smart_rescale(sdf, site)
        sdf = validate_data_contract(sdf, site)
        
        sdf = sdf[sdf['date'] <= FORECAST_CUTOFF_DATE]
        sdf['is_observed'] = True
        
        # 2. Gap Fill (Feb 21 -> Feb 28)
        last_obs_date = sdf['date'].max()
        if last_obs_date < FORECAST_CUTOFF_DATE:
            gap_dates = pd.date_range(start=last_obs_date + pd.Timedelta(days=1), end=FORECAST_CUTOFF_DATE)
            gap_df = pd.DataFrame({'date': gap_dates, 'site': site, 'is_observed': False})
            gap_df['month_day'] = gap_df['date'].dt.strftime('%m-%d')
            gap_df = gap_df.merge(normals, on=['site', 'month_day'], how='left')
            
            # 7-day Linear Decay Bridge from last observed values
            last_vals = sdf.iloc[-1][['TAVG', 'TMAX', 'TMIN']]
            bridge_len = min(7, len(gap_df))
            for i in range(bridge_len):
                alpha = (i + 1) / (bridge_len + 1)
                for col in ['TAVG', 'TMAX', 'TMIN']:
                    gap_df.loc[i, col] = (1 - alpha) * last_vals[col] + alpha * gap_df.loc[i, col]
            
            sdf = pd.concat([sdf, gap_df], ignore_index=True)
            
        # 3. Climatology Padding (March 1 -> May 31)
        pad_dates = pd.date_range(start=FORECAST_CUTOFF_DATE + pd.Timedelta(days=1), end='2026-05-31')
        pad_df = pd.DataFrame({'date': pad_dates, 'site': site, 'is_observed': False})
        pad_df['month_day'] = pad_df['date'].dt.strftime('%m-%d')
        pad_df = pad_df.merge(normals, on=['site', 'month_day'], how='left')
        
        # 3B. Seasonal Anomaly Perturbation (Patch P0-B)
        if anomaly_engine:
            for i, row in pad_df.iterrows():
                m = row['date'].month
                y = row['date'].year
                for var in ['TAVG', 'TMAX', 'TMIN']:
                    anom = anomaly_engine.get_monthly_anomaly(site, y, m, var)
                    pad_df.loc[i, var] += anom
                
                # Enforce physical constraints: TMIN <= TAVG <= TMAX
                pad_df.loc[i, 'TMAX'] = max(pad_df.loc[i, 'TMAX'], pad_df.loc[i, 'TAVG'], pad_df.loc[i, 'TMIN'])
                pad_df.loc[i, 'TMIN'] = min(pad_df.loc[i, 'TMIN'], pad_df.loc[i, 'TAVG'], pad_df.loc[i, 'TMAX'])
                # Re-center TAVG if it was pushed out of bounds
                pad_df.loc[i, 'TAVG'] = np.clip(pad_df.loc[i, 'TAVG'], pad_df.loc[i, 'TMIN'], pad_df.loc[i, 'TMAX'])

        full_path = pd.concat([sdf, pad_df], ignore_index=True)
        assert not full_path['site'].isna().any(), f"Vanishing site labels detected for {site}"
        
        full_path['doy'] = full_path['date'].dt.dayofyear
        
        # Teleconnections: Climatological Fill Post-Cutoff
        for c in ['ao_30d', 'nao_30d', 'oni_30d']:
             full_path[c] = full_path[c].ffill().fillna(0.0)
            
        # Bio-Thermal Compute (V5 Engine)
        full_path = compute_bio_thermal_path(full_path)
        
        processed.append(full_path)
        
    return pd.concat(processed)
        
def prepare_noise_pool(df_daily):
    """
    Computes daily residuals (observed - climatology) from historical data
    to be used for stochastic perturbation of the forecast.
    """
    df = df_daily.copy()
    df['month_day'] = df['date'].dt.strftime('%m-%d')
    
    cols = ['TAVG', 'TMAX', 'TMIN']
    # Calculate historical normals per site/month_day
    normals = df.groupby(['site', 'month_day'])[cols].mean().reset_index()
    df = df.merge(normals, on=['site', 'month_day'], suffixes=('', '_normal'))
    
    for col in cols:
        df[f'{col}_resid'] = df[col] - df[f'{col}_normal']
        
    return df[['site', 'month_day', 'TAVG_resid', 'TMAX_resid', 'TMIN_resid']]

# Orchestrator 

def load_data_orchestrator(features_path, targets_dir):
    df_raw = pd.read_csv(features_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    # 1. Target Standardization
    all_targets = []
    for site in SITE_DEFS.keys():
        fname = site.replace('newyorkcity', 'nyc')
        t_file = f"{targets_dir}/{fname}.csv"
        if os.path.exists(t_file):
            df_t = pd.read_csv(t_file)
            df_t.rename(columns={'year': 'bio_year', 'location': 'site'}, inplace=True, errors='ignore')
            df_t = df_t[['bio_year', 'bloom_doy', 'site']]
            df_t['site'] = site
            all_targets.append(df_t)
    df_targets = pd.concat(all_targets, ignore_index=True).drop_duplicates(subset=['bio_year', 'site'])

    # 2. Daily Path Engine
    processed_daily = []
    for site in df_raw['site'].unique():
        if site not in SITE_DEFS: continue
        site_df = df_raw[df_raw['site'] == site].copy()
        
        site_df, scaled = smart_rescale(site_df, site)
        site_df = validate_data_contract(site_df, site)
        
        site_res = []
        for byear, group in site_df.groupby('bio_year'):
            res = compute_bio_thermal_path(group)
            res['bio_year'] = byear
            site_res.append(res)
        processed_daily.append(pd.concat(site_res))
    
    df_daily = pd.concat(processed_daily)
    
    # 3. Aggregated Training Table (Bloom Snapshots)
    training_rows = []
    for _, row in df_targets.iterrows():
        path = df_daily[(df_daily['bio_year'] == row['bio_year']) & (df_daily['site'] == row['site'])]
        if path.empty: continue
        
        target_doy = int(round(row['bloom_doy']))
        bloom_row = path[path['doy'] == target_doy]
        
        if bloom_row.empty:
            # Bounded nearest-day fallback (±3 days)
            offsets = [1, -1, 2, -2, 3, -3]
            for off in offsets:
                fallback_row = path[path['doy'] == target_doy + off]
                if not fallback_row.empty:
                    bloom_row = fallback_row
                    break
                    
        if not bloom_row.empty:
            feat = bloom_row.iloc[0].copy() # Snapshots features at bloom
            feat['bloom_doy'] = row['bloom_doy']
            # Explicit Semantic Locks (proxy paths always present)
            feat['gdd0_cum_at_bloom'] = float(feat['gdd0_cum'])
            feat['chill7_cum_at_bloom'] = float(feat['chill7_cum'])

            # Biological locks (preferred if present; fall back to proxies)
            # cp/gdd come from feature_engineer (Chilling Hours or DCM-derived), but may be absent in older artifacts.
            if 'cp' in feat.index and pd.notna(feat['cp']):
                feat['cp_at_bloom'] = float(feat['cp'])
            else:
                feat['cp_at_bloom'] = float(feat['chill7_cum'])  # fallback proxy

            if 'gdd' in feat.index and pd.notna(feat['gdd']):
                feat['gdd_at_bloom'] = float(feat['gdd'])
            else:
                feat['gdd_at_bloom'] = float(feat['gdd0_cum'])   # fallback proxy

            training_rows.append(feat)
                        
    df_train_agg = pd.DataFrame(training_rows)
    
    return df_train_agg, df_daily, df_targets

def compute_site_training_stats(df_targets: pd.DataFrame):
    """
    Compute per-site training counts and historical median bloom DOY from observed targets.
    Counts are #distinct bio_year rows per site.
    Medians are median(bloom_doy) per site.
    """
    required = {'site', 'bio_year', 'bloom_doy'}
    missing = required - set(df_targets.columns)
    if missing:
        raise ValueError(f"df_targets missing required columns: {sorted(missing)}")

    t = df_targets.drop_duplicates(subset=['site', 'bio_year']).copy()
    counts = t.groupby('site')['bio_year'].nunique().to_dict()
    medians = t.groupby('site')['bloom_doy'].median().round().astype(int).to_dict()
    return counts, medians


def blend_with_climatology(site: str, mech_pred_doy: float, site_counts: dict, site_medians: dict) -> int:
    """
    w = min(n / CLIMATOLOGY_ANCHOR_DENOM_YEARS, 1)
    pred = w * mech_pred + (1 - w) * median
    """
    if site not in site_counts or site not in site_medians:
        return int(round(mech_pred_doy))  # unseen site fallback

    n = float(site_counts[site])
    w = min(n / float(CLIMATOLOGY_ANCHOR_DENOM_YEARS), 1.0)
    median = float(site_medians[site])
    return int(round(w * float(mech_pred_doy) + (1.0 - w) * median))

def main():
    print("Initializing V5.0 God-Tier Architecture")

    # -----------------------------
    # Submission Controls (P1/P2)
    # -----------------------------
    USE_RESIDUAL_STACKER = False          # Priority 2: disable stacker until rebuilt
    USE_CLIMATOLOGY_ANCHOR = True         # Priority 1: anchor sparse sites
    CLIMATOLOGY_ANCHOR_DENOM_YEARS = 10.0 # w = min(n/denom, 1)

    def _compute_site_training_stats(df_siteyear: pd.DataFrame):
        """
        Compute per-site support (n unique bio_year) + historical median bloom_doy
        from the *actual training table* df_agg/train_df (not raw targets spanning centuries).
        Requires columns: site, bio_year, bloom_doy
        """
        req = {'site', 'bio_year', 'bloom_doy'}
        missing = req - set(df_siteyear.columns)
        if missing:
            raise ValueError(f"Training df missing required columns: {sorted(missing)}")

        t = df_siteyear.dropna(subset=['bloom_doy']).drop_duplicates(subset=['site', 'bio_year']).copy()
        counts = t.groupby('site')['bio_year'].nunique().to_dict()
        medians = t.groupby('site')['bloom_doy'].median().round().astype(int).to_dict()
        return counts, medians

    def _blend_with_climatology(site: str, pred_doy: float, site_counts: dict, site_medians: dict) -> int:
        """
        Blend prediction toward site median based on support:
          w = min(n/denom, 1)
          blended = w*pred + (1-w)*median
        """
        if (site not in site_counts) or (site not in site_medians):
            return int(round(pred_doy))
        n = float(site_counts[site])
        w = min(n / float(CLIMATOLOGY_ANCHOR_DENOM_YEARS), 1.0)
        med = float(site_medians[site])
        return int(round(w * float(pred_doy) + (1.0 - w) * med))

    # -----------------------------
    # 1) Data Ingestion
    # -----------------------------
    df_agg, df_daily, df_targets = load_data_orchestrator('features_train.csv', 'data')

    # Global climatology stats based on the *actual training table* df_agg
    site_counts_global, site_medians_global = _compute_site_training_stats(df_agg)

    # -----------------------------
    # 2) Empirical Anchors
    # -----------------------------
    site_anchors = extract_empirical_anchors(df_daily, df_targets)
    print("Empirical Anchors (GDD0):", site_anchors)

    sites = sorted(df_agg['site'].unique())
    sites_map = {s: i for i, s in enumerate(sites)}
    coords = {'site': sites}

    # -----------------------------
    # 3) Expanding Window Validation (Time-Safe)
    # -----------------------------
    validation_years = sorted([y for y in df_agg['bio_year'].unique() if y >= 2015])

    audit_results = []
    print(f"\nStarting Expanding Window Audit (2015-2024)...")

    for test_year in validation_years:
        if test_year > 2024:
            continue

        train_mask = df_agg['bio_year'] < test_year
        test_mask = df_agg['bio_year'] == test_year

        train_df = df_agg[train_mask].copy()
        test_df = df_agg[test_mask].copy()

        if train_df.empty or test_df.empty:
            continue

        # Fold-specific climatology stats (based on available training support in this fold)
        fold_counts, fold_medians = _compute_site_training_stats(train_df)

        # 2.1 Fix: Time-safe anchors for audit loop
        site_anchors_fold = extract_empirical_anchors(
            df_daily[df_daily['bio_year'] < test_year],
            df_targets[df_targets['bio_year'] < test_year]
        )

        # A) Fit Mechanistic
        with build_hierarchical_model(train_df, site_anchors_fold, sites_map, coords) as model:
            trace = pm.sample(
                1000, tune=1000, cores=1, target_accept=0.999,
                progressbar=False, random_seed=42
            )

        post = trace.posterior.mean(dim=['chain', 'draw'])
        chill_stats = train_df.groupby('site')['chill7_cum_at_bloom'].agg(['mean', 'std']).to_dict('index')

        # B) Infer t_mech on Train (Standardized Space)
        t_mech_train = []
        for _, r in train_df.iterrows():
            site = r['site']
            mu_c = chill_stats[site]['mean']
            sd_c = chill_stats[site]['std'] if chill_stats[site]['std'] > 0 else 1.0

            b_hat = post['b_site'].sel(site=site).values
            a_hat = post['a_site'].sel(site=site).values

            path = df_daily[(df_daily['site'] == site) & (df_daily['bio_year'] == r['bio_year'])]
            tm = simulate_bloom_doy(path, a_hat, b_hat, mu_c, sd_c)
            t_mech_train.append(tm)

        train_df['t_mech'] = t_mech_train
        train_df['resid'] = train_df['bloom_doy'] - train_df['t_mech']

        # C) Residual Model (disabled by default)
        xgbr, sparse_sites = (None, set())
        if USE_RESIDUAL_STACKER:
            xgbr, sparse_sites = train_residual_stacker(train_df)

        # D) Predict on Test
        for _, r in test_df.iterrows():
            site = r['site']

            # 2.1 Fix: skip if site not in this fold's training set (late debut)
            if site not in chill_stats:
                continue

            mu_c = chill_stats[site]['mean']
            sd_c = chill_stats[site]['std'] if chill_stats[site]['std'] > 0 else 1.0

            b_hat = post['b_site'].sel(site=site).values
            a_hat = post['a_site'].sel(site=site).values

            path = df_daily[(df_daily['site'] == site) & (df_daily['bio_year'] == r['bio_year'])]
            tm = simulate_bloom_doy(path, a_hat, b_hat, mu_c, sd_c)

            # Residual ML Correction (disabled unless flag enabled)
            if (not USE_RESIDUAL_STACKER) or (xgbr is None) or (site in sparse_sites):
                resid_pred = 0.0
            else:
                bloom_row = path[path['doy'] == int(tm)]
                if bloom_row.empty:
                    bloom_row = path.iloc[-1:]
                feats = bloom_row[['ao_30d', 'nao_30d', 'oni_30d', 'vpd_14d']]
                resid_pred = float(xgbr.predict(feats)[0])

            pred_mech = float(tm)
            pred_hybrid = float(tm) + float(resid_pred)

            # Priority 1: climatology anchor for sparse sites (applied to final preds)
            if USE_CLIMATOLOGY_ANCHOR:
                pred_mech = _blend_with_climatology(site, pred_mech, fold_counts, fold_medians)
                pred_hybrid = _blend_with_climatology(site, pred_hybrid, fold_counts, fold_medians)

            audit_results.append({
                'site': site,
                'year': int(test_year),
                'mae_mech': abs(float(r['bloom_doy']) - float(pred_mech)),
                'mae_hybrid': abs(float(r['bloom_doy']) - float(pred_hybrid))
            })

    # Report
    audit_df = pd.DataFrame(audit_results)
    print("\n=== Ablation Report (Expanding Window) ===")
    if not audit_df.empty:
        print(audit_df.groupby('site')[['mae_mech', 'mae_hybrid']].mean())
        print("Overall:", audit_df[['mae_mech', 'mae_hybrid']].mean())
    else:
        print("No audit results produced (check validation_years / data coverage).")

    # -----------------------------
    # 4) Final Production Run
    # -----------------------------
    print("\nTraining Final Production Model (All History)...")
    final_train = df_agg[df_agg['bio_year'] <= 2025].copy()

    with build_hierarchical_model(final_train, site_anchors, sites_map, coords) as model:
        idata = pm.sample(
            2000, tune=2000, cores=1, target_accept=0.999,
            progressbar=False, random_seed=2026
        )

    post_f = idata.posterior.mean(dim=['chain', 'draw'])
    chill_stats_final = final_train.groupby('site')['chill7_cum_at_bloom'].agg(['mean', 'std']).to_dict('index')

    # Re-sim mechanics on full set (Standardized Space)
    t_mech_final = []
    for _, r in final_train.iterrows():
        site = r['site']
        mu_c = chill_stats_final[site]['mean']
        sd_c = chill_stats_final[site]['std'] if chill_stats_final[site]['std'] > 0 else 1.0

        b_hat = post_f['b_site'].sel(site=site).values
        a_hat = post_f['a_site'].sel(site=site).values

        path = df_daily[(df_daily['site'] == site) & (df_daily['bio_year'] == r['bio_year'])]
        tm = simulate_bloom_doy(path, a_hat, b_hat, mu_c, sd_c)
        t_mech_final.append(tm)

    final_train['t_mech'] = t_mech_final
    final_train['resid'] = final_train['bloom_doy'] - final_train['t_mech']

    # Final production stacker (disabled unless flag enabled)
    xgbr_f, sparse_sites_f = (None, set())
    if USE_RESIDUAL_STACKER:
        xgbr_f, sparse_sites_f = train_residual_stacker(final_train)

    # -----------------------------
    # 5) 2026 Forecast
    # -----------------------------
    print("\nGenerating 2026 Forecast (Patch P0-B Ensemble)...")

    # Initialize Anomaly Engine
    try:
        anomaly_engine = AnomalyEngine(allow_fallback=ALLOW_CLIMATOLOGY_FALLBACK)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        return

    noise_pool = prepare_noise_pool(df_daily)

    df_raw_forecast = pd.read_csv('features_2026_forecast.csv')
    df_raw_forecast['date'] = pd.to_datetime(df_raw_forecast['date'])

    # Baseline forecast (normals + monthly anomalies)
    df_forecast_base = sanitize_forecast_features(df_raw_forecast, df_daily, anomaly_engine=anomaly_engine)

    predictions = []
    post_draws = idata.posterior

    cutoff_ts = pd.to_datetime(FORECAST_CUTOFF_DATE)

    for site in SITE_DEFS.keys():
        site_path_base = df_forecast_base[df_forecast_base['site'] == site].copy()
        if site_path_base.empty:
            continue

        mu_c = chill_stats_final[site]['mean']
        sd_c = chill_stats_final[site]['std'] if chill_stats_final[site]['std'] > 0 else 1.0

        sims = []
        fail_count = 0

        print(f"  Running ensemble for {site}...")
        for _ in range(ENSEMBLE_SIZE):
            # 1) Sample Model Parameters
            d = np.random.randint(0, post_draws.draw.size)
            c = np.random.randint(0, post_draws.chain.size)
            b_hat = post_draws['b_site'].sel(site=site, draw=d, chain=c).values
            a_hat = post_draws['a_site'].sel(site=site, draw=d, chain=c).values

            # 2) Sample Stochastic Path
            perturbed_path = site_path_base.copy()

            # Perturb post-cutoff padding only
            pad_mask = (perturbed_path['is_observed'] == False) & (perturbed_path['date'] > cutoff_ts)

            pad_dates = perturbed_path.loc[pad_mask, 'date'].dt.strftime('%m-%d')
            for m_d in pad_dates.unique():
                day_mask = pad_mask & (perturbed_path['date'].dt.strftime('%m-%d') == m_d)
                pool_slice = noise_pool[(noise_pool['site'] == site) & (noise_pool['month_day'] == m_d)]
                if not pool_slice.empty:
                    resids = pool_slice.sample(n=int(day_mask.sum()), replace=True)
                    perturbed_path.loc[day_mask, 'TAVG'] += resids['TAVG_resid'].values
                    perturbed_path.loc[day_mask, 'TMAX'] += resids['TMAX_resid'].values
                    perturbed_path.loc[day_mask, 'TMIN'] += resids['TMIN_resid'].values

            # Enforce physical constraints: TMIN <= TAVG <= TMAX
            perturbed_path['TMAX'] = np.maximum(perturbed_path['TMAX'], perturbed_path['TAVG'])
            perturbed_path['TMIN'] = np.minimum(perturbed_path['TMIN'], perturbed_path['TAVG'])

            # Recalculate cumulative bio-thermal paths
            perturbed_path = compute_bio_thermal_path(perturbed_path)

            # 3) Simulate
            tm = simulate_bloom_doy(perturbed_path, a_hat, b_hat, mu_c, sd_c)
            if tm == MAY_31_DOY:
                fail_count += 1

            # 4) Residual ML correction (disabled unless flag enabled)
            if (not USE_RESIDUAL_STACKER) or (xgbr_f is None) or (site in sparse_sites_f):
                resid = 0.0
            else:
                bloom_row = perturbed_path[perturbed_path['doy'] == int(tm)]
                if bloom_row.empty:
                    bloom_row = perturbed_path.iloc[-1:]
                feats = bloom_row[['ao_30d', 'nao_30d', 'oni_30d', 'vpd_14d']]
                resid = float(xgbr_f.predict(feats)[0])

            sims.append(float(tm) + float(resid))

        # Log failure rate
        fail_rate = fail_count / float(ENSEMBLE_SIZE)
        print(f"    Fail rate (DOY={MAY_31_DOY}): {fail_rate:.1%}")
        if fail_rate > 0.1:
            print(f"    WARNING: High reachability failure rate for {site}")

        final_doy = int(round(np.median(sims)))

        # Priority 1: climatology anchor (global stats)
        if USE_CLIMATOLOGY_ANCHOR:
            final_doy = _blend_with_climatology(site, final_doy, site_counts_global, site_medians_global)

        predictions.append({
            'location': site.replace('newyorkcity', 'nyc'),
            'year': 2026,
            'prediction': final_doy
        })

    sub = pd.DataFrame(predictions)
    print("\n=== Final Submission ===")
    print(sub)
    sub.to_csv('submission_2026.csv', index=False)
    
if __name__ == "__main__":
    main()