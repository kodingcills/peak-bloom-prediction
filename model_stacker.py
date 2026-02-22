import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
import xgboost as xgb
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# --- Configuration and Constants ---
# Dynamic Chill Model Constants (from Sprint 1)
A0 = 1.395e5
A1 = 2.567e18
E0 = 12400
E1 = 41400

# Site-specific thermal priors (T_base for GDD) and bloom definitions
SITE_DEFS = {
    'washingtondc': {'lat': 38.85, 't_base': 5, 'cp_threshold': 45, 'bloom_def': 70},
    'kyoto': {'lat': 35.01, 't_base': 0, 'cp_threshold': 38, 'bloom_def': 100},
    'liestal': {'lat': 47.48, 't_base': 0, 'cp_threshold': 55, 'bloom_def': 25},
    'vancouver': {'lat': 49.25, 't_base': 5, 'cp_threshold': 48, 'bloom_def': 70},
    'newyorkcity': {'lat': 40.77, 't_base': 5, 'cp_threshold': 45, 'bloom_def': 70}
}
NPN_PATH = 'data/USA-NPN_status_intensity_observations_data.csv'
NYC_OFFSET_DAYS = 0.5 # From Sprint 1 analysis

# --- Helper Functions (Copied/Adapted from feature_engineer.py) ---
def calculate_photoperiod(lat, doy):
    phi = np.radians(lat)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    cos_h = (np.sin(np.radians(-6.0)) - np.sin(phi) * np.sin(delta)) / (np.cos(phi) * np.cos(delta))
    cos_h = np.clip(cos_h, -1, 1)
    return 2 * np.degrees(np.arccos(cos_h)) / 15.0

# This function is not directly used in model_stacker for GDD calculation,
# but rather the pre-computed GDD from features_train.csv
def dynamic_chill_model_daily_agg(tmin, tmax):
    # Simplified version for single day aggregation, assuming pre-computed hourly
    # For actual PyMC integration, this would need to be an Aesara op or simulated.
    # We will rely on the pre-computed 'cp' feature from feature_engineer.py
    return 0 # Placeholder

# --- Data Loading and Preprocessing ---
def load_and_prepare_data(features_path, targets_dir, nyc_offset_days):
    df_features = pd.read_csv(features_path)
    df_features['date'] = pd.to_datetime(df_features['date'])

    all_targets = []
    for site_name, meta in SITE_DEFS.items():
        target_file = f"{targets_dir}/{site_name.replace('newyorkcity', 'nyc')}.csv"
        if os.path.exists(target_file):
            df_target = pd.read_csv(target_file)
            df_target['bloom_date'] = pd.to_datetime(df_target['bloom_date'])
            df_target = df_target[['year', 'bloom_doy', 'location']]
            df_target.rename(columns={'year': 'bio_year', 'location': 'site'}, inplace=True)
            
            # Filter targets for the current site_name
            df_target = df_target[df_target['site'].str.contains(site_name.replace('newyorkcity', 'nyc'), case=False)]
            
            all_targets.append(df_target)

    if not all_targets:
        raise FileNotFoundError("No target CSV files found or parsed correctly.")

    df_targets = pd.concat(all_targets, ignore_index=True)
    df_targets = df_targets.drop_duplicates(subset=['bio_year', 'site'])

    # Apply NYC offset
    df_targets.loc[df_targets['site'] == 'newyorkcity', 'bloom_doy'] += nyc_offset_days

    df_yearly_aggregated_features = []
    
    # Iterate through unique bloom events (bio_year, site) in df_targets
    for idx, target_row in df_targets.iterrows():
        bio_year = target_row['bio_year']
        site = target_row['site']
        observed_bloom_doy = target_row['bloom_doy']
        
        # Filter daily features for the specific bio_year and site
        site_year_features = df_features[
            (df_features['bio_year'] == bio_year) &
            (df_features['site'] == site)
        ].copy()
        
        if site_year_features.empty:
            continue
        
        # Find the row corresponding to the observed bloom_doy
        # Use abs(doy - observed_bloom_doy) to handle potential off-by-one from float conversion
        bloom_features_candidates = site_year_features[
            (site_year_features['doy'] >= observed_bloom_doy - 1) &
            (site_year_features['doy'] <= observed_bloom_doy + 1)
        ]
        
        if not bloom_features_candidates.empty:
            # Take the closest doy if exact match not found
            features_at_bloom = bloom_features_candidates.iloc[(bloom_features_candidates['doy'] - observed_bloom_doy).abs().argsort()[:1]].iloc[0].copy()
            features_at_bloom['bloom_doy'] = observed_bloom_doy # Keep the original observed bloom_doy
            df_yearly_aggregated_features.append(features_at_bloom)

    if not df_yearly_aggregated_features:
        return pd.DataFrame()
        
    df_merged = pd.DataFrame(df_yearly_aggregated_features)
    
    return df_merged

# --- PyMC Layer 1: Hierarchical Bayesian Model ---
def build_hierarchical_model(data, site_coords, sites_map):
    with pm.Model(coords=site_coords) as hierarchical_model:
        # Map site indices
        site_idx = data['site'].map(sites_map).values
        
        # Non-centered hierarchical intercept
        # Centered: intercept_site = pm.Normal('intercept_site', mu=intercept_global, sigma=tau_intercept, dims='site')
        # Non-centered: intercept_site = intercept_global + intercept_offset * tau_intercept
        intercept_global = pm.Normal('intercept_global', mu=90, sigma=10)
        tau_intercept = pm.HalfCauchy('tau_intercept', beta=1)
        intercept_offset = pm.Normal('intercept_offset', mu=0, sigma=1, dims='site')
        a_site = pm.Deterministic('a_site', intercept_global + intercept_offset * tau_intercept, dims='site')

        # Global coefficient for photoperiod (not site-specific)
        b_photoperiod = pm.Normal('b_photoperiod', mu=0, sigma=0.5)

        # Global coefficient for cp
        b_cp = pm.Normal('b_cp', mu=0, sigma=0.1)

        # Global coefficient for gdd
        b_gdd = pm.Normal('b_gdd', mu=0, sigma=0.01)
        
        # Expected bloom_doy (mechanistic prediction)
        mu_bloom = (
            a_site[site_idx] +
            b_photoperiod * data['photoperiod'].values +
            b_cp * data['cp'].values +
            b_gdd * data['gdd'].values
        )

        # Precision weighting for Vancouver
        weights = np.ones(len(data))
        vancouver_mask = data['site'] == 'vancouver'
        weights[vancouver_mask] = 0.2 # Lower weight for Vancouver

        # Site-specific observational noise (sigma)
        sigma = pm.HalfNormal('sigma', sigma=5, dims='site')
        
        # Likelihood
        pm.Normal('obs', mu=mu_bloom, sigma=sigma[site_idx] / np.sqrt(weights), observed=data['bloom_doy'].values)

    return hierarchical_model

# --- Main Execution ---
def main():
    print("Starting Sprint 2: Hierarchical Prediction Engine")

    # Load data
    df_raw = load_and_prepare_data('features_train.csv', 'data', NYC_OFFSET_DAYS)
    df_raw = df_raw.dropna(subset=['bloom_doy', 'photoperiod', 'cp', 'gdd', 'ao_30d', 'nao_30d', 'oni_30d'])
    
    # Filter for LOYO-CV years
    df_train_loyo = df_raw[df_raw['bio_year'] >= 1880].copy()

    # Create site mappings for PyMC
    sites = df_train_loyo['site'].unique()
    sites_map = {site: i for i, site in enumerate(sites)}
    site_coords = {'site': sites}

    # Prepare data for LOYO CV
    all_years = sorted(df_train_loyo['bio_year'].unique())
    loyo_results = []
    
    # --- LOYO Cross-Validation ---
    print("""
Starting LOYO Cross-Validation (years 1990-2024)...""")
    for test_year in all_years:
        if test_year < 1990 or test_year > 2024: continue # Filter years to 1990-2024
        print(f"  Processing test year: {test_year}")
        
        train_data = df_train_loyo[df_train_loyo['bio_year'] != test_year].copy()
        test_data = df_train_loyo[df_train_loyo['bio_year'] == test_year].copy()

        if train_data.empty or test_data.empty:
            print(f"    Skipping {test_year} due to insufficient data.")
            continue
        
        # Build and sample PyMC model for Layer 1
        with build_hierarchical_model(train_data, site_coords, sites_map) as model:
            idata = pm.sample(2000, tune=2000, cores=1, return_inferencedata=True, random_seed=42, target_accept=0.98)
        
        # Extract posterior means for Layer 1 parameters
        posterior_means = idata.posterior.mean(dim=['chain', 'draw'])
        
        # Calculate Layer 1 predictions for test_data with global cp and gdd
        test_site_idx = test_data['site'].map(sites_map).values
        layer1_predictions = (
            posterior_means['a_site'].values[test_site_idx] +
            posterior_means['b_photoperiod'].values * test_data['photoperiod'].values +
            posterior_means['b_cp'].values * test_data['cp'].values +
            posterior_means['b_gdd'].values * test_data['gdd'].values
        )

        test_data['layer1_pred_doy'] = layer1_predictions
        test_data['layer1_residual'] = test_data['bloom_doy'] - test_data['layer1_pred_doy']

        # --- XGBoost Layer 2 (Residual Model) ---
        xgb_train_features = train_data[['ao_30d', 'nao_30d', 'oni_30d']].fillna(0) # Include ONI
        xgb_train_target = train_data['bloom_doy'] - (
             posterior_means['a_site'].values[train_data['site'].map(sites_map).values] +
             posterior_means['b_photoperiod'].values * train_data['photoperiod'].values +
             posterior_means['b_cp'].values * train_data['cp'].values +
             posterior_means['b_gdd'].values * train_data['gdd'].values
        )

        # XGBoost Model with specified hyperparameters
        xgbr = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=2,
            reg_lambda=5.0, # L2 regularization
            eta=0.05,       # Learning rate
            n_estimators=100, # Can tune this
            random_state=42
        )
        xgbr.fit(xgb_train_features, xgb_train_target)

        xgb_test_features = test_data[['ao_30d', 'nao_30d', 'oni_30d']].fillna(0) # Include ONI
        layer2_residuals = xgbr.predict(xgb_test_features)
        
        test_data['layer2_pred_residual'] = layer2_residuals
        test_data['hybrid_pred_doy'] = test_data['layer1_pred_doy'] + test_data['layer2_pred_residual']


        # Calculate MAE for this test year
        for site in test_data['site'].unique():
            site_test_data = test_data[test_data['site'] == site]
            if not site_test_data.empty:
                mae_mechanistic = np.mean(np.abs(site_test_data['bloom_doy'] - site_test_data['layer1_pred_doy']))
                mae_hybrid = np.mean(np.abs(site_test_data['bloom_doy'] - site_test_data['hybrid_pred_doy']))
                
                loyo_results.append({
                    'bio_year': test_year,
                    'site': site,
                    'mae_mechanistic': mae_mechanistic,
                    'mae_hybrid': mae_hybrid
                })
    
    df_loyo_results = pd.DataFrame(loyo_results)
    print("""
LOYO CV Results (MAE per site per year):""")
    print(df_loyo_results)

    print("""
Average MAE across years (Mechanistic-Only vs. Hybrid-Stacker):""")
    avg_mae = df_loyo_results.groupby('site')[['mae_mechanistic', 'mae_hybrid']].mean()
    print(avg_mae)

    # --- 2026 Forecast Generation ---
    print("""
Generating 2026 Forecast...""")
    df_forecast_features = pd.read_csv('features_2026_forecast.csv')
    df_forecast_features['date'] = pd.to_datetime(df_forecast_features['date'])
    
    # Re-train on all available historical data (up to bio_year 2024 for bloom_doy)
    final_train_data = df_raw[df_raw['bio_year'] < 2025].dropna(subset=['bloom_doy', 'photoperiod', 'cp', 'gdd', 'ao_30d', 'nao_30d', 'oni_30d']).copy()

    # Build and sample PyMC model for Layer 1 with all training data
    with build_hierarchical_model(final_train_data, site_coords, sites_map) as final_model:
        idata_final = pm.sample(2000, tune=2000, cores=1, return_inferencedata=True, random_seed=42, target_accept=0.98)
    
    # Extract posterior means for Layer 1 parameters for final model
    posterior_means_final = idata_final.posterior.mean(dim=['chain', 'draw'])

    # Retrain XGBoost on all final_train_data residuals from the final PyMC model
    final_xgb_train_features = final_train_data[['ao_30d', 'nao_30d', 'oni_30d']].fillna(0)
    final_xgb_train_target_residuals = final_train_data['bloom_doy'] - (
        posterior_means_final['a_site'].values[final_train_data['site'].map(sites_map).values] +
        posterior_means_final['b_photoperiod'].values * final_train_data['photoperiod'].values +
        posterior_means_final['b_cp'].values * final_train_data['cp'].values +
        posterior_means_final['b_gdd'].values * final_train_data['gdd'].values
    )
    
    final_xgbr = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=2,
        reg_lambda=5.0, # L2 regularization
        eta=0.05,       # Learning rate
        n_estimators=100,
        random_state=42
    )
    final_xgbr.fit(final_xgb_train_features, final_xgb_train_target_residuals)

    # --- True Monte Carlo Simulation for 2026 Forecast ---
    def run_2026_stochastic_forecast(idata, features_2026, xgbr_model, sites_map):
        """
        Simulates 1,000 versions of Spring 2026 to find the most likely bloom date.
        """
        posterior = idata.posterior
        mc_forecasts = []

        for site in SITE_DEFS.keys(): # Iterate over actual site names
            # Get the 2026 "Bridge" data (Observed + Normals)
            site_path = features_2026[features_2026['site'] == site].sort_values('doy')
            
            if site_path.empty:
                print(f"  Warning: No forecast data for site {site}. Skipping.")
                continue

            # Sample 1,000 versions of the tree's internal 'threshold' from the posterior
            # The 'a_site' is site-specific, b_gdd and b_cp are global
            simulated_bloom_days = []
            
            # Ensure sites_map has the current site
            if site not in sites_map:
                print(f"  Warning: Site {site} not in sites_map from training. Skipping forecast.")
                continue
            
            site_idx_val = sites_map[site] # Get the integer index for the site
            
            for i in range(1000):
                d = np.random.randint(0, posterior.draw.size)
                c = np.random.randint(0, posterior.chain.size)
                
                # Extract this specific 'version' of the biological requirements
                # For non-centered, use a_site as the intercept
                intercept = posterior['a_site'].sel(site=sites[site_idx_val], draw=d, chain=c).values # Correctly select site
                b_gdd = posterior['b_gdd'].sel(draw=d, chain=c).values
                b_cp = posterior['b_cp'].sel(draw=d, chain=c).values
                b_photoperiod = posterior['b_photoperiod'].sel(draw=d, chain=c).values
                
                # Calculate expected bloom_doy for each day in the site_path
                # This needs to be adjusted for the threshold crossing logic, not direct prediction.
                # The provided example code implies that bloom_signal is the predicted DOY.
                
                # Find when the tree's threshold is triggered by the 2026 weather
                # This logic is based on: bloom_doy = intercept + b_photoperiod * photoperiod + b_gdd * gdd + b_cp * cp
                # We need to find the first 'doy' for which this equation holds true for accumulated values.
                
                # The provided logic: site_path['bloom_signal'] = (intercept + b_gdd * site_path['gdd'] + b_cp * site_path['cp'])
                # This means it's finding the first day where actual doy >= predicted doy.
                # Let's add photoperiod back to this 'bloom_signal' calculation.
                
                site_path['bloom_signal'] = (
                    intercept + 
                    b_photoperiod * site_path['photoperiod'] +
                    b_gdd * site_path['gdd'] + 
                    b_cp * site_path['cp']
                )
                
                # Biological 'Tripwire': The first day where DOY >= predicted DOY
                # This assumes bloom_signal is the required DOY.
                # We need to find the *first day* where the predicted bloom signal is met.
                
                # This needs to be robust to cases where bloom_signal might be very low.
                # Let's define bloom_idx by when the current day (doy) exceeds the calculated signal.
                
                # The original design was "bloom occurs when F* GDDs are met", not a direct regression.
                # However, the user provided this specific run_2026_stochastic_forecast which is a regression type prediction.
                # Let's stick to the provided logic here.
                
                # The first day where the observed doy crosses the predicted doy.
                bloom_indices = site_path[site_path['doy'] >= site_path['bloom_signal']].index
                if not bloom_indices.empty:
                    bloom_idx = bloom_indices[0] # First day where actual DOY meets or exceeds predicted DOY
                    bloom_doy_pred_mechanistic = site_path.loc[bloom_idx, 'doy']
                else:
                    # If no bloom_idx found, take the last doy in the path or a reasonable default
                    bloom_doy_pred_mechanistic = site_path['doy'].max() # Fallback

                # Apply XGBoost atmospheric correction (AO/NAO/ONI signal) for that day
                # Need to get the feature values for the predicted bloom day
                atmos_features = site_path[site_path['doy'] == bloom_doy_pred_mechanistic][['ao_30d', 'nao_30d', 'oni_30d']].fillna(0)
                if not atmos_features.empty:
                    atmos = atmos_features.iloc[0].values.reshape(1, -1)
                    correction = xgbr_model.predict(atmos)[0]
                else:
                    correction = 0 # No correction if no features available for the day
                
                simulated_bloom_days.append(bloom_doy_pred_mechanistic + correction)
            
            mc_forecasts.append({
                'site': site,
                'predicted_bloom_doy': np.median(simulated_bloom_days),
                'lower_90': np.percentile(simulated_bloom_days, 5),
                'upper_90': np.percentile(simulated_bloom_days, 95)
            })
            
        return pd.DataFrame(mc_forecasts)

    # Run the stochastic forecast
    df_submission = run_2026_stochastic_forecast(idata_final, df_forecast_features, final_xgbr, sites_map)
    df_submission['bio_year'] = 2026 
    df_submission.to_csv('submission_2026.csv', index=False)
    print("""
2026 Forecast saved to submission_2026.csv""")

if __name__ == "__main__":
    main()

