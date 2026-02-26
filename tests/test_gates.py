import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import logging

# Import the module to be tested
sys.path.append(os.getcwd())
import model_stacker as ms
from seasonal_anomaly_engine import AnomalyEngine

class TestCompilerGates(unittest.TestCase):

    def setUp(self):
        # Create dummy dataframes for testing
        self.site = 'washingtondc'
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.df_mock = pd.DataFrame({
            'date': dates,
            'site': self.site,
            'TAVG': np.random.uniform(5, 25, len(dates)), # Celsius range
            'TMAX': np.random.uniform(10, 30, len(dates)),
            'TMIN': np.random.uniform(0, 20, len(dates)),
            'ao_30d': np.random.uniform(-1, 1, len(dates)),
            'nao_30d': np.random.uniform(-1, 1, len(dates)),
            'oni_30d': np.random.uniform(-1, 1, len(dates)),
            'bio_year': 2020
        })
        self.df_mock['doy'] = self.df_mock['date'].dt.dayofyear
        # Pre-compute bio-thermal paths
        self.df_mock = ms.compute_bio_thermal_path(self.df_mock)
        self.df_mock['bio_year'] = 2020 # Re-add bio_year as it might be lost in path compute if logic changes
        self.df_mock['site'] = self.site

    def test_smart_rescale_tenths(self):
        """ Test if smart_rescale correctly identifies and scales tenths data """
        df_tenths = self.df_mock.copy()
        df_tenths['TMAX'] = df_tenths['TMAX'] / 10.0
        df_tenths['TMIN'] = df_tenths['TMIN'] / 10.0
        df_tenths['TAVG'] = df_tenths['TAVG'] / 10.0
        
        summer_mask = df_tenths['date'].dt.month.isin([7,8])
        df_tenths.loc[summer_mask, 'TMAX'] = 2.5 
        
        scaled_df, was_scaled = ms.smart_rescale(df_tenths, self.site)
        self.assertTrue(was_scaled, "Should have detected tenths scaling")
        self.assertGreater(scaled_df['TMAX'].max(), 15.0, "Should be scaled back to Celsius")

    def test_smart_rescale_celsius(self):
        """ Test if smart_rescale leaves Celsius data alone """
        df_c = self.df_mock.copy()
        summer_mask = df_c['date'].dt.month.isin([7,8])
        df_c.loc[summer_mask, 'TMAX'] = 30.0 
        
        scaled_df, was_scaled = ms.smart_rescale(df_c, self.site)
        self.assertFalse(was_scaled, "Should NOT have scaled Celsius data")
        self.assertEqual(scaled_df['TMAX'].max(), 30.0)

    def test_gdd_monotonicity(self):
        """ Test if GDD accumulation is strictly non-decreasing """
        df = ms.compute_bio_thermal_path(self.df_mock)
        diffs = df['gdd0_cum'].diff().dropna()
        self.assertTrue((diffs >= 0).all(), "GDD must be monotonic")

    def test_cutoff_enforcement(self):
        """ Test if forecast sanitization strictly cuts off after Feb 28 """
        dates = pd.date_range(start='2026-01-01', end='2026-03-15', freq='D')
        df_forecast = pd.DataFrame({
            'date': dates,
            'site': 'washingtondc',
            'TAVG': 10.0, 'TMAX': 15.0, 'TMIN': 5.0,
            'ao_30d': 0.1, 'nao_30d': 0.2, 'oni_30d': 0.3
        })
        
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        cutoff = pd.to_datetime("2026-02-28")
        
        # A) No post-cutoff observed leakage
        post_cutoff = sanitized[sanitized['date'] > cutoff]
        self.assertTrue((post_cutoff['is_observed'] == False).all(), 
                        "Data after cutoff must not be marked observed")
        
        # B) Padding reaches May 31
        self.assertEqual(sanitized['date'].max(), pd.to_datetime("2026-05-31"))
        
        # C) No site label NaNs in padding
        self.assertFalse(sanitized['site'].isna().any(), "Site labels should not be NaN in padding")

    def test_feb_gap_and_climatology(self):
        """ Test Feb 22-28 gap fill and teleconnection climatology fill """
        # Ends on Feb 21
        df_forecast = pd.DataFrame({
            'date': pd.to_datetime(['2026-02-20', '2026-02-21']),
            'site': 'washingtondc',
            'TAVG': [10.0, 12.0], 'TMAX': [15.0, 17.0], 'TMIN': [5.0, 7.0],
            'ao_30d': [1.0, 1.0], 'nao_30d': [0.5, 0.5], 'oni_30d': [0.2, 0.2]
        })
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        
        # A) Feb 22-28 existence and No NaNs in TAVG/TMAX/TMIN
        gap = sanitized[(sanitized['date'] >= '2026-02-22') & (sanitized['date'] <= '2026-02-28')]
        self.assertEqual(len(gap), 7, "Feb gap should have 7 rows")
        self.assertFalse(gap[['TAVG', 'TMAX', 'TMIN']].isna().any().any(), "Gap temps should not be NaN")
        
        # B) Teleconnection fill post-cutoff (ffill from normals/observed)
        may_data = sanitized[sanitized['date'] == '2026-05-31']
        self.assertFalse(may_data[['ao_30d', 'nao_30d', 'oni_30d']].isna().any().any(), 
                        "May teleconnections should be filled (climatology/ffill)")

    def test_anchor_fallback_behavior(self):
        """ Test if anchor extraction correctly handles missing exact DOY with ±3 day fallback """
        # Targets with a DOY that is missing in daily path
        df_targets = pd.DataFrame({
            'bio_year': [2020],
            'site': ['washingtondc'],
            'bloom_doy': [100.0]
        })
        
        # Path MISSING DOY 100, but has DOY 101
        df_path = self.df_mock.copy()
        df_path = df_path[df_path['doy'] != 100].copy()
        
        # Use a unique value for a fallback day (e.g., 101)
        unique_val = 9999.0
        df_path.loc[df_path['doy'] == 101, 'gdd0_cum'] = unique_val
        
        anchors = ms.extract_empirical_anchors(df_path, df_targets)
        self.assertEqual(anchors['washingtondc'], unique_val, "Should have fallen back to DOY 101 (within ±3 days)")

class TestAnomalyPerturbation(unittest.TestCase):
    def setUp(self):
        self.site = 'washingtondc'
        # Normals: TAVG=10, TMAX=15, TMIN=5
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.df_train = pd.DataFrame({
            'date': dates,
            'site': self.site,
            'TAVG': 10.0, 'TMAX': 15.0, 'TMIN': 5.0,
            'ao_30d': 0.0, 'nao_30d': 0.0, 'oni_30d': 0.0
        })
        
        # Forecast: observed up to Feb 28
        f_dates = pd.date_range(start='2026-01-01', end='2026-02-28', freq='D')
        self.df_fc = pd.DataFrame({
            'date': f_dates,
            'site': self.site,
            'TAVG': 10.0, 'TMAX': 15.0, 'TMIN': 5.0,
            'ao_30d': 0.0, 'nao_30d': 0.0, 'oni_30d': 0.0
        })

    def test_anomaly_application(self):
        """ T1: Post-cutoff anomaly application correctness """
        class MockEngine:
            def get_monthly_anomaly(self, site, year, month, var):
                if month == 3: return 2.0  # +2C in March
                return 0.0
        
        sanitized = ms.sanitize_forecast_features(self.df_fc, self.df_train, anomaly_engine=MockEngine())
        
        march_1 = sanitized[sanitized['date'] == '2026-03-01'].iloc[0]
        feb_28 = sanitized[sanitized['date'] == '2026-02-28'].iloc[0]
        
        # Feb 28 should be unchanged (observed)
        self.assertEqual(feb_28['TAVG'], 10.0)
        # March 1 should be shifted by +2.0
        self.assertEqual(march_1['TAVG'], 12.0)
        self.assertEqual(march_1['TMAX'], 17.0)
        self.assertEqual(march_1['TMIN'], 7.0)

    def test_physical_constraints(self):
        """ T1: TMIN <= TAVG <= TMAX holds after perturbation """
        class ExtremeEngine:
            def get_monthly_anomaly(self, site, year, month, var):
                if var == 'TMIN': return 20.0 # Push TMIN above others
                return 0.0
        
        sanitized = ms.sanitize_forecast_features(self.df_fc, self.df_train, anomaly_engine=ExtremeEngine())
        march_1 = sanitized[sanitized['date'] == '2026-03-01'].iloc[0]
        
        self.assertTrue(march_1['TMIN'] <= march_1['TAVG'] <= march_1['TMAX'], 
                        f"Inversion detected: {march_1['TMIN']} > {march_1['TAVG']} or {march_1['TAVG']} > {march_1['TMAX']}")

    def test_fallback_behavior(self):
        """ T3: Explicit missing-anomaly behavior (robust even if CSV exists in repo) """
        import seasonal_anomaly_engine as sae

        original_path = sae.ANOMALY_CSV_PATH
        try:
            # Force a missing anomaly file path for this test
            sae.ANOMALY_CSV_PATH = "data/external/__definitely_missing_seasonal_anomalies__.csv"

            # Test 1: Fallback disabled -> should hard-fail
            with self.assertRaises(FileNotFoundError):
                sae.AnomalyEngine(allow_fallback=False)

            # Test 2: Fallback enabled -> should not fail and should flag fallback
            engine = sae.AnomalyEngine(allow_fallback=True)
            self.assertTrue(engine.fallback_triggered)

            # Test sanitization with fallback engine (should use deterministic climatology)
            sanitized = ms.sanitize_forecast_features(self.df_fc, self.df_train, anomaly_engine=engine)
            march_1 = sanitized[sanitized['date'] == '2026-03-01'].iloc[0]
            self.assertEqual(march_1['TAVG'], 10.0, "Should use unperturbed climatology in fallback")

        finally:
            # Always restore original path to avoid side effects on other tests
            sae.ANOMALY_CSV_PATH = original_path
if __name__ == '__main__':
    unittest.main()
