import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import the module to be tested
sys.path.append(os.getcwd())
import model_stacker as ms

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
            'bio_year': 2020
        })
        self.df_mock['doy'] = self.df_mock['date'].dt.dayofyear

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
            'ao_30d': 0, 'nao_30d': 0, 'oni_30d': 0
        })
        
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        cutoff = pd.to_datetime("2026-02-28")
        
        self.assertTrue((sanitized[sanitized['date'] > cutoff]['is_observed'] == False).all(), 
                        "Data after cutoff must not be marked observed")
        self.assertEqual(sanitized['date'].max(), pd.to_datetime("2026-05-31"))

    def test_site_label_retention(self):
        """ Test if site labels are retained after padding """
        df_forecast = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-01', '2026-02-21']),
            'site': 'washingtondc',
            'TAVG': [10.0, 12.0], 'TMAX': [15.0, 17.0], 'TMIN': [5.0, 7.0],
            'ao_30d': 0, 'nao_30d': 0, 'oni_30d': 0
        })
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        self.assertFalse(sanitized['site'].isna().any(), "Site labels should not be NaN")
        self.assertTrue((sanitized['site'] == 'washingtondc').all(), "Site labels should remain 'washingtondc'")

    def test_feb_gap_completeness(self):
        """ Test if the Feb 22-28 gap is filled with valid temps """
        # Ends on Feb 21
        df_forecast = pd.DataFrame({
            'date': pd.to_datetime(['2026-02-20', '2026-02-21']),
            'site': 'washingtondc',
            'TAVG': [10.0, 12.0], 'TMAX': [15.0, 17.0], 'TMIN': [5.0, 7.0],
            'ao_30d': 0, 'nao_30d': 0, 'oni_30d': 0
        })
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        gap = sanitized[(sanitized['date'] >= '2026-02-22') & (sanitized['date'] <= '2026-02-28')]
        
        self.assertEqual(len(gap), 7, "Feb gap should have 7 rows")
        self.assertFalse(gap[['TAVG', 'TMAX', 'TMIN']].isna().any().any(), "Gap temps should not be NaN")
        self.assertTrue((gap['TMAX'] >= gap['TMIN']).all(), "TMAX must be >= TMIN in gap")

    def test_reachability_sanity(self):
        """ Test if anchors are reachable by May 31 in mock data """
        df = ms.compute_bio_thermal_path(self.df_mock)
        max_gdd = df['gdd0_cum'].max()
        # Anchor shouldn't be astronomically high
        anchor = 500.0
        self.assertGreater(max_gdd, anchor, "Plausible anchor should be reachable by end of year")

if __name__ == '__main__':
    unittest.main()
