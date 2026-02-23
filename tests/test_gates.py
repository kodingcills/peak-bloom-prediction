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
        self.site = 'test_site'
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
        # Create tenths data (max < 15.0 in summer)
        df_tenths = self.df_mock.copy()
        df_tenths['TMAX'] = df_tenths['TMAX'] / 10.0
        df_tenths['TMIN'] = df_tenths['TMIN'] / 10.0
        df_tenths['TAVG'] = df_tenths['TAVG'] / 10.0
        
        # Inject summer data to be sure
        summer_mask = df_tenths['date'].dt.month.isin([7,8])
        df_tenths.loc[summer_mask, 'TMAX'] = 2.5 # 2.5C in tenths = 25C. Logic checks raw value < 15.
        
        scaled_df, was_scaled = ms.smart_rescale(df_tenths, self.site)
        self.assertTrue(was_scaled, "Should have detected tenths scaling")
        self.assertGreater(scaled_df['TMAX'].max(), 15.0, "Should be scaled back to Celsius")

    def test_smart_rescale_celsius(self):
        """ Test if smart_rescale leaves Celsius data alone """
        df_c = self.df_mock.copy()
        # Ensure summer max is high enough
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
        # Mock forecast data going into March
        dates = pd.date_range(start='2026-01-01', end='2026-03-15', freq='D')
        df_forecast = pd.DataFrame({
            'date': dates,
            'site': 'washingtondc', # Must be a valid site key
            'TAVG': 10.0, 'TMAX': 15.0, 'TMIN': 5.0,
            'ao_30d': 0, 'nao_30d': 0, 'oni_30d': 0
        })
        
        # Mock training data for normals
        df_train = self.df_mock.copy()
        df_train['site'] = 'washingtondc'
        
        sanitized = ms.sanitize_forecast_features(df_forecast, df_train)
        
        # Check that is_observed is False after Feb 28
        cutoff = pd.to_datetime("2026-02-28")
        observed_mask = sanitized['date'] <= cutoff
        
        # The logic in sanitize_forecast_features marks rows from file as is_observed=True 
        # IF they are before cutoff. Rows after cutoff come from padding and are False.
        # However, if the input df has rows after cutoff, they should be DROPPED or Ignored 
        # in favor of climatology? The code slices: sdf = sdf[sdf['date'] <= FORECAST_CUTOFF_DATE]
        
        self.assertTrue((sanitized[sanitized['date'] > cutoff]['is_observed'] == False).all(), 
                        "Data after cutoff must not be marked observed")
        
        # Verify no data from the original 'future' rows leaked in as observed
        # The original had data up to March 15. The slice should remove them.
        # The function pads from cutoff+1 to May 31.
        self.assertEqual(sanitized['date'].max(), pd.to_datetime("2026-05-31"))

if __name__ == '__main__':
    unittest.main()
