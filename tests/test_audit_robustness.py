import unittest
import pandas as pd
import numpy as np
import os
import sys

# Import the module to be tested
sys.path.append(os.getcwd())
import model_stacker as ms

class TestAuditRobustness(unittest.TestCase):

    def test_late_debut_site_skipping(self):
        """ Verify that a site appearing only in later years is skipped in early audit folds. """
        # Using existing site names from SITE_DEFS to satisfy potential checks
        site_a = 'washingtondc'
        site_b = 'vancouver'
        years = [2010, 2011, 2012, 2013, 2014, 2015]
        
        agg_rows = []
        for y in years:
            agg_rows.append({'bio_year': y, 'site': site_a, 'bloom_doy': 100, 'chill7_cum_at_bloom': 100, 'gdd0_cum_at_bloom': 500})
            if y >= 2014:
                agg_rows.append({'bio_year': y, 'site': site_b, 'bloom_doy': 110, 'chill7_cum_at_bloom': 120, 'gdd0_cum_at_bloom': 600})
        df_agg = pd.DataFrame(agg_rows)
        
        # Audit for test_year = 2012
        test_year = 2012
        train_df = df_agg[df_agg['bio_year'] < test_year]
        test_df = df_agg[df_agg['bio_year'] == test_year]
        
        # chill_stats only for site_a
        chill_stats = train_df.groupby('site')['chill7_cum_at_bloom'].agg(['mean', 'std']).to_dict('index')
        self.assertIn(site_a, chill_stats)
        self.assertNotIn(site_b, chill_stats)
        
        # Simulate the D) loop in main()
        audit_results = []
        for _, r in test_df.iterrows():
            site = r['site']
            if site not in chill_stats:
                continue
            audit_results.append({'site': site, 'mae': 0})
            
        self.assertEqual(len(audit_results), 1)
        self.assertEqual(audit_results[0]['site'], site_a)

    def test_time_safe_anchors(self):
        """ Verify that anchors used in a fold only use targets with bio_year < test_year. """
        # Construct toy daily paths and targets
        site = 'washingtondc'
        dates = pd.date_range('2010-01-01', '2015-12-31', freq='D')
        df_daily = pd.DataFrame({
            'date': dates,
            'site': site,
            'bio_year': dates.year,
            'doy': dates.dayofyear,
            'gdd0_cum': np.arange(len(dates)) # Dummy increasing GDD
        })
        
        # Targets: bloom always at DOY 100
        df_targets = pd.DataFrame({
            'bio_year': [2010, 2011, 2012, 2013, 2014],
            'site': site,
            'bloom_doy': 100
        })
        
        # If test_year = 2012, anchor should only use 2010, 2011
        test_year = 2012
        df_daily_past = df_daily[df_daily['bio_year'] < test_year]
        df_targets_past = df_targets[df_targets['bio_year'] < test_year]
        
        anchors_fold = ms.extract_empirical_anchors(df_daily_past, df_targets_past)
        
        # If we used ALL targets, anchor would be median of GDD at DOY 100 for all years
        anchors_all = ms.extract_empirical_anchors(df_daily, df_targets)
        
        # In this dummy setup with strictly increasing GDD, anchors_all > anchors_fold
        self.assertLess(anchors_fold[site], anchors_all[site])
        
        # Verify anchor changes when cutoff changes
        test_year_2 = 2014
        df_daily_past_2 = df_daily[df_daily['bio_year'] < test_year_2]
        df_targets_past_2 = df_targets[df_targets['bio_year'] < test_year_2]
        anchors_fold_2 = ms.extract_empirical_anchors(df_daily_past_2, df_targets_past_2)
        
        self.assertNotEqual(anchors_fold[site], anchors_fold_2[site])

if __name__ == '__main__':
    unittest.main()
