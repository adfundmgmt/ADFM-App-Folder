import unittest
import numpy as np
import pandas as pd
from adfm_core.global_macro import equity_snapshot, period_snapshot, comparison_period, country_rows, parse_world_bank

def s(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))

class GlobalMacroTests(unittest.TestCase):
    def test_ytd_uses_prior_year_end_and_excludes_today(self):
        r = equity_snapshot(s(['2025-12-31','2026-01-02','2026-09-08','2026-09-09'], [100,110,120,999]), pd.Timestamp('2026-09-09'), 'YTD')
        self.assertAlmostEqual(r['Value'],20)
        self.assertEqual(r['Baseline'],'2025-12-31')
    def test_stale_quote_never_colored(self):
        r = equity_snapshot(s(['2026-07-01','2026-08-01'],[100,120]),pd.Timestamp('2026-09-09'),'1M')
        self.assertTrue(np.isnan(r['Value']))
        self.assertEqual(r['Status'],'Stale quote')
    def test_missing_baseline_never_shortens_horizon(self):
        r = equity_snapshot(s(['2026-09-01','2026-09-08'],[100,120]),pd.Timestamp('2026-09-09'),'1M')
        self.assertTrue(np.isnan(r['Value']))
    def test_yield_bp_and_negative_yields(self):
        r = period_snapshot(s(['2026-05-01','2026-06-01'],[-.25,.10]),pd.Timestamp('2026-09-09'),'M','Change',1)
        self.assertAlmostEqual(r['Value'],35)
    def test_missing_month_not_bridged(self):
        r = period_snapshot(s(['2026-04-01','2026-06-01'],[3,4]),pd.Timestamp('2026-09-09'),'M','Change',1)
        self.assertTrue(np.isnan(r['Value']))
    def test_annual_change_is_pp_and_forecast_excluded(self):
        r = period_snapshot(s(['2024-12-31','2025-12-31','2026-12-31'],[6,5,4]),pd.Timestamp('2026-09-09'),'Y','Change',1)
        self.assertEqual(r['Value'],-1)
        self.assertEqual(r['Period'],'2025')
    def test_common_period_and_gray_missing(self):
        x = {'USA':s(['2025-12-31'],[3]), 'GBR':s(['2024-12-31','2025-12-31'],[2,1]),'CHN':s(['2024-12-31','2025-12-31'],[4,5]),'FRA':s(['2024-12-31','2025-12-31'],[1,2]),'DEU':s(['2024-12-31'],[0])}
        p = comparison_period(x,pd.Timestamp('2026-09-09'),'Y')
        self.assertEqual(p,'2025')
        rows=country_rows(x,pd.Timestamp('2026-09-09'),'GDP growth',period=p)
        self.assertEqual(len(rows),19)
        self.assertTrue(rows.set_index('ISO').loc[['DEU','ARG'],'Value'].isna().all())
    def test_worldbank_null_not_zero(self):
        payload=[{'pages':1},[{'countryiso3code':'USA','indicator':{'id':'X'},'date':'2025','value':None},{'countryiso3code':'DEU','indicator':{'id':'X'},'date':'2025','value':0}]]
        out=parse_world_bank(payload,'X')
        self.assertNotIn('USA',out)
        self.assertEqual(out['DEU'].iloc[0],0)
        with self.assertRaises(ValueError):
            parse_world_bank([{'pages':2},[]],'X')

if __name__ == '__main__':
    unittest.main()
