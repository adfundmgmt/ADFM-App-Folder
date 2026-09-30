import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adfm_core import bond_event_study as study


class BondResearchUpgradeTests(unittest.TestCase):
    def test_failed_breakout_must_reenter_original_level(self):
        rates = pd.Series([4.] * 24 + [5., 4.5, 3.9], index=pd.date_range('2020-01-01', periods=27, freq='MS'))
        frame = study.signal_frame(rates, 'monthly', 'Failed Breakout', {'memory': 3, 'reversal_components': 0})
        self.assertFalse(frame['Signal'].iloc[-2], '4.5 remains above the original 4.0 breakout level')
        self.assertTrue(frame['Signal'].iloc[-1])

    def test_monthly_signals_are_dated_at_month_end_availability_proxy(self):
        frame = study.signal_frame(pd.Series([4., 4.1], index=pd.to_datetime(['2026-01-01', '2026-03-01'])), 'monthly', 'Early Warning')
        self.assertEqual(frame.index[0], pd.Timestamp('2026-01-31'))
        self.assertTrue(np.isnan(frame.loc['2026-02-28', 'Yield']))

    def test_outcomes_include_excursions_and_sparse_intervals_require_all_months(self):
        rates = pd.Series([4., 4.5, 3., 3.5], index=pd.date_range('2025-01-01', periods=4, freq='MS'))
        _, history = study.event_summary(rates, pd.DatetimeIndex(['2025-01-01']), 'monthly')
        self.assertEqual(history.loc[0, '3M adverse (bp)'], 50.)
        self.assertEqual(history.loc[0, '3M favorable (bp)'], -100.)
        _, missing = study.event_summary(rates.drop(rates.index[1]), pd.DatetimeIndex(['2025-01-01']), 'monthly')
        self.assertTrue(np.isnan(missing.loc[0, '3M adverse (bp)']))

    def test_regime_controls_ci_and_purged_holdout(self):
        rates = pd.Series(3 + np.sin(np.arange(240) / 5) + np.arange(240) / 1000, index=pd.date_range('2000-01-01', periods=240, freq='MS'))
        events = rates.index[[40, 41, 80, 120, 155, 170, 190, 210]]
        summary, history = study.event_summary(rates, events, 'monthly')
        self.assertEqual(summary.loc['Independent N', '3M'], 7)
        self.assertGreater(summary.loc['Control N', '3M'], 0)
        self.assertLessEqual(summary.loc['Edge CI low', '3M'], summary.loc['Median edge', '3M'])
        self.assertGreaterEqual(summary.loc['Edge CI high', '3M'], summary.loc['Median edge', '3M'])
        self.assertEqual(summary.loc['Train N', '3M'] + summary.loc['Holdout N', '3M'], 7)
        self.assertIn('small sample', summary.attrs['cautions']['3M'])
        controls = summary.attrs['controls']['3M']
        self.assertTrue(all(b - a >= 3 for a, b in zip(controls, controls[1:], strict=False)))
        regimes = study.regime_labels(study.monthly_history(rates), 'monthly')
        for event_pos, control_pos in summary.attrs['matches']['3M']:
            self.assertEqual(regimes.iloc[event_pos], regimes.iloc[control_pos])
            self.assertTrue(all(abs(control_pos - ep) >= 3 for ep in [40, 41, 80, 120, 155, 170, 190, 210]))
        self.assertTrue(set(history['Partition']) <= {'Train', 'Holdout'})


class DailySovereignUpgradeTests(unittest.TestCase):
    def test_official_daily_cache_is_filtered_and_fresh_refresh_wins(self):
        from adfm_core import sovereign_daily as daily
        rows = pd.DataFrame({'date': pd.to_datetime(['2026-09-01', '2026-09-01', '2026-09-01']), 'ccy': ['USD', 'NZD', 'EUR'], 'tenor': ['10Y'] * 3, 'value': [4., 4.5, 3.], 'source': ['us_treasury', 'interest_co_nz_b2', 'ecb_yc'], 'fetched_at': [pd.Timestamp('2026-09-02')] * 3})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'yields.parquet'
            rows.to_parquet(path)
            fresh = rows.iloc[[0]].copy()
            fresh['value'] = 4.2
            def fail():
                raise RuntimeError('offline')
            panel, status = daily.load_daily_sovereigns(cache_path=path, fetchers={'United States': lambda: fresh, 'Canada': fail}, today=pd.Timestamp('2026-09-03'))
        self.assertEqual(panel['United States'].iloc[-1], 4.2)
        self.assertNotIn('Euro area', panel)
        self.assertTrue(panel['New Zealand'].dropna().empty)
        self.assertIn('offline', status.set_index('country').loc['Canada', 'error'])
        self.assertEqual(status.set_index('country').loc['United States', 'basis'], 'Treasury par constant-maturity curve')

    def test_invalid_direct_data_leaves_valid_cache_with_failure(self):
        from adfm_core import sovereign_daily as daily
        rows = pd.DataFrame({'date': pd.to_datetime(['2026-09-01']), 'ccy': ['CAD'], 'tenor': ['10Y'], 'value': [4.], 'source': ['boc_valet'], 'fetched_at': [pd.Timestamp('2026-09-02')]})
        bad = rows.copy()
        bad['tenor'] = '2Y'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'yields.parquet'
            rows.to_parquet(path)
            panel, status = daily.load_daily_sovereigns(cache_path=path, fetchers={'Canada': lambda: bad}, today=pd.Timestamp('2026-09-03'))
        self.assertEqual(panel['Canada'].iloc[-1], 4.)
        self.assertTrue(status.set_index('country').loc['Canada', 'error'])



class BondPageUpgradeTests(unittest.TestCase):
    def tearDown(self):
        import streamlit as st
        st.cache_data.clear()

    def test_daily_and_monthly_are_separate_sortable_views(self):
        from unittest.mock import patch

        import streamlit as st
        from streamlit.testing.v1 import AppTest

        from adfm_core.sovereign_daily import DAILY_SOVEREIGNS
        st.cache_data.clear()
        def fred(symbols, **kwargs):
            index = pd.date_range('2020-01-01', periods=70, freq='MS')
            return pd.DataFrame({s: 3 + np.sin(np.arange(len(index)) / 4) for s in symbols}, index=index), pd.DataFrame()
        index = pd.bdate_range('2023-01-01', periods=300)
        official = pd.DataFrame({item[0]: 4 + np.sin(np.arange(len(index)) / 20) for item in DAILY_SOVEREIGNS}, index=index)
        status = pd.DataFrame([{'country': x[0], 'basis': x[3], 'source': x[2], 'error': '', 'url': x[4]} for x in DAILY_SOVEREIGNS])
        with patch('adfm_core.primary_data.fetch_fred_symbols', side_effect=fred), patch('adfm_core.sovereign_daily.load_daily_sovereigns', return_value=(official, status)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'pages/2_Global_Macro_Regime.py')).run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any('Instrument' in item.value.columns for item in app.dataframe))
            app.selectbox(key='bond_view').set_value('Global sovereign').run(timeout=30)
            self.assertEqual(app.radio(key='bond_sovereign_frequency').value, 'Daily official')
            app.radio(key='bond_sovereign_frequency').set_value('Monthly OECD').run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.selectbox(key='bond_period').value, 'Max')
            app.radio(key='bond_sovereign_frequency').set_value('Daily official').run(timeout=30)
            self.assertEqual(list(app.exception), [])
            table = next(item.value for item in app.dataframe if 'Instrument' in item.value.columns)
            self.assertEqual(len(table), 8)
            self.assertIn('Government nominal spot curve', table['Source / basis'].tolist())
            self.assertEqual(len(app.get('plotly_chart')), 1)


class BondPerformanceTests(unittest.TestCase):
    def test_max_daily_history_statistics_finish_within_five_seconds(self):
        import time
        rates = pd.Series(4 + np.sin(np.arange(16000) / 50), index=pd.bdate_range('1960-01-01', periods=16000))
        start = time.perf_counter()
        summary, _ = study.event_summary(rates, rates.index[250::200], 'daily')
        elapsed = time.perf_counter() - start
        self.assertEqual(summary.loc['Independent N', '3M'], 79)
        self.assertLess(elapsed, 5., f'Max history calculation took {elapsed:.2f}s')


class SovereignDeadlineTests(unittest.TestCase):
    def test_slow_source_deadline_preserves_fast_country(self):
        import time

        from adfm_core.sovereign_daily import load_daily_sovereigns
        today = pd.Timestamp('2026-09-03')
        fresh = pd.DataFrame({'date': [today], 'ccy': ['USD'], 'tenor': ['10Y'], 'value': [4.], 'source': ['us_treasury'], 'fetched_at': [today]})
        def slow():
            time.sleep(.15)
            return pd.DataFrame()
        with tempfile.TemporaryDirectory() as temp:
            start = time.perf_counter()
            panel, status = load_daily_sovereigns(cache_path=Path(temp) / 'missing.parquet', fetchers={'Canada': slow, 'United States': lambda: fresh}, today=today, timeout=.02)
            self.assertLess(time.perf_counter() - start, .1)
        self.assertEqual(panel['United States'].iloc[-1], 4.)
        self.assertIn('deadline', status.set_index('country').loc['Canada', 'error'])



class BondAvailabilityTests(unittest.TestCase):
    def test_monthly_snapshot_labels_month_end_as_unverified_proxy(self):
        from adfm_core.bond_monitor import monthly_snapshot
        snapshot = monthly_snapshot(pd.Series([4.], index=pd.to_datetime(['2026-08-01'])), pd.Timestamp('2026-09-10'))
        self.assertEqual(snapshot['Availability'], '2026-08-31 · release unverified')

    def test_monthly_shaped_daily_cache_is_not_accepted(self):
        from adfm_core.sovereign_daily import load_daily_sovereigns
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        rows = pd.DataFrame({'date': dates, 'ccy': 'USD', 'tenor': '10Y', 'value': 4., 'source': 'us_treasury', 'fetched_at': pd.Timestamp('2026-09-01')})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'yields.parquet'
            rows.to_parquet(path)
            panel, _ = load_daily_sovereigns(cache_path=path, fetchers={}, today=pd.Timestamp('2026-09-03'))
        self.assertTrue(panel['United States'].dropna().empty)


class FixedBreakoutWindowTests(unittest.TestCase):
    def test_each_new_breakout_keeps_its_original_level(self):
        rates = pd.Series([4.] * 24 + [5., 6., 4.5], index=pd.date_range('2020-01-01', periods=27, freq='MS'))
        frame = study.signal_frame(rates, 'monthly', 'Failed Breakout', {'memory': 3, 'reversal_components': 0})
        self.assertTrue(frame['Signal'].iloc[-1], 'the second breakout has failed its original 5.0 level while the first remains above 4.0')

    def test_confirmation_window_includes_last_permitted_observation(self):
        rates = pd.Series([4.] * 24 + [5., 4.7, 4.5, 3.9], index=pd.date_range('2020-01-01', periods=28, freq='MS'))
        frame = study.signal_frame(rates, 'monthly', 'Failed Breakout', {'memory': 3, 'reversal_components': 0})
        self.assertTrue(frame['Signal'].iloc[-1])



class SovereignPanelOrderingTests(unittest.TestCase):
    def test_panel_is_chronological_across_different_history_spans(self):
        from adfm_core.sovereign_daily import load_daily_sovereigns
        dates = pd.to_datetime(['2026-09-01', '2025-01-02'])
        rows = pd.DataFrame({'date': dates, 'ccy': ['USD', 'CAD'], 'tenor': '10Y', 'value': [4., 3.], 'source': ['us_treasury', 'boc_valet'], 'fetched_at': pd.Timestamp('2026-09-01')})
        with tempfile.TemporaryDirectory() as temp:
            panel, _ = load_daily_sovereigns(cache_path=Path(temp) / 'missing.parquet', fetchers={'United States': lambda: rows, 'Canada': lambda: rows}, today=pd.Timestamp('2026-09-03'))
        self.assertTrue(panel.index.is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
