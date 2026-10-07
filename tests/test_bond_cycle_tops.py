import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adfm_core.bond_event_study import event_dates, signal_frame
from adfm_core.fred_store import read_record

ROOT = Path(__file__).resolve().parents[1]


class CycleTopTests(unittest.TestCase):
    def test_reference_audit_reports_a_miss_without_fabricating_an_alert(self):
        from adfm_core.bond_top_audit import audit_reference_tops
        # Flat yields cannot satisfy a prior-rise/reversal rule even though
        # the calendar contains all ten reference dates.
        rates = pd.Series(4., index=pd.bdate_range('1953-01-01', '2024-12-31'))
        audit = audit_reference_tops(rates, 'daily', 'Cycle Top')
        self.assertTrue(audit['Status'].eq('Missed').all())
        self.assertTrue(audit['Alert date'].isna().all())

    def test_monthly_rule_captures_all_ten_reference_episodes(self):
        rates = pd.read_csv(ROOT / 'tests/fixtures/GS10_1953_2024.csv', index_col=0, parse_dates=True).iloc[:, 0]
        frame = signal_frame(rates, 'monthly', 'Cycle Top')
        dates = event_dates(frame, 1)
        # Independent episode windows: confirmation must occur within 3 months
        # AFTER the peak, rather than matching an unrelated earlier warning.
        for peak in ('1960-01-31', '1966-08-31', '1970-05-31', '1975-09-30',
                     '1981-09-30', '1984-06-30', '1994-11-30', '2000-01-31',
                     '2007-06-30', '2023-10-31'):
            with self.subTest(peak=peak):
                start = pd.Timestamp(peak)
                self.assertTrue(((dates > start) & (dates <= start + pd.DateOffset(months=3))).any())

    def test_daily_rule_captures_nine_peaks_within_ten_observed_sessions(self):
        rates, _ = read_record(ROOT / 'data/fred/DGS10.json.gz', 'DGS10')
        frame = signal_frame(rates, 'daily', 'Cycle Top')
        dates = event_dates(frame, 1)
        for peak in ('1966-08-29', '1970-05-26', '1975-09-16', '1981-09-30',
                     '1984-05-30', '1994-11-07', '2000-01-20', '2007-06-12', '2023-10-19'):
            with self.subTest(peak=peak):
                pos = frame.index.get_loc(pd.Timestamp(peak))
                window = frame.index[pos + 1:pos + 11]
                self.assertTrue(len(dates.intersection(window)) > 0)

    def test_rising_yields_never_confirm_and_future_data_cannot_change_history(self):
        rates = pd.Series(np.linspace(1., 5., 400), index=pd.bdate_range('2018-01-01', periods=400))
        frame = signal_frame(rates, 'daily', 'Cycle Top')
        self.assertFalse(frame.Signal.any())
        future = pd.Series([4.8, 4.7, 4.6], index=pd.bdate_range(rates.index[-1] + pd.offsets.BDay(), periods=3))
        extended = signal_frame(pd.concat([rates, future]), 'daily', 'Cycle Top')
        pd.testing.assert_frame_equal(frame, extended.iloc[:len(frame)])
        self.assertTrue(extended.Signal.iloc[-3:].any())

    def test_one_alert_per_peak_then_rearm_after_a_new_high(self):
        values = [3.] * 240 + list(np.linspace(3., 4., 20)) + [3.8, 3.7, 3.6, 4.1, 3.6, 3.5]
        rates = pd.Series(values, index=pd.bdate_range('2000-01-01', periods=len(values)))
        frame = signal_frame(rates, 'daily', 'Cycle Top')
        self.assertEqual(int(frame.Signal.iloc[-6:].sum()), 2)
        self.assertTrue(frame.Signal.iloc[-6])
        self.assertFalse(frame.Signal.iloc[-5])
        self.assertTrue(frame.Signal.iloc[-2])

    def test_missing_month_cancels_pending_confirmation(self):
        values = [3.] * 24 + [3.5, 3.8, 4., np.nan, 3.7]
        rates = pd.Series(values, index=pd.date_range('2000-01-01', periods=len(values), freq='MS'))
        frame = signal_frame(rates, 'monthly', 'Cycle Top')
        self.assertFalse(frame.Signal.iloc[-1])

    def test_page_renders_ten_peak_audit_and_keeps_monthly_history_separate(self):
        from unittest.mock import patch

        import streamlit as st
        from streamlit.testing.v1 import AppTest

        def fred(symbols, **kwargs):
            values = {}
            for symbol in symbols:
                record = read_record(ROOT / f'data/fred/{symbol}.json.gz', symbol)
                if record:
                    values[symbol] = record[0]
            return pd.DataFrame(values), pd.DataFrame()

        st.cache_data.clear()
        try:
            with patch('adfm_core.primary_data.fetch_fred_symbols', side_effect=fred):
                app = AppTest.from_file(str(ROOT / 'pages/2_Global_Macro_Regime.py')).run(timeout=30)
                self.assertEqual(list(app.exception), [])
                audit = next(item.value for item in app.dataframe if 'Episode' in item.value.columns)
                self.assertEqual(int(audit.Status.eq('Captured').sum()), 10)
                self.assertEqual(audit.Basis.iloc[0], 'GS10 monthly average')
                self.assertEqual(audit['Peak date'].iloc[0], '1960-01-31')
                self.assertEqual(audit['Peak yield (%)'].iloc[-1], 4.98)
                app.radio(key='bond_us_frequency').set_value('Monthly since 1953').run(timeout=30)
                self.assertEqual(list(app.exception), [])
                audit = next(item.value for item in app.dataframe if 'Episode' in item.value.columns)
                self.assertEqual(int(audit.Status.eq('Captured').sum()), 10)
                self.assertEqual(audit['Peak yield (%)'].iloc[-1], 4.80)
                self.assertTrue(audit.Basis.eq('GS10 monthly average').all())
        finally:
            st.cache_data.clear()


if __name__ == '__main__':
    unittest.main()
