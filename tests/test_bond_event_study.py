import unittest

import numpy as np
import pandas as pd

from adfm_core.bond_event_study import event_summary, signal_frame


class BondEventStudyTests(unittest.TestCase):
    def test_yield_exhaustion_requires_reversal_after_a_setup(self):
        dates = pd.date_range("2018-01-01", periods=65, freq="MS")
        values = np.r_[np.linspace(-0.5, 0.0, 48), np.linspace(0.0, 2.0, 12),
                       [2.4, 2.6, 2.1, 1.8, 1.6]]
        rates = pd.Series(values, index=dates)
        settings = dict(change_pctile=60, trend_z=0.2, rsi=55, vol_pctile=0,
                        memory=3, reversal_components=2)
        early = signal_frame(rates, "monthly", "Early Warning", settings=settings)
        confirmed = signal_frame(rates, "monthly", "Confirmed Exhaustion", settings=settings)
        self.assertTrue(early["Signal"].iloc[60:62].any())
        self.assertTrue(early["Watch"].iloc[60])
        self.assertFalse(confirmed["Signal"].iloc[61])
        self.assertTrue(confirmed["Signal"].iloc[62])
        self.assertTrue(np.isfinite(confirmed["TrendZ"].iloc[-1]))

    def test_signals_do_not_use_future_observations(self):
        rates = pd.Series(np.linspace(1, 4, 70) + np.sin(np.arange(70)),
                          index=pd.date_range("2017-01-01", periods=70, freq="MS"))
        first = signal_frame(rates.iloc[:60], "monthly", "Early Warning")
        full = signal_frame(rates, "monthly", "Early Warning")
        pd.testing.assert_frame_equal(first, full.iloc[:60], check_freq=False)

    def test_forward_yield_change_is_in_bp_and_negative_favors_yield_top(self):
        rates = pd.Series([3.0, 4.0, 3.5, 3.2, 3.3, 3.2],
                          index=pd.date_range("2026-01-01", periods=6, freq="MS"))
        events = pd.DatetimeIndex([pd.Timestamp("2026-02-01")])
        summary, history = event_summary(rates, events, "monthly")
        self.assertEqual(history.loc[0, "1M"], -50)
        self.assertEqual(summary.loc["Signal median", "1M"], -50)
        self.assertEqual(summary.loc["% lower yield", "1M"], 100)
        self.assertEqual(summary.loc["Independent N", "1M"], 1)

    def test_missing_month_does_not_become_one_month_forward_outcome(self):
        rates = pd.Series([4.0, 3.0], index=pd.to_datetime(["2026-01-01", "2026-03-01"]))
        _, history = event_summary(rates, pd.DatetimeIndex(["2026-01-01"]), "monthly")
        self.assertTrue(np.isnan(history.loc[0, "1M"]))


if __name__ == "__main__":
    unittest.main()
