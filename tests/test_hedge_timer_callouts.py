"""Contracts for the actual red-dot hedge events, independent of Watch scores."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core import hedge_timer_model as model
from adfm_core.hedge_timer_data import callout_session_inputs

ROOT = Path(__file__).resolve().parents[1]


def history():
    close = pd.read_csv(ROOT / "data/hedge_timer/research_inputs.csv", index_col="Date", parse_dates=True)
    close = callout_session_inputs(close)
    bars = pd.read_csv(ROOT / "data/hedge_timer/research_indices_ohlc.csv", header=[0, 1], index_col=0, parse_dates=True)
    return close, bars


def actual_callouts(close, ticker):
    return model.compute_callouts(close, ticker)


class RedDotContracts(unittest.TestCase):
    def test_unused_legacy_volatility_inputs_do_not_block_hedge_alerts(self):
        close, _ = history()
        close = close.drop(columns=["^VVIX", "^VIX3M"])
        result = actual_callouts(close, model.SPX_TICKER)
        self.assertTrue(result.loc["2020":, "Inputs valid"].all())
        self.assertTrue(result.loc["2020-02-21", "Callout"])

    def test_event_calendar_excludes_holidays_and_preserves_unknown_sessions(self):
        observed = pd.DataFrame({"^GSPC": [100.0] * 4},
                                index=pd.to_datetime(["2026-05-22", "2026-05-25", "2026-05-26", "2026-05-28"]))
        aligned = callout_session_inputs(observed)
        self.assertEqual(aligned.index.tolist(), pd.to_datetime(
            ["2026-05-22", "2026-05-26", "2026-05-27", "2026-05-28"]
        ).tolist())
        self.assertTrue(aligned.loc["2026-05-27"].isna().all())
        self.assertEqual(aligned.loc["2026-05-26", "^GSPC"], 100.0)

    def test_missing_index_recovery_session_cannot_manufacture_a_new_dot(self):
        close, _ = history()
        close.loc["2020-04-27", model.NDX_TICKER] = float("nan")
        result = actual_callouts(callout_session_inputs(close), model.SPX_TICKER)
        self.assertFalse(result.loc["2020-04-27", "Recovery"])
        self.assertFalse(result.loc["2020-05-12", "Callout"])

    def test_page_audits_the_dates_of_the_actual_plotted_red_dots(self):
        close, bars = history()
        raw = bars.swaplevel(axis=1)
        for ticker in model.TICKERS:
            raw[(ticker, "Close")] = close[ticker]
        plots = []

        def capture(figure, **kwargs):
            ax = figure.axes[0]
            plots.append([tuple(point) for collection in ax.collections for point in collection.get_offsets()])

        with patch("adfm_core.market_data.download_market_data", return_value=raw), patch("streamlit.pyplot", side_effect=capture):
            app = AppTest.from_file(str(ROOT / "pages/21_Hedge_Timer.py")).run(timeout=30)
            self.assertEqual(app.radio[1].value, 2)
            plots.clear()
            app.radio[1].set_value(10).run(timeout=30)
        self.assertFalse(app.exception)
        audit = next(table.value for table in app.dataframe if "Captured" in table.value)
        date_column = "First alert"
        self.assertEqual(audit.iloc[0][date_column], "2020-02-21")
        chart_dates = close.index[-2520:]
        dots = {chart_dates[int(point[0])] for point in plots[0]}
        self.assertTrue(set(pd.to_datetime(audit[date_column])).issubset(dots))
        self.assertEqual(len(plots), 1)

    def test_actual_spx_dots_capture_all_seven_intraday_legs_within_five_sessions_before_peak_or_three_percent(self):
        close, bars = history()
        dots = actual_callouts(close, model.SPX_TICKER)["Callout"]
        audit = model.episode_audit("SPX", bars.xs(model.SPX_TICKER, level=1, axis=1), dots, lookback=5)
        self.assertEqual(len(audit), 7)
        self.assertEqual(int(audit["Captured"].sum()), 7)
        self.assertTrue((audit["Loss at warning"] >= -0.03 - 1e-12).all())
        # Every credited date must be an actual plotted event, never a carried state.
        self.assertTrue(dots.reindex(audit["First warning"]).all())

    def test_covid_and_each_2022_leg_have_a_timely_new_dot(self):
        close, _ = history()
        dots = actual_callouts(close, model.SPX_TICKER)["Callout"]
        for start, end in [("2020-02-19", "2020-02-21"), ("2022-01-04", "2022-01-05"),
                           ("2022-03-29", "2022-04-05"), ("2022-08-16", "2022-08-19")]:
            with self.subTest(start=start):
                self.assertTrue(dots.loc[start:end].any())

    def test_pre_peak_divergence_catches_september_2020_before_gap(self):
        close, _ = history()
        dots = actual_callouts(close, model.SPX_TICKER)["Callout"]
        self.assertTrue(dots.loc["2020-08-31":"2020-09-02"].any())

    def test_callouts_do_not_repaint_when_future_prices_are_added(self):
        close, _ = history()
        for ticker in (model.SPX_TICKER, model.NDX_TICKER):
            full = actual_callouts(close, ticker)
            for end in ("2020-02-21", "2020-08-31", "2022-01-05", "2022-03-31", "2022-08-19"):
                with self.subTest(ticker=ticker, end=end):
                    prefix = actual_callouts(close.loc[:end], ticker)
                    pd.testing.assert_frame_equal(prefix, full.reindex(prefix.index))

    def test_weak_breadth_alone_does_not_print_a_dot_in_a_rising_market(self):
        close, _ = history()
        dots = actual_callouts(close, model.SPX_TICKER)["Callout"]
        self.assertFalse(dots.loc["2020-08-18":"2020-08-28"].any())

    def test_repeat_candidate_needs_sustained_recovery_before_another_dot(self):
        index = pd.bdate_range("2020-01-02", periods=14)
        candidate = pd.Series([True, False, True, False, True, False, False, False,
                               True, False, False, False, False, True], index=index)
        recovery = pd.Series([False] * 5 + [True, True, True] + [False] * 6, index=index)
        actual = model.debounce_callouts(candidate, recovery, pd.Series(True, index=index), minimum_spacing=5)
        dots = actual["Callout"]
        self.assertEqual(dots[dots].index.tolist(), [index[0], index[8]])

    def test_missing_input_cannot_create_an_event_or_recovery(self):
        close, _ = history()
        close.loc[pd.Timestamp("2020-02-21"), "^VIX"] = float("nan")
        result = actual_callouts(close, model.SPX_TICKER)
        self.assertFalse(result.loc["2020-02-21", "Callout"])
        self.assertFalse(result.loc["2020-02-21", "Inputs valid"])
        self.assertFalse(result.loc["2020-02-21", "Recovery"])

    def test_nonpositive_input_is_unavailable(self):
        close, _ = history()
        close.loc[pd.Timestamp("2020-02-21"), "^VIX"] = -15.0
        result = actual_callouts(close, model.SPX_TICKER)
        self.assertFalse(result.loc["2020-02-21", "Inputs valid"])

    def test_malformed_and_infinite_history_is_treated_as_missing(self):
        close, _ = history()
        missing = close.copy()
        missing.loc["2020-02-20", "^VIX"] = float("nan")
        expected = actual_callouts(missing, model.SPX_TICKER)
        for value in ("unavailable", float("inf"), -15.0):
            with self.subTest(value=value):
                malformed = close.copy()
                malformed["^VIX"] = malformed["^VIX"].astype(object)
                malformed.loc["2020-02-20", "^VIX"] = value
                pd.testing.assert_frame_equal(actual_callouts(malformed, model.SPX_TICKER), expected)

    def test_spacing_does_not_rearm_without_recovery(self):
        index = pd.bdate_range("2020-01-02", periods=30)
        candidate = pd.Series(True, index=index)
        actual = model.debounce_callouts(candidate, pd.Series(False, index=index), candidate)
        self.assertEqual(actual.index[actual["Callout"]].tolist(), [index[0]])

    def test_invalid_day_interrupts_consecutive_recovery(self):
        index = pd.bdate_range("2020-01-02", periods=9)
        candidate = pd.Series([True, False, False, False, True, False, False, True, False], index=index)
        recovery = pd.Series([False, True, True, True, True, True, True, False, False], index=index)
        valid = pd.Series(True, index=index)
        valid.iloc[3] = False
        actual = model.debounce_callouts(candidate, recovery, valid, minimum_spacing=5)
        self.assertEqual(actual.index[actual["Callout"]].tolist(), [index[0], index[7]])

    def test_minimum_spacing_is_enforced_after_recovery(self):
        index = pd.bdate_range("2020-01-02", periods=20)
        series = pd.Series(True, index=index)
        actual = model.debounce_callouts(series, series, series)
        self.assertEqual(actual.index[actual["Callout"]].tolist(), [index[0], index[10]])

    def test_ndx_and_old_weights_cannot_change_spx_events(self):
        close, _ = history()
        original = actual_callouts(close, model.SPX_TICKER)
        close[model.NDX_TICKER] *= 10
        with patch.object(model, "WATCH_COMPONENTS", ()), patch.object(model, "CONFIRM_COMPONENTS", ()):
            altered = actual_callouts(close, model.SPX_TICKER)
        pd.testing.assert_frame_equal(original, altered)

    def test_deep_decline_cannot_generate_a_new_early_callout(self):
        close, _ = history()
        close.loc[pd.Timestamp("2020-02-21"), model.SPX_TICKER] = 3200.0
        result = actual_callouts(close, model.SPX_TICKER)
        self.assertFalse(result.loc["2020-02-21", "Callout"])


if __name__ == "__main__":
    unittest.main()
