"""Early-warning contracts using real provider history and hand-checked paths."""
from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.hedge_timer_data import load_hedge_inputs
from adfm_core.hedge_timer_model import (
    FROZEN_WATCH_THRESHOLD,
    NDX_TICKER,
    SPX_TICKER,
    TICKERS,
    calibrate_watch_threshold,
    compute_scores,
    episode_audit,
    find_drawdown_episodes,
    onset,
    watch_signal,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages/21_Hedge_Timer.py"


def research():
    close = pd.read_csv(ROOT / "data/hedge_timer/research_inputs.csv", index_col="Date", parse_dates=True)
    close = close.loc[close[SPX_TICKER].notna() & close[NDX_TICKER].notna()]
    bars = pd.read_csv(ROOT / "data/hedge_timer/research_indices_ohlc.csv", header=[0, 1], index_col=0, parse_dates=True)
    return close, bars


def provider_panel():
    close, bars = research()
    raw = bars.swaplevel(axis=1)
    for ticker in TICKERS:
        raw[(ticker, "Close")] = close[ticker]
    return raw


class EarlyWarningContracts(unittest.TestCase):
    def path(self):
        idx = pd.bdate_range("2020-01-02", periods=100)
        px = pd.Series(100.0, index=idx)
        px.iloc[40:51] = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 88]
        return idx, px

    def test_nine_percent_warning_is_late(self):
        idx, px = self.path()
        warnings = pd.Series(False, index=idx)
        warnings.iloc[49] = True
        self.assertFalse(episode_audit("SPX", px, warnings).iloc[0]["Captured"])

    def test_three_percent_warning_qualifies_but_rebound_does_not_reopen_deadline(self):
        idx, px = self.path()
        px.iloc[40:52] = [100, 99, 98, 97, 95, 98, 96, 94, 92, 91, 88, 87]
        for day, captured in [(43, True), (45, False)]:
            with self.subTest(day=day):
                warnings = pd.Series(False, index=idx)
                warnings.iloc[day] = True
                self.assertEqual(bool(episode_audit("SPX", px, warnings).iloc[0]["Captured"]), captured)

    def test_expiring_old_peak_cannot_shrink_depth(self):
        idx = pd.bdate_range("2020-01-02", periods=150)
        px = pd.Series(100.0, index=idx)
        px.iloc[30:40] = [99, 98, 96, 94, 92, 90, 89, 88, 87, 86]
        px.iloc[40:140] = 86.0
        px.iloc[140:] = 80.0
        episodes = find_drawdown_episodes(px)
        self.assertEqual(len(episodes), 1)
        self.assertAlmostEqual(episodes[0][3], -0.20)

    def test_ndx_cannot_change_spx_fit(self):
        idx, px = self.path()
        score = pd.Series(0.0, index=idx)
        score.iloc[41:51] = 44.0
        first = calibrate_watch_threshold(score, px, score, px)
        second = calibrate_watch_threshold(score, px, score * 0, px)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["threshold"], 44.0)

    def test_price_retreat_warns_before_old_five_percent_trigger(self):
        idx = pd.bdate_range("2019-01-02", periods=300)
        px = pd.Series([100.0 + day / 3 for day in range(300)], index=idx)
        px.iloc[-3:] = [200.0, 198.0, 196.5]
        _, _, _, conditions = compute_scores(pd.DataFrame({SPX_TICKER: px}), SPX_TICKER)
        self.assertTrue(conditions["drawdown_velocity"].iloc[-1])

    def test_spx_history_includes_covid_and_each_2022_leg(self):
        close, _ = research()
        score, _, _, _ = compute_scores(close, SPX_TICKER)
        fit = calibrate_watch_threshold(score, close[SPX_TICKER])
        self.assertEqual(fit["spx_coverage"], 1.0)
        self.assertEqual(fit["threshold"], FROZEN_WATCH_THRESHOLD)
        active = watch_signal(score, FROZEN_WATCH_THRESHOLD)
        audit = episode_audit("SPX", close[SPX_TICKER], onset(active), warning_active=active)
        self.assertEqual(audit["Peak"].dt.strftime("%Y-%m-%d").tolist(), [
            "2020-02-19", "2022-01-03", "2022-03-29", "2022-08-16", "2023-07-31", "2025-02-19",
        ])
        self.assertTrue(audit["Captured"].all())
        self.assertTrue((audit["Loss at warning"] >= -0.03 - 1e-12).all())
        self.assertLess(active.loc["2020":].mean(), 0.75)

    def test_frozen_spx_rules_capture_intraday_spx_and_transfer_ndx(self):
        close, bars = research()
        for ticker, count in [(SPX_TICKER, 7), (NDX_TICKER, 17)]:
            with self.subTest(ticker=ticker):
                score, _, _, _ = compute_scores(close, ticker)
                active = watch_signal(score, FROZEN_WATCH_THRESHOLD)
                audit = episode_audit(ticker, bars.xs(ticker, level=1, axis=1), onset(active), warning_active=active)
                self.assertEqual(len(audit), count)
                self.assertTrue(audit["Captured"].all())
                self.assertTrue((audit["Loss at warning"] >= -0.03 - 1e-12).all())

    def test_future_prices_do_not_change_warning_inputs(self):
        close, _ = research()
        for ticker in [SPX_TICKER, NDX_TICKER]:
            full, _, _, _ = compute_scores(close, ticker)
            for end in ["2020-02-21", "2022-01-14", "2022-04-05", "2022-08-19"]:
                with self.subTest(ticker=ticker, end=end):
                    prefix, _, _, _ = compute_scores(close.loc[:end], ticker)
                    pd.testing.assert_series_equal(prefix, full.reindex(prefix.index))

    def test_existing_warning_at_peak_qualifies_without_new_onset(self):
        idx, px = self.path()
        active = pd.Series(True, index=idx)
        audit = episode_audit("SPX", px, onset(active), warning_active=active)
        self.assertTrue(audit.iloc[0]["Captured"])
        self.assertEqual(audit.iloc[0]["Timing"], "Active at peak")

    def test_intraday_ten_percent_decline_is_not_lost_by_closes(self):
        idx = pd.bdate_range("2020-01-02", periods=80)
        bars = pd.DataFrame({"Close": 100.0, "High": 100.0, "Low": 100.0}, index=idx)
        bars.loc[idx[40], ["Close", "High", "Low"]] = [100.0, 102.0, 100.0]
        bars.loc[idx[41:50], ["Close", "High", "Low"]] = [92.0, 94.0, 90.0]
        episodes = find_drawdown_episodes(bars)
        self.assertEqual(len(episodes), 1)
        self.assertAlmostEqual(episodes[0][3], -0.11764705882352944)

    def test_rearm_day_alert_after_three_percent_loss_is_late(self):
        idx = pd.bdate_range("2020-01-02", periods=80)
        bars = pd.DataFrame({"Close": 100.0, "High": 100.0, "Low": 100.0}, index=idx)
        bars.iloc[40:44] = [[95, 100, 95], [89, 90, 89], [99, 103, 89], [89, 90, 89]]
        bars.iloc[44:] = [89, 90, 89]
        active = pd.Series(False, index=idx)
        active.iloc[42] = True
        audit = episode_audit("SPX", bars, onset(active), warning_active=active)
        self.assertEqual(len(audit), 2)
        self.assertFalse(audit.iloc[1]["Captured"])
        self.assertLess(audit.iloc[1]["Loss at warning"], -0.03)

    def test_all_nan_outage_preserves_browsable_snapshot_without_live_claim(self):
        raw = pd.DataFrame({("^GSPC", "Close"): [np.nan], ("^NDX", "Close"): [np.nan]}, index=pd.to_datetime(["2026-10-02"]))
        with patch("adfm_core.market_data.download_market_data", return_value=raw):
            panel, _, health = load_hedge_inputs(["^GSPC", "^NDX"], date(2016, 1, 1),
                now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"),
                research_snapshot_path=ROOT / "data/hedge_timer")
        self.assertIn(pd.Timestamp("2020-02-19"), panel.index)
        self.assertIn("^GSPC High", panel)
        self.assertTrue(health["Status"].eq("Research snapshot").all())

    def test_episode_browser_includes_covid_and_early_spring_2022_warning(self):
        with patch("adfm_core.market_data.download_market_data", return_value=provider_panel()):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            self.assertFalse(app.exception)
            browser = app.selectbox(key="hedge_episode")
            self.assertTrue(any("2020-02-19" in option for option in browser.options))
            browser.set_value(4).run(timeout=30)
        self.assertFalse(app.exception)
        audit = next(table.value for table in app.dataframe if "Captured" in table.value)
        self.assertEqual(len(audit), 7)
        self.assertTrue(audit["Captured"].eq("Yes").all())
        self.assertTrue(any("2022-03-31" in item.value for item in app.caption))

    def test_missing_historical_range_cannot_certify_intraday_coverage(self):
        raw = provider_panel()
        raw.loc[pd.Timestamp("2020-03-23"), ("^GSPC", "Low")] = np.nan
        with patch("adfm_core.market_data.download_market_data", return_value=raw):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("intraday coverage cannot be certified" in item.value for item in app.warning))
        audit = next(table.value for table in app.dataframe if "Captured" in table.value)
        self.assertEqual(len(audit), 6)


if __name__ == "__main__":
    unittest.main()
