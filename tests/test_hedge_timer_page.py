"""Integration guards for the Hedge Timer Streamlit page."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.catalog import sidebar_guide_for_page, tool_for_page
from adfm_core.hedge_timer_data import callout_session_inputs
from adfm_core.hedge_timer_model import TICKERS, compute_callouts
from adfm_core.market_data import _last_completed_us_session

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "21_Hedge_Timer.py"


class HedgeTimerPageTests(unittest.TestCase):
    def test_missing_index_session_is_preserved_in_recovery_inputs(self) -> None:
        endpoint = _last_completed_us_session()
        dates = pd.bdate_range(end=endpoint, periods=450)
        raw = pd.DataFrame(
            {(ticker, "Close"): np.linspace(100.0, 150.0, len(dates)) for ticker in TICKERS},
            index=dates,
        )
        missing_date = callout_session_inputs(raw).index[-20]
        raw.loc[missing_date, ("^NDX", "Close")] = np.nan
        event_frames = []

        def capture(frame, ticker):
            event_frames.append(frame.copy())
            return compute_callouts(frame, ticker)

        with patch("adfm_core.market_data.download_market_data", return_value=raw), patch(
            "adfm_core.hedge_timer_model.compute_callouts", side_effect=capture
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(event_frames), 2)
        for frame in event_frames:
            self.assertIn(missing_date, frame.index)
            self.assertTrue(pd.isna(frame.loc[missing_date, "^NDX"]))

    def test_invalid_latest_close_blocks_a_previously_latched_signal(self) -> None:
        endpoint = _last_completed_us_session()
        dates = pd.bdate_range(end=endpoint, periods=450)
        raw = pd.DataFrame(
            {(ticker, "Close"): np.linspace(100.0, 150.0, len(dates)) for ticker in TICKERS},
            index=dates,
        )
        for ticker in ("^GSPC", "^NDX"):
            raw.loc[dates[-2:], (ticker, "Close")] = 147.5
        raw.loc[dates[-2], ("^VIX", "Close")] = 180.0
        for invalid in (0.0, -15.0, float("inf")):
            with self.subTest(invalid=invalid):
                raw.loc[dates[-1], ("^VIX", "Close")] = invalid
                with patch("adfm_core.market_data.download_market_data", return_value=raw):
                    app = AppTest.from_file(str(PAGE)).run(timeout=30)
                self.assertFalse(app.exception)
                displayed = "\n".join(item.value for item in app.markdown)
                self.assertIn("hedge-unavailable", displayed)
                self.assertNotIn("Fresh-short gate: <b>Open</b>", displayed)
                if np.isfinite(invalid):
                    self.assertIn("Last callout <b>NA</b>", displayed)
                self.assertTrue(any("^VIX" in item.value for item in app.warning))

    def test_missing_recent_close_is_recovered_before_displaying_signal(self) -> None:
        endpoint = _last_completed_us_session()
        dates = pd.bdate_range(end=endpoint, periods=450)
        raw = pd.DataFrame(
            { (ticker, "Close"): np.linspace(100.0, 150.0, len(dates)) for ticker in TICKERS },
            index=dates,
        )
        raw.loc[endpoint, ("^VIX9D", "Close")] = np.nan
        recovery = pd.DataFrame({"Close": [150.0]}, index=[endpoint])

        def provider(tickers, **kwargs):
            return raw if len(tickers) == len(TICKERS) else recovery

        with patch("adfm_core.market_data.download_market_data", side_effect=provider):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
        self.assertFalse(app.exception)
        displayed = "\n".join(item.value for item in app.markdown)
        self.assertNotIn("Unavailable: stale or missing inputs", displayed)
        self.assertIn(endpoint.date().isoformat(), "\n".join(item.value for item in app.caption))

    def test_unrecovered_input_is_identified_without_publishing_partial_score(self) -> None:
        endpoint = _last_completed_us_session()
        dates = pd.bdate_range(end=endpoint, periods=450)
        raw = pd.DataFrame(
            { (ticker, "Close"): np.linspace(100.0, 150.0, len(dates)) for ticker in TICKERS },
            index=dates,
        )
        raw.loc[endpoint, ("XLU", "Close")] = np.nan

        def provider(tickers, **kwargs):
            return raw if len(tickers) == len(TICKERS) else pd.DataFrame()

        with patch("adfm_core.market_data.download_market_data", side_effect=provider):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
        self.assertFalse(app.exception)
        displayed = "\n".join(item.value for item in app.markdown)
        self.assertIn("Price <b>149.89</b>", displayed)
        self.assertNotIn("Price <b>150.00</b>", displayed)
        self.assertIn("Fresh-short gate: <b>Blocked</b>", displayed)
        self.assertTrue(any("XLU" in item.value for item in app.warning))

    def test_older_than_previous_session_stays_unavailable(self) -> None:
        endpoint = _last_completed_us_session()
        dates = pd.bdate_range(end=endpoint, periods=450)
        raw = pd.DataFrame(
            {(ticker, "Close"): np.linspace(100.0, 150.0, len(dates)) for ticker in TICKERS},
            index=dates,
        )
        raw.loc[dates[-3:], ("XLU", "Close")] = np.nan

        def provider(tickers, **kwargs):
            return raw if len(tickers) == len(TICKERS) else pd.DataFrame()

        with patch("adfm_core.market_data.download_market_data", side_effect=provider):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
        self.assertFalse(app.exception)
        displayed = "\n".join(item.value for item in app.markdown)
        self.assertIn("hedge-unavailable", displayed)
        self.assertIn("Last callout <b>NA</b>", displayed)

    def test_page_uses_recall_model_and_keeps_spx_ndx_separate(self) -> None:
        source = PAGE.read_text(encoding="utf-8")

        self.assertIn("from adfm_core.hedge_timer_model import", source)
        self.assertIn("FROZEN_CALLOUT_RULES", source)
        self.assertIn("episode_audit", source)
        self.assertIn("warning_summary", source)
        self.assertNotIn("pick_target_today", source)
        self.assertNotIn('"TLT"', source)

    def test_chart_shows_only_confirmation_dots_on_price(self) -> None:
        source = PAGE.read_text(encoding="utf-8")

        self.assertNotIn('label="Hedge Score"', source)
        self.assertIn('label="Confirmed"', source)
        self.assertIn("ax_price.scatter(", source)
        self.assertNotIn("ax_score", source)
        self.assertNotIn('marker="^"', source)
        self.assertNotIn("ax_price.axvline", source)
        self.assertNotIn("ax_price.axvspan", source)
        self.assertNotIn("ax_score.scatter(", source)
        self.assertNotIn('label="Hedge Watch onset"', source)
        self.assertNotIn('label="Short allowed onset"', source)
        self.assertNotIn('label="Confirmation"', source)

    def test_about_metadata_matches_the_recall_model(self) -> None:
        tool = tool_for_page("21_Hedge_Timer.py")
        guide = sidebar_guide_for_page("21_Hedge_Timer.py")

        self.assertIsNotNone(tool)
        self.assertIsNotNone(guide)
        assert tool is not None
        assert guide is not None
        self.assertIn("actual red dots", tool.description.lower())
        self.assertIn("10%+", tool.description)
        self.assertIn("IWM", tool.primary_inputs)
        self.assertIn("sector ETFs", tool.primary_inputs)
        self.assertIn("confirmed red dots", " ".join(guide.read_order))
        self.assertIn("local", " ".join(guide.read_order).lower())


if __name__ == "__main__":
    unittest.main()
