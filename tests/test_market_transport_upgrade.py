"""Transport contract regressions: removing recovery/freshness gates breaks these."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, Timer
from time import monotonic
from unittest.mock import patch

import numpy as np
import pandas as pd

from adfm_core import market_data, observability


def bars(values):
    return pd.DataFrame({"Close": values, "Volume": [100] * len(values)}, index=pd.date_range("2026-09-25", periods=len(values)))


class TransportUpgradeTests(unittest.TestCase):
    def test_same_symbol_cache_wait_cannot_exceed_callers_deadline(self):
        entered, release = Event(), Event()
        def provider(tickers, **kwargs):
            entered.set()
            release.wait(.3)
            return bars([10, 11])
        try:
            with patch.object(market_data.yf, "download", side_effect=provider), ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self.download, "SPY", retries=1, recovery_budget_seconds=.5)
                self.assertTrue(entered.wait(.5))
                started = monotonic()
                second = self.download("SPY", retries=1, recovery_budget_seconds=.03)
                self.assertLess(monotonic() - started, .12)
                self.assertTrue(second.empty)
                release.set()
                self.assertFalse(first.result().empty)
        finally:
            release.set()

    def test_transport_limits_do_not_invalidate_successful_observation_cache(self):
        with patch.object(market_data.yf, "download", return_value=bars([10, 11])) as provider:
            first = self.download("SPY", period="1y", retries=1, timeout=8, recovery_budget_seconds=20)
            second = self.download("SPY", period="1y", retries=1, timeout=3, recovery_budget_seconds=5)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(provider.call_count, 1)

    def test_healthy_concurrent_requests_wait_within_their_own_budget(self):
        entered, release = Event(), Event()
        def provider(tickers, **kwargs):
            if tickers == ["SPY"]:
                entered.set()
                release.wait(.5)
            return bars([10, 11])
        timer = Timer(.05, release.set)
        try:
            with patch.object(market_data.yf, "download", side_effect=provider), ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self.download, "SPY", retries=1, recovery_budget_seconds=.5)
                self.assertTrue(entered.wait(.5))
                timer.start()
                second = self.download("QQQ", retries=1, recovery_budget_seconds=.5)
                self.assertFalse(second.empty)
                self.assertFalse(first.result().empty)
        finally:
            release.set()
            if timer.ident is not None:
                timer.join()

    def setUp(self):
        if hasattr(market_data, "_cached_download"):
            market_data._cached_download.clear()
        if hasattr(market_data, "_LAST_GOOD"):
            market_data._LAST_GOOD.clear()

    def download(self, *args, **kwargs):
        self.assertTrue(hasattr(market_data, "download_market_data"), "shared date-range transport must exist")
        kwargs.setdefault("now", datetime(2026, 9, 30, 10))
        return market_data.download_market_data(*args, **kwargs)

    def test_null_provider_response_is_unavailable_without_losing_the_page(self):
        with patch.object(market_data.yf, "download", return_value=None):
            result = self.download("SPY", retries=1)
        self.assertTrue(result.empty)
        self.assertEqual(result.attrs["market_data_health"]["SPY"]["status"], "unavailable")

    def test_slow_provider_is_hard_bounded_without_spawning_more_inflight_calls(self):
        release = Event()
        calls = []

        def provider(tickers, **kwargs):
            calls.append(tickers)
            release.wait(0.3)
            return bars([10, 11])

        try:
            with patch.object(market_data.yf, "download", side_effect=provider):
                started = monotonic()
                timed_out = self.download("SPY", retries=1, recovery_budget_seconds=0.03)
                self.assertTrue(timed_out.empty, "a provider result arriving after the deadline must be discarded")
                blocked = self.download("QQQ", retries=1, recovery_budget_seconds=0.03)
                self.assertTrue(blocked.empty)
                self.assertLess(monotonic() - started, 0.2)
                self.assertEqual(len(calls), 1, "a still-running provider request must prevent additional workers")
        finally:
            release.set()
            # Join the worker through its lock so the fixture cannot leak into
            # the next test, which may use a different provider response.
            if hasattr(market_data, "_PROVIDER_CALL_LOCK"):
                with market_data._PROVIDER_CALL_LOCK:
                    pass

        with patch.object(market_data.yf, "download", return_value=bars([20, 21])):
            recovered = self.download("SPY", retries=1, recovery_budget_seconds=0.03)
        self.assertEqual(recovered.Close.iloc[-1], 21)

    def test_unavailable_result_retries_on_next_call_without_manual_refresh(self):
        with patch.object(market_data.yf, "download", side_effect=[pd.DataFrame(), bars([20, 21])]):
            unavailable = self.download("SPY", retries=1)
            recovered = self.download("SPY", retries=1)
        self.assertTrue(unavailable.empty)
        self.assertFalse(recovered.empty, "the next call must retry an unavailable request")
        self.assertEqual(recovered.Close.iloc[-1], 21)
        self.assertEqual(recovered.attrs["market_data_health"]["SPY"]["status"], "ok")

    def test_partial_failure_retries_missing_symbol_on_next_call(self):
        partial = pd.concat({"SPY": bars([10, 11])}, axis=1).swaplevel(axis=1)
        complete = pd.concat({"SPY": bars([10, 11]), "QQQ": bars([20, 21])}, axis=1).swaplevel(axis=1)
        with patch.object(market_data.yf, "download", side_effect=[partial, pd.DataFrame(), complete]):
            first = self.download(["SPY", "QQQ"], retries=1)
            recovered = self.download(["SPY", "QQQ"], retries=1)
        self.assertEqual(first[("Close", "SPY")].iloc[-1], 11)
        self.assertTrue(first[("Close", "QQQ")].isna().all())
        self.assertEqual(recovered[("Close", "QQQ")].iloc[-1], 21)

    def test_bulk_owner_can_defer_symbol_recovery_without_caching_missing_data(self):
        partial = pd.concat({"SPY": bars([10, 11])}, axis=1).swaplevel(axis=1)
        complete = pd.concat({"SPY": bars([10, 11]), "QQQ": bars([20, 21])}, axis=1).swaplevel(axis=1)
        bulk_responses = iter([partial, complete])

        def provider(tickers, **kwargs):
            if len(tickers) == 1:
                return bars([99, 100])
            return next(bulk_responses)

        with patch.object(market_data.yf, "download", side_effect=provider):
            first = self.download(["SPY", "QQQ"], retries=1, recover_missing=False)
            recovered = self.download(["SPY", "QQQ"], retries=1, recover_missing=False)
        self.assertEqual(first[("Close", "SPY")].iloc[-1], 11)
        self.assertTrue(first[("Close", "QQQ")].isna().all())
        self.assertEqual(recovered[("Close", "QQQ")].iloc[-1], 21)

    def test_last_good_delivery_does_not_cache_outage_over_provider_recovery(self):
        with patch.object(market_data.yf, "download", return_value=bars([10, 11])):
            self.download("SPY", retries=1)
        market_data._cached_download.clear()
        with patch.object(market_data.yf, "download", side_effect=[pd.DataFrame(), bars([20, 21])]):
            fallback = self.download("SPY", retries=1)
            recovered = self.download("SPY", retries=1)
        self.assertEqual(fallback.Close.iloc[-1], 11)
        self.assertEqual(fallback.attrs["market_data_health"]["SPY"]["status"], "last_good")
        self.assertEqual(recovered.Close.iloc[-1], 21)
        self.assertEqual(recovered.attrs["market_data_health"]["SPY"]["status"], "ok")

    def test_completed_daily_cache_refetches_at_actual_session_settlement(self):
        cases = [
            ("2026-09-28", "2026-09-29", datetime(2026, 9, 29, 15, 50), datetime(2026, 9, 29, 16, 20)),
            ("2026-11-25", "2026-11-27", datetime(2026, 11, 27, 12, 50), datetime(2026, 11, 27, 13, 20)),
        ]
        for previous, today, before, after in cases:
            for typed in (True, False):
                with self.subTest(day=today, typed=typed):
                    self.setUp()
                    provisional = bars([10, 11])
                    provisional.index = pd.to_datetime([previous, today])
                    provisional["Open"] = provisional["High"] = provisional["Low"] = provisional.Close
                    settled = provisional.copy()
                    settled.loc[today, ["Open", "High", "Low", "Close"]] = 20
                    with patch.object(market_data.yf, "download", side_effect=[provisional, settled]):
                        if typed:
                            first, _ = market_data.fetch_daily_ohlcv(("SPY",), now=before)
                            second, _ = market_data.fetch_daily_ohlcv(("SPY",), now=after)
                            first, second = first["SPY"], second["SPY"]
                        else:
                            first = self.download("SPY", completed_only=True, session_timezone="America/New_York", now=before, retries=1)
                            second = self.download("SPY", completed_only=True, session_timezone="America/New_York", now=after, retries=1)
                    self.assertEqual(first.Close.tolist(), [10])
                    self.assertEqual(second.Close.tolist(), [10, 20])
                    self.assertEqual(second.attrs["market_data_health"].get("SPY", second.attrs["market_data_health"])["as_of"], today)

    def test_settlement_outage_cannot_promote_provisional_last_good_bar(self):
        provisional = bars([10, 11])
        provisional.index = pd.to_datetime(["2026-09-28", "2026-09-29"])
        provisional["Open"] = provisional["High"] = provisional["Low"] = provisional.Close
        with patch.object(market_data.yf, "download", side_effect=[provisional, pd.DataFrame()]):
            market_data.fetch_daily_ohlcv(("SPY",), now=datetime(2026, 9, 29, 15, 50))
            result = self.download("SPY", period="3y", interval="1d", auto_adjust=False, group_by="column", threads=True, completed_only=True, session_timezone="America/New_York", now=datetime(2026, 9, 29, 16, 20), retries=1)
        self.assertTrue(result.empty or result.index[-1] < pd.Timestamp("2026-09-29"))

    def test_date_range_raw_volume_missing_endpoint_and_provenance(self):
        source = bars([10, 11, np.nan])
        source["Adj Close"] = [5, 5.5, np.nan]
        with patch.object(market_data.yf, "download", return_value=source):
            result = self.download("SPY", start="2026-09-25", end="2026-09-29", auto_adjust=False, retries=1)
        self.assertEqual(result.Volume.tolist(), [100] * 3)
        self.assertTrue(pd.isna(result.Close.iloc[-1]))
        self.assertEqual(result.attrs["market_data_health"]["SPY"]["as_of"], "2026-09-26")
        self.assertFalse(result.attrs["auto_adjust"])

    def test_partial_failure_recovers_symbol_and_retains_unavailable_column(self):
        def provider(tickers, **kwargs):
            if len(tickers) > 1:
                return pd.concat({"SPY": bars([10, 11]), "QQQ": bars([np.nan, np.nan])}, axis=1).swaplevel(axis=1)
            return bars([20, 21]) if tickers[0] == "QQQ" else pd.DataFrame()
        with patch.object(market_data.yf, "download", side_effect=provider):
            result = self.download(["SPY", "QQQ", "BAD"], period="1y", retries=1)
        self.assertEqual(result[("Close", "QQQ")].iloc[-1], 21)
        self.assertTrue(result[("Close", "BAD")].isna().all())
        self.assertEqual(result.attrs["market_data_health"]["BAD"]["status"], "unavailable")

    def test_ticker_group_adjusted_intraday_preserves_time_and_foreign_endpoint(self):
        source = bars([10, 11])
        source.index = pd.date_range("2026-09-29 09:00", periods=2, freq="h")
        with patch.object(market_data.yf, "download", return_value=source):
            result = self.download("7203.T", interval="1h", auto_adjust=True, group_by="ticker", retries=1)
        self.assertEqual(result.index[-1].hour, 10)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.attrs["auto_adjust"])

    def test_completion_policy_does_not_guess_foreign_or_futures_close(self):
        source = bars([10, 11])
        source.index = pd.date_range("2026-09-28", periods=2)
        with patch.object(market_data.yf, "download", return_value=source):
            result = self.download("GC=F", completed_only=True, now=datetime(2026, 9, 29, 10), retries=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.attrs["completion_policy"], "unverified_exchange_close")

    def test_explicit_us_completed_only_and_default_raw(self):
        source = bars([10, 11])
        source.index = pd.date_range("2026-09-28", periods=2)
        with patch.object(market_data.yf, "download", return_value=source):
            result = self.download("SPY", completed_only=True, session_timezone="America/New_York", now=datetime(2026, 9, 29, 10), retries=1)
        self.assertEqual(len(result), 1)

    def test_provider_and_cache_delivery_observability_is_bounded_and_symbol_free(self):
        with patch.object(market_data.yf, "download", return_value=bars([10, 11])):
            self.download("PRIVATE", retries=1)
            self.download("PRIVATE", retries=1)
        self.assertTrue(hasattr(observability, "performance_events"))
        events = observability.performance_events()
        self.assertEqual(events[-1]["cache_hit"], True)
        self.assertNotIn("PRIVATE", str(events))
        self.assertLessEqual(len(events), 100)

    def test_typed_daily_completion_recognizes_us_but_preserves_asian_futures(self):
        source = bars([10, 11])
        source.index = pd.date_range("2026-09-28", periods=2)
        source["Open"] = source["High"] = source["Low"] = source.Close
        raw = pd.concat({symbol: source for symbol in ["SPY", "7203.T", "GC=F"]}, axis=1).swaplevel(axis=1)
        with patch.object(market_data.yf, "download", return_value=raw):
            frames, _ = market_data.fetch_daily_ohlcv(("SPY", "7203.T", "GC=F"), now=datetime(2026, 9, 29, 10))
        self.assertEqual(len(frames["SPY"]), 1)
        self.assertEqual(len(frames["7203.T"]), 2)
        self.assertEqual(len(frames["GC=F"]), 2)
        self.assertEqual(frames["7203.T"].attrs["completion_policy"], "unverified_exchange_close")

    def test_us_completion_uses_actual_early_close_calendar(self):
        self.assertTrue(hasattr(market_data, "completed_daily_observations"))
        source = bars([10, 11])
        source.index = pd.to_datetime(["2026-11-25", "2026-11-27"])
        result = market_data.completed_daily_observations(source, "SPY", now=datetime(2026, 11, 27, 14))
        self.assertEqual(len(result), 2)
        before = market_data.completed_daily_observations(source, "SPY", now=datetime(2026, 11, 27, 12))
        self.assertEqual(len(before), 1)

    def test_complete_provider_outage_has_no_individual_request_fanout(self):
        calls = []
        def provider(tickers, **kwargs):
            calls.append(tickers)
            return pd.DataFrame()
        with patch.object(market_data.yf, "download", side_effect=provider), patch.object(market_data.time, "sleep"):
            result = self.download([f"SYMBOL{i}" for i in range(100)], retries=2)
        self.assertTrue(result.empty)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(item["status"] == "unavailable" for item in result.attrs["market_data_health"].values()))

    def test_recovery_deadline_preserves_partial_data_without_new_requests(self):
        calls = []
        clock = [0.]
        def provider(tickers, **kwargs):
            calls.append(tickers)
            if len(calls) == 1:
                clock[0] += 20.
                return pd.concat({"SPY": bars([10, 11])}, axis=1).swaplevel(axis=1)
            clock[0] += 6.
            return pd.DataFrame()
        with patch.object(market_data.yf, "download", side_effect=provider), patch.object(market_data.time, "perf_counter", side_effect=lambda: clock[0]):
            result = self.download(["SPY", "BAD", "BAD2"], retries=3, recovery_budget_seconds=25)
        self.assertEqual(result[("Close", "SPY")].iloc[-1], 11)
        self.assertTrue(result[("Close", "BAD")].isna().all())
        self.assertEqual(len(calls), 2)

    def test_expired_cache_outage_recovers_only_exact_observed_last_good(self):
        source = bars([10, 11, np.nan])
        with patch.object(market_data.yf, "download", return_value=source):
            first = self.download("SPY", period="1y", retries=1)
        market_data._cached_download.clear()
        with patch.object(market_data.yf, "download", return_value=pd.DataFrame()):
            recovered = self.download("SPY", period="1y", retries=1)
        pd.testing.assert_frame_equal(first, recovered)
        self.assertEqual(recovered.attrs["market_data_health"]["SPY"]["status"], "last_good")
        self.assertTrue(pd.isna(recovered.Close.iloc[-1]))
        self.assertLessEqual(len(market_data._LAST_GOOD), 16)

    def test_health_summary_uses_actual_observed_close_not_missing_endpoint(self):
        event = observability.record_data_load("Yahoo Finance", {"SPY": bars([10, 11, np.nan])}, ["SPY"])
        self.assertEqual(event.data_through, "2026-09-26")

    def test_required_missing_or_stale_observation_blocks_current_signal(self):
        self.assertTrue(hasattr(market_data, "required_inputs_fresh"))
        frame = pd.DataFrame({"SPY": [1, 2, 3], "HYG": [1, 2, np.nan]}, index=pd.date_range("2026-09-25", periods=3))
        self.assertFalse(market_data.required_inputs_fresh(frame, ["SPY", "HYG"]))
        self.assertFalse(market_data.required_inputs_fresh(frame, ["SPY"], now=pd.Timestamp("2026-10-10")))
        self.assertTrue(market_data.required_inputs_fresh(frame, ["SPY"], now=pd.Timestamp("2026-09-28")))

    def test_benchmark_fill_never_extends_missing_endpoint(self):
        source = pd.DataFrame({"Close": [1, np.nan, 3]}, index=pd.date_range("2026-09-25", periods=3))
        result = market_data.align_to_benchmark_calendar(source, pd.date_range("2026-09-25", periods=4), forward_fill_limit=2)
        self.assertEqual(result.Close.iloc[1], 1)
        self.assertTrue(pd.isna(result.Close.iloc[-1]))


class OwnedPageAlignmentTests(unittest.TestCase):
    def test_credit_ratios_do_not_extend_stale_endpoint(self):
        from test_page_accuracy import page_functions
        functions = page_functions("5_Credit_Conditions_Monitor.py", {"ratio_frame"}, {"fill_short_calendar_gaps": market_data.fill_short_calendar_gaps})
        frame = pd.DataFrame({"HYG": [10., 11., np.nan], "LQD": [20., 21., 22.], "JNK": [10., 11., 12.]}, index=pd.date_range("2026-09-25", periods=3))
        result = functions["ratio_frame"](frame)
        self.assertTrue(pd.isna(result.reindex(frame.index)["HYG/LQD"].iloc[-1]))

    def test_memory_proxy_alignment_does_not_extend_stale_endpoint(self):
        from test_page_accuracy import page_functions
        functions = page_functions("23_Market_Memory_Explorer.py", {"build_feature_frame", "rolling_max_drawdown_array", "bucket_vix_value"}, {"DEFAULT_HORIZONS": [5, 21, 63, 252], "fill_short_calendar_gaps": market_data.fill_short_calendar_gaps})
        prices = pd.Series(np.linspace(100, 150, 300), index=pd.bdate_range("2025-01-01", periods=300))
        vix = pd.Series(20., index=prices.index[:-2])
        result = functions["build_feature_frame"](prices, {"vix": vix})
        self.assertTrue(pd.isna(result.vix.iloc[-1]))
        self.assertEqual(result.vix_bucket.iloc[-1], "unknown")
