"""Exercise actual page loaders against slow and partially available providers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from test_loader_recovery import scope as basket_scope
from test_page_accuracy import page_functions

from adfm_core import market_data


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration

    perf_counter = monotonic

    def monotonic_ns(self):
        return int(self.elapsed * 1_000_000_000)


class PageDownloadDeadlineTests(unittest.TestCase):
    def loader(self, page, clock, provider):
        extra = {"time": clock, "download_market_data": provider}
        if page == "baskets":
            namespace = basket_scope
            overrides = {
                **extra,
                "compatible_snapshot": lambda *args: (pd.DataFrame(), {}),
                "save_last_good_levels": lambda *args: None,
            }
            return namespace, overrides, lambda: namespace["fetch_daily_levels"](
                ["SPY"] + [f"T{i:03}" for i in range(89)],
                pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
            )
        if page == "sector":
            extra.update(DOWNLOAD_CHUNK_SIZE=30, DOWNLOAD_RETRIES=3, BENCHMARKS={"SPY": "S&P 500"})
            namespace = page_functions("7_Sector_Breadth_and_Rotation.py", {"_download_batch", "fetch_prices"}, extra)
            return namespace, {}, lambda: namespace["fetch_prices"](tuple(["SPY"] + [f"T{i:03}" for i in range(89)]))
        if page == "flows":
            namespace = page_functions("14_ETF_Flow_Pressure_Proxy.py", {
                "chunked", "strip_tz_from_index", "normalize_ohlcv", "extract_ticker_frame", "safe_yf_download", "fetch_prices",
            }, extra)
            return namespace, {}, lambda: namespace["fetch_prices"](
                tuple(["SPY"] + [f"T{i:03}" for i in range(89)]),
                pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), "test",
            )
        filename = "8_Equity_Leadership_&_Rotation.py" if page == "leadership" else "11_Cross-Asset_Ratio_Chartbook.py"
        namespace = page_functions(filename, {"clean_ticker", "unique_keep_order", "chunked", "fetch_closes"}, extra)
        return namespace, {}, lambda: namespace["fetch_closes"](
            tuple(["SPY"] + [f"T{i:03}" for i in range(89)]),
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
        )

    @staticmethod
    def raw_prices(symbols):
        columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], symbols])
        return pd.DataFrame(100.0, index=pd.to_datetime(["2026-01-05", "2026-01-06"]), columns=columns)

    def test_outages_share_one_budget_across_chunks_retries_and_fallbacks(self):
        for page in ("baskets", "sector", "leadership", "ratios", "flows"):
            with self.subTest(page=page):
                clock = Clock()

                def unavailable(tickers, clock=clock, **kwargs):
                    clock.elapsed += min(12.0, kwargs.get("recovery_budget_seconds", 25.0))
                    return pd.DataFrame()

                namespace, overrides, load = self.loader(page, clock, unavailable)
                with patch.dict(namespace, overrides):
                    if page == "baskets":
                        with self.assertRaises(namespace["PriceFeedUnavailable"]):
                            load()
                    else:
                        result = load()
                        if page == "flows":
                            self.assertTrue(all(frame.empty for frame in result.values()))
                            self.assertEqual(len(result), 90)
                        else:
                            self.assertTrue(result.empty)
                self.assertLessEqual(clock.elapsed, 25.0)

    def test_deadline_keeps_observations_without_fabricating_missing_symbols(self):
        for page in ("baskets", "sector", "leadership", "ratios", "flows"):
            with self.subTest(page=page):
                clock = Clock()

                def partial(tickers, clock=clock, **kwargs):
                    clock.elapsed += 25.0
                    return self.raw_prices(["SPY"])

                namespace, overrides, load = self.loader(page, clock, partial)
                with patch.dict(namespace, overrides):
                    result = load()
                if page == "baskets":
                    result, metadata = result
                    self.assertEqual(len(metadata["missing_tickers"]), 89)
                if page == "flows":
                    self.assertEqual(result["SPY"]["Close"].tolist(), [100.0, 100.0])
                    self.assertTrue(all(frame.empty for symbol, frame in result.items() if symbol != "SPY"))
                else:
                    self.assertEqual(result.columns.tolist(), ["SPY"])
                    self.assertEqual(result["SPY"].tolist(), [100.0, 100.0])
                self.assertLessEqual(clock.elapsed, 25.0)

    def test_available_provider_still_loads_the_entire_universe(self):
        for page in ("baskets", "sector", "leadership", "ratios", "flows"):
            with self.subTest(page=page):
                clock = Clock()

                def available(tickers, clock=clock, **kwargs):
                    clock.elapsed += 1.0
                    return self.raw_prices(tickers)

                namespace, overrides, load = self.loader(page, clock, available)
                with patch.dict(namespace, overrides):
                    result = load()
                if page == "baskets":
                    result, metadata = result
                    self.assertEqual(metadata["missing_tickers"], [])
                if page == "flows":
                    self.assertEqual(len(result), 90)
                    self.assertTrue(all(len(frame) == 2 for frame in result.values()))
                else:
                    self.assertEqual(result.shape, (2, 90))

    def test_basket_benchmark_survives_deadline_in_large_alphabetical_universe(self):
        clock = Clock()

        def first_batch_only(tickers, **kwargs):
            clock.elapsed += kwargs["recovery_budget_seconds"]
            return self.raw_prices(tickers)

        namespace, overrides, _ = self.loader("baskets", clock, first_batch_only)
        symbols = [f"A{i:04}" for i in range(1100)] + ["SPY"]
        with patch.dict(namespace, overrides):
            result, metadata = namespace["fetch_daily_levels"](
                symbols, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
            )
        self.assertIn("SPY", result)
        self.assertEqual(result["SPY"].tolist(), [100.0, 100.0])
        self.assertEqual(metadata["returned_tickers"], 45)
        self.assertEqual(len(metadata["missing_tickers"]), 1056)
        self.assertLessEqual(clock.elapsed, 25.0)

    def test_basket_partial_response_without_benchmark_is_not_usable(self):
        clock = Clock()

        def no_benchmark(tickers, **kwargs):
            clock.elapsed += kwargs["recovery_budget_seconds"]
            return self.raw_prices([symbol for symbol in tickers if symbol != "SPY"])

        namespace, overrides, load = self.loader("baskets", clock, no_benchmark)
        with patch.dict(namespace, overrides):
            with self.assertRaises(namespace["PriceFeedUnavailable"]):
                load()
        self.assertLessEqual(clock.elapsed, 25.0)

    def test_basket_bulk_sweep_reaches_later_symbols_before_retrying_invalid_ones(self):
        clock = Clock()
        symbols = ["SPY"] + [f"A{i:04}" for i in range(1100)]
        unavailable = {f"A{i:04}" for i in range(0, 1100, 45)}

        def provider(tickers, kwargs, deadline):
            waves = (len(tickers) + kwargs["threads"] - 1) // kwargs["threads"]
            clock.elapsed += min(8.0 if len(tickers) == 1 else 0.25 * waves, deadline - clock.elapsed)
            return self.raw_prices([symbol for symbol in tickers if symbol not in unavailable])

        market_data._cached_download.clear()
        market_data._LAST_GOOD.clear()
        namespace, overrides, _ = self.loader("baskets", clock, market_data.download_market_data)
        try:
            with patch.dict(namespace, overrides), patch.object(market_data, "time", clock), \
                    patch.object(market_data, "_download_before_deadline", side_effect=provider):
                result, metadata = namespace["fetch_daily_levels"](
                    symbols, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
                )
            self.assertIn("A1099", result)
            self.assertGreaterEqual(metadata["returned_tickers"], 1075)
            self.assertTrue(set(metadata["missing_tickers"]).issubset(unavailable))
            self.assertLessEqual(clock.elapsed, 25.0)
        finally:
            market_data._cached_download.clear()
            market_data._LAST_GOOD.clear()

    def test_complete_large_basket_sweep_keeps_successful_chunks_for_warm_load(self):
        clock = Clock()
        symbols = ["SPY"] + [f"A{i:04}" for i in range(1204)]
        quote = {"value": 100.0}

        def provider(tickers, kwargs, deadline):
            clock.elapsed += 0.5
            return self.raw_prices(tickers) * (quote["value"] / 100.0)

        market_data._cached_download.clear()
        market_data._LAST_GOOD.clear()
        namespace, overrides, _ = self.loader("baskets", clock, market_data.download_market_data)
        try:
            with patch.dict(namespace, overrides), patch.object(market_data, "time", clock), \
                    patch.object(market_data, "_download_before_deadline", side_effect=provider):
                first, _ = namespace["fetch_daily_levels"](
                    symbols, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
                )
                quote["value"] = 200.0
                elapsed = clock.elapsed
                warm, metadata = namespace["fetch_daily_levels"](
                    symbols, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
                )
            pd.testing.assert_frame_equal(warm, first)
            self.assertEqual(metadata["returned_tickers"], 1205)
            self.assertEqual(clock.elapsed, elapsed)
        finally:
            market_data._cached_download.clear()
            market_data._LAST_GOOD.clear()

    def test_seasonality_regimes_share_one_budget_across_symbols_and_aliases(self):
        clock = Clock()

        def unavailable(symbol, start, end, **kwargs):
            clock.elapsed += kwargs.get("recovery_budget_seconds", 25.0)
            return None

        namespace = page_functions("24_Monthly_Seasonality_Explorer.py", {"fetch_regime_market_series"},
            {"time": clock, "_today": lambda: pd.Timestamp("2026-09-30"),
             "_yf_download": unavailable, "DXY_FALLBACKS": ["DX=F", "UUP"]})
        self.assertTrue(namespace["fetch_regime_market_series"]("2020-01-01", "2026-09-30").empty)
        self.assertLessEqual(clock.elapsed, 25.0)

    def test_seasonality_price_fallback_has_no_duplicate_yahoo_retry(self):
        clock = Clock()
        requested = []

        def unavailable(symbol, start, end, **kwargs):
            requested.append(symbol)
            clock.elapsed += kwargs.get("recovery_budget_seconds", 25.0)
            return None

        namespace = page_functions("24_Monthly_Seasonality_Explorer.py", {"fetch_prices"},
            {"time": clock, "_today": lambda: pd.Timestamp("2026-09-30"),
             "_clean_symbol": str, "_yf_download": unavailable,
             "_fred_series": lambda *args, **kwargs: None, "FALLBACK_MAP": {"^GSPC": "SP500"}})
        self.assertIsNone(namespace["fetch_prices"]("SPY", "2020-01-01", "2026-09-30"))
        self.assertLessEqual(clock.elapsed, 15.0)
        self.assertEqual(len(requested), len(set(requested)))

    def test_options_calendar_fallback_respects_shared_deadline(self):
        clock = Clock()

        def stalled(key, fetch, *, deadline, **kwargs):
            clock.elapsed = max(clock.elapsed, deadline)
            raise TimeoutError("pending")

        source = SimpleNamespace(call=stalled)
        namespace = page_functions("16_Options_Positioning_Compass.py",
            {"fetch_expirations", "fetch_cboe_snapshot", "fetch_cboe_expirations", "available_expirations"},
            {"time": clock, "YAHOO_OPTIONS": source, "CBOE_OPTIONS": source,
             "expirations_from_cboe": lambda frame: ()})
        for symbol in ("SPY", "QQQ", "IWM", "DIA"):
            self.assertEqual(namespace["available_expirations"](symbol, deadline=20.0), ())
        self.assertLessEqual(clock.elapsed, 20.0)

    def test_optional_alfred_timeout_keeps_all_regimes_unknown(self):
        clock = Clock()

        def unavailable(key, fetch, *, deadline, **kwargs):
            clock.elapsed = deadline
            raise TimeoutError("ALFRED stalled")

        namespace = page_functions("24_Monthly_Seasonality_Explorer.py", {"fetch_decision_regime_data"},
            {"time": clock, "SEASONALITY_FRED": SimpleNamespace(call=unavailable)})
        result = namespace["fetch_decision_regime_data"]("2020-01-01", "2026-09-30")
        self.assertTrue(result["fed_regime"].eq("Unknown").all())
        self.assertIn("TimeoutError", result.attrs["availability_error"])
        self.assertLessEqual(clock.elapsed, 25.0)


if __name__ == "__main__":
    unittest.main()
