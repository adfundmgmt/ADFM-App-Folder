"""Exercise actual page loaders against slow and partially available providers."""

import unittest
from unittest.mock import patch

import pandas as pd
from test_loader_recovery import scope as basket_scope
from test_page_accuracy import page_functions


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


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


if __name__ == "__main__":
    unittest.main()
