"""Focused regression tests for the Page 19 deterministic analytics engine."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_momentum_scanner import (
    SECTOR_WEIGHTS, SIGNAL_COLUMNS, adjusted_ohlcv, rank_change, score_scanner,
    diagnostic_table, sector_leadership, security_features, signal_snapshot, valid_ohlcv_reason,
)
from adfm_sector_rotation_config import MAJOR_SECTORS, SUBSECTOR_ROWS


def make_frame(periods: int = 330, drift: float = 0.001, volume: float = 2_000_000) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = 100 * np.cumprod(np.repeat(1 + drift, periods))
    return pd.DataFrame({"Open": close * .997, "High": close * 1.012, "Low": close * .988, "Close": close, "Adj Close": close, "Volume": np.repeat(volume, periods)}, index=index)


class MomentumScannerTests(unittest.TestCase):
    def test_every_documented_signal_has_a_boolean_output(self) -> None:
        """Covers VOL, MA FAN, ATR SQZ, CLOSE HI, 52W HI, VDU, HH/HL,
        W.EMA, OBV+, NR7, U/D VOL, SQUEEZE, and supplemental SPRING."""
        result = signal_snapshot(make_frame())
        for signal in SIGNAL_COLUMNS + ["SPRING"]:
            self.assertIn(signal, result)
            self.assertIsInstance(result[signal], (bool, np.bool_))

    def test_signal_count_is_exactly_the_sum_of_twelve_core_flags(self) -> None:
        frame = make_frame()
        features = security_features("TEST", frame, frame["Close"] * .98)
        self.assertEqual(features["Signal Count"], sum(int(features[signal]) for signal in SIGNAL_COLUMNS))
        self.assertGreaterEqual(features["Signal Count"], 0)
        self.assertLessEqual(features["Signal Count"], 12)

    def test_adjusted_ohlcv_scales_prices_and_volume_without_forward_fill(self) -> None:
        raw = make_frame(252)
        raw["Adj Close"] = raw["Close"] * .5
        raw.iloc[10, raw.columns.get_loc("Close")] = np.nan
        adjusted = adjusted_ohlcv(raw)
        self.assertAlmostEqual(adjusted["Open"].iloc[0], raw["Open"].iloc[0] * .5)
        self.assertAlmostEqual(adjusted["Volume"].iloc[0], raw["Volume"].iloc[0] * 2)
        self.assertTrue(pd.isna(adjusted["Close"].iloc[10]))

    def test_scoring_is_bounded_with_missing_sector_relative_data(self) -> None:
        frame = make_frame()
        rows = []
        for ticker, multiplier in [("AAA", 1.0), ("BBB", .95), ("CCC", 1.05)]:
            row = security_features(ticker, frame * multiplier, frame["Close"] * .97)
            if ticker == "CCC": row["Sector RS 21D"] = np.nan
            rows.append(row)
        scored = score_scanner(pd.DataFrame(rows))
        self.assertTrue(scored["Expansion Score"].between(0, 100).all())
        self.assertTrue(scored["Relative Strength Score"].notna().all())

    def test_stale_security_is_excluded_using_benchmark_sessions(self) -> None:
        benchmark = make_frame()
        self.assertEqual(valid_ohlcv_reason(benchmark.iloc[:-4], benchmark.index), "Stale (> 3 benchmark sessions)")

    def test_rank_change_is_positive_when_a_security_improves(self) -> None:
        current = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Rank": [1, 2]})
        prior = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Rank": [3, 1]})
        changed = rank_change(current, prior)
        self.assertEqual(changed.loc[changed["Ticker"] == "AAA", "Rank change over five sessions"].iloc[0], 2)
        self.assertEqual(changed.loc[changed["Ticker"] == "BBB", "Rank change over five sessions"].iloc[0], -1)

    def test_diagnostics_materialize_walk_forward_outcomes(self) -> None:
        frame = make_frame()
        diagnostics = diagnostic_table({"SPY": frame * .98, "AAA": frame}, "SPY", max_sessions=20, min_dollar_volume=1)
        self.assertFalse(diagnostics.empty)
        self.assertTrue({"Next-session close-to-open gap", "Five-session forward return", "MFE over 5 sessions", "MAE over 5 sessions", "Observations"}.issubset(diagnostics.columns))

    def test_sector_aggregation_is_median_based_and_weights_sum_to_100(self) -> None:
        frames = {"SPY": make_frame(drift=.0004)}
        for index, ticker in enumerate(MAJOR_SECTORS): frames[ticker] = make_frame(drift=.0005 + index * .00001)
        for index, item in enumerate(row for row in SUBSECTOR_ROWS if row["Tier"] == "Core"):
            frames[item["Ticker"]] = make_frame(drift=.00035 + index * .000005)
        sectors, subsectors = sector_leadership(frames, "SPY", include_thematic=False)
        self.assertEqual(len(sectors), len(MAJOR_SECTORS))
        self.assertEqual(set(sectors["Sector"]), set(MAJOR_SECTORS.values()))
        self.assertEqual(sum(SECTOR_WEIGHTS.values()), 100.0)
        technology = subsectors[subsectors["Sector Group"] == "Technology"]
        observed = sectors.loc[sectors["Sector"] == "Technology", "Median Subsector RS 1M"].iloc[0]
        self.assertAlmostEqual(observed, technology["RS 1M"].median())


if __name__ == "__main__":
    unittest.main()
