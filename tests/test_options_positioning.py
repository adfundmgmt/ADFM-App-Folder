import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from adfm_core.options_positioning import (
    add_cross_sectional_ranks,
    black_scholes_delta,
    black_scholes_price,
    build_positioning_commentary,
    implied_volatility_from_price,
    option_snapshot,
    ordinal,
    prepare_chain,
)


def sample_chain(strikes, ivs, volumes, open_interest):
    return pd.DataFrame(
        {
            "contractSymbol": [f"TEST{i}" for i in range(len(strikes))],
            "strike": strikes,
            "lastPrice": [2.0] * len(strikes),
            "bid": [1.8] * len(strikes),
            "ask": [2.2] * len(strikes),
            "volume": volumes,
            "openInterest": open_interest,
            "impliedVolatility": ivs,
        }
    )


class OptionsPositioningTests(unittest.TestCase):
    def test_ordinal_formats_special_teen_suffixes(self):
        self.assertEqual(ordinal(21.2), "21st")
        self.assertEqual(ordinal(32.0), "32nd")
        self.assertEqual(ordinal(13.0), "13th")

    def test_black_scholes_call_and_put_delta_obey_parity_without_dividend(self):
        call = black_scholes_delta(100, 100, 0.5, 0.2, option_type="call")
        put = black_scholes_delta(100, 100, 0.5, 0.2, option_type="put")

        self.assertTrue(np.isclose(call - put, 1.0))
        self.assertGreater(call, 0)
        self.assertLess(put, 0)

    def test_prepare_chain_prefers_valid_midquote_for_premium_activity(self):
        chain = sample_chain([100], [0.2], [10], [50])

        result = prepare_chain(chain, "call")

        self.assertTrue(math.isclose(result.loc[0, "mid"], 2.0))
        self.assertTrue(math.isclose(result.loc[0, "premium_activity"], 2_000.0))

    def test_implied_volatility_solver_recovers_model_input(self):
        option_price = black_scholes_price(
            100,
            105,
            45 / 365,
            0.28,
            option_type="call",
        )

        result = implied_volatility_from_price(
            option_price,
            100,
            105,
            45 / 365,
            option_type="call",
        )

        self.assertTrue(np.isclose(result, 0.28, atol=1e-6))

    def test_prepare_chain_repairs_placeholder_iv_from_price(self):
        price = black_scholes_price(100, 100, 30 / 365, 0.25, option_type="call")
        chain = sample_chain([100], [0.00001], [10], [50])
        chain.loc[0, ["bid", "ask", "lastPrice"]] = [0.0, 0.0, price]

        result = prepare_chain(chain, "call", spot=100, time_years=30 / 365)

        self.assertTrue(np.isclose(result.loc[0, "impliedVolatility"], 0.25, atol=1e-6))
        self.assertEqual(result.loc[0, "iv_source"], "Solved from price")

    def test_option_snapshot_builds_skew_and_put_call_ratios(self):
        calls = sample_chain(
            [100, 105, 110, 115],
            [0.20, 0.21, 0.22, 0.24],
            [20, 20, 10, 5],
            [100, 100, 50, 25],
        )
        puts = sample_chain(
            [85, 90, 95, 100],
            [0.30, 0.27, 0.24, 0.22],
            [10, 20, 30, 50],
            [50, 75, 100, 125],
        )

        result = option_snapshot(
            calls,
            puts,
            spot=100,
            expiry="2026-06-19",
            as_of=date(2026, 5, 1),
            risk_free_rate=0.04,
        )

        self.assertTrue(np.isfinite(result["atm_iv"]))
        self.assertTrue(np.isfinite(result["put_skew"]))
        self.assertTrue(math.isclose(result["put_call_volume"], 2.0))
        self.assertTrue(math.isclose(result["put_call_oi"], 350 / 275))

    def test_cross_sectional_ranks_use_current_peer_group(self):
        frame = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "atm_iv": [0.20, 0.30, 0.40],
                "realized_vol_21d": [0.15, 0.15, 0.15],
                "put_skew": [0.01, 0.02, 0.03],
            }
        )

        result = add_cross_sectional_ranks(frame)

        self.assertTrue(np.isclose(result.loc[0, "iv_richness_percentile"], 100 / 6))
        self.assertTrue(np.isclose(result.loc[2, "put_skew_percentile"], 500 / 6))

    def test_commentary_discloses_cross_sectional_basis_and_positioning_proxy(self):
        commentary = build_positioning_commentary(
            {
                "ticker": "QQQ",
                "atm_iv": 0.25,
                "iv_richness_percentile": 60.0,
                "put_skew": 0.04,
                "put_skew_percentile": 80.0,
                "put_call_volume": 1.2,
                "put_call_oi": 0.9,
            }
        )

        self.assertIn("cross-sectional percentile", commentary)
        self.assertIn("do not identify whether trades opened or closed", commentary)


if __name__ == "__main__":
    unittest.main()
