import unittest

import pandas as pd

from adfm_core import futures_contracts as contracts


class FuturesContractsTests(unittest.TestCase):
    def test_quote_unit_controls_exact_multiplier_and_signed_pnl(self):
        expected = {"CL=F": 1000, "NG=F": 10000, "GC=F": 100, "MGC=F": 10,
                    "SI=F": 5000, "SIL=F": 1000, "HG=F": 25000, "ZC=F": 50, "ZS=F": 50}
        for symbol, multiplier in expected.items():
            with self.subTest(symbol=symbol):
                spec = contracts.get_contract_spec(symbol)
                self.assertEqual(spec.multiplier, multiplier)
                self.assertTrue(spec.spec_url.startswith("https://www.cmegroup.com/"))
                self.assertEqual(contracts.futures_pnl(symbol, -2, 3), -6 * multiplier)
        self.assertIsNone(contracts.get_contract_spec("UNKNOWN=F"))
        self.assertIsNone(contracts.get_contract_spec("LBS=F"))
        with self.assertRaises(ValueError):
            contracts.futures_pnl("UNKNOWN=F", 1, 1)

    def test_discontinuity_candidates_do_not_rewrite_real_prices(self):
        prices = pd.Series([100.0] * 30 + [150.0, 149.0, -1.0], index=pd.bdate_range("2020-01-01", periods=33))
        original = prices.copy()
        candidates = contracts.discontinuity_candidates(prices)
        self.assertIn(prices.index[30], candidates.index)
        self.assertIn(prices.index[32], candidates.index)
        self.assertIn("possible roll or market move", candidates.loc[prices.index[30], "Reason"])
        pd.testing.assert_series_equal(prices, original)


if __name__ == "__main__":
    unittest.main()
