from __future__ import annotations

import unittest

from adfm_core.data_registry import (
    COCKPIT_PROXIES,
    MARKET_SERIES,
    PRIMARY_MACRO_SERIES,
    market_symbols,
    registry_by_key,
)


class DataRegistryTests(unittest.TestCase):
    def test_registry_keys_and_market_symbols_are_unique(self):
        definitions = MARKET_SERIES + PRIMARY_MACRO_SERIES
        self.assertEqual(len(definitions), len(registry_by_key()))
        self.assertEqual(len(market_symbols()), len(set(market_symbols())))

    def test_every_proxy_resolves_to_registered_market_symbols(self):
        symbols = set(market_symbols())
        for proxy in COCKPIT_PROXIES:
            self.assertIn(proxy.numerator, symbols)
            if proxy.denominator:
                self.assertIn(proxy.denominator, symbols)
            self.assertIn(proxy.direction, {-1, 1})


if __name__ == "__main__":
    unittest.main()
