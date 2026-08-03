import unittest

from adfm_core.options_sources import (
    cboe_payload_to_chains,
    expirations_from_cboe,
    parse_occ_option_symbol,
    select_cboe_expiry,
)


class OptionsSourcesTests(unittest.TestCase):
    def test_parse_occ_option_symbol(self):
        self.assertEqual(
            parse_occ_option_symbol("QQQ260918C00690000"),
            ("2026-09-18", "call", 690.0),
        )

    def test_cboe_payload_normalizes_and_selects_expiration(self):
        payload = {
            "timestamp": "2026-08-03 14:20:17",
            "data": {
                "symbol": "QQQ",
                "current_price": 691.57,
                "options": [
                    {
                        "option": "QQQ260918C00690000",
                        "bid": 24.1,
                        "ask": 24.4,
                        "iv": 0.225,
                        "open_interest": 100,
                        "volume": 12,
                        "last_trade_price": 24.2,
                        "last_trade_time": "2026-08-03T10:05:15",
                    },
                    {
                        "option": "QQQ260918P00690000",
                        "bid": 22.0,
                        "ask": 22.3,
                        "iv": 0.235,
                        "open_interest": 120,
                        "volume": 15,
                        "last_trade_price": 22.1,
                        "last_trade_time": "2026-08-03T10:05:15",
                    },
                ],
            },
        }

        frame, underlying, timestamp = cboe_payload_to_chains(payload)
        calls, puts = select_cboe_expiry(frame, "2026-09-18")

        self.assertEqual(expirations_from_cboe(frame), ("2026-09-18",))
        self.assertEqual(underlying["regularMarketPrice"], 691.57)
        self.assertEqual(timestamp, "2026-08-03 14:20:17")
        self.assertEqual(calls.iloc[0]["contractSymbol"], "QQQ260918C00690000")
        self.assertEqual(puts.iloc[0]["impliedVolatility"], 0.235)
        self.assertEqual(calls.iloc[0]["iv_source"], "Cboe")


if __name__ == "__main__":
    unittest.main()
