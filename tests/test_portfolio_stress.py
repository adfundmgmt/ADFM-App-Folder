"""Hand-calculated financial signs, input completeness and European model checks."""

from __future__ import annotations

import unittest

import pandas as pd

from adfm_core.portfolio_stress import (
    Scenario,
    european_option,
    portfolio_scenarios,
    validate_holdings,
)

DATE = "2026-09-29"


def holding(kind="shares", **changes):
    row = dict(
        symbol="TEST",
        kind=kind,
        quantity=10,
        multiplier=1,
        mark=100,
        mark_date=DATE,
        currency="USD",
        fx_rate=1,
        factor="equity",
    )
    row.update(changes)
    return row


class PortfolioStressTests(unittest.TestCase):
    def test_missing_currency_rejects_entire_portfolio(self):
        with self.assertRaisesRegex(ValueError, "currency"):
            validate_holdings(pd.DataFrame([holding(currency=float("nan"))]), DATE)

    def run_portfolio(self, rows, scenario, nav=100000):
        return portfolio_scenarios(pd.DataFrame(rows), nav, DATE, [scenario]).iloc[0]

    def test_signed_shares_and_fx_convert_simultaneously(self):
        # EUR shares: -10 * (100*.9*1.2*1.1 -100*1.2)=+12;
        # long EUR cash: 1000 * (1.2*1.1-1.2)=+120.
        rows = [
            holding(quantity=-10, currency="EUR", fx_rate=1.2),
            holding("fx", symbol="EURUSD", quantity=1000, mark=1.2, factor="fx"),
        ]
        result = self.run_portfolio(rows, Scenario("combined", equity=-0.1, fx=0.1))
        self.assertAlmostEqual(result["P&L USD"], 132)
        self.assertAlmostEqual(result["NAV impact %"], 0.132)
        self.assertAlmostEqual(result["Scenario NAV USD"], 100132)
        self.assertAlmostEqual(result["Gross equity USD"], 1200)
        self.assertAlmostEqual(result["Gross FX USD"], 2400)

    def test_short_crude_and_heating_oil_quote_multipliers(self):
        rows = [
            holding(
                "futures",
                symbol="CL=F",
                quantity=-2,
                multiplier=1000,
                mark=70,
                factor="commodity",
            ),
            holding(
                "futures",
                symbol="HO=F",
                quantity=1,
                multiplier=42000,
                mark=2.5,
                factor="commodity",
            ),
        ]
        result = self.run_portfolio(rows, Scenario("oil down", commodity=-0.1))
        self.assertAlmostEqual(result["P&L USD"], 3500)
        self.assertAlmostEqual(result["Gross commodity USD"], 245000)

    def test_dv01_and_convexity_units_and_short_sign(self):
        # -2 * (-50*20 + .5*.3*20²) = 1880 dollars.
        result = self.run_portfolio(
            [
                holding(
                    "dv01", quantity=-2, dv01=50, convexity=0.3, factor="yield", mark=0
                )
            ],
            Scenario("rates", yield_bp=20),
        )
        self.assertAlmostEqual(result["P&L USD"], 1880)
        self.assertAlmostEqual(result["Gross DV01 USD/bp"], 100)

    def test_european_call_put_known_values_and_parity(self):
        call = european_option(100, 100, 1, 0.2, 0.05, 0, "call")
        put = european_option(100, 100, 1, 0.2, 0.05, 0, "put")
        self.assertAlmostEqual(call, 10.450583572185565)
        self.assertAlmostEqual(put, 5.573526022256971)

    def test_option_pnl_anchored_to_observed_mark_and_signed_contracts(self):
        options = [
            holding(
                "call",
                quantity=-2,
                multiplier=100,
                mark=12,
                underlying=100,
                strike=100,
                expiry="2027-09-29",
                volatility=0.2,
                rate=0.05,
                dividend_yield=0,
            )
        ]
        base = self.run_portfolio(options, Scenario("base"))
        self.assertEqual(base["P&L USD"], 0)
        shock = self.run_portfolio(options, Scenario("vol", iv=0.1))
        # Known European prices sigma .2:10.45058357, sigma .3:14.23125479.
        self.assertAlmostEqual(shock["P&L USD"], -756.1342439108, places=5)
        self.assertGreater(shock["Gross vega USD/vol point"], 0)

    def test_combined_options_and_factors_all_contribute(self):
        rows = [
            holding(quantity=10),
            holding("fx", quantity=-1000, mark=1.2, factor="fx"),
            holding(
                "futures",
                symbol="CL=F",
                quantity=1,
                multiplier=1000,
                mark=70,
                factor="commodity",
            ),
            holding("dv01", quantity=1, factor="yield", mark=0, dv01=20),
        ]
        result = self.run_portfolio(
            rows,
            Scenario(
                "joint", equity=-0.1, fx=0.1, commodity=-0.2, yield_bp=10, iv=0.05
            ),
        )
        self.assertAlmostEqual(result["P&L USD"], -14420)
        self.assertAlmostEqual(result["Scenario NAV USD"], 85580)

    def test_margin_missing_is_unavailable_never_partial_total(self):
        result = self.run_portfolio(
            [holding(margin_rate=0.5), holding(quantity=-2)],
            Scenario("down", equity=-0.1),
        )
        self.assertTrue(pd.isna(result["Estimated margin USD"]))
        self.assertTrue(pd.isna(result["Estimated cash buffer USD"]))
        self.assertIn("unavailable", result["Margin basis"])
        self.assertAlmostEqual(result["Loss funding USD"], 80)

    def test_supplied_margin_estimate_and_cash_buffer(self):
        result = self.run_portfolio(
            [holding(margin_rate=0.5)], Scenario("up", equity=0.1)
        )
        self.assertAlmostEqual(result["Estimated margin USD"], 550)
        self.assertAlmostEqual(result["Estimated cash buffer USD"], 50)
        self.assertIn("supplied", result["Margin basis"])

    def test_invalid_row_aborts_entire_portfolio(self):
        for change in [
            dict(mark=None),
            dict(quantity=0),
            dict(quantity=float("inf")),
            dict(mark=-1),
            dict(multiplier=0),
            dict(currency="EUR", fx_rate=None),
            dict(mark_date="2026-09-28"),
            dict(mark_date="2026-10-01"),
            dict(kind="unknown"),
            dict(factor="commodity"),
            dict(currency="USD", fx_rate=2),
            dict(direction="Long", quantity=-1),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_portfolio([holding(), holding(**change)], Scenario("base"))

    def test_bad_options_and_contract_quantities_rejected(self):
        row = holding(
            "put",
            quantity=1,
            multiplier=100,
            mark=5,
            underlying=100,
            strike=100,
            expiry="2027-09-29",
            volatility=0.2,
            rate=0.03,
            dividend_yield=0,
        )
        for change in [
            dict(expiry=DATE),
            dict(expiry="2026-09-28"),
            dict(volatility=0),
            dict(underlying=None),
            dict(strike=0),
            dict(quantity=0.5),
            dict(mark=101),
            dict(rate=None),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_holdings(pd.DataFrame([{**row, **change}]), DATE)
        with self.assertRaises(ValueError):
            validate_holdings(
                pd.DataFrame(
                    [
                        holding(
                            "futures", symbol="CL=F", multiplier=1, factor="commodity"
                        )
                    ]
                ),
                DATE,
            )

    def test_foreign_futures_payoff_translates_without_funded_notional_pnl(self):
        row = holding(
            "futures",
            symbol="CUSTOM",
            quantity=-2,
            multiplier=50,
            mark=10,
            factor="commodity",
            currency="EUR",
            fx_rate=1.2,
        )
        base_fx = self.run_portfolio([row], Scenario("FX alone", fx=0.1))
        combined = self.run_portfolio(
            [row], Scenario("combined", fx=0.1, commodity=-0.2)
        )
        self.assertEqual(base_fx["P&L USD"], 0)
        self.assertAlmostEqual(combined["P&L USD"], 264)

    def test_known_usd_futures_reject_inconsistent_quote_currency(self):
        with self.assertRaises(ValueError):
            self.run_portfolio(
                [
                    holding(
                        "futures",
                        symbol="CL=F",
                        multiplier=1000,
                        factor="commodity",
                        currency="EUR",
                        fx_rate=1.2,
                    )
                ],
                Scenario("base"),
            )

    def test_rates_etf_put_yield_reprices_underlying_with_supplied_dv01(self):
        row = holding(
            "put",
            symbol="TLT",
            quantity=1,
            multiplier=100,
            mark=8,
            underlying=100,
            strike=100,
            expiry="2027-09-29",
            volatility=0.2,
            rate=0,
            dividend_yield=0,
            factor="rates",
            underlying_dv01=0.05,
            underlying_convexity=0,
        )
        result = self.run_portfolio([row], Scenario("rates", equity=0.2, yield_bp=100))
        # Underlying = 100-.05*100 = 95, risk-free rate=.01.
        # European put changes 7.965567455406 -> 9.892823260527.
        self.assertAlmostEqual(result["P&L USD"], 192.72558051216, places=5)
        self.assertEqual(result["Gross equity USD"], 0)
        self.assertAlmostEqual(result["Gross rates USD"], 10000)
        self.assertGreater(result["Gross DV01 USD/bp"], 0)

    def test_rates_shares_and_explicit_symbol_targets(self):
        shares = holding(
            symbol="TLT",
            factor="rates",
            underlying_dv01=0.05,
            underlying_convexity=0.0001,
        )
        result = self.run_portfolio(
            [shares], Scenario("rates", equity=0.2, yield_bp=100)
        )
        self.assertAlmostEqual(result["P&L USD"], -45)
        target = self.run_portfolio(
            [shares], Scenario("target", equity=0.2, yield_bp=100, targets={"TLT": 95})
        )
        self.assertAlmostEqual(target["P&L USD"], -50)
        for scenario in [
            Scenario("unknown", targets={"MISSING": 95}),
            Scenario("bad", targets={"TLT": 0}),
        ]:
            with self.assertRaises(ValueError):
                self.run_portfolio([shares], scenario)

    def test_rates_without_sensitivity_requires_target_when_yields_change(self):
        row = holding(symbol="TLT", factor="rates")
        with self.assertRaises(ValueError):
            self.run_portfolio([row], Scenario("rates", yield_bp=100))
        result = self.run_portfolio(
            [row], Scenario("target", yield_bp=100, targets={"TLT": 95})
        )
        self.assertAlmostEqual(result["P&L USD"], -50)

    def test_nonfinite_result_and_future_valuation_dates_rejected(self):
        from datetime import date, timedelta

        future = (date.today() + timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            validate_holdings(pd.DataFrame([holding(mark_date=future)]), future)
        with self.assertRaises(ValueError):
            self.run_portfolio(
                [holding(quantity=1e308)], Scenario("large", equity=1e308)
            )

    def test_empty_nav_dates_and_nonphysical_scenarios_rejected(self):
        with self.assertRaises(ValueError):
            self.run_portfolio([], Scenario("base"))
        for nav in [0, -100, float("nan")]:
            with self.subTest(nav=nav), self.assertRaises(ValueError):
                self.run_portfolio([holding()], Scenario("base"), nav=nav)
        for shock in [
            Scenario("bad", equity=-1),
            Scenario("bad", fx=-1),
            Scenario("bad", commodity=-1),
            Scenario("bad", iv=float("nan")),
        ]:
            with self.subTest(shock=shock), self.assertRaises(ValueError):
                self.run_portfolio([holding()], shock)
        with self.assertRaises(ValueError):
            validate_holdings(pd.DataFrame([holding()]), "invalid")
        with self.assertRaises(ValueError):
            self.run_portfolio(
                [
                    holding(
                        "call",
                        multiplier=100,
                        underlying=100,
                        strike=100,
                        expiry="2027-09-29",
                        volatility=0.1,
                        rate=0,
                        dividend_yield=0,
                    )
                ],
                Scenario("bad vol", iv=-0.2),
            )


if __name__ == "__main__":
    unittest.main()
