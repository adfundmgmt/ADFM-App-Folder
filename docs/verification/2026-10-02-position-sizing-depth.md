# Position Sizing Lab depth update, October 2, 2026

The main page retains one exposure visual, one linked volatility/sizing chart, and one comparison table. Added 10/20/60-session volatility context, the selected window's percentile versus 252 lagged observations, and its change over 20 sessions. Reference size remains inverse volatility. Permitted size is the minimum of reference size, the exposure ceiling, and an optional invalidation-price loss-budget cap. The hero names the binding limit without suggesting a trade.

Invalidation distance is measured from the latest unadjusted close, with separate long/short validation and ticker/direction-specific price inputs. Zero disables the optional cap. A zero NAV loss budget permits no exposure. The historical exposure chart applies today's exposure caps throughout for illustration; it is not a strategy backtest.

The table compares current, half, uncapped volatility reference and permitted exposures under identical shocks. Historical adverse moves use up to ten years of adjusted OHLCV: average worst 5% daily and five-session returns, worst day and worst gap. Short losses negate the cumulative underlying return. Missing/insufficient data remains unavailable. Dollar notional requires explicitly entered NAV. Scenario P&L is position-only, at fixed initial notional, excluding costs and portfolio offsets.

## Validation

34 focused tests passed on the pinned runtime: volatility sizing (11), portfolio-stress/sizing UI (7), original position-sizing analysis (6), palette (4), and repository standards (6). They verify risk invariants, no lookahead, directional tails, compounded weekly returns, loss-budget caps, wrong-side rejection, short notionals and lazy stress loading. Full compilation and both CI lint commands passed.

The initial complete suite stalled in the provider-dependent basket UI test and was interrupted. A subsequent run excluded that test and the basket transport test, both previously observed to invoke unrelated live provider requests. Of 433 tests run, 426 passed and seven failed. All seven are the previously baseline-confirmed failures listed in `2026-10-01-position-sizing.md`: basket partial response, two catalog/documentation expectations, Underwriter layout and three Options Positioning expectations. No new broad-suite failures were observed. A clean full-suite result is not claimed.

Invalidation budgets assume execution at the entered price; gaps, slippage and costs may exceed the budget. Historical tails and two-sigma scenarios are descriptive, not forecasts or loss guarantees. Options/futures still require instrument-specific risk treatment.
