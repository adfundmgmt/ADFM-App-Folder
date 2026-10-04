# Hedge Timer actual red-dot verification

The plotted signal no longer requires a weighted long-term trend confirmation. The same event series drives the chart, sidebar counts, episode audit, downloads, and historical driver dates. Carried signal states never earn capture credit. The pre-peak window is now five sessions, rather than twenty, and a post-peak event must occur before the first daily range loss beyond 3%. A price recovery cannot reopen that historical deadline.

| Intraday high/low audit, 2020 through 2026-10-02 | Previous actual red dots, same five-session test | New actual red dots |
| --- | --- | --- |
| SPX early captures | 0/7 | 7/7 |
| NDX early captures | 0/17 | 5/17 |
| SPX event count | 48 | 39 |
| NDX event count | 46 | 34 |

The earlier 1/7 SPX and 3/17 NDX red-dot measurements used a twenty-session pre-peak window and could credit an ongoing state. The earlier 7/7 and 17/17 headline counts measured the separate Watch layer. Neither is a current red-dot result.

| SPX intraday peak | Actual new red-dot date | Loss from peak at callout |
| --- | --- | --- |
| 2020-02-19 | 2020-02-21 | 1.64% |
| 2020-09-02 | 2020-08-31 | Before peak |
| 2022-01-04 | 2022-01-05 | 2.45% |
| 2022-03-29 | 2022-03-31 | 2.31% |
| 2022-08-16 | 2022-08-19 | 2.24% |
| 2023-07-27 | 2023-08-02 | 2.03% |
| 2025-02-19 | 2025-02-21 | 2.18% |

SPX has 30 mature non-capturing false callouts, one late/repeat callout, and one pending outcome. NDX has 24 mature non-capturing false callouts, four late/repeats, and one pending outcome. "False" uses the stated major-drawdown definition; it does not assess whether a hedge helped in a smaller correction. Pending callouts have fewer than sixty completed follow-up sessions. These costs are deliberately shown beside recall.

## Frozen rules

Three alternative triggers share one recovery latch:

1. **Price break plus corroboration:** close below the previous three-session close low, between 1.25% and 3% below the rolling twenty-session close high, and at least two distinct risk groups.
2. **Shock:** at least a 1.25% one-day index loss, VIX up at least 15% on the day, and close no more than 3% below the twenty-session close high.
3. **Near-high divergence:** index within 0.5% of its twenty-session close high and up over five sessions, breadth weak, VIX up at least 15% over five sessions and at least 5% above its twenty-session average. This is required for the September 2020 pre-peak warning; waiting for a price break misses its intraday deadline.

Breadth counts once if RSP/SPY or IWM/SPY is below its hundred-session average and declining over twenty sessions, or sector participation deteriorates under the existing 55%/73% weak-sector rule. Volatility counts once if VIX rises 5% in a day, 10% in three sessions, reaches 1.15 times its twenty-session average, or VIX9D/VIX reaches one. Credit counts once if HYG/LQD retreats at least 1% from its ten-session high and declines over three sessions. These are proxies, not constituent-level breadth or duration-neutral credit spreads.

After a dot, another requires at least ten sessions plus three consecutive known closes above EMA10, within 1% of the twenty-session high, with VIX not rising over three sessions. Time alone cannot rearm it. Unknown, nonpositive, or infinite inputs cannot emit or contribute to recovery. MA50, MA200, weekly momentum, and realized-volatility expansion do not trigger the new red dots. RSI and the late-stage gate remain independent limits on fresh shorts, not peak predictors.

## Research and limits

Rules were fitted using SPX only. Broad SPX price/volatility/recovery probes established the 1.25% retreat and five-session pre-peak window; a bounded 108-candidate credit/shock/spacing refinement then prioritizes full SPX recall, fewer mature false callouts, fewer late/repeat callouts, and fewer total dots. NDX is evaluated only after selection and never changes the chosen rules. Reproduce that final refinement with:

```bash
python scripts/calibrate_hedge_callouts.py
```

The selected parameters and transfer counts are in [callout_calibration.json](../data/hedge_timer/callout_calibration.json), and every SPX/NDX episode is in [callout_drawdown_audit.csv](../data/hedge_timer/callout_drawdown_audit.csv). Sources are the existing dated Yahoo adjusted-close and genuine index range checkpoints. These retrospective provider prices may be revised. Causal prefix replay confirms that adding later prices does not repaint signals; it does not establish out-of-time predictive performance. SPX full coverage is an in-sample fit, and NDX misses demonstrate that full recall does not transfer.

Actual high-yield OAS history was tested as a source candidate. The available FRED CSV returned only 2023-10-03 through 2026-10-01, without publication-vintage evidence covering COVID or 2022. It was not introduced into the production trigger or silently backfilled. Credit remains explicitly labelled HYG/LQD proxy.

The new regression tests first failed against the previous red-dot rule for COVID, all three 2022 legs, September 2020, repeated crossings, and chart/audit date agreement. Missing-data, invalid-price, reset, minimum-spacing, causal-prefix, and independence from NDX/legacy weights are checked. Production availability and independent review are verified separately before completion.
