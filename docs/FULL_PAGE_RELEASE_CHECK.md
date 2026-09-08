# All-page release check — 2026-09-08

All 25 cataloged tools and Home were visited in the deployed app before this
release. Each page was observed until its default analysis completed or an
explicit failure was shown. The Underwriter was run for AAPL, and the 13F
security screen was run for INTC rather than stopping at their landing forms.
Source code, calculation paths, and existing test coverage were reviewed
alongside the browser checks. This checks the main workflow of each tool, not
every possible symbol or combination of controls.

## Browser evidence before release

| # | Tool | Result |
|---:|---|---|
| 1 | Public Equities Baskets | Consolidated panel and full 314-basket map completed. |
| 2 | Global Macro Regime | Completed; 30/30 market proxies and 9/9 macro series loaded. |
| 3 | Liquidity Conditions Monitor | Primary liquidity and confirmation view completed. |
| 4 | Yield Curve Rates Regime Monitor | Stopped on an empty Yahoo response; fixed with official-source recovery. |
| 5 | Credit Conditions Monitor | Credit and sovereign-rate view completed. |
| 6 | Currency Tension Engine | Persisted snapshot, pillar scores, and source notes rendered. |
| 7 | Sector Breadth and Rotation | Completed with disclosed eligibility of 56/59 tickers. |
| 8 | Equity Leadership & Rotation | Rotation map and relationship charts completed. |
| 9 | ADFM Underwriter | AAPL run completed with SEC-driven financials and valuation. |
| 10 | ADFM Chart Terminal | Default technical chart completed. |
| 11 | Cross-Asset Ratio Chartbook | Grouped ratio charts completed. |
| 12 | Rate of Change Regime Explorer | Completed; completed-session data through September 4. |
| 13 | Relative Volatility Lab | NDX/SPX realized and implied comparisons completed. |
| 14 | ETF Flow Pressure Proxy | Default flow-proxy dashboard completed. |
| 15 | Volume Based Sentiment Indicator | Dollar-volume view completed; Yahoo Chart API through September 4. |
| 16 | Options Positioning Compass | Default positioning dashboard completed. |
| 17 | SEC 13F Exposure Browser | INTC screen completed: 1,130 managers for March 31, 2026 holdings. |
| 18 | CFTC Positioning Monitor | Rendered but showed physical reports only through August 26, 2025; truncation fixed. |
| 19 | Market Stress Composite | Default stress view completed. |
| 20 | Catalyst Calendar | Event calendar and contextual charts completed. |
| 21 | Hedge Timer | Default hedge decision and historical views completed. |
| 22 | Position Sizing Lab | AAPL simulation and sizing view completed; 11,525 sessions. |
| 23 | Market Memory Explorer | Calendar-year analogs and historical context completed. |
| 24 | Monthly Seasonality Explorer | Default seasonal charts and tables completed. |
| 25 | Commodity Event Study | Default exhaustion study and forward-outcome tables completed. |

## Additional corrections in this release

- CFTC requests now page beyond the first 50,000 ascending records. A direct
  read from the official API returned **64,407 records, 405 contract codes, and
  a latest report date of September 1, 2026**. Reports older than 21 days are
  excluded from current rankings with a visible explanation.
- Treasury curves recover from an unavailable or unusably incomplete Yahoo
  curve using Federal Reserve/FRED DGS3MO, DGS5, DGS10, and DGS30. The source
  and date are disclosed, the curve is not mixed across providers, and the
  3M investment-basis versus bill-discount distinction is explicit.
- The two ratio chartbooks bridge at most two interior missing observations,
  without extending any price past its last observed date. Cross-asset YTD
  returns use the prior-year close, including the first trading day's move.
- Missing moving-average inputs no longer become a bearish leadership trend.
  Missing recession or policy-change history becomes Unknown, and missing
  foreign-market prices no longer dilute stress breadth as false negatives.
- Market Memory converts percent yields to basis points correctly, including
  legacy Yahoo quotes, and calculates prior-year anchors without repeated
  scans of the entire history.
- The initial audit's shared date, adjustment, return, integrity, basket-cache,
  retry, cache-limit, and vectorized-scoring corrections are included.

## Validation and limits

The full local test suite passes, with shared/engine coverage above the existing
45% release gate. Compilation, dependency consistency, shared-code lint, and
fatal-error lint across every page pass. Added fixtures cover CFTC pagination,
the actual Treasury page with official fallback and both optional source views,
calendar gaps, prior-year anchors, basis points, missing regimes, and breadth.

The initial helper benchmarks showed 184× and 233× speedups with equivalent
results. These are not end-to-end production page timings. Provider latency,
Streamlit cold starts, and concurrent-session memory need ongoing measurement.
Exact historical publication-time/vintage data and every provider value have
not been independently reconciled. CFTC commodity-study release dates still
use an explicitly described normal Friday publication assumption.

## Source references

- [Federal Reserve/FRED DGS10](https://fred.stlouisfed.org/series/DGS10)
- [Federal Reserve/FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)
- [CFTC Public Reporting Environment](https://publicreporting.cftc.gov/)

The user explicitly authorized publishing and deploying this release after
the complete page audit. Post-deployment observations belong in the delivered
audit report after the release is live.
