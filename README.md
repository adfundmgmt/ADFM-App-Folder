# ADFM Analytics Platform

ADFM's internal Streamlit toolkit for daily market monitoring, technical analysis, macro regimes, risk management, and portfolio decision support. Run the application from [Home.py](Home.py); its tool map is the source of truth for the catalog below.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run Home.py
```

For development checks, install `requirements-dev.txt` and run the test suite:

```bash
python -m pip install -r requirements-dev.txt
coverage run --source=adfm_core,cte -m unittest discover -s tests -p "test_*.py" -q
coverage report --fail-under=45
python -m ruff check --select E,F,I,B --ignore E501 Home.py adfm_core cte scripts tests
python -m ruff check --select E9,F63,F7,F82 pages adfm_sector_rotation_config.py
```

## Tool catalog

The application contains 21 tools, in the same order and groups shown on the Home page.

| # | Home-page tool | Primary purpose | Primary inputs |
|---:|---|---|---|
| 1 | ADFM Public Equities Baskets | Compares ADFM equity baskets across leadership, trend strength, dispersion, and benchmark-relative performance. | Internal basket definitions; Yahoo Finance market data |
| 2 | Sector Breadth and Rotation | Measures participation and sector rotation to identify where equity strength is broadening or narrowing. | Yahoo Finance sector and subsector ETFs |
| 3 | Factor Momentum Leadership | Ranks factor momentum to highlight which styles are leading, fading, or inflecting. | Yahoo Finance ETF prices |
| 4 | Rate of Change Dashboard | Tracks multi-horizon rate-of-change regimes for fast reads on momentum, acceleration, and trend pressure. | Yahoo Finance daily OHLCV |
| 5 | Technical Chart Explorer | Explores multi-timeframe chart structure, trend, momentum, volatility bands, and key moving averages. | Yahoo Finance OHLCV |
| 6 | Ratio Charts | Uses relative-strength ratios to compare assets, sectors, credit, factors, and risk appetite proxies. | Yahoo Finance adjusted close history |
| 7 | Market Memory Explorer | Surfaces historical analogs to contextualize the current tape against prior return paths and regimes. | Yahoo Finance market history |
| 8 | Monthly Seasonality Explorer | Shows recurring monthly return and volatility patterns by asset, index, sector, or commodity. | Yahoo Finance; FRED for selected series and regime tags |
| 9 | Volume Based Sentiment Indicator | Reads conviction, participation, and sentiment using volume-regime signals across major liquid assets. | Yahoo Finance adjusted OHLCV; provider fallback where available |
| 10 | ETF Flows Dashboard | Tracks ETF flow-pressure proxies to monitor allocation shifts across macro, equity, and thematic exposures. | Yahoo Finance OHLCV |
| 11 | Global Macro Regime Dashboard | Combines growth, inflation, policy, financial conditions, and market signals into a broad macro-regime read. | Yahoo Finance market proxies |
| 12 | Yield Curve + Rates Regime Monitor | Tracks the Treasury curve, real yields, breakevens, and bull/bear steepener or flattener regimes. | Yahoo Finance rate and market proxies |
| 13 | Credit Conditions Dashboard | Monitors credit spreads, credit ETF ratios, regional banks, loans, EM debt, and financial conditions. | Yahoo Finance market proxies |
| 14 | Liquidity Tracker | Monitors major liquidity drivers including the Fed balance sheet, RRP, TGA, policy rates, and financial conditions. | Yahoo Finance; Federal Reserve FCI-G data |
| 15 | Market Stress Composite | Builds a cross-asset stress score across equities, credit, commodities, FX, rates, breadth, and dispersion. | Yahoo Finance; local last-good cache on provider failure |
| 16 | Event Risk + Catalyst Calendar | Maps upcoming macro catalysts, options windows, Treasury supply, earnings season, and custom event risks. | Yahoo Finance market proxies; configured calendar data |
| 17 | Hedge Timer | Provides tactical timing cues for adding, holding, reducing, or rolling portfolio hedges. | Yahoo Finance; FRED regime inputs |
| 18 | Currency Tension Engine | Maps currencies across trajectory and valuation-policy stretch, with carry, pillar scores, overlays, and daily risk flags. | Persisted Currency Tension Engine snapshot and configured adapters |
| 19 | Relative Volatility Lab | Decomposes selectable realized-volatility ratios and compares them with implied volatility, acceleration, downside, semiconductor, and breadth diagnostics. | Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available |
| 20 | Options Positioning Compass | Maps current implied-volatility richness, downside skew, term structure, and aggregate option activity, with a price-derived volatility fallback when chains are unavailable. | Yahoo Finance current option chains and adjusted close history |
| 21 | SEC 13F Exposure Browser | Ranks institutional managers by a selected security's share of their disclosed Form 13F portfolio. | SEC Form 13F bulk data sets; SEC company ticker directory |

## Tool groups

| Group | Tools |
|---|---|
| Equity Leadership | Public Equities Baskets; Sector Breadth and Rotation; Factor Momentum Leadership; Rate of Change Dashboard |
| Technicals + Analogs | Technical Chart Explorer; Ratio Charts; Market Memory Explorer; Monthly Seasonality Explorer; Relative Volatility Lab |
| Flows + Sentiment | ETF Flows Dashboard; Volume Based Sentiment Indicator; Options Positioning Compass; SEC 13F Exposure Browser |
| Macro + Rates | Global Macro Regime Dashboard; Yield Curve + Rates Regime Monitor; Credit Conditions Dashboard; Liquidity Tracker; Currency Tension Engine |
| Risk + Catalysts | Market Stress Composite; Event Risk + Catalyst Calendar; Hedge Timer |

## Shared application foundations

The `adfm_core` package is the incremental shared layer for common functionality. It currently provides:

- Daily OHLCV loading with ticker normalization, batching, retries, individual fallback, raw-observation preservation, and completed-session handling.
- A centralized market and macro series registry, plus a primary-source FRED adapter with per-series diagnostics.
- Benchmark-calendar alignment, adjusted-price handling, stale-session checks, safe ratios, and close panels.
- A data-integrity policy and diagnostics report for eligible, stale, thin-history, and invalid series.
- Causal PM command-center scores, cross-asset group summaries, movers, and an atomic point-in-time signal ledger.
- Reusable Rate of Change calculations and chart-axis helpers.

The Rate of Change Dashboard, Global Macro Regime Dashboard, and Liquidity Tracker use these foundations. Other pages are being migrated incrementally so their established layouts and calculations remain stable. See [the architecture guide](docs/ARCHITECTURE.md) for the data-source and scoring policies.

## Data-use notes

- Market data are provider supplied and may be delayed, revised, unavailable, or incomplete.
- The 13F browser stores prepared public SEC releases in the ignored `data/13f/` cache. Deployments can set `ADFM_13F_CACHE_DIR` for persistent storage and `ADFM_SEC_USER_AGENT` for an organization-specific SEC request identity.
- Signals and dashboards are deterministic analytical tools, not investment advice or a guarantee of future returns.
- Pages should surface their own as-of date and source context. Where a data field is unavailable, the application should leave it blank rather than fabricate a value.
- Client, holdings, positions, account, and credential data must not be committed. See [the security policy](SECURITY.md).

## Quality checks

GitHub Actions runs dependency consistency, compilation, coverage-gated tests, strict shared-code lint, and fatal-error lint across every page. Dependabot reviews Python and workflow updates weekly. The scheduled Currency Tension Engine import validates all required files and schemas and emits a hash manifest before promoting a snapshot.
