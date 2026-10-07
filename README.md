# ADFM Analytics Platform

ADFM's internal Streamlit toolkit for daily market monitoring, technical analysis, macro regimes, risk management, and portfolio decision support. Run the application from [Home.py](Home.py); its tool map is the source of truth for the catalog below. Visible navigation titles are intentionally decoupled from legacy page filenames so routes can remain stable while naming stays concise and consistent.

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

The application contains 23 tools, in the same order and groups shown on the Home page.

| # | Home-page tool | Primary purpose | Primary inputs |
|---:|---|---|---|
| 1 | Equity Baskets | Compares ADFM equity baskets across leadership, trend strength, dispersion, and benchmark-relative performance. | Internal basket definitions; Yahoo Finance market data |
| 2 | Global Bond Monitor | Tracks yield exhaustion setups, confirmed tops, historical forward moves and baseline outcomes across U.S. rates, credit spreads and global sovereign markets. | Federal Reserve and ICE BofA via FRED; OECD monthly sovereign rates via FRED |
| 3 | Liquidity | Separates the level and marginal impulse of system liquidity across Fed plumbing, overnight funding, credit transmission, and market confirmation. | Federal Reserve H.4.1; New York Fed; FRED; Yahoo Finance proxies |
| 4 | Rates & Yield Curve | Tracks outright Treasury yields, curve spreads, and bull/bear steepener or flattener regimes. | Federal Reserve / FRED nominal and real Treasury yields and inflation breakevens; Yahoo nominal-curve fallback |
| 5 | Credit Conditions | Monitors credit spreads, credit ETF ratios, regional banks, loans, EM debt, and financial conditions. | Yahoo Finance market proxies |
| 6 | FX Regime | Maps currencies across trajectory and valuation-policy stretch, with carry, pillar scores, overlays, and daily risk flags. | Persisted Currency Tension Engine snapshot and configured adapters |
| 7 | Sector Rotation | Measures participation and sector rotation to identify where equity strength is broadening or narrowing. | Yahoo Finance sector and subsector ETFs |
| 8 | Equity Leadership | Maps all 11 S&P 500 sectors versus SPY, five China/U.S. relationships, three breadth and alternative-weighting ratios, and six inter-sector relationships. | Yahoo Finance adjusted ETF and index prices |
| 9 | Equity Underwriter | Calculates filing-driven valuation, per-share growth, margins, returns, liquidity, capital structure, issuer-credit ratios, debt maturities, and recent SEC events. | SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history |
| 10 | Chart Terminal | Explores multi-timeframe chart structure, trend, momentum, volatility bands, and key moving averages. | Yahoo Finance OHLCV |
| 11 | Cross-Asset Ratios | Displays 38 duration, crisis-hedge, commodity, credit, funding, and financial-intermediary ratios as a grouped scrollable chartbook, plus custom relationships. | Yahoo Finance adjusted close history |
| 12 | Momentum & Rate of Change | Tracks multi-horizon rate-of-change regimes for fast reads on momentum, acceleration, and trend pressure. | Yahoo Finance daily OHLCV |
| 13 | Relative Volatility | Decomposes selectable realized-volatility ratios and compares them with implied volatility, acceleration, downside, semiconductor, and breadth diagnostics. | Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available |
| 14 | ETF Flow Pressure | Combines reported weekly ETF net issuance with normalized daily trading pressure across tactical ETF exposures. | Investment Company Institute; Yahoo Finance OHLCV |
| 15 | Volume Sentiment | Reads conviction, participation, and sentiment using volume-regime signals across major liquid assets. | Yahoo Finance adjusted OHLCV; provider fallback where available |
| 16 | Options Relative Value | Maps underlying price trend against the volatility premium to identify markets where implied volatility screens rich or cheap to recent realized volatility. | Yahoo Finance current option chains and adjusted close history; Cboe delayed option-chain fallback |
| 18 | CFTC Positioning | Scans financial and physical futures for crowded longs, crowded shorts, and sharp weekly positioning shifts, with historical percentile and z-score context. | CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts |
| 19 | Market Stress | Builds a cross-asset stress score across equities, credit, commodities, FX, rates, breadth, and dispersion. | Yahoo Finance; local last-good cache on provider failure |
| 20 | Catalyst Calendar | Maps upcoming macro catalysts, options windows, Treasury supply, earnings season, and custom event risks. | Official agency calendars; recurring market-calendar rules; Yahoo Finance market proxies |
| 21 | Hedge Timing | Plots SPX-fitted hedge callouts from price breaks and distinct breadth, volatility, and credit risk groups, with shock and divergence triggers, recovery resets, and an audit of actual red dots in 10%+ local-peak drawdowns. NDX uses the same frozen rules; missed and late events remain visible. | Yahoo Finance adjusted closes for S&P 500, Nasdaq-100, SPY, RSP, IWM, HYG, LQD, all 11 S&P 500 sector ETFs, VIX, VIX9D, VIX3M, and VVIX; index daily highs and lows |
| 23 | Market Memory | Surfaces historical analogs to contextualize the current tape against prior return paths and regimes. | Yahoo Finance market history |
| 24 | Seasonality | Shows recurring monthly return and volatility patterns by asset, index, sector, or commodity. | Yahoo Finance; FRED for selected series and regime tags |
| 25 | Commodity Event Study | Marks repeatable commodity price events and measures historical forward returns and drawdowns across Yahoo Finance futures histories. | Yahoo Finance daily continuous-futures price history |

## Tool groups

| Group | Tools |
|---|---|
| Macro Regime | Global Macro; Liquidity; Rates & Yield Curve; Credit Conditions; FX Regime |
| Equity Discovery | Equity Baskets |
| Equity Leadership | Sector Rotation; Equity Leadership |
| Fundamental Research | Equity Underwriter |
| Technical Confirmation | Chart Terminal; Cross-Asset Ratios; Momentum & Rate of Change; Relative Volatility |
| Positioning + Flows | ETF Flow Pressure; Volume Sentiment; Options Relative Value; CFTC Positioning |
| Risk + Execution | Market Stress; Catalyst Calendar; Hedge Timing |
| Historical Context | Market Memory; Seasonality; Commodity Event Study |

## Shared application foundations

The `adfm_core` package is the incremental shared layer for common functionality. It currently provides:

- Daily OHLCV loading with ticker normalization, batching, retries, individual fallback, raw-observation preservation, and completed-session handling.
- A centralized market and macro series registry, plus a primary-source FRED adapter with per-series diagnostics.
- Benchmark-calendar alignment, adjusted-price handling, stale-session checks, safe ratios, and close panels.
- A data-integrity policy and diagnostics report for eligible, stale, thin-history, and invalid series.
- Causal PM command-center scores, cross-asset group summaries, movers, and an atomic point-in-time signal ledger.
- Reusable Rate of Change calculations and chart-axis helpers.
- Historical conviction-based position sizing, target/invalidation first-touch analysis, earnings-event risk, liquidity caps, and an interactive compounding simulation built from observed holding-period outcomes.
- SEC EDGAR ticker resolution, XBRL concept normalization, stand-alone-quarter reconstruction, filing provenance, current valuation, and issuer-credit calculations.
- CFTC Commitments of Traders retrieval, cohort normalization, open-interest-adjusted crowding percentiles, z-scores, weekly changes, and mapped futures price overlays.

Daily market transport is shared across the tools while page adapters retain their established calculation and calendar rules. See [the architecture guide](docs/ARCHITECTURE.md) for data-source, historical-availability, portfolio-privacy and presentation policies.

## Performance measurement

Run default-page cold/warm render measurements in isolated processes:

```bash
python scripts/benchmark_tools.py --all --jobs 2 --timeout 45 --output /tmp/adfm-benchmark.json
```

Each page reports runtime failures, provider-error/warning counts, cold and warm elapsed time, and peak process memory. These are local Streamlit AppTest measurements, not production browser latency or concurrent-user load tests. A rendered source-unavailable state is distinguished from a Python runtime failure; it is not successful source reconciliation. The measurements do not include holdings or account data.

## Data-use notes

- The bond monitor's default Cycle Top profile uses one causal trailing-high/reversal rule across history. Its ten requested U.S. 10-year peak episodes are calibration cases, not an independent validation set. The visible audit reports actual alert dates after each observed peak, delays, yield distance and subsequent outcomes; the all-alert statistics retain unsuccessful calls. GS10 monthly business-day averages since 1953 remain separate from DGS10 daily observations since 1962. January 1960 is a monthly audit with unverified publication timing, never a fabricated daily observation. The 1974–75 reference window reports its higher peak.
- Portfolio stress uses a dated CSV, explicit USD NAV and supplied marks. Uploads stay in the current session. Options use European model changes anchored to the supplied premium; rates ETFs require supplied sensitivities or explicit underlying price targets. Margin and cash-buffer outputs are estimates from supplied broker inputs.
- Daily sovereign histories retain their precise curve bases and remain separate from monthly OECD averages. Stale or unavailable observations do not receive current signal labels.
- Known-at-month-start seasonality uses prior market observations and ALFRED release/revision records. `FRED_API_KEY` is required for the official release-history API; missing historical availability stays unknown. Revised descriptive studies remain available. Strict CFTC timing excludes dates without verified actual release records.
- Public commodity captures run through `.github/workflows/capture-public-signals.yml` and preserve scheduled versions in `data/signals/`. Set `ADFM_SIGNAL_LEDGER_PATH` to a persistent location for runtime captures. Continuous-futures histories are price studies; actual contract P&L requires the correct traded contract and quote multiplier.
- Market data are provider supplied and may be delayed, revised, unavailable, or incomplete.
- CFTC positioning is a weekly Tuesday snapshot normally released Friday; it is not a real-time flow feed. Dollar notional is shown only for contracts with explicit mapped multipliers.
- Signals and dashboards are deterministic analytical tools, not investment advice or a guarantee of future returns.
- Pages should surface their own as-of date and source context. Where a data field is unavailable, the application should leave it blank rather than fabricate a value.
- Client, holdings, positions, account, and credential data must not be committed. See [the security policy](SECURITY.md).

## Quality checks

GitHub Actions runs dependency consistency, compilation, coverage-gated tests, strict shared-code lint, and fatal-error lint across every page. Dependabot reviews Python and workflow updates weekly. The scheduled Currency Tension Engine import validates all required files and schemas and emits a hash manifest before promoting a snapshot.
