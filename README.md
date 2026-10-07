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
| 1 | Equity Baskets | Compares thematic equity baskets by performance, trend, leadership and dispersion against a selected benchmark. | Internal basket definitions; Yahoo Finance market data |
| 7 | Sector Breadth and Rotation | Shows where equity participation is broadening or narrowing and which sectors are gaining or losing leadership. | Yahoo Finance sector and subsector ETFs |
| 8 | Equity Relative Strength | Ranks sector, regional and breadth relationships to show which equity exposures are outperforming their counterparts. | Yahoo Finance adjusted ETF and index prices |
| 9 | Equity Underwriter | Examines company valuation, growth, profitability, balance-sheet strength and debt using financial statements and recent market prices. | SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history |
| 2 | Bond Cycle Monitor | Examines yield exhaustion, reversal signals and historical outcomes across Treasury yields, credit spreads and global sovereign rates. | Federal Reserve and ICE BofA via FRED; OECD monthly sovereign rates via FRED |
| 3 | Liquidity Conditions | Tracks changes in central-bank liquidity, funding conditions and credit transmission, alongside market confirmation and financial conditions. | Federal Reserve H.4.1; New York Fed rates and RRP; ICE BofA OAS via FRED; broad dollar; real yields; Yahoo Finance confirmation proxies |
| 4 | Rates and Yield Curve | Tracks Treasury yields, real yields, inflation expectations and changes in the shape of the yield curve. | Federal Reserve / FRED nominal and real Treasury yields and inflation breakevens; Yahoo nominal-curve fallback |
| 5 | Credit Conditions | Compares global government-yield moves, corporate credit spreads and funding costs, with market indicators of financial stress. | ICE BofA corporate OAS and U.S. Treasury yields via Federal Reserve FRED; Yahoo Finance market confirmation; Trading Economics or fresh Stooq sovereign yields with OECD/FRED structural fallback |
| 6 | FX Valuation and Trend | Compares currency trends with valuation, policy and carry to identify stretched or changing market conditions. | Persisted Currency Tension Engine snapshot and configured adapters |
| 10 | Chart Terminal | Explores price history, trend, momentum, volatility and chart structure across assets and multiple time horizons. | Yahoo Finance OHLCV |
| 11 | Cross-Asset Ratios | Charts relative performance across rates, equities, commodities, credit and currencies, including relationships selected by the user. | Yahoo Finance adjusted close history |
| 12 | Momentum | Compares price trends, returns and acceleration across several horizons to identify strengthening or weakening market momentum. | Yahoo Finance daily OHLCV |
| 13 | Relative Volatility | Compares two assets' realized volatility, its historical distribution and optional implied-volatility measures over a selected window. | Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available |
| 14 | ETF Trading Pressure | Ranks dollar-weighted ETF trading pressure across the full tactical universe with pressure versus ADV, historical percentile and price-pressure divergence context. | Yahoo Finance OHLCV |
| 15 | Volume Participation | Shows unusually heavy or quiet trading participation alongside price trends and the historical outcomes of similar sessions. | Yahoo Finance adjusted OHLCV; provider fallback where available |
| 16 | Options Relative Value | Compares price trends with implied versus realized volatility to identify relatively rich or cheap option premiums. | Yahoo Finance current option chains and adjusted close history; Cboe delayed option-chain fallback |
| 18 | CFTC Positioning | Tracks futures positioning, historical crowding and weekly changes across major financial and commodity contracts using CFTC reports. | CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts |
| 19 | Market Stress | Tracks stress across equities, credit, rates, currencies and commodities to identify broader changes in market conditions. | Yahoo Finance; local last-good cache on provider failure |
| 20 | Catalyst Calendar | Charts upcoming economic releases, central-bank decisions and market-calendar events, with dates, categories and sources in one table. | Official agency calendars; recurring market-calendar rules; Yahoo Finance market proxies |
| 21 | Drawdown Risk | Monitors price, breadth, volatility and credit conditions for drawdown warnings in the S&P 500 and Nasdaq-100. | Yahoo Finance adjusted closes for S&P 500, Nasdaq-100, SPY, RSP, IWM, HYG, LQD, all 11 S&P 500 sector ETFs, VIX, VIX9D, VIX3M, and VVIX; index daily highs and lows |
| 23 | Historical Analogs | Finds historical return paths resembling the current market and compares what followed across different periods and regimes. | Yahoo Finance market history |
| 24 | Seasonality | Compares recurring monthly return and volatility patterns across assets, with historical distributions and optional regime filters. | Yahoo Finance; FRED for selected series and regime tags |
| 25 | Commodity Exhaustion | Studies extended commodity moves, reversal confirmation and subsequent returns to assess whether potential tops held historically. | Yahoo Finance daily continuous-futures history; CFTC Disaggregated Managed Money positioning where mapped |

## Tool groups

| Group | Tools |
|---|---|
| Equity Research | Equity Baskets; Sector Breadth and Rotation; Equity Relative Strength; Equity Underwriter |
| Macro Regime | Bond Cycle Monitor; Liquidity Conditions; Rates and Yield Curve; Credit Conditions; FX Valuation and Trend |
| Technical Confirmation | Chart Terminal; Cross-Asset Ratios; Momentum; Relative Volatility |
| Positioning and Flows | ETF Trading Pressure; Volume Participation; Options Relative Value; CFTC Positioning |
| Risk and Catalysts | Market Stress; Catalyst Calendar; Drawdown Risk |
| Historical Context | Historical Analogs; Seasonality; Commodity Exhaustion |

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
