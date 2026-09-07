# Source inventory

Baseline: `ac2bd39d8371e959c778437519d390e1891c1d08`. 25 pages plus Home. 94 Python files inspected.

Static call counts are audit clues, not the number of rendered charts or controls. Conditional paths and imported renderers must be checked during each page migration.

| Page | File lines | Python dependencies (transitive) | Data sources |
|---|---:|---|---|
| 1. ADFM Public Equities Baskets | 3426 | Page-local functions | Internal basket definitions; Yahoo Finance market data |
| 2. Global Macro Regime | 858 | `data_registry.py`, `market_data.py`, `primary_data.py` | Federal Reserve / FRED primary macro series; Yahoo Finance market proxies |
| 3. Liquidity Conditions Monitor | 703 | `_liquidity_tracker_base.py`, `market_data.py` | Federal Reserve H.4.1; New York Fed rates and RRP; ICE BofA OAS via FRED; broad dollar; real yields; Yahoo Finance confirmation proxies |
| 4. Yield Curve Rates Regime Monitor | 1144 | Page-local functions | Yahoo Finance Treasury yield symbols: ^IRX, ^FVX, ^TNX, ^TYX |
| 5. Credit Conditions Monitor | 1499 | `data_registry.py`, `market_data.py`, `primary_data.py` | ICE BofA corporate OAS and U.S. Treasury yields via Federal Reserve FRED; Yahoo Finance market confirmation; Trading Economics or fresh Stooq sovereign yields with OECD/FRED structural fallback |
| 6. Currency Tension Engine | 1051 | `adapters/base.py`, `commentary/narrator.py`, `config.py`, `dashboard/plots.py`, `flags/notes.py`, `flags/overlays.py`, `flags/positioning.py`, `scoring/compositor.py`, `scoring/engine.py`, `scoring/history.py`, `transform/features.py`, `transform/pairwise.py`, `transform/zscore.py` | Persisted Currency Tension Engine snapshot and configured adapters |
| 7. Sector Breadth and Rotation | 1648 | `adfm_sector_rotation_config.py` | Yahoo Finance sector and subsector ETFs |
| 8. Equity Leadership & Rotation | 464 | `leadership.py` | Yahoo Finance adjusted ETF and index prices |
| 9. ADFM Underwriter | 1176 | `market_data.py`, `sec_fundamentals.py` | SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history |
| 10. ADFM Chart Terminal | 3277 | `chart_patterns.py` | Yahoo Finance OHLCV |
| 11. Cross-Asset Ratio Chartbook | 991 | Page-local functions | Yahoo Finance adjusted close history |
| 12. Rate of Change Regime Explorer | 438 | `data_integrity.py`, `market_data.py`, `rate_of_change.py` | Yahoo Finance daily OHLCV |
| 13. Relative Volatility Lab | 653 | `market_data.py`, `relative_volatility.py` | Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available |
| 14. ETF Flow Pressure Proxy | 1304 | Page-local functions | Yahoo Finance OHLCV |
| 15. Volume Based Sentiment Indicator | 1848 | `market_data.py`, `regime_math.py` | Yahoo Finance adjusted OHLCV; provider fallback where available |
| 16. Options Positioning Compass | 664 | `market_data.py`, `options_positioning.py`, `options_sources.py`, `relative_volatility.py` | Yahoo Finance current option chains and adjusted close history |
| 17. SEC 13F Exposure Browser | 18 | `sec_13f.py`, `sec_13f_browser.py`, `sec_13f_corrected.py` | SEC Form 13F bulk data sets; SEC company ticker directory |
| 18. CFTC Positioning Monitor | 496 | `cftc_positioning.py`, `market_data.py` | CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts |
| 19. Market Stress Composite | 685 | Page-local functions | Yahoo Finance; local last-good cache on provider failure |
| 20. Catalyst Calendar | 4 | `catalyst_calendar_exact_page.py`, `catalyst_calendar_official_page.py`, `catalyst_calendar_page.py`, `data_registry.py`, `primary_data.py` | Official agency calendars; recurring market-calendar rules; Yahoo Finance market proxies |
| 21. Hedge Timer | 1122 | Page-local functions | Yahoo Finance; FRED regime inputs |
| 22. Position Sizing Lab | 881 | `market_data.py`, `position_sizing.py` | Yahoo Finance adjusted OHLCV, earnings dates, and liquid cross-asset proxies |
| 23. Market Memory Explorer | 1661 | Page-local functions | Yahoo Finance market history |
| 24. Monthly Seasonality Explorer | 1841 | `monthly_returns_matrix.py` | Yahoo Finance; FRED for selected series and regime tags |
| 25. Commodity Event Study | 18 | `cftc_positioning.py`, `commodity_top_exhaustion_page.py` | Yahoo Finance daily continuous-futures history; CFTC Disaggregated Managed Money positioning where mapped |

See `source-audit.json` for full control/default expressions, functions, decorator TTLs, state/callback references, URLs, and hashes for **every** file. Presentation imports also execute `adfm_core/__init__.py`; its global patches are not included in the transitive counts above.
