# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

- Restored scheduled currency-snapshot validation by installing its pinned Python dependencies, and restored the SEC 13F release refresh by running it as a module with the complete pinned application dependencies.
- All-page release audit: CFTC now retrieves every response page (the former 50,000-row cap omitted the latest year of physical-commodity positioning) and rejects stale reports from current rankings. The Treasury curve recovers from Yahoo failures using a fully disclosed, unmixed Federal Reserve / FRED constant-maturity curve.
- Ratio chartbooks now bridge only short interior gaps and retain actual observation endpoints; cross-asset YTD returns include the first trading day. Missing leadership trend, recession, policy-change, and market-stress breadth inputs no longer become false signals. Market Memory now converts Treasury yield changes to basis points correctly and computes prior-year anchors without repeatedly scanning the full history.
- Corrected shared daily date normalization, duplicate selection, timezone-aware session cutoffs, dividend-adjusted price/volume treatment, missing return endpoints, and rejection of infinite observations or negative volume.
- Accelerated previous-observation percentile and grouped-composite calculations while retaining their weighting, tie, missing-data, and coverage rules; bounded shared/basket caches and made download timeouts explicit.
- Basket loading now retries completely empty ticker columns and identifies partial recovery from the last-good cache in source diagnostics.
- Restored four basket-map checks to unittest discovery and added regression coverage for market data, basket recovery, and commodity-study causality and forward outcomes.

- Public Equities Baskets now includes the missing global bank, insurer, industrial, defense, grid, semiconductor, AI-hardware, energy-royalty, wealth-management, custody-bank, and BDC cohorts; geographic baskets are organized into continental groups rather than one flat country list.
- Standardized every tool sidebar around one concise purpose, three-step reading order, key caveat, and primary-input block; reordered the flat page list so SEC 13F and CFTC sit with Positioning + Flows before Risk + Execution and Historical Context.
- Equity Leadership & Rotation now formats all relative-return percentages to exactly two decimal places in Rotation Map hover labels and chart captions.
- Equity Leadership & Rotation now opens with the Rotation Map and renders all 11 S&P 500 sectors versus SPY, five China/U.S. relationships, three breadth and alternative-weighting ratios, and six inter-sector relationships as expanded charts; the unavailable Russell 2000 equal-weight index pair was replaced with a live RWJ/IJR small-cap weighting comparison.
- Cross-Asset Ratio Chartbook now renders 38 focused ratios across duration and crisis hedges, commodities versus equity indices, credit and funding, and financial-intermediary baskets; default single-stock ratios were removed.

### Added

- SEC 13F Exposure Browser (Page 17) with ticker-to-CUSIP resolution, amendment-aware filing consolidation, a $1 billion default minimum portfolio filter, manager rankings by disclosed portfolio weight, reported market value, or shares, searchable holdings, CSV export, and EDGAR filing links.
- ADFM Underwriter (Page 22) with an ADFM-named route, SEC EDGAR financial normalization, complete-window 50-day and 200-day price averages, compact color-coded valuation and quality cards, collapsed calculation methodology, issuer-credit measures, debt maturities, recent filings, and source-audit links.
- Options Positioning Compass (Page 20) with Yahoo Finance option-chain analytics, generated commentary, volatility/skew ranks, term structure, IV surface, estimated premium activity, and a price-derived fallback when hosted option endpoints are unavailable.
- Relative Volatility Lab (Page 19).
- PM command center with causal cross-asset regime, breadth, impulse, dispersion, confidence, and mover signals.
- Centralized market/macro data registry, primary-source FRED adapter, and point-in-time signal ledger.
- Shared core modules for market data, data integrity, Rate of Change calculations, catalog metadata, UI primitives, and session data-load status.
- Currency snapshot schema validation and hash manifest before data promotion.
- Continuous integration, coverage gating, regression coverage, weekly dependency updates, security policy, architecture guide, and release-review templates.
- Reproducible direct-dependency constraints and repository-wide standards checks.

### Changed

- Factor Momentum Leadership is now Equity Leadership & Rotation: a scored 25-relationship scanner with family rankings, a four-state rotation map, a multi-horizon heatmap, a styled leaderboard, and one selected historical drill-down instead of 25 duplicate full charts.
- Ratio Charts is now Cross-Asset Ratio Chartbook, with a default focused relationship view, six institutional chart families, an optional two-column full chartbook, cleaner default moving averages, and retained custom ratios.
- SEC 13F Exposure Browser now supports direct filing-manager search by name or CIK, with Duquesne Family Office LLC (CIK 0001536411) as the manager-mode baseline and faster manager-specific portfolio loading.
- Home now uses stable internal route links instead of deployment-sensitive `st.page_link` validation, and the Home/sidebar order follows a single research workflow from public-equity discovery through regime, fundamentals, technical confirmation, positioning, risk execution, and historical context.
- Public Equities Baskets deployment refreshed so the live app reflects the current basket map, which excludes the legacy Private Robotics Access Vehicles (BOT) entry.
- The Options Positioning Compass price-history fallback now uses an intuitive left-to-right volatility scale, top-to-bottom upside/downside scale, matching directional colors, and shorter commentary.
- Home-page tool cards now navigate directly to all 23 tools.
- README catalog now reflects the 23 tools exposed from `Home.py`.
- The catalog and Home navigation now expose 23 tools.
- Rate of Change Dashboard now uses shared calculation and daily-data helpers.
- Global Macro Regime Dashboard and Liquidity Tracker now use the shared market-data loader and preserve missing observations.
- Currency snapshot commits now run application CI.
- All 19 Streamlit pages now use the shared ADFM footer component.
