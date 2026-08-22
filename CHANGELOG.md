# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

- Equity Leadership & Rotation now opens with the Rotation Map and renders every selected relationship as an expanded chart; the current read, family summary, momentum matrix, ranking table, and single-detail selector were removed.
- Cross-Asset Ratio Chartbook now renders all selected families and all 50 default ratios immediately, removing the focused-versus-full view gate.

### Added

- SEC 13F Exposure Browser (Page 23) with ticker-to-CUSIP resolution, amendment-aware filing consolidation, a $1 billion default minimum portfolio filter, manager rankings by disclosed portfolio weight, reported market value, or shares, searchable holdings, CSV export, and EDGAR filing links.
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
