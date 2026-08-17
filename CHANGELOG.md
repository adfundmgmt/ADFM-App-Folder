# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

### Added

- CFTC Positioning Monitor (Page 17) with cross-asset TFF and Disaggregated COT scanning, cohort-level net positioning, open-interest-normalized crowding percentiles and z-scores, weekly change rankings, cohort history, mapped price overlays, and dollar-notional estimates where contract multipliers are explicitly supported.
- SEC 13F Exposure Browser (Page 18) with ticker-to-CUSIP resolution, amendment-aware filing consolidation, a $1 billion default minimum portfolio filter, manager rankings by disclosed portfolio weight, reported market value, or shares, searchable holdings, CSV export, and EDGAR filing links.
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

- Home now uses stable internal route links instead of deployment-sensitive `st.page_link` validation, and the Home/sidebar order follows a single research workflow from public-equity discovery through regime, fundamentals, technical confirmation, positioning, risk execution, and historical context.
- Public Equities Baskets deployment refreshed so the live app reflects the current basket map, which excludes the legacy Private Robotics Access Vehicles (BOT) entry.
- The Options Positioning Compass price-history fallback now uses an intuitive left-to-right volatility scale, top-to-bottom upside/downside scale, matching directional colors, and shorter commentary.
- Home-page tool cards now navigate directly to all 24 tools.
- README catalog now reflects the 24 tools exposed from `Home.py`.
- The catalog and Home navigation now expose 24 tools.
- Rate of Change Dashboard now uses shared calculation and daily-data helpers.
- Global Macro Regime Dashboard and Liquidity Tracker now use the shared market-data loader and preserve missing observations.
- Currency snapshot commits now run application CI.
- All 19 Streamlit pages now use the shared ADFM footer component.
