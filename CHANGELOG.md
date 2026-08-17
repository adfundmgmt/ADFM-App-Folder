# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

### Added

- SEC 13F Exposure Browser (Page 21) with ticker-to-CUSIP resolution, amendment-aware filing consolidation, a $1 billion default minimum portfolio filter, and manager rankings by disclosed portfolio weight, reported market value, or shares.
- Options Positioning Compass (Page 20) with Yahoo Finance option-chain analytics, generated commentary, volatility/skew ranks, term structure, IV surface, estimated premium activity, and a price-derived fallback when hosted option endpoints are unavailable.
- Relative Volatility Lab (Page 19).
- PM command center with causal cross-asset regime, breadth, impulse, dispersion, confidence, and mover signals.
- Centralized market/macro data registry, primary-source FRED adapter, and point-in-time signal ledger.
- Shared core modules for market data, data integrity, Rate of Change calculations, catalog metadata, UI primitives, and session data-load status.
- Currency snapshot schema validation and hash manifest before data promotion.
- Continuous integration, coverage gating, regression coverage, weekly dependency updates, security policy, architecture guide, and release-review templates.
- Reproducible direct-dependency constraints and repository-wide standards checks.

### Changed

- The Options Positioning Compass price-history fallback now uses an intuitive left-to-right volatility scale, top-to-bottom upside/downside scale, matching directional colors, and shorter commentary.
- Home-page tool cards now navigate directly to all 21 tools.
- README catalog now reflects the 21 tools exposed from `Home.py`.
- The catalog and Home navigation now expose 21 tools.
- Rate of Change Dashboard now uses shared calculation and daily-data helpers.
- Global Macro Regime Dashboard and Liquidity Tracker now use the shared market-data loader and preserve missing observations.
- Currency snapshot commits now run application CI.
- All 21 Streamlit pages now use the shared ADFM footer component.
