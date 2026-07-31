# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

### Added

- Momentum Scanner (Page 19).
- Relative Volatility Lab (Page 20).
- PM command center with causal cross-asset regime, breadth, impulse, dispersion, confidence, and mover signals.
- Six PM research and operating tools covering the daily brief, attribution, historical analogs, primary sources, decision journaling, reliability, and threshold alerts.
- Causal signal-history reconstruction with forward hit rate, return, drawdown, turnover, and bounded evidence-weight diagnostics.
- Historical regime matching across rates, inflation, liquidity, dollar, credit, volatility, and breadth with forward cross-asset return distributions.
- Institutional primary-source registry for Treasury, BLS, BEA, Fed, ECB, BOJ, BOE, CFTC, EIA, and SEC data.
- Centralized market/macro data registry, primary-source FRED adapter, and point-in-time signal ledger.
- Shared core modules for market data, data integrity, Rate of Change calculations, catalog metadata, UI primitives, and session data-load status.
- Currency snapshot schema validation and hash manifest before data promotion.
- Continuous integration, coverage gating, regression coverage, weekly dependency updates, security policy, architecture guide, and release-review templates.
- Reproducible direct-dependency constraints and repository-wide standards checks.

### Changed

- Home-page tool cards now navigate directly to all 20 tools.
- README catalog now reflects the 20 tools exposed from `Home.py`.
- The catalog and Home navigation now expose 26 tools, including a dedicated PM Research + Operations group.
- Rate of Change Dashboard now uses shared calculation and daily-data helpers.
- Global Macro Regime Dashboard and Liquidity Tracker now use the shared market-data loader and preserve missing observations.
- Currency snapshot commits now run application CI.
- All 20 Streamlit pages now use the shared ADFM footer component.
