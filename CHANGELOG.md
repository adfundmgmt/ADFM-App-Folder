# Changelog

All notable application changes are recorded here. Dates use ISO 8601.

## Unreleased

### Added

- Momentum Scanner (Page 19).
- Relative Volatility Lab (Page 20).
- PM command center with causal cross-asset regime, breadth, impulse, dispersion, confidence, and mover signals.
- Cross-Asset Correlation Lab (Page 21) for correlation matrices, beta term structure, market-mode concentration, effective independent bets, regime history, conditional correlations, and pair diagnostics.
- Centralized market/macro data registry, primary-source FRED adapter, and point-in-time signal ledger.
- Shared core modules for market data, data integrity, Rate of Change calculations, catalog metadata, UI primitives, and session data-load status.
- Currency snapshot schema validation and hash manifest before data promotion.
- Continuous integration, coverage gating, regression coverage, weekly dependency updates, security policy, architecture guide, and release-review templates.
- Reproducible direct-dependency constraints and repository-wide standards checks.

### Changed

- Home-page tool cards now navigate directly to all 21 tools.
- README catalog now reflects the 21 tools exposed from `Home.py`.
- The catalog and Home navigation now expose 21 tools.
- Rate of Change Dashboard now uses shared calculation and daily-data helpers.
- Global Macro Regime Dashboard and Liquidity Tracker now use the shared market-data loader and preserve missing observations.
- Currency snapshot commits now run application CI.
- All 20 Streamlit pages now use the shared ADFM footer component.
