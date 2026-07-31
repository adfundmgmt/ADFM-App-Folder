# ADFM development and release guide

## Page standards

Each page should retain raw provider observations until after data-quality checks, disclose its source and as-of date, and avoid forward-filling OHLCV used for pattern recognition, range, volume, or gap calculations. Forward fill is permitted only when an explicitly documented ratio or alignment calculation requires it.

Use `adfm_core` for common market-data, integrity, UI, export, and catalog functionality. New pages belong in `pages/` and must be added to `adfm_core.catalog.TOOL_CATALOG`; `Home.py` and the README derive their tool lists from that catalog.

## Before opening a pull request

```bash
coverage run --source=adfm_core,cte -m unittest discover -s tests -p "test_*.py" -q
coverage report --show-missing --fail-under=45
python -m ruff check --select E,F,I,B --ignore E501 Home.py adfm_core cte scripts tests
python -m ruff check --select E9,F63,F7,F82 pages adfm_momentum_scanner.py adfm_sector_rotation_config.py
python -m compileall -q Home.py pages adfm_core cte scripts tests
```

Review the data and calculation checklist in the pull-request template. A change that modifies signals, lookback windows, benchmark alignment, or adjusted-price treatment needs a fixture-based regression test.

## Releases

1. Update `CHANGELOG.md` with user-facing changes and calculation/data-policy changes.
2. Confirm CI and the affected Streamlit page pass.
3. Merge through a reviewed pull request.
4. Tag the release from `main` when a deployment process is established.

## New shared signals and sources

Register cross-asset symbols and primary-source macro series in
`adfm_core.data_registry`. PM-level signals must be causal: calculations for a
date may use observations on or before that date only. Add deterministic tests
for sign conventions, sparse histories, and missing data.

Do not add page-local Yahoo Finance loaders. Extend `adfm_core.market_data`
instead so retries, completed-session handling, and diagnostics remain
consistent. For official FRED series, use `adfm_core.primary_data`.

## Currency snapshot changes

Before promoting a Currency Tension Engine snapshot, run:

```bash
python scripts/validate_currency_snapshot.py data/cache
```

This validates file presence, nonempty Parquet data, required columns, currency
codes, and JSON readability, then generates `snapshot_manifest.json`.
