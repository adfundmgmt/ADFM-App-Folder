# ADFM development and release guide

## Page standards

Each page should retain raw provider observations until after data-quality checks, disclose its source and as-of date, and avoid forward-filling OHLCV used for pattern recognition, range, volume, or gap calculations. Forward fill is permitted only when an explicitly documented ratio or alignment calculation requires it.

Use `adfm_core` for common market-data, integrity, UI, export, and catalog functionality. New pages belong in `pages/` and must be added to `adfm_core.catalog.TOOL_CATALOG`; `Home.py` and the README derive their tool lists from that catalog.

## Before opening a pull request

```bash
python -m unittest discover -s tests -p "test_*.py" -q
python -m ruff check --select E,F,I,B --ignore E501 adfm_core tests
python -m compileall -q Home.py pages adfm_core tests
```

Review the data and calculation checklist in the pull-request template. A change that modifies signals, lookback windows, benchmark alignment, or adjusted-price treatment needs a fixture-based regression test.

## Releases

1. Update `CHANGELOG.md` with user-facing changes and calculation/data-policy changes.
2. Confirm CI and the affected Streamlit page pass.
3. Merge through a reviewed pull request.
4. Tag the release from `main` when a deployment process is established.
