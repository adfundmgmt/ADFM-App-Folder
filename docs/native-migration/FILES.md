# First migration implementation files

The complete code is committed in the two actual project repositories. This manifest identifies exact application-shell files, with no illustrative placeholders.

## Existing website: create/change

- `app/tools/layout.tsx`
- `app/tools/page.tsx`
- `app/tools/catalog.ts`
- `app/tools/analytics.css`
- `worker/analytics-gateway.ts`
- `worker/index.ts`
- `tests/analytics-gateway.test.mjs`
- `app/tools/_components/data-table.tsx`
- `app/tools/_components/leadership.tsx`
- `app/tools/_components/macro.tsx`
- `app/tools/_components/multi-choice.tsx`
- `app/tools/_components/overview.tsx`
- `app/tools/_components/plotly-chart.tsx`
- `app/tools/_components/ratios.tsx`
- `app/tools/_components/roc-explorer.tsx`
- `app/tools/_components/shell.tsx`
- `app/tools/_components/use-analysis.ts`
- `app/tools/_components/volatility.tsx`
- `app/tools/cross-asset-ratio-chartbook/page.tsx`
- `app/tools/equity-leadership-and-rotation/page.tsx`
- `app/tools/global-macro-regime/page.tsx`
- `app/tools/rate-of-change-regime-explorer/page.tsx`
- `app/tools/relative-volatility-lab/page.tsx`
- `public/tools-assets/adfm-logo.png`
- `public/tools-assets/plotly-3.6.0.min.js`

## Python first slice: Rate of Change

- `adfm_api/main.py`
- `adfm_engine/services.py`
- `adfm_engine/analytics/rate_of_change.py`
- `adfm_engine/data/market.py`
- `adfm_engine/data/integrity.py`
- `adfm_engine/charts/rate_of_change.py`
- `adfm_engine/cache.py`
- `adfm_engine/serialization.py`
- `adfm_engine/palette.py`
- `requirements-native.txt`
- `requirements-native-dev.txt`
- `deploy/Dockerfile`
- `deploy/compose.yaml`
- `render.yaml`
- `scripts/package_native_runtime.py`
- `native_tests/test_roc_parity.py`
- `native_tests/reference_roc.py`
- `native_tests/fixtures/roc_baseline.json`
- `.github/workflows/native-analytics.yml`

Original `adfm_core/rate_of_change.py` and `adfm_core/palette.py` re-export the extracted engine during migration. They are not included in the new runtime. Subsequent page-specific files follow the same data/analytics/charts/service separation.
