# ADFM Web Platform Migration

This directory contains the non-Streamlit replacement surface for the ADFM analytics platform.

- `web/`: Next.js / React frontend.
- `api/`: FastAPI backend that calls the existing Python analytics engine.

The legacy Streamlit application remains untouched during migration so each native page can be compared against the established implementation before it is retired.

## Current migration status

The first native tool is `Rate of Change Regime Explorer`.

Its browser route is `/tools/rate-of-change`, and its API route is `/api/tools/rate-of-change`.

The API reuses:

- `adfm_core.market_data.fetch_daily_ohlcv`
- `adfm_core.data_integrity.build_data_quality_report`
- `adfm_core.rate_of_change.compute_features`

The React frontend owns the controls and Plotly rendering. It does not iframe, redirect to, or open Streamlit.

## Run locally

From the repository root, start the API:

```bash
python -m pip install -r apps/api/requirements.txt
uvicorn apps.api.app.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd apps/web
npm install
NEXT_PUBLIC_ADFM_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

## Environment

API:

```text
ADFM_CORS_ORIGINS=http://localhost:3000
```

Frontend:

```text
NEXT_PUBLIC_ADFM_API_URL=http://localhost:8000
```

For production these should be set to the ADFM web and API origins, or the services can be placed behind one reverse proxy and served under the same domain.

## Migration rule

Do not add a tool to the live navigation until the native implementation reproduces the established data path, calculations, controls, diagnostics, and chart behavior without relying on Streamlit for presentation.
