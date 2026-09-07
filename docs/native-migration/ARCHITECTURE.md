# ADFM native application migration

## Decision and current boundary

The existing **adfundmgmt.com website becomes the application**. Its actual source already uses React, TypeScript and the Next-compatible Vinext runtime on a Cloudflare Worker. Extend that project under `app/tools/`; do not introduce a second public frontend. Run CPython, FastAPI and the reusable `adfm_engine` package on Render. The website Worker forwards `/tools/api/*` to the protected API over HTTPS. Browsers remain on adfundmgmt.com.

The first five page implementations are complete and have fixture parity tests. The remaining twenty require sequential migration. The current website deployment has NOT been replaced with this incomplete branch. A protected staging Python service exists at `https://adfm-python-api.onrender.com`; the runtime installs only native requirements and runs from an allowlisted package directory. This is not completion of the whole platform.

## Source of truth and inventory

Baseline GitHub revision: `ac2bd39d8371e959c778437519d390e1891c1d08`.

[INVENTORY.md](INVENTORY.md) enumerates all 25 pages and transitive Python dependencies and providers. Home is a separate entrypoint. [source-audit.json](source-audit.json) records 94 original Python files, hashes, imports, function boundaries, all statically discovered controls/defaults, caches, state accesses, callbacks, URLs and filesystem operations. The JSON is reproducible with `python scripts/audit_native_migration.py`. It is a static audit, not proof that every conditional runtime branch works. Frozen source fixtures and each page's parity tests establish the narrower runtime claims.

Original Home is a directory. Its replacement includes a real cross-asset pulse using the same Python ROC engine for SPX, Nasdaq, Treasuries, gold, USD/JPY and crude, each with its own observation date, missing-data state and direct native navigation. Add major macro, liquidity, credit, stress and positioning signals as their engines pass parity. Never infer a healthy signal from a failed data pull.

## Runtime design

| Component | Runs where | Responsibility |
|---|---|---|
| React/TypeScript frontend | Existing website Worker and static assets | Sidebar, Home, controls, tables, accessible navigation, Plotly interaction, downloads |
| Same-origin gateway | Existing website Worker | Route/method allowlist, identity/access check, bounded POST body, private service token, response/error handling |
| FastAPI | Single Render Python web service | Typed request validation, authentication, API serialization, resource limits, health and request identifiers |
| `adfm_engine.data` | Python | Provider adapters, date normalization, diagnostics, explicit source-specific cache policies |
| `adfm_engine.analytics` | Python | Original formulas and signal classifications; injectable data for replay |
| `adfm_engine.charts` | Python | Original Plotly figure transformations, hover templates, axis ranges and precision |
| Page services | Python | Orchestration and versioned results; callable without HTTP by reports/jobs/AI |
| Scheduled ingestion, later | Existing GitHub Actions initially; Render cron if needed | CTE snapshots and SEC bulk refresh, atomic publish of validated artifacts |
| Persistent storage, later | Object storage for Parquet/CSV snapshots; optional small database | Last-good validated datasets and provenance; saved portfolios/watchlists/session simulations only when required |

No Celery, Kubernetes, Redis, or database is needed for the five initial stateless page slices. Use one API process with bounded, per-key coalescing TTL caches and copy-on-read values. Preserve distinct loader semantics; cache keys include relevant controls and date bounds. Do not put user-specific inputs or private holdings in a shared unscoped cache. Introduce a shared cache only when multiple workers need one. Explicitly surface stale/failed data, including last-good timestamps when that workflow is migrated.

CPU-intensive analog searches, large universes and model jobs need resource profiling as their pages migrate. Prefer bounded background jobs returning job IDs for genuinely long work. Keep algorithms in the same engine; do not introduce a second scheduled-job implementation of the math.

## Repository structure and ownership

The existing website and GitHub analytics project are separate established repositories. Keep each source in its actual repository and connect them via the API contract. Avoid copying a second website into the analytics repository.

```text
ADFM-App-Folder/
  adfm_engine/
    analytics/       # pure calculations, universes and signal rules
    data/            # provider adapters and provenance
    charts/          # Python Plotly transformations
    *_service.py     # reusable page result orchestration
    cache.py
    serialization.py
    palette.py
  adfm_api/main.py
  native_tests/
    fixtures/        # frozen original sources, test-only
    test_*_parity.py # original-page replay and boundary cases
  requirements-native.txt
  requirements-native-dev.txt
  scripts/package_native_runtime.py
  scripts/audit_native_migration.py
  deploy/Dockerfile
  deploy/compose.yaml
  render.yaml
  docs/native-migration/
  .github/workflows/native-analytics.yml
  Home.py, pages/, adfm_core/ # legacy reference during migration; excluded from runtime

adfm-website/
  app/tools/
    layout.tsx
    page.tsx
    catalog.ts
    analytics.css
    _components/
    <tool-slug>/page.tsx
  worker/analytics-gateway.ts
  worker/index.ts
  public/tools-assets/
  tests/analytics-gateway.test.mjs
```

The runtime packager copies only `adfm_engine` and `adfm_api`; the Docker alternative copies the same. It rejects imports of `streamlit`, `adfm_core`, `pages`, or `Home`. Original UI source exists only for migration review/testing, never as an executing shim. After all pages pass, remove obsolete UI files and Streamlit CI/dependencies from the active tree, keeping their Git history. The original moved pure math modules re-export the single engine implementation to avoid maintaining two formulas while the reference application remains available.

## Migration sequence and gates

The order favors small representative vertical slices, then macro and provider foundations, then wider/stateful models. It is a sequence, not parallel page rebuilding.

| Order | Original page | Status |
|---:|---|---|
| 1 | 12. Rate of Change Regime Explorer | Native implementation; 50 initial math/chart/API/cache checks |
| 2 | 8. Equity Leadership & Rotation | Native implementation; 26 parity checks |
| 3 | 13. Relative Volatility Lab | Native implementation; 17 parity checks |
| 4 | 11. Cross-Asset Ratio Chartbook | Native implementation; 22 parity checks |
| 5 | 2. Global Macro Regime | Native implementation; 25 parity/provider checks |
| 6 | 4. Yield Curve Rates Regime Monitor | Audited; next |
| 7 | 3. Liquidity Conditions Monitor | Audited |
| 8 | 5. Credit Conditions Monitor | Audited |
| 9 | 18. CFTC Positioning Monitor | Audited |
| 10 | 16. Options Positioning Compass | Audited |
| 11 | 9. ADFM Underwriter | Audited |
| 12 | 17. SEC 13F Exposure Browser | Audited |
| 13 | 20. Catalyst Calendar | Audited |
| 14 | 19. Market Stress Composite | Audited |
| 15 | 21. Hedge Timer | Audited |
| 16 | 22. Position Sizing Lab | Audited |
| 17 | 7. Sector Breadth and Rotation | Audited |
| 18 | 1. ADFM Public Equities Baskets | Audited |
| 19 | 15. Volume Based Sentiment Indicator | Audited |
| 20 | 14. ETF Flow Pressure Proxy | Audited |
| 21 | 23. Market Memory Explorer | Audited |
| 22 | 24. Monthly Seasonality Explorer | Audited |
| 23 | 25. Commodity Event Study | Audited |
| 24 | 10. ADFM Chart Terminal | Audited |
| 25 | 6. Currency Tension Engine | Audited |

ROC is first because its bounded inputs exercise the entire migration contract: provider and trading-session handling, history controls, reusable Python formulas, multi-panel Plotly charts, hover precision, API transport, missing history and downloadable rows. It exposes calculation drift quickly without starting with the largest stateful page.

For every next page: capture source and imported execution paths; extract provider/calculation/signal/chart functions; implement API and frontend controls; replay the original page on identical fixtures; compare all calculations, tables, chart traces/layouts/hover precision/downloads; verify missing/stale data and state transitions; run native build and type checks. Only then begin another page. Fixture tests do not replace final live-source and user-workflow acceptance.

## Hidden behavior and concrete migration risks

- `adfm_core/__init__.py` patches Streamlit globally using call-stack inspection. It modifies theme behavior, Position Sizing button appearance and suppresses deprecated seasonality 3D terrain. Preserve user-visible behavior deliberately, not this mechanism.
- Liquidity executes part of `_liquidity_tracker_base.py` using `exec`. Extract the actual bound functions and constants before replacing the outer page.
- Calendar's four-line page imports an official wrapper that replaces `exact._dated_calendar` in another module. Its explicit official calendar currently covers September 2026 to January 2027. Preserve dates with provenance; outside coverage, show missing official coverage instead of inventing releases.
- Commodity Event Study actually imports `commodity_top_exhaustion_page.py`. The separate `commodity_event_study_page.py` is not the active page specification.
- SEC 13F's thin page delegates through the corrected resolver and bulk Parquet loader. Do not migrate only the visible 18-line entrypoint.
- Position Sizing uses a timed Streamlit fragment, persistent RNG seed, bankroll arrays and continuous-block simulation state. Replace it with explicit state transitions and reproducible seeds, not unrelated rerandomized HTTP calls.
- Chart Terminal has session versions/defaults, watchlist mutations and queued control callbacks. Underwriter executes its form on submit. Seasonality resets linked controls in callbacks. These are functional behaviors, not decoration.
- CTE depends on a persisted upstream `smileys21/currency_tension_tool-main` snapshot refreshed by GitHub Actions at 07:15 UTC weekdays, plus pillar/overlay/history transforms. The schedule and snapshot schema belong in the migration. Optional commentary needs `ANTHROPIC_API_KEY`; eStat and FRED adapters need their respective credentials.
- Shared daily market data uses unadjusted raw bars and a 16:15 New York completed-session cutoff without forward fill. Leadership and Ratio Chartbook instead request adjusted prices and forward fill. Preserve each page's semantics; globally unifying them would change results.
- Global Macro's Fed net liquidity subtracts TGA in millions and RRP in billions multiplied by 1,000. Voting thresholds and calendar-day lookbacks are preserved exactly.
- Relative Volatility's original methodology text refers to percentile/5-day change information that is not calculated/displayed by the active implementation. Text is preserved; no new investment metrics were invented during migration.
- Original Streamlit dataframe controls become native sortable/searchable tables and CSV downloads. Python Plotly figure JSON is displayed by the matching local Plotly.js distribution; no frontend chart library reinterprets analytics values.
- Primary FRED retrieval now reads the same FRED CSV observations directly with explicit timeouts and independent bounded requests, avoiding a broad pandas-datareader dependency. Missing observations remain missing, series failures stay isolated, and status order is preserved. Transport timing/cache mechanics differ; investment calculations do not.

## Validation evidence and limits

Each frozen original page is executed only inside test code with its data-provider boundary replaced. Native code uses no Streamlit adapter. Tests compare complete Plotly JSON objects, calculations, rankings, table records and exact downloadable CSV where present. Fixtures cover gaps, unavailable symbols, short history, multi-decade lookbacks, controls and fixed-universe rank behavior.

On September 7, 2026 the protected Render API returned HTTP 200 for `/health/live`, HTTP 401 for an unauthenticated analytics request, and an authenticated SPY 3-year ROC response with 754 observations, September 4 data, no warnings and nine chart traces. The earlier local Yahoo probe was rate limited, so live-provider reliability still needs monitoring from the actual host. No browser workflow parity or final 25-page acceptance has been claimed.

Before the website cutover: finish all pages; run live provider checks and stateful workflow acceptance; verify all original controls/downloads; confirm private identity access on the custom domain; profile large pages; deploy the complete website version; verify native navigation; retire Streamlit hosting and old scheduled entrypoints. Do not use partially completed availability flags as evidence of a full platform migration.
