# ADFM native analytics — 15-page release

User revised scope on September 7, 2026: finalize the first 15 analytics pages plus Home; defer the other ten. Do not resume the full 25-page migration without a new request.

## Included
1. Rate of Change Regime Explorer
2. Equity Leadership & Rotation
3. Relative Volatility Lab
4. Cross-Asset Ratio Chartbook
5. Global Macro Regime
6. Yield Curve Rates Regime Monitor
7. Liquidity Conditions Monitor
8. Credit Conditions Monitor
9. CFTC Positioning Monitor
10. Options Positioning Compass
11. ADFM Underwriter
12. SEC 13F Exposure Browser
13. Catalyst Calendar
14. Market Stress Composite
15. Hedge Timer

Home contains cross-asset momentum observations with individual observation dates and links to all 15 tools. Other catalog entries are hidden. No production path imports, embeds or launches Streamlit.

## Validation and differences
The original reference revision is ac2bd39d8371e959c778437519d390e1891c1d08. Native tests execute original calculations and, for several pages, whole original pages using a test-only capture. These captures are never packaged into production.

Underwriter preserves all original model and chart function bodies, with six original fundamentals tests and eight integration/equivalence checks. Its empty source-audit crash is fixed. The 13F browser preserves the corrected effective-holdings denominator and amendment semantics; streaming computes all portfolio denominators without loading the whole release. Eighteen focused tests pass. Duplicate manager names drill down by CIK rather than ambiguous names. The original pre-January-2023 report-period value convention remains unchanged.

Calendar: 31 tests compare original cards, complete figures and all displayed tables across six horizons and five VIX regimes. Confirmed macro dates cover September–December 2026 and January 27, 2027 FOMC only; missing future dates are explicitly reported rather than fabricated. An empty out-of-range custom calendar no longer crashes.

Market Stress: 31 tests compare full original figures, scores, global moves and data-health tables across all lookbacks, overlays, speeds, normalization extremes and missing inputs. Auto overlay selection remains retrospective, as in the original.

Hedge Timer: six tests verify original scores, calibration, gates, forward statistics, episode tables and chart vertices across all five lookbacks and missing optional inputs. The old static Matplotlib image becomes an interactive Plotly chart. Trading-session spacing, prices, MAs, gradient score segments, thresholds and onset dates are preserved; fonts and drawing primitives necessarily differ. The episode image becomes a searchable, sortable, downloadable native table. In-sample calibration is explicitly labelled.

## Release verification
- Full native suite: 420 passed, 2 warnings (483.85 seconds). Subsequent capacity protection: 4 passed. No additional investment-model changes followed the full suite.
- Frontend TypeScript check passed. Production Worker build passed with all 15 analytics routes plus Home. Gateway/render checks: 5 passed.
- Native runtime packaging and import exclusion passed; dependency check passed.
- Website source 0a9708737ae0614bc2fed784cffaa60d94408306 saved as website version 19, archive-backed. Not published: do not cut over the public website until the API host is stable.
- Fifteen-page API source 31711bcf254e46b92c39255644d56c1112e2380d deployed live on Render; then live checks exposed the host-capacity blocker.
- Live responses passed for ROC, Home overview, Leadership (25 charts), Relative Volatility, Macro, Ratios (38 charts), Yields and Credit. These are smoke checks, not claims that every optional provider observation is complete.
- During simultaneous data loading and an SEC bulk preparation, the 512 MB free instance exceeded memory and restarted. Render's email supplied by the user confirms the memory-limit event. Liquidity, CFTC and subsequent remaining requests returned gateway failures during this event; their live acceptance remains incomplete.
- Stop broad live testing until the instance is upgraded. The capacity protection now refuses SEC bulk work on cgroups below 1 GB before allocation, so restart recovery cannot repeatedly launch the same oversized job on this tier.
- Required next action: existing Render service to Standard 2 GB; 10 GB disk at /var/data; ADFM_DATA_DIR=/var/data; health check /health/live. Installed connector cannot change plans/create disks. After the user applies these host settings, retry only the failed live checks and SEC job, then publish the existing saved website version 19. Do not rebuild or remigrate the 15 pages.


## Infrastructure
Existing Sites website serves /tools and the same-origin /tools/api gateway. Existing Render Python service runs one Uvicorn process. The 13F background queue uses a single worker thread and SQLite in ADFM_DATA_DIR. Production should mount a persistent disk there to preserve jobs and prepared SEC archives across deployments. Free Render remains a staging tier until an always-on instance and persistent storage are configured. No Redis, separate worker service or Postgres is required for this release.

## Deferred
Public Equities Baskets; Currency Tension Engine; Sector Breadth and Rotation; Chart Terminal; ETF Flow Pressure; Volume Sentiment; Position Sizing; Market Memory; Monthly Seasonality; Commodity Event Study.
