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

## Remaining release gates
Whole-suite test, production build, live provider smoke tests, and matched website/API deployment must be recorded below. Do not claim these are complete based only on per-page checks.

## Infrastructure
Existing Sites website serves /tools and the same-origin /tools/api gateway. Existing Render Python service runs one Uvicorn process. The 13F background queue uses a single worker thread and SQLite in ADFM_DATA_DIR. Production should mount a persistent disk there to preserve jobs and prepared SEC archives across deployments. Free Render remains a staging tier until an always-on instance and persistent storage are configured. No Redis, separate worker service or Postgres is required for this release.

## Deferred
Public Equities Baskets; Currency Tension Engine; Sector Breadth and Rotation; Chart Terminal; ETF Flow Pressure; Volume Sentiment; Position Sizing; Market Memory; Monthly Seasonality; Commodity Event Study.
