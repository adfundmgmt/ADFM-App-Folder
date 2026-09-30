# ADFM Tools Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Independent tasks use parallel-agent work with exclusive file ownership.

**Goal:** Improve source freshness, research validity, load performance and portfolio stress analysis across the existing Streamlit toolkit.

**Architecture:** Extend the current shared core and preserve compatible adapters for existing pages. Pure calculation modules own research and risk math; Streamlit pages retain consolidated tables and stable routes. Independent changes are integrated into one tested release.

**Tech Stack:** Python 3.12, pandas 3.0.1, Streamlit 1.58.0, yfinance 1.4.1, unittest, coverage, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-tools-data-risk-design.md`

## Global Constraints

- Existing Streamlit app and all 25 routes remain available.
- One main sortable table, selected detail chart, collapsed methodology; no new KPI cards, manual refresh controls, recommendation engine or AI commentary.
- No fabricated current observations, historical availability dates, option marks or broker margin.
- User uploads remain session-local and never enter git or public telemetry.
- Preserve raw volume, observation dates, adjustment provenance and missing endpoints.
- Existing CI coverage floor is 45%; all required checks must pass before publishing.

## Review Focus

- Multi-symbol partial failure: valid symbols render while unavailable symbols remain missing.
- Stale series and exchange closures: alignment never invents an observed endpoint.
- Sparse monthly observations and revised macro series: historical dates cannot predate availability.
- Signed futures/options and invalid inputs: combined scenario P&L is economically correct or explicitly unavailable.
- Concurrent signal writes and warm-page reruns: no lost original captures or unbounded cache growth.

## Task A: Shared data, freshness and performance

**Files:** `adfm_core/market_data.py`, `adfm_core/observability.py`, new performance/freshness helpers, and market-loading pages except baskets, Chart Terminal, bonds, sizing, seasonality and commodity renderer.
**Produces:** `download_market_data(tickers, **kwargs) -> pd.DataFrame`, supporting legacy Yahoo-shaped response layouts and explicit completed-only daily behavior. Other tasks import this wrapper in place of direct `yf.download`.
- [ ] Add failing tests for daily/date-range transport, raw and adjusted layouts, partial failure, bounded interior fills, fresh input requirements and instrumentation.
- [ ] Observe failures, implement the adapter and migrate owned loaders; preserve rendering and source checks.
- [ ] Run focused tests, report commands/results and remaining exact-source restrictions.

## Task B: Daily sovereign coverage and bond statistics

**Files:** `adfm_core/bond_event_study.py`, `adfm_core/bond_monitor.py`, new daily sovereign helper, `pages/2_Global_Macro_Regime.py`, bond tests.
- [ ] Add failing tests for fixed breakout levels, monthly availability, non-overlap, confidence intervals, adverse excursions, holdout partitions and daily/monthly separation.
- [ ] Implement official-source daily coverage and richer statistics; maintain Max default, consolidated sortable table and selected chart.
- [ ] Run focused tests and source reconciliation; document availability/source bases accurately.

## Task C: Publication-aware history, signal ledger and futures

**Files:** `adfm_core/signal_ledger.py`, new point-in-time and futures helpers, `adfm_core/commodity_top_exhaustion_page.py`, `pages/24_Monthly_Seasonality_Explorer.py`, scheduled capture script/workflow and associated tests.
- [ ] Add failing tests for immutable/idempotent captures, write locking, known release-date alignment, unknown dates, vintage exclusion and futures contract metadata/multipliers.
- [ ] Implement durable capture and publication-aware histories, migrate owned loaders to Task A wrapper, surface futures roll context and collapse auxiliary methodology.
- [ ] Run focused tests; retain explicit limitations where historical publication records cannot be verified.

## Task D: Portfolio stress table

**Files:** new `adfm_core/portfolio_stress.py`, new focused UI helper, `pages/22_Position_Sizing_Lab.py`, tests.
- [ ] Add failing tests for signed shares/FX/futures, DV01 plus convexity, option repricing, missing marks, invalid dates/rows and combined NAV impacts.
- [ ] Implement dated session-only CSV input, documented schema/template and one scenario table; retain existing individual-position analytical capability in collapsed detail.
- [ ] Run tests with verified hand calculations and a deterministic Streamlit smoke fixture.

## Task E: Baskets/Chart Terminal maintenance and layout

**Files:** `pages/1_ADFM_Public_Equities_Baskets.py`, `pages/10_ADFM_Chart_Terminal.py`, focused extracted helpers, associated tests.
- [ ] Pin existing calculation/layout behavior with fixtures before extracting transport/calculation code.
- [ ] Consume Task A wrapper, preserve table sorting and dynamic MACD, expose observed date and prior-close change/signal age when meaningful, calculate selected chart details lazily and collapse methodology.
- [ ] Run existing basket/pattern/page tests and deterministic smoke checks.

## Task F: Integration and release

**Files:** performance benchmark/smoke scripts, docs, changelog, central layout helpers as needed.
- [ ] Review all task reports and diffs; resolve interface mismatches and add meaningful cross-task checks.
- [ ] Run full coverage, compilation, dependency check, required lint, deterministic page smoke checks and measured provider/page benchmarks.
- [ ] Independent review, fix findings, verify again; publish through a branch and PR with CI, merging only passing changes.

## Execution record

The user explicitly requested implementation of the entire recommendation set. Proceed without another scope-approval checkpoint. A clean isolated clone is on `feat/tools-data-risk-upgrade-20260929`; baseline commit is `b70765d40a95014d80d6f3219f109375fb0eed84`. Task reports and exact test evidence are retained in `docs/superpowers/reports/2026-09-29/`.
