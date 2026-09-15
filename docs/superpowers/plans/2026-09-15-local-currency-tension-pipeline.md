# Local Currency Tension Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Currency Tension Engine's dependency on `smileys21/currency_tension_tool-main` and build validated production snapshots entirely inside `ADFM-App-Folder`.

**Architecture:** Keep the existing `cte/` engine and adapters. Add the missing local ingestion runner, make the scheduled workflow execute ingestion → history → scoring → optional commentary → validation locally, and make the Streamlit page read freshness from the validator manifest rather than commentary metadata.

**Tech Stack:** Python 3.12, pandas, pyarrow, yfinance, requests, openpyxl, optional Anthropic SDK, GitHub Actions, Streamlit.

**Spec:** `docs/superpowers/specs/2026-09-15-local-currency-tension-pipeline-design.md`

## Global Constraints

- `ADFM-App-Folder` must be the only GitHub repository in the Currency Tension production data path.
- Preserve the existing CTE scoring mathematics and historical replay outputs.
- `FRED_API_KEY` and `ESTAT_APP_ID` are required for scheduled production ingestion.
- Commentary is optional and must never control data freshness.
- A failed refresh must leave the last-good committed cache available.

---

### Task 1: Lock the migration contract with tests

**Files:**
- Modify: `tests/test_snapshot_validation.py`
- Create: `tests/test_currency_tension_workflow.py`
- Create: `tests/test_currency_snapshot_meta.py`

**Interfaces:**
- Consumes: existing validator and workflow file.
- Produces: tests requiring local workflow commands, optional commentary validation, and manifest-based freshness parsing.

- [ ] Add a validator test that builds all required parquet/text outputs except commentary and asserts validation succeeds.
- [ ] Add a workflow contract test that asserts `.github/workflows/sync_currency_tension_snapshot.yml` contains `python -m scripts.backfill --daily`, `python -m cte.scoring.engine`, and local validation, while excluding `smileys21` and `raw.githubusercontent.com`.
- [ ] Add a metadata test for `cte.snapshot_meta.snapshot_generated_at(cache_dir)` using `snapshot_manifest.json.validated_at_utc`.
- [ ] Run the three targeted tests and verify they fail for the expected missing behavior.

### Task 2: Add local ingestion and workflow-only dependencies

**Files:**
- Create: `scripts/backfill.py`
- Create: `requirements-cte.txt`

**Interfaces:**
- Consumes: `cte.adapters.*`, `cte.store.merge_cache`, environment variables `FRED_API_KEY` and `ESTAT_APP_ID`.
- Produces: refreshed `macro_backbone`, `fx_spot`, `yields`, `reer`, and `tff` caches.

- [ ] Port the existing ingestion runner into this repository without any GitHub-repository dependency.
- [ ] Keep `--daily` as the incremental scheduled mode and full-history behavior as the default manual/cold-start mode.
- [ ] Add a workflow-only requirements file that includes the app requirements plus `openpyxl` and `anthropic`.
- [ ] Compile `scripts/backfill.py` and run its import-level tests.

### Task 3: Make validation and freshness independent of commentary

**Files:**
- Modify: `scripts/validate_currency_snapshot.py`
- Create: `cte/snapshot_meta.py`
- Modify: `pages/6_Currency_Tension_Engine.py`

**Interfaces:**
- Produces: `snapshot_manifest.json` with `validated_at_utc`; `snapshot_generated_at(cache_dir) -> Optional[pd.Timestamp]`.

- [ ] Change commentary files from required snapshot files to optional validated artifacts when present.
- [ ] Implement manifest timestamp parsing in `cte.snapshot_meta` with a safe fallback to snapshot-history dates.
- [ ] Replace the page's commentary-based freshness helper with the new module.
- [ ] Run targeted validator/metadata tests and verify they pass.

### Task 4: Replace the cross-repository sync workflow

**Files:**
- Modify: `.github/workflows/sync_currency_tension_snapshot.yml`

**Interfaces:**
- Consumes: `requirements-cte.txt`, `scripts.backfill`, `scripts.backfill_history`, `cte.scoring.engine`, `scripts.validate_currency_snapshot`.
- Produces: validated committed `data/cache` in the same repository.

- [ ] Replace the raw GitHub download stage with local ingestion using `FRED_API_KEY` and `ESTAT_APP_ID`.
- [ ] Run local history seeding and scoring.
- [ ] Regenerate commentary only when `ANTHROPIC_API_KEY` exists; otherwise delete cached commentary artifacts before validation.
- [ ] Validate `data/cache` locally and force-stage the curated snapshot files plus manifest.
- [ ] Preserve rebase/retry push handling for concurrent scheduled jobs.
- [ ] Run the workflow contract test and repository search for `smileys21`.

### Task 5: Full verification and merge

**Files:**
- No new production files.

- [ ] Run compile checks for `pages`, `cte`, `scripts`, and tests.
- [ ] Run the targeted Currency Tension tests plus existing `test_cte_math.py`.
- [ ] Run the full repository test suite and distinguish baseline failures from migration regressions.
- [ ] Open a pull request, inspect changed files, and merge only if the branch is mergeable and no new regression is found.