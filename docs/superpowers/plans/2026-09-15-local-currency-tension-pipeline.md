# Local Currency Tension Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Currency Tension Engine's dependency on the legacy external Currency Tension repository and build validated production snapshots entirely inside `ADFM-App-Folder`.

**Architecture:** Keep the existing `cte/` engine and adapters. Add the missing local ingestion runner, make the scheduled workflow execute ingestion → history → scoring → optional commentary → validation locally, source Japan headline CPI through OECD, and make the Streamlit page read freshness from the validator manifest rather than commentary metadata.

**Tech Stack:** Python 3.12, pandas, pyarrow, yfinance, requests, openpyxl, optional Anthropic SDK, GitHub Actions, Streamlit.

**Spec:** `docs/superpowers/specs/2026-09-15-local-currency-tension-pipeline-design.md`

## Global Constraints

- `ADFM-App-Folder` must be the only GitHub repository in the Currency Tension production data path.
- Preserve the existing CTE scoring mathematics and historical replay outputs.
- `FRED_API_KEY` is the only required scheduled-ingestion secret; Japan CPI comes from OECD.
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
- Produces: tests requiring local workflow commands, optional commentary validation, OECD Japan CPI, and manifest-based freshness parsing.

- [x] Add a validator test that builds all required parquet/text outputs except commentary and asserts validation succeeds.
- [x] Add a workflow contract test requiring local ingestion/scoring/validation while excluding cross-repository raw GitHub downloads.
- [x] Add a metadata test for `cte.snapshot_meta.snapshot_generated_at(cache_dir)` using `snapshot_manifest.json.validated_at_utc`.
- [x] Verify the tests fail against the pre-migration behavior for the expected reasons.

### Task 2: Add local ingestion and workflow-only dependencies

**Files:**
- Create: `scripts/backfill.py`
- Create: `requirements-cte.txt`
- Modify: `cte/adapters/oecd.py`
- Modify: `cte/adapters/macro.py`

**Interfaces:**
- Consumes: `cte.adapters.*`, `cte.store.merge_cache`, environment variable `FRED_API_KEY`.
- Produces: refreshed `macro_backbone`, `fx_spot`, `yields`, `reer`, and `tff` caches.

- [x] Port the existing ingestion runner into this repository without any GitHub-repository dependency.
- [x] Keep `--daily` as the incremental scheduled mode and full-history behavior as the default manual/cold-start mode.
- [x] Add a workflow-only requirements file that includes the app requirements plus `openpyxl` and `anthropic`.
- [x] Move Japan headline CPI onto the existing OECD national CPI adapter so the production job does not require e-Stat credentials.

### Task 3: Make validation and freshness independent of commentary

**Files:**
- Modify: `scripts/validate_currency_snapshot.py`
- Create: `cte/snapshot_meta.py`
- Modify: `pages/6_Currency_Tension_Engine.py`

**Interfaces:**
- Produces: `snapshot_manifest.json` with `validated_at_utc`; `snapshot_generated_at(cache_dir) -> Optional[pd.Timestamp]`.

- [x] Change commentary files from required snapshot files to optional validated artifacts when present.
- [x] Implement manifest timestamp parsing in `cte.snapshot_meta` with a safe fallback to snapshot-history dates.
- [x] Replace the page's commentary-based freshness helper with the new module.
- [x] Add a regression test that the page uses manifest freshness.

### Task 4: Replace the cross-repository sync workflow

**Files:**
- Modify: `.github/workflows/sync_currency_tension_snapshot.yml`

**Interfaces:**
- Consumes: `requirements-cte.txt`, `scripts.backfill`, `scripts.backfill_history`, `cte.scoring.engine`, `scripts.validate_currency_snapshot`.
- Produces: validated committed `data/cache` in the same repository.

- [x] Replace the raw GitHub download stage with local ingestion using `FRED_API_KEY`.
- [x] Run local history seeding and scoring.
- [x] Regenerate commentary only when `ANTHROPIC_API_KEY` exists; otherwise delete cached commentary artifacts before validation.
- [x] Validate `data/cache` locally and force-stage the curated snapshot files plus manifest.
- [x] Preserve rebase/retry push handling for concurrent scheduled jobs.

### Task 5: Full verification and merge

**Files:**
- No new production files.

- [ ] Run fresh compile and test verification on the final branch head.
- [ ] Confirm migration-specific tests pass and only known baseline failures remain in the full repository suite.
- [ ] Inspect the final PR diff and confirm the workflow contains no legacy raw-GitHub source dependency.
- [ ] Mark the pull request ready and merge only if the branch remains mergeable.