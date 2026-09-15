# Local Currency Tension Pipeline Design

## Goal

Make `adfundmgmt/ADFM-App-Folder` the sole code and data source for the Currency Tension Engine so the production Streamlit page no longer reads from, downloads from, or depends on the legacy external Currency Tension repository.

## Architecture

The existing `cte/` package remains the scoring implementation. A local `scripts/backfill.py` becomes the ingestion entrypoint, using the adapters already present in this repository to refresh macro, FX, sovereign-yield, REER and CFTC inputs into `data/cache`. The scheduled Currency Tension workflow then runs local ingestion, seeds history when needed, rebuilds the scoring snapshot, optionally regenerates commentary, validates the finished snapshot, and commits the validated cache back to this repository.

No network call may target another GitHub repository. External data providers remain the existing primary sources used by the adapters: FRED, OECD, BIS, Eurostat, UK ONS, CFTC, Yahoo Finance and national debt-management offices. Japan headline CPI is sourced from the same OECD national CPI dataflow used for the other non-FRED legs, eliminating the prior e-Stat application credential from production.

## Data Flow

1. GitHub Actions checks out `ADFM-App-Folder` with full history.
2. Install the workflow-only CTE ingestion dependencies.
3. Run `python -m scripts.backfill --daily` using `FRED_API_KEY`.
4. Run `python -m scripts.backfill_history --if-missing` to preserve historical replay support.
5. Run `python -m cte.scoring.engine` to build current maps, pillars, overlays, carry grids, warnings and history outputs.
6. If `ANTHROPIC_API_KEY` is configured, regenerate cached Daily Read commentary. If it is absent, remove cached commentary files so stale AI text is never presented as current.
7. Run `python scripts/validate_currency_snapshot.py data/cache` to validate schemas and write `data/cache/snapshot_manifest.json`.
8. Commit the controlled `data/cache` outputs to `main`, rebasing and retrying if another scheduled data job raced with the push.

## Freshness Contract

`cte.snapshot_meta.snapshot_generated_at()` treats `snapshot_manifest.json.validated_at_utc` as the production snapshot timestamp, with snapshot history as a fallback. Commentary metadata is optional and is not the source-of-truth freshness record.

## Failure Behavior

`FRED_API_KEY` is required for a complete production refresh. The workflow fails rather than publishing a newly validated snapshot when required ingestion cannot produce the complete output set. Existing last-good committed cache remains available to Streamlit if the scheduled job fails.

Commentary is optional. Its absence must not fail snapshot validation. If commentary is not regenerated, cached commentary from an older snapshot must not remain visible.

## Validation

Automated tests must verify that the Currency Tension workflow contains no legacy cross-repository raw GitHub download, invokes the local ingestion/scoring/validation pipeline, accepts a valid snapshot without commentary, sources Japan headline CPI through OECD without an e-Stat credential, and resolves snapshot freshness from the manifest timestamp. Existing CTE mathematical tests and repository compilation remain unchanged.