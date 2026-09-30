# ADFM Analytics Platform architecture

## Application layers

| Layer | Responsibility |
|---|---|
| `Home.py` | Lightweight grouped tool directory and stable page navigation |
| `pages/` | Focused analytical tools and page-specific presentation |
| `adfm_core/` | Shared market loading, source registry, integrity policy, scoring, signal history, catalog, and UI |
| `cte/` | Currency Tension Engine adapters, transformations, scoring, overlays, persistence, and commentary |
| `data/cache/` | Validated public-source Currency Tension Engine snapshot used by the deployed app |
| `data/last_good/` | Local, ignored continuity data such as the PM signal ledger |

## Data-source policy

1. Use a primary source when an official API is available. The shared FRED
   adapter retrieves rates and liquidity series with per-series diagnostics.
2. Use market proxies through the shared market loader. It batches requests,
   retries failures, normalizes symbols, and reports stale or missing data.
3. Keep missing observations missing until a calculation explicitly documents
   an alignment rule. Do not forward-fill OHLCV used for gaps, ranges, volume,
   patterns, or turning points.
4. Show the source, latest observation, freshness, and limitation close to the
   resulting signal.
5. Filing-driven company analysis uses SEC EDGAR Company Facts and submissions.
   Stand-alone quarters may be derived only by subtracting two disclosed YTD
   observations with the same fiscal-period start. Current multiples keep the
   market-price observation separate from the filing denominator and expose the
   calculation formula and source concept.

## Analytical signals and point-in-time records

Shared scoring functions convert registered cross-asset proxies into causal
percentiles. Home intentionally performs no provider downloads or scoring.
Individual tools are responsible for the dates, eligibility, and interpretation
of their own signals. Historical observations, their publication dates, and
capture timestamps are distinct fields; revised descriptive research must not
be represented as a contemporaneous trading backtest.

Provider transport is centralized in `adfm_core.market_data`; legacy page
adapters preserve their response shape and explicitly choose daily completion
and calendar-alignment rules. Performance diagnostics contain aggregate timings,
cache delivery, request counts, and process peak memory, never account positions.

Portfolio stress analysis is pure scenario math on a dated, session-only CSV.
It uses supplied marks, signed quantities, FX conversion, contract multipliers,
DV01/convexity and option-model changes. Broker margin, assignment and executable
prices remain separate from scenario estimates. Uploaded data is never persisted
by the application or scheduled public-data workflows.

The presentation contract is a primary consolidated table, selected detail
charts and collapsed secondary analysis/methodology. Existing page routes remain
stable even when calculations are extracted into smaller core modules.

## Snapshot promotion

The scheduled Currency Tension Engine workflow downloads all files into a
temporary directory, validates required files and schemas, generates SHA-256
metadata, and only then promotes the snapshot to `data/cache/`. The resulting
commit runs the normal application CI.

## Repository boundary

The codebase is designed for internal deployment. No client, holding, position,
or credential data belongs in Git. Repository visibility and branch-protection
settings should be managed at the GitHub organization level, with `main`
requiring a reviewed pull request and passing CI.
