# ADFM Analytics Platform architecture

## Application layers

| Layer | Responsibility |
|---|---|
| `Home.py` | PM command center, regime read, movers, confidence, and navigation |
| `pages/` | Focused analytical tools and page-specific presentation |
| `adfm_core/` | Shared market loading, source registry, integrity policy, scoring, signal history, catalog, and UI |
| `cte/` | Currency Tension Engine adapters, transformations, scoring, overlays, persistence, and commentary |
| `data/cache/` | Validated public-source Currency Tension Engine snapshot used by the deployed app |
| `data/last_good/` | Local, ignored continuity data such as the PM signal ledger |

## PM research and operating layer

Pages 21–26 are additive operating tools. They do not change calculations or
controls on the original 20 analytical pages.

- Signal attribution reconstructs causal weekly snapshots and reconciles each
  composite to its weighted inputs, prior values, and timestamps.
- Performance diagnostics evaluate subsequent 1-week, 1-month, and 3-month
  constructive-proxy returns, hit rate, drawdown, and turnover. Evidence weights
  are bounded research proposals and never change production weights automatically.
- Regime analogs compare rates, inflation, liquidity, dollar, credit, volatility,
  and breadth configurations. Results preserve the full forward-return
  distribution across SPY, TLT, UUP, DBC, and HYG.
- The decision journal stores entry-state evidence locally and outside Git.
  Reviews separate thesis, timing, sizing, execution, and luck.
- Threshold alerts persist active keys locally and surface only newly crossed
  conditions. External notification delivery remains disabled until an approved
  destination and credentials are configured.

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

## PM command-center scoring

The home page converts the registered cross-asset proxies into causal percentile
scores: each observation is ranked only against information available before that
date. Constructive signals point in the same direction, so the composite,
breadth, impulse, and group scores are directly comparable. The local
point-in-time ledger records dated snapshots atomically for change analysis.

The command center is an orientation layer, not a portfolio optimizer. Individual
tool pages remain the source for detailed diagnostics and trade-level judgment.

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
