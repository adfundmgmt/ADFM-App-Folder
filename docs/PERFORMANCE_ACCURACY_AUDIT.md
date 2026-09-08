# Performance and accuracy audit — 2026-09-08

Reviewed against `ac2bd39` on `main`, using the canonical checkout. The live Home,
Rate of Change, and Public Equities Baskets pages were observed rendering in the
browser. This is a code and targeted live-page audit, not certification of every
source value or a production load test.

## Corrections

| Area | Verified problem | Result |
|---|---|---|
| Daily dates | Duplicate mask was calculated before sorting but applied after sorting; timezone conversion could shift foreign daily labels to the preceding date. | Keep the final provider row for each exchange date, then sort; normalize daily labels without changing their local date. |
| Completed sessions | An explicitly supplied UTC clock was compared directly with the New York cutoff. | Convert aware clocks to New York before applying the existing 16:15 policy. |
| Adjustments | The adjusted-close factor changed share volume, including dividend effects; entirely missing adjusted prices silently became raw prices. | Preserve provider volume and keep missing adjusted observations unavailable. |
| Returns | Removing missing rows before selecting endpoints compressed the return horizon. | Use the requested observation endpoints; missing, infinite, or zero-denominator inputs return unavailable. |
| Integrity | Infinite prices and negative volume could pass validation. | Exclude affected series with explicit reasons. |
| Scoring | Percentiles constructed a Series for each rolling window; composites created row objects repeatedly. | Use pandas rolling ranks and vectorized weighted sums, with reference-equivalence checks. |
| Downloads | All-null basket ticker columns counted as successful and skipped recovery. | Drop all-null response columns before identifying missing tickers. |
| Backup data | Filling holes in an existing ticker from disk could still be labeled wholly live. | Report mixed live/cache provenance whenever a cached observation is used. |
| Resources | Shared and basket result caches had no entry bound. | Limit shared entries to 128 and basket panels to 16; use explicit 10-second Yahoo request timeouts. These are request settings, not total-page time limits. |
| Tests | Four function-style basket tests were skipped by unittest discovery. | Include them through unittest's discovery hook. |

## Measurement

Validation: 174 tests pass; shared/engine coverage is 47%, above the existing
45% gate. Dependency consistency, compilation, strict shared-code lint, fatal
page lint, and whitespace checks pass. Six new edge-case regressions were first
observed failing against the baseline before the corresponding fixes.

Local Python 3.12.14, pandas 3.0.1, Streamlit 1.58.0, yfinance 1.4.1.
Median of three repetitions on deterministic synthetic fixtures. The old and new
outputs were compared with pandas equality assertions, including missing values.

| Calculation | Fixture | Before | After | Speedup |
|---|---|---:|---:|---:|
| Previous-only percentile | 5,000 observations, 252-session window, ties and gaps | 0.655 s | 0.00355 s | 184× |
| Grouped composite | 2,000 dates, 24 proxies, six groups, 10% missing | 11.023 s | 0.0473 s | 233× |

These are helper benchmarks. Provider latency, Streamlit cold starts, chart
serialization, and browser rendering are outside these measurements. The
percentile helper is used by Volume Based Sentiment; not every page uses these
shared scoring helpers.

## Remaining priorities

1. **Finish shared-loader adoption.** Many pages retain independent Yahoo loaders,
   so cache reuse, retries, and completed-session policies differ. Examples are
   Equity Leadership, Cross-Asset Ratio Chartbook, Hedge Timer, and Market Memory.
   Migrate with page-specific fixtures and visible source/as-of checks.
2. **Set per-series freshness limits.** The two ratio chartbooks and Hedge Timer
   still have unlimited forward filling. Require current observed endpoints for
   signals, distinguish permitted calendar alignment from stale quotes, and make
   the as-of date visible next to each result. Simply deleting fill calls would
   change the sampling rules and needs separate regression work.
3. **Improve historical publication timing.** The commodity study estimates CFTC
   publication as Tuesday plus three days. Holiday delays and exceptional delayed
   releases need an actual publication calendar for strict point-in-time research.
   The new tests establish price causality and correct forward outcomes, not exact
   historical release timing. Macro backtests also need vintage data if they are
   intended to represent what was known at the time.
4. **Measure production performance.** Record cold/warm page timings, provider
   timings, cache hits, memory, and concurrent-session behavior for all 25 tools.
   The browser checks confirm three pages rendered but do not measure a production
   latency distribution or validate Streamlit Cloud resource allocation.
5. **Expand coverage and source reconciliation.** The original 155-test suite
   passed but covered only 44% of shared/engine code, below the existing 45% gate.
   Added checks cover previously skipped baskets and commodity outcomes. Further
   work should reconcile representative filing, FRED, CFTC, option-chain, and
   market-price outputs to their source records; treat current constituents as
   potentially survivorship-biased in historical basket studies.

## References

- [Yahoo Finance adapter adjustment implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py): `auto_adjust` adjusts OHLC prices while preserving volume.
- [Streamlit caching guidance](https://docs.streamlit.io/develop/concepts/architecture/caching): TTL and entry limits manage freshness and cache memory.

Production release remains subject to the repository's reviewed-PR process.
