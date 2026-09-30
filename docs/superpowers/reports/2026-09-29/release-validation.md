# Tools data and portfolio-risk release validation

Validated 2026-09-30 on Python 3.12 with the repository requirements and constraints. Existing Streamlit application and all 25 tool routes are retained.

## Release checks

- Full unittest discovery: **407 tests passed in 63.198 seconds**.
- Coverage over `adfm_core,cte`: **66%**, above the 45% CI threshold.
- Dependency consistency, compilation of all application/page/test sources, strict shared-code/script/test lint, fatal page lint and `git diff --check`: passed.
- Independent math review: no unresolved important findings after regressions corrected missing portfolio currency, standard futures price-target parsing and stale commodity current labels.
- Independent final transport review: 26 transport tests and three page deadline tests passed. Same-symbol concurrent request with a 0.03-second budget returned in 0.034 seconds instead of waiting 0.308 seconds behind Streamlit's network cache lock. Normal concurrent requests, successful-cache reuse across varying request limits, session-settlement refetching and cache clearing were verified.
- Options interaction and missing-data regressions cover independently opened downloads, missing price histories, valid fallback spots, malformed strikes and missing underlying prices.

Provider calls run outside cache locks, with one bounded in-flight worker and a deadline that includes waiting for that worker. Successful raw observations use at most 16 cache entries and 64 MiB; separate last-good storage has the same limits. Failed/degraded results are retried on subsequent delivery. Provisional observations cannot cross the US session completion epoch as completed prices. Five larger batch loaders share one 25-second budget across their chunks, retries and fallbacks.

## Source and model limits

Representative official daily sovereign tails were reconciled against Treasury, Bank of Canada, Bundesbank, Bank of England, Japan MOF and RBA sources during implementation. New Zealand direct access was unavailable and a nonofficial substitute was rejected. Swiss cached observations were stale. Each country's table retains exact curve basis, observation date and availability; daily and monthly histories are never spliced.

Monthly OECD values remain revised descriptive observations with an explicitly unverified month-end availability proxy. Known-at-month-start seasonality requires ALFRED release/revision records and a configured FRED API key. Unknown actual CFTC release dates are excluded in strict mode; schedules remain assumptions in the optional descriptive mode. These controls do not create unavailable historical vintages.

Portfolio inputs remain private and session-only. Option P&L uses instantaneous European model changes anchored to supplied premiums; early exercise, assignment, trade liquidity and broker portfolio-margin rules are outside that approximation. Rates exposures require supplied sensitivities or an explicit underlying price target. Margin totals remain unavailable when any required broker input is missing.

Public provider outages are source limitations, not successful quote reconciliation. Local AppTest measurements are not production browser latency or a concurrent-user load test. Controlled fixtures verify the retained analytical paths and lazy sections when providers supply valid observations.
