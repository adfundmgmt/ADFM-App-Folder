# Task E — Baskets and Chart Terminal

Implemented in the existing page routes. No commits or pushes performed.

## Files

- `pages/1_ADFM_Public_Equities_Baskets.py`
- `pages/10_ADFM_Chart_Terminal.py`
- `adfm_core/basket_calculations.py`: coverage-aware returns, reliable sessions, uninterrupted indicator segments, DMA and observation summary.
- `adfm_core/chart_terminal_data.py`: Yahoo layout normalization and the existing RSI/MACD/Bollinger/ATR/moving-average calculations, independent of Streamlit and providers.
- `adfm_core/basket_cache.py`: newest-16 public-price snapshot-pair retention.
- `tests/test_basket_chart_maintenance.py`
- `tests/test_basket_data_loading.py`: existing AST harness now injects the shared transport; original behavior assertions retained.

Page function adapters preserve existing offline AST consumers. These are focused calculations, not copies of either page monolith.

## Behavior

Both pages consume `download_market_data`. Independent `Ticker.history` transport was replaced with a period fallback through the shared wrapper. Basket metadata still uses Yahoo metadata APIs because the wrapper covers price transport. Chart caches have a 64-entry limit per loader, basket history/metadata caches 16 entries, and disk snapshots retain the newest 16 pairs. Empty chart loads raise within the cached loader so transient outages can recover on the next rerun.

Basket completion still uses the original exchange-calendar closes plus 15-minute grace, including early closes, and passes the resulting end-exclusive date into transport. Mixed foreign/FX batches explicitly use `completed_only=False`; they never inherit an assumed US cutoff. Reliable-session filtering, verified foreign holiday carry, coverage floors and dynamic MACD/EMA settings retain existing behavior. One demonstrated correction: FX conversion now fills at most two interior alignment dates and leaves a missing trailing exchange-rate endpoint missing. It no longer fabricates trailing USD closes from a stale exchange rate.

The consolidated sortable compact table remains the default. `%Close` is the return from the prior reliable observed close, explicitly labeled as spanning omitted provider sessions; row hover notes show each basket return's actual observation date and calendar age. It remains N/A when the endpoint is missing or only an inception anchor exists. A collapsed selected-basket detail computes one basket chart plus SPY only when requested. Optional definition and methodology sections start collapsed.

Chart Terminal retains its selected ticker chart and analytical overlays. The KPI strip is replaced by a compact price/prior-bar-return line and a collapsed statistics table. The observed bar date, calendar age, adjustment basis, raw-volume basis and provisional-bar limitation are explicit. Signal matrix/memo calculations run only after requesting detail; pattern/comparison secondary outputs are collapsed. A second demonstrated correction: horizon/YTD returns remain N/A without an observed prior anchor, rather than substituting a future quote or first available year quote.

## Red/green evidence

All commands below use `/tmp/adfm-review-venv/bin/python` in the repository.

1. `-m unittest tests.test_basket_chart_maintenance -v`: before extraction, two preservation fixtures passed and four extracted-helper tests failed because the new modules were absent. Initial test-harness missing-AST-line-number errors were corrected before implementation. The original three-symbol outage fixture was corrected to four symbols to preserve the existing 70% universe threshold; production threshold was unchanged. After extraction plus existing integrity fixtures: 20 tests passed.
2. `-m unittest tests.test_basket_chart_maintenance.MaintenanceIntegrationTests -v`: two expected failures before migration/layout changes: shared transport was not called, and observed-bar caption was absent. Afterwards the focused maintenance/loading/integrity run passed 25 tests.
3. `-m unittest tests.test_basket_chart_maintenance.BasketFreshnessIntegrationTests -v`: three expected failures before the corresponding changes: selected-chart control absent, trailing FX endpoint filled, `%Close` absent. After implementation the maintenance/loading/integrity/pattern run passed 40 tests.
4. Observation/inception and disk retention tests: two expected failures (inception showed a prior-close change; retention module absent), followed by two passing tests.
5. `-m unittest tests.test_basket_chart_maintenance.ChartRecoveryTests -q`: failed because an empty chart response stayed cached on the next rerun; after exceptions prevent empty-cache writes, one test passed.
6. `-m unittest tests.test_basket_chart_maintenance.ChartReturnEndpointTests -q`: two expected failures, returning `100.82995636352837` for an unavailable prior anchor and `0.0` for unavailable YTD. After removing forward/year-inception substitutions, two tests passed.

Final focused command:

```sh
/tmp/adfm-review-venv/bin/python -m unittest tests.test_basket_chart_maintenance tests.test_basket_data_loading tests.test_public_equities_integrity_regressions tests.test_public_equities_basket_map tests.test_chart_patterns tests.test_ui_theme -v
```

**58 tests passed in 12.596 seconds.** Exact final stdout: `/tmp/adfm-task-e-focused.log`.

Deterministic Streamlit AppTest fixtures exercised both real page scripts with provider-boundary fixtures, including the full basket definition map, no default basket chart, one selected basket chart, no default signal matrix, requested signal matrix, retained raw volume, warm reruns without extra provider calls, and transient empty-chart recovery. Basket smoke runs in a temporary working directory so generated public-price cache files do not remain in the checkout.

Final scoped checks passed:

```sh
/tmp/adfm-review-venv/bin/ruff check --select E,F,I,B --ignore E501 adfm_core/basket_calculations.py adfm_core/basket_cache.py adfm_core/chart_terminal_data.py tests/test_basket_chart_maintenance.py tests/test_basket_data_loading.py
/tmp/adfm-review-venv/bin/ruff check --select E9,F63,F7,F82 pages/1_ADFM_Public_Equities_Baskets.py pages/10_ADFM_Chart_Terminal.py
/tmp/adfm-review-venv/bin/python -m compileall -q pages/1_ADFM_Public_Equities_Baskets.py pages/10_ADFM_Chart_Terminal.py adfm_core/basket_calculations.py adfm_core/basket_cache.py adfm_core/chart_terminal_data.py tests/test_basket_chart_maintenance.py
```

`git diff --check` on owned tracked page/test changes also passed.

## Full-suite snapshots and limits

Full command was run twice while other tasks were actively changing shared files:

```sh
/tmp/adfm-review-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

First snapshot: **362 tests, seven errors**, exact stdout `/tmp/adfm-task-e-suite.log`:

- `test_commodity_exhaustion.CommodityExhaustionTests.test_contract_loader_preserves_raw_close_and_volume_and_completes_sessions`
- `test_commodity_exhaustion.CommodityExhaustionTests.test_strict_publication_dates_exclude_unverified_history`
- `test_page_accuracy.HistoricalPageAccuracyTests.test_market_memory_yield_changes_use_basis_points`
- `test_portfolio_stress.PortfolioStressTests.test_rates_etf_put_yield_reprices_underlying_with_supplied_dv01`
- `test_portfolio_stress.PortfolioStressTests.test_rates_shares_and_explicit_symbol_targets`
- `test_portfolio_stress.PortfolioStressTests.test_rates_without_sensitivity_requires_target_when_yields_change`
- `test_seasonality_information.SeasonalityInformationTests.test_decision_regimes_use_prior_month_and_missing_trend_stays_unknown`

Final snapshot: **374 tests in 23.731 seconds, two failures**, exact stdout `/tmp/adfm-task-e-suite-final.log`:

- `test_bond_upgrade.FixedBreakoutWindowTests.test_confirmation_window_includes_last_permitted_observation`
- `test_bond_upgrade.FixedBreakoutWindowTests.test_each_new_breakout_keeps_its_original_level`

No Task E test failed in either snapshot. These external failures were reported to the parent for integration; this report does not claim the release suite is green.

The chart handles domestic equities, foreign instruments, FX and crypto without validated per-symbol exchange metadata. Its bars are labeled observed/provisional, not falsely certified as completed. Calendar age is not an exchange-session staleness measure. Basket row observation dates describe the coverage-qualified basket return, not a claim that every constituent traded that date. No historical signal-age capture is invented. Existing vendor adjusted-price/rebalancing/roll limitations remain.

Live provider reconciliation and browser screenshots were not performed in this task. AppTest logs retain existing Streamlit warnings about initialized selectbox defaults, bare-mode caching contexts and legacy `components.v1.html`; the sortable table's existing iframe and HTML sorting are retained. Root integration remains responsible for full-suite coverage/release checks.
