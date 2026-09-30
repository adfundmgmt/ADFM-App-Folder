# Task layout implementation

Implemented the approved consolidated-table layout in the assigned files. All original routes and analytical calculations remain available.

| Surface | Result |
| --- | --- |
| Shared UI | `render_kpi_cards` keeps its signature but emits escaped inline status instead of cards. Shared sidebar About content is collapsed. Footer exposes aggregate delivery diagnostics through the shared lazy helper. Whole-page timing is handled by the root implementation around navigation, avoiding partial or duplicate header/footer timing. |
| Underwriter | One sortable valuation/company-snapshot metric table retains Section, Metric, Value, Formula and Context. Values/context retain the existing positive/caution/negative color bands. Selected issuer price history, financials, credit, growth/read-through and filing/source audits are lazy collapsed sections. |
| SEC 13F | Holdings and manager navigation remain primary, with customizable columns, search and CSV export. Exposure chart is lazy collapsed; methodology is collapsed. Existing summary facts now use the inline compatibility renderer. |
| Catalyst | Upcoming events table is primary. Five cards removed; volatility setup remains a concise caption. Charts, market backdrop, macro requests/prints and event details are lazy collapsed. Dates and scores remain typed for meaningful chronological/numeric sorting. Live official-calendar route still delegates to this renderer. Base module changes are limited to shared Yahoo transport. |
| Currency Tension | Original user-designed map and sortable ranking remain primary. Currency details and pillar heatmap are lazy diagnostics, with no tabs or Daily Read UI. Commentary engine itself remains available unchanged. |
| Liquidity | Primary level/impulse chart remains visible. FCI-G request/chart, driver charts, combined primary/market component audit, source diagnostics and history export are lazy collapsed. Removed audit tabs. |

Expanded Underwriter smoke exposed an existing Streamlit magic issue with multiline conditional expressions around annual and balance-sheet `metric_table` rendering: magic attempted to parse an incomplete first line while displaying the expression result. Explicit `if`/`else` blocks preserve behavior and pass expanded-view smoke.

## Red/green evidence

- `/tmp/adfm-review-venv/bin/python -m unittest tests.test_layout_upgrade -v`: initial three expected failures for missing inline status, missing metric table and eager closed macro requests. After implementation: green.
- `/tmp/adfm-review-venv/bin/python -m unittest tests.test_sec_13f_page tests.test_ui_theme tests.test_layout_upgrade -q`: updated holder-table expectation failed against the original three tabs; then passed with the table-first lazy chart. About test verifies collapsed shared content.
- Controlled regression verification temporarily restored the two original owned page files, ran the FX and Liquidity methods in `tests.test_layout_upgrade`, then restored the edited files in `finally`: two expected failures (closed pillar heatmap executed; original audit tabs present). Edited pages then passed.
- Expanded Underwriter fixture test failed with the reproducible Streamlit magic `SyntaxError`, then passed after explicit conditional blocks.
- Catalyst first-view test failed on string dates, then passed with native dates and numeric risk scores.

## Final verification

- `MPLCONFIGDIR=/tmp/adfm-matplotlib /tmp/adfm-review-venv/bin/python -m unittest tests.test_layout_upgrade tests.test_sec_13f_page tests.test_ui_theme -q`: **21 passed**, 5.550 seconds. Includes closed and opened Catalyst/Underwriter/Liquidity sections, preserved FX map/ranking, closed and opened 13F exposure chart, and manager CIK navigation. Logs: `/tmp/adfm-layout-focused.log`.
- Assigned core/test strict lint (`ruff check --select E,F,I,B --ignore E501`) passed.
- Assigned page fatal lint (`ruff check --select E9,F63,F7,F82`) passed.
- Assigned-file `compileall` and `git diff --check` passed.
- Full discovery (`MPLCONFIGDIR=/tmp/adfm-matplotlib /tmp/adfm-review-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q`) ran **370 tests in 22.218 seconds** while other tasks were changing shared files. It reported the following three concurrent incomplete-upgrade failures, communicated to root/owners:
  - `test_typed_daily_completion_recognizes_us_but_preserves_asian_futures`: `fetch_daily_ohlcv(now=...)` unsupported at that moment.
  - `test_us_completion_uses_actual_early_close_calendar`: `completed_daily_observations` absent at that moment.
  - `test_decision_regimes_use_prior_month_and_missing_trend_stays_unknown`: `information_mode` unsupported at that moment.
  Shared-data owner subsequently reported both transport regressions green; root is running the final integrated discovery/coverage checks. Full log: `/tmp/adfm-layout-suite.log`.

Tests use real Streamlit AppTest rendering and calculation functions with provider fixtures at external boundaries. Streamlit's AppTest expander element has no click/toggle API, so opened-view smoke sets the real expander's initial `expanded=True` while preserving its normal `.open` behavior and contents. Existing bare-mode/deprecation warnings remain; there are no exceptions in the final owned fixture runs. No commit or push was made.
