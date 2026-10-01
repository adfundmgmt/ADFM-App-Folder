# Position Sizing Lab verification, October 1, 2026

The page now opens into inverse-volatility sizing. Exposure is the lesser of the user ceiling and base exposure times historical normal volatility divided by recent volatility. The 252-observation normal-volatility baseline ends before the recent return window. Historical sizing uses contemporaneous data. Portfolio stress remains available as an optional, lazily loaded tool.

## Passing checks

27 focused tests pass with the repository-pinned Streamlit 1.58.0, Plotly 6.8.0 and pandas 3.0.1: volatility sizing (5), original sizing analytics (6), portfolio-stress and sizing UI (6), shared palette (4), and repository standards (6). Full source compilation and the CI lint commands pass.

The deployed page was inspected in a browser. Its default TLT data populated through October 1. Changing current exposure from 10% to 20% updated the exposure visual, relative reduction, chart reference line and NAV-risk table.

## Broader suite limitations

An initial run of all 428 tests used Streamlit 1.64.0 and returned 11 failures and 2 errors. Two failures introduced by this page, shared palette and standardized About This Tool, were corrected and their test modules now pass on the pinned runtime. A clean full-suite result is not claimed.

These seven failures were reproduced on the unchanged parent commit 3499e9e with the pinned runtime:

- `test_basket_data_loading.BasketDataLoadingTests.test_partial_response_does_not_freeze_missing_constituents_on_next_load`
- `test_documentation.DocumentationTests.test_catalog_follows_the_research_workflow`
- `test_documentation.DocumentationTests.test_readme_catalog_matches_the_shared_tool_catalog`
- `test_layout_upgrade.LayoutUpgradeTests.test_underwriter_default_is_dense_overview_with_visible_market_and_annual_charts`
- `test_options_positioning_page.OptionsPositioningPageTests.test_failed_highlight_ticker_does_not_hide_loaded_peer_map`
- `test_options_positioning_page.OptionsPositioningPageTests.test_first_view_is_one_four_quadrant_chart_and_one_compact_table`
- `test_options_positioning_page.OptionsPositioningPageTests.test_source_no_longer_contains_old_detail_surfaces`

Other initial broad-run failures/errors were:

- `test_basket_chart_maintenance.MaintenanceIntegrationTests.test_basket_download_uses_shared_transport_without_a_guessed_us_cutoff`: mock was not called; the basket loader attempted provider requests.
- `test_basket_chart_maintenance.BasketFreshnessIntegrationTests.test_basket_page_keeps_one_table_and_selected_chart_on_demand`: timed out while Yahoo returned rate limits. This was not rerun because it requests a large unrelated basket universe.
- `test_home_command_center.HomeResearchDirectoryTests.test_home_renders_ordered_native_page_links`: newer Streamlit resolved the relative script path against the tests directory. The Home page was not changed.
- `test_layout_upgrade.LayoutUpgradeTests.test_fx_preserves_map_and_ranking_without_daily_read_or_closed_diagnostics`: passed when rerun on pinned Streamlit.

## Model scope

The tool sizes equity/ETF instrument exposure, not an entire portfolio. Realized volatility does not forecast returns, capture correlated portfolio loss, or cap jumps. NAV defaults to zero and is optional; exposure inputs are illustrative defaults, not inferred holdings. Stale or insufficient history is labelled and unavailable history cannot generate a sizing result.
