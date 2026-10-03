# Hedge Timer verification

Verified 2026-10-02 New York time. Implementation commit: 413695de17c35d3f672f15c796978c9422c37e99.

The live index feeds included October 2, while sixteen ETF inputs ended October 1. The page now calculates and displays every signal metric from the latest complete common session, labels that date, and blocks fresh shorts until current inputs agree. A fallback older than one NYSE session remains unavailable. Targeted recovery preserves full rolling history and bypasses stale cached deliveries on every attempt.

## Checks

- 20 Hedge Timer data, model, and Streamlit tests passed.
- 27 shared transport regressions passed.
- Application compilation and both repository lint commands passed.
- Independent review approved.
- Live S&P 500 and Nasdaq panels displayed valid October 1 signals; Nasdaq ten-year chart and controls rendered successfully.
- Full suite: 447 tests, 8 failures and 3 errors. Coverage: 64%, above the repository's 45% requirement.
- Each of those same eleven failures reproduced on unchanged base cff2dcecc002787ef440e57a96c923652e5c05c7. They concern other pages and existing catalog assertions.

## Existing broader-suite failures

- ERROR: `test_basket_chart_maintenance.BasketFreshnessIntegrationTests.test_basket_page_keeps_one_table_and_selected_chart_on_demand`
- ERROR: `test_layout_upgrade.LayoutUpgradeTests.test_catalyst_first_view_is_table_without_secondary_provider_requests`
- ERROR: `test_layout_upgrade.LayoutUpgradeTests.test_catalyst_open_details_retain_charts_macro_prints_and_event_sources`
- FAIL: `test_basket_chart_maintenance.MaintenanceIntegrationTests.test_basket_download_uses_shared_transport_without_a_guessed_us_cutoff`
- FAIL: `test_basket_data_loading.BasketDataLoadingTests.test_partial_response_does_not_freeze_missing_constituents_on_next_load`
- FAIL: `test_documentation.DocumentationTests.test_catalog_follows_the_research_workflow`
- FAIL: `test_documentation.DocumentationTests.test_readme_catalog_matches_the_shared_tool_catalog`
- FAIL: `test_layout_upgrade.LayoutUpgradeTests.test_underwriter_default_is_dense_overview_with_visible_market_and_annual_charts`
- FAIL: `test_options_positioning_page.OptionsPositioningPageTests.test_failed_highlight_ticker_does_not_hide_loaded_peer_map`
- FAIL: `test_options_positioning_page.OptionsPositioningPageTests.test_first_view_is_one_four_quadrant_chart_and_one_compact_table`
- FAIL: `test_options_positioning_page.OptionsPositioningPageTests.test_source_no_longer_contains_old_detail_surfaces`

