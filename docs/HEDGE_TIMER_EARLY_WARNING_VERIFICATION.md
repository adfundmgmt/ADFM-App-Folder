# Hedge Timer early-warning verification

Rules are fitted on SPX only through 2026-10-02 and frozen at20/100. NDX uses identical metrics, weights and thresholds. The price-retreat watch metric now activates at a1.5% fall from the20-session high, a2% five-session fall, or a3% ten-session fall; other watch metrics retain their original definitions and weights.

A qualifying warning begins within20 sessions before the peak, is already active at the peak, or begins before the first loss beyond3%. Separate local-peak legs rearm after a10% close rebound from the trough. Historical highs/lows are used for outcome auditing only, not warning scores. Rebounds do not reopen the deadline. An existing warning credited at a peak is explicitly an observation of its state, not a new alert.

| Basis | SPX early captures | NDX unchanged rules |
| --- | --- | --- |
| Intraday high/low |7/7|17/17|
| Daily close |6/6|13/13|

| SPX intraday peak | Trough | Historical warning | Loss at warning |
| --- | --- | --- | --- |
|2020-02-19|2020-03-23|2020-01-21|Before peak|
|2020-09-02|2020-09-24|2020-08-21|Before peak|
|2022-01-04|2022-02-24|2022-01-05|2.45%|
|2022-03-29|2022-06-17|2022-03-31|2.31%|
|2022-08-16|2022-10-13|2022-08-19|2.24%|
|2023-07-27|2023-10-27|2023-07-07|Before peak|
|2025-02-19|2025-04-07|2025-01-27|Before peak|

SPX spends about61% of sessions in warning; NDX about66%. The app reports mature false alarms separately from late/repeat alerts and pending outcomes with fewer than60 completed follow-up sessions. Historical recall is fitted evidence, not a future prediction guarantee. NDX is a cross-index transfer check, not a temporally independent validation.

Input source: [Yahoo Finance research checkpoint](../data/hedge_timer/research_source.json), with [22 close series](../data/hedge_timer/research_inputs.csv) and [actual index daily ranges](../data/hedge_timer/research_indices_ohlc.csv). Missing intraday ranges trigger explicit daily-close fallback and block intraday certification. Incomplete historical signal inputs are disclosed. A total live outage uses the dated research snapshot for browsing while blocking current signals and fresh shorts.

The reviewed local implementation passed34 focused tests, compilation, dependency checks and lint; independent read-only review approved the fixes. Initial full-repository verification ran459 tests with64% coverage and the same11 pre-existing unrelated failures below. The execution environment disconnected during a later verification run, so the recovery branch must pass fresh GitHub-hosted Hedge Timer checks before publication.

Pre-existing unrelated errors:
- test_basket_chart_maintenance.BasketFreshnessIntegrationTests.test_basket_page_keeps_one_table_and_selected_chart_on_demand
- test_layout_upgrade.LayoutUpgradeTests.test_catalyst_first_view_is_table_without_secondary_provider_requests
- test_layout_upgrade.LayoutUpgradeTests.test_catalyst_open_details_retain_charts_macro_prints_and_event_sources

Pre-existing unrelated failures:
- test_basket_chart_maintenance.MaintenanceIntegrationTests.test_basket_download_uses_shared_transport_without_a_guessed_us_cutoff
- test_basket_data_loading.BasketDataLoadingTests.test_partial_response_does_not_freeze_missing_constituents_on_next_load
- test_documentation.DocumentationTests.test_catalog_follows_the_research_workflow
- test_documentation.DocumentationTests.test_readme_catalog_matches_the_shared_tool_catalog
- test_layout_upgrade.LayoutUpgradeTests.test_underwriter_default_is_dense_overview_with_visible_market_and_annual_charts
- test_options_positioning_page.OptionsPositioningPageTests.test_failed_highlight_ticker_does_not_hide_loaded_peer_map
- test_options_positioning_page.OptionsPositioningPageTests.test_first_view_is_one_four_quadrant_chart_and_one_compact_table
- test_options_positioning_page.OptionsPositioningPageTests.test_source_no_longer_contains_old_detail_surfaces

Production originally showed Streamlit's memory resource-limit page. Live deployment verification remains a separate requirement.
