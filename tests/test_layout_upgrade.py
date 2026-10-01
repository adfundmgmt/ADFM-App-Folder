"""Offline UI regressions for compact status and lazy secondary analysis."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.ui import render_kpi_cards

ROOT = Path(__file__).resolve().parents[1]


class LayoutUpgradeTests(unittest.TestCase):
    def test_kpi_compatibility_renders_escaped_inline_status(self):
        with patch("adfm_core.ui.st.markdown") as markdown:
            render_kpi_cards([("<Issuer>", "4 & 5", 'A "quoted" source')])
        body = markdown.call_args.args[0]
        assert "adfm-status" in body
        assert "kpi-card" not in body and "kpi-grid" not in body
        assert "&lt;Issuer&gt;" in body and "4 &amp; 5" in body
        assert "&quot;quoted&quot;" in body


    def test_underwriter_renderer_keeps_metric_formula_and_context_in_sortable_table(self):
        path = ROOT / "pages/9_ADFM_Underwriter.py"
        node = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "render_underwriter_cards")
        script = "import pandas as pd\nimport streamlit as st\n"
        script += ast.unparse(node) + "\n"
        script += "render_underwriter_cards([{'Section': 'Valuation', 'Metric': 'P/E', 'Value': '12.0x', 'Formula': 'Price divided by diluted EPS', 'Context': 'Favorable', 'Tone': 'positive'}])\n"
        app = AppTest.from_string(script).run()
        assert not app.exception
        assert len(app.dataframe) == 1
        frame = app.dataframe[0].value
        assert list(frame.columns) == ["Section", "Metric", "Value", "Formula", "Context"]
        assert frame.loc[0, "Formula"] == "Price divided by diluted EPS"
        assert frame.loc[0, "Context"] == "Favorable"
        assert not any("underwriter-metric-card" in block.value for block in app.markdown)


    def test_catalyst_first_view_is_table_without_secondary_provider_requests(self):
        from adfm_core import catalyst_calendar_page as base

        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=500)
        market = pd.DataFrame({ticker: range(100, 600) for ticker in base.MARKET_TICKERS}, index=dates)
        with (
            patch.object(base, "_fetch_market", return_value=market),
            patch.object(base, "_fetch_macro", side_effect=AssertionError("Closed macro section fetched")),
            patch.object(base, "_timeline", side_effect=AssertionError("Closed chart section computed")),
            patch.object(base, "_heatmap", side_effect=AssertionError("Closed backdrop chart computed")),
        ):
            app = AppTest.from_string("from adfm_core.catalyst_calendar_official_page import render_catalyst_calendar\nrender_catalyst_calendar()").run()
        assert not app.exception
        assert not app.tabs
        assert len(app.dataframe) == 1
        assert {"Date", "Event", "Source", "Risk Score", "Action"} <= set(app.dataframe[0].value.columns)
        assert not app.get("plotly_chart")
        assert not isinstance(app.dataframe[0].value.loc[0, "Date"], str)
        assert pd.api.types.is_numeric_dtype(app.dataframe[0].value["Risk Score"])
        assert not any("metric-card" in block.value and "Next Catalyst" in block.value for block in app.markdown)
        assert {"Catalyst charts and market backdrop", "Latest macro prints", "Full event details"} <= {item.label for item in app.expander}

    def test_footer_exposes_lazy_aggregate_diagnostics(self):
        app = AppTest.from_string("""
from adfm_core.ui import PageHeader, render_page_header, render_footer
from adfm_core.observability import performance_events, page_timer
with page_timer():
    render_page_header(PageHeader(title="Test", description="Test", eyebrow="Test"))
render_footer()
assert performance_events()[-1]["operation"] == "page"
assert set(performance_events()[-1]) == {"operation", "elapsed_seconds", "cache_hit", "requested_count", "failed_count", "peak_memory_mib"}
""").run()
        assert not app.exception
        assert "Data delivery diagnostics" in [item.label for item in app.expander]
        assert not app.dataframe

    def test_fx_preserves_map_and_ranking_without_daily_read_or_closed_diagnostics(self):
        tm = pd.DataFrame({"ccy": ["USD", "EUR"], "axis1_fundamental_struct": [0.2, -0.3], "axis2_stretch_struct": [-0.1, 0.4]})
        pillars = pd.DataFrame({"ccy": ["USD", "EUR"], "F_growth": [0.2, -0.3]})
        def cached(name):
            return {"tension_map": tm, "pillar_scores": pillars}.get(name)
        with (
            patch("cte.adapters.base.read_cache", side_effect=cached),
            patch("cte.dashboard.plots.pillar_heatmap_fig", side_effect=AssertionError("Closed diagnostics computed")),
        ):
            app = AppTest.from_file(str(ROOT / "pages/6_Currency_Tension_Engine.py")).run()
        assert not app.exception
        assert not app.tabs
        assert len(app.dataframe) == 1
        assert list(app.dataframe[0].value["FX"]) == ["USD", "EUR"]
        assert len(app.get("imgs")) == 1
        assert "Diagnostics" in [item.label for item in app.expander]
        assert not any("Daily Read" in item.label for item in app.expander)

    def test_liquidity_default_shows_fcig_and_primary_drivers_without_lower_sections(self):
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1000)
        values = 100 + np.linspace(0, 10, len(dates)) + np.sin(np.arange(len(dates)) / 20)
        def fred(symbol, *args, **kwargs):
            return SimpleNamespace(series=pd.Series(values, index=dates, name=symbol), metadata={})
        def prices(tickers, **kwargs):
            return ({ticker: pd.DataFrame({"Close": values}, index=dates) for ticker in tickers}, {})
        fcig = pd.DataFrame({"Date": dates[::20], "FCI-G": np.sin(np.arange(len(dates[::20])))})
        response = SimpleNamespace(content=fcig.to_csv(index=False).encode(), raise_for_status=lambda: None)
        with (
            patch("adfm_core.fred_store.FredStore.get", side_effect=fred),
            patch("adfm_core.market_data.fetch_daily_ohlcv", side_effect=prices),
            patch("requests.get", return_value=response),
        ):
            app = AppTest.from_file(str(ROOT / "pages/3_Liquidity_Conditions_Monitor.py")).run(timeout=30)
        assert not app.exception
        assert not app.tabs
        assert len(app.get("plotly_chart")) == 4
        expander_labels = {item.label for item in app.expander}
        assert "Federal Reserve FCI-G overlay" not in expander_labels
        assert "Primary liquidity drivers" not in expander_labels
        assert "Component audit" not in expander_labels
        assert "Source diagnostics" not in expander_labels
        assert "Download history" not in expander_labels

    def test_underwriter_default_table_and_open_financial_credit_source_sections(self):
        import streamlit as st

        from tests.test_sec_fundamentals import company_facts_payload

        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
        prices = pd.DataFrame({"Close": np.full(len(dates), 10.0)}, index=dates)
        directory = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        original_expander = st.expander
        def opened(label, *args, **kwargs):
            if label in {"Selected issuer price history", "Operating trajectory and issuer read-through", "Financials", "Credit", "Filings & Sources"}:
                kwargs["expanded"] = True
            return original_expander(label, *args, **kwargs)
        with (
            patch("adfm_core.sec_fundamentals.SecClient.company_tickers", return_value=directory),
            patch("adfm_core.sec_fundamentals.SecClient.company_facts", return_value=company_facts_payload()),
            patch("adfm_core.sec_fundamentals.SecClient.submissions", return_value={}),
            patch("adfm_core.market_data.fetch_daily_ohlcv", return_value=({"AAPL": prices}, {})),
        ):
            app = AppTest.from_file(str(ROOT / "pages/9_ADFM_Underwriter.py")).run()
            app = app.button[0].click().run()
            assert not app.exception
            assert not app.tabs
            assert len(app.dataframe) == 1
            main = app.dataframe[0].value
            assert {"Section", "Metric", "Formula", "Context"} <= set(main.columns)
            assert len(main) > 20
            assert not app.get("plotly_chart")
            with patch("streamlit.expander", side_effect=opened):
                app = app.run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 2
        assert len(app.dataframe) >= 6
        assert any("Reported growth" in item.value for item in app.markdown)
        assert any("Source audit" in item.value for item in app.markdown)

    def test_catalyst_open_details_retain_charts_macro_prints_and_event_sources(self):
        import streamlit as st

        from adfm_core import catalyst_calendar_page as base

        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=500)
        market = pd.DataFrame({ticker: range(100, 600) for ticker in base.MARKET_TICKERS}, index=dates)
        macro = pd.DataFrame({"cpi": [100.0, 102.0, 104.0, 105.0]}, index=pd.to_datetime(["2025-07-01", "2025-08-01", "2026-07-01", "2026-08-01"]))
        original_expander = st.expander
        def opened(label, *args, **kwargs):
            if label in {"Catalyst charts and market backdrop", "Latest macro prints", "Full event details"}:
                kwargs["expanded"] = True
            return original_expander(label, *args, **kwargs)
        with (
            patch.object(base, "_fetch_market", return_value=market),
            patch.object(base, "_fetch_macro", return_value=(macro, pd.DataFrame())),
            patch("streamlit.expander", side_effect=opened),
        ):
            app = AppTest.from_string("from adfm_core.catalyst_calendar_official_page import render_catalyst_calendar\nrender_catalyst_calendar()").run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 2
        assert len(app.dataframe) == 3
        assert "Why It Matters" in app.dataframe[-1].value.columns
        assert "Headline CPI YoY" in app.dataframe[1].value["Catalyst"].tolist()

