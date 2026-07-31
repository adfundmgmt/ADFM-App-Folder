from __future__ import annotations

import streamlit as st

from adfm_core.primary_data import fetch_fred_series
from adfm_core.source_registry import source_capability_table
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_section_header,
)

st.set_page_config(page_title="Primary Source Monitor", layout="wide")
inject_explorer_style()


@st.cache_data(ttl=3600, show_spinner=False)
def load_primary_status():
    panel, diagnostics = fetch_fred_series(start="2000-01-01")
    return panel, diagnostics


panel, diagnostics = load_primary_status()
render_page_header(
    PageHeader(
        title="Primary Source Monitor",
        description="Track official-source coverage, adapter readiness, authentication requirements, revisions, and current FRED observations.",
        as_of=(
            f"FRED data through {panel.dropna(how='all').index.max().date()}"
            if not panel.dropna(how="all").empty
            else "FRED data unavailable"
        ),
        source_note="Official US and international statistical and regulatory systems",
    )
)

render_section_header(
    "Live primary-series health",
    "Each FRED series is requested independently so one failure does not erase the macro panel.",
)
st.dataframe(diagnostics, width="stretch", hide_index=True)

render_section_header(
    "Institutional source architecture",
    "Treasury, BLS, BEA, Fed, ECB, BOJ, BOE, CFTC, EIA, and SEC are registered behind one source contract.",
)
capabilities = source_capability_table()
st.dataframe(capabilities, width="stretch", hide_index=True)

st.info(
    "Adapters marked Registered have source metadata, endpoint, authentication, and revision rules defined. "
    "They remain disabled until the required API keys, SEC User-Agent, and production series selections are configured."
)

render_footer()
