from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Page config and styling
# ============================================================
st.set_page_config(page_title="Composite Chart Quality Score", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
    .meta {color: #6b7280; font-size: 0.92rem; margin-top: -0.35rem; margin-bottom: 1rem;}
    .section-rule {border: 0; border-top: 1px solid #e5e7eb; margin: 1.25rem 0 1.1rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CCQS_APP_URL = "https://composite-chart-quality-score.streamlit.app"


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Surface the live Composite Chart Quality Score dashboard inside the ADFM app.

        **Architecture:** The CCQS repo remains the compute and data layer. This page is intentionally lightweight so the ADFM app does not download the full CCQS parquet cache on every page load.

        **Use case:** Technical quality screening across equities, themes, leadership tiers, chart states, setups, and score changes.
        """
    )
    st.divider()
    st.markdown("### Controls")
    embed_height = st.slider("Embedded dashboard height", min_value=700, max_value=1800, value=1150, step=50)
    show_explanation = st.checkbox("Show implementation note", value=False)


# ============================================================
# Header
# ============================================================
st.title("Composite Chart Quality Score")
st.caption("Live CCQS dashboard embedded inside the ADFM app.")

st.markdown(
    f"<div class='meta'>Source: {CCQS_APP_URL} · compute and daily refresh remain in the CCQS repository</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 5])
with left:
    st.link_button("Open CCQS", CCQS_APP_URL, use_container_width=True)

if show_explanation:
    st.info(
        "This page deliberately embeds the live CCQS app instead of pulling the CCQS parquet files into the ADFM app at runtime. "
        "The prior version attempted to read multiple remote parquet files from GitHub on first load, which made Streamlit Cloud stall. "
        "The cleaner long-term version is to have the CCQS repo publish a small daily snapshot CSV for the ADFM app to consume."
    )

st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)

# ============================================================
# Embedded app
# ============================================================
components.iframe(
    src=CCQS_APP_URL,
    height=embed_height,
    scrolling=True,
)
