from __future__ import annotations

import streamlit as st


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
    .ccqs-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 20px;
        background: #ffffff;
        min-height: 170px;
    }
    .ccqs-card h3 {margin-top: 0; margin-bottom: 0.45rem; font-size: 1.05rem;}
    .ccqs-card p {color: #4b5563; font-size: 0.94rem; line-height: 1.45; margin-bottom: 0;}
    .ccqs-muted {color: #6b7280; font-size: 0.90rem;}
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
        **Purpose:** Launch the live Composite Chart Quality Score dashboard from inside the ADFM app.

        **Architecture:** The CCQS repo remains the compute and data layer. This page stays lightweight and avoids pulling the full CCQS parquet cache into the ADFM app.

        **Use case:** Technical quality screening across equities, themes, leadership tiers, chart states, setups, and score changes.
        """
    )
    st.divider()
    show_note = st.checkbox("Show implementation note", value=False)


# ============================================================
# Header
# ============================================================
st.title("Composite Chart Quality Score")
st.caption("Launch the live CCQS technical-quality dashboard in a separate tab.")

st.markdown(
    f"<div class='meta'>Source: {CCQS_APP_URL} · compute and daily refresh remain in the CCQS repository</div>",
    unsafe_allow_html=True,
)

button_col, spacer_col = st.columns([1, 5])
with button_col:
    st.link_button("Open CCQS", CCQS_APP_URL, use_container_width=True)

if show_note:
    st.info(
        "The embedded iframe was removed because Streamlit's share redirect chain can break inside another Streamlit app. "
        "Opening CCQS in a separate tab is faster and more stable. The cleaner long-term version is for the CCQS repo to publish a tiny daily CSV snapshot that this page can read directly."
    )

st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)

# ============================================================
# Lightweight useful output
# ============================================================
st.markdown("## How to Use CCQS")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="ccqs-card">
            <h3>Start with Top Stocks</h3>
            <p>Use the ranked ticker table to find names with strong technical quality, clean leadership, and favorable setup labels. This is the fastest way to move from broad risk appetite to single-name candidates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="ccqs-card">
            <h3>Check Themes Next</h3>
            <p>Theme CCQS helps identify where leadership is concentrated. Strong theme score plus strong constituent breadth is a better setup than an isolated single-stock move.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="ccqs-card">
            <h3>Use the Detail Panel</h3>
            <p>After a ticker screens well, inspect the CCQS trajectory, component contributions, setup label, state, and leadership tier. The goal is to separate durable trend quality from a short-term price spike.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Practical Workflow")
st.markdown(
    """
    **Risk-on tape:** start with Elite Leader and Strong Leader names, then favor Trending, Pullback, or Consolidating states with high CCQS and constructive setup labels.

    **Choppy tape:** focus on theme strength and breadth first. A high-scoring ticker inside a weak theme is lower quality than a high-scoring ticker inside a strong, broad theme.

    **Risk-off tape:** use CCQS defensively. Look for score deterioration, failed breakouts, fading leadership, and negative score changes before adding exposure.
    """
)

st.markdown("### Interpretation")
interpretation = {
    "CCQS 80+": "High-quality technical profile. Usually worth deeper chart and catalyst review.",
    "CCQS 55-80": "Constructive but less dominant. Needs confirmation from theme strength, setup, and tape.",
    "CCQS below 55": "Lower-quality technical profile. Avoid forcing long exposure unless there is a clear reversal or special-situation catalyst.",
    "Leadership Tier": "Relative-strength context. Elite and Strong Leaders carry the cleanest trend quality.",
    "State": "Current chart regime. Trending and Pullback tend to be more actionable than No Edge or Breaking Down.",
    "Setup": "Pattern label. Use it to distinguish breakout, reclaim, tight base, pullback, extended, and breakdown behavior.",
}

st.dataframe(
    [{"Signal": k, "How to Read It": v} for k, v in interpretation.items()],
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    "<div class='ccqs-muted'>CCQS is a screening layer, not a position-sizing model. The output should be combined with macro regime, liquidity, fundamentals, positioning, catalysts, and risk limits.</div>",
    unsafe_allow_html=True,
)
