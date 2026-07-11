"""Currency Tension Dashboard — Streamlit dashboard (spec §14).

Reads the persisted engine snapshot from cache and renders the two-axis tension map,
inflection warnings, pillar breakdown, carry grid, overlays, and currency detail.
A rebuild control refreshes the snapshot from the latest cached data when the required
raw cache and API credentials are available.

Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import streamlit as st

from cte.adapters.base import read_cache
from cte.config import CACHE_DIR
from cte.dashboard.plots import (
    carry_heatmap_fig,
    pillar_heatmap_fig,
    tension_map_fig,
)


# =========================================================
# Page config and visual system
# =========================================================
st.set_page_config(
    page_title="Currency Tension Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

TITLE = "Currency Tension Dashboard"
SUBTITLE = (
    "Fundamental trajectory, valuation and policy stretch, pairwise carry, "
    "and regime warnings across major currencies."
)

TEXT = "#2d3142"
SUBTLE = "#7a8493"
SIDEBAR_BG = "#f3f6fa"
CONTROL_BG = "#f5f7fb"
BORDER = "#e0e6ef"
CARD_BG = "#ffffff"
SOFT_BG = "#fafbfd"

CUSTOM_CSS = f"""
<style>
    /* Main canvas */
    .block-container {{
        padding-top: 2.15rem;
        padding-bottom: 3.0rem;
        max-width: 1560px;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT};
        font-weight: 700;
        letter-spacing: -0.015em;
    }}

    p, li {{
        line-height: 1.55;
    }}

    div[data-testid="stCaptionContainer"] {{
        color: {SUBTLE};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] > div {{
        padding-top: 1.15rem;
    }}

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {TEXT};
        font-weight: 750;
        letter-spacing: -0.01em;
    }}

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {{
        color: #465264;
        font-size: 0.93rem;
        line-height: 1.55;
    }}

    /* Hero */
    .hero-title {{
        color: {TEXT};
        font-size: 2.65rem;
        line-height: 1.04;
        font-weight: 800;
        letter-spacing: -0.045em;
        margin: 0 0 0.28rem 0;
    }}

    .hero-caption {{
        color: {SUBTLE};
        font-size: 1.0rem;
        line-height: 1.55;
        margin-bottom: 1.55rem;
        max-width: 980px;
    }}

    .section-kicker {{
        color: #697586;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }}

    .section-title {{
        color: {TEXT};
        font-size: 1.18rem;
        font-weight: 750;
        letter-spacing: -0.015em;
        margin-bottom: 0.12rem;
    }}

    .section-copy {{
        color: {SUBTLE};
        font-size: 0.90rem;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }}

    .tool-divider {{
        margin-top: 1.15rem;
        margin-bottom: 1.15rem;
        border-top: 1px solid #d8dee8;
    }}

    /* Input labels and controls */
    div[data-testid="stRadio"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stButton"] label {{
        color: #536171;
    }}

    div[data-testid="stRadio"] > label,
    div[data-testid="stSelectbox"] > label {{
        color: #536171;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    div[data-baseweb="select"] > div {{
        min-height: 42px;
        border-radius: 12px;
        background-color: {CONTROL_BG};
        border-color: {BORDER};
    }}

    div[role="radiogroup"] {{
        gap: 0.35rem;
    }}

    div[role="radiogroup"] label {{
        border-radius: 10px;
    }}

    /* Buttons */
    div.stButton > button {{
        width: 100%;
        min-height: 42px;
        border-radius: 11px;
        border: 1px solid #cfd7e3;
        background: #ffffff;
        color: {TEXT};
        font-weight: 700;
    }}

    div.stButton > button:hover {{
        border-color: #aeb9c8;
        color: #1f2937;
        background: #f8fafc;
    }}

    /* Containers and panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER} !important;
        border-radius: 15px !important;
        background: {CARD_BG};
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    }}

    .chart-panel {{
        margin-top: 0.25rem;
    }}

    .soft-note {{
        background: {SOFT_BG};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 0.82rem 0.95rem;
        color: #465264;
        font-size: 0.91rem;
        line-height: 1.5;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {SOFT_BG};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 0.85rem 1rem;
    }}

    div[data-testid="stMetricLabel"] {{
        color: #667085;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT};
        font-weight: 750;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: #667085;
        font-weight: 650;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {TEXT};
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: #596579;
    }}

    /* Expanders, tables, and charts */
    details {{
        border-color: {BORDER} !important;
        border-radius: 12px !important;
        background: #ffffff;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    .stPlotlyChart,
    div[data-testid="stImage"] img {{
        background: #ffffff;
        border-radius: 12px;
    }}

    /* Reduce default rule weight */
    hr {{
        border-color: #e6eaf0;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# Data access
# =========================================================
def _grid(name: str) -> Optional[pd.DataFrame]:
    grid = read_cache(name)
    return grid.set_index(grid.columns[0]) if grid is not None else None


@st.cache_data(show_spinner=False)
def _load():
    tension_map = read_cache("tension_map")
    pillars = read_cache("pillar_scores")
    overlays = read_cache("overlays")

    warning_path = CACHE_DIR / "warnings.json"
    warnings = (
        json.loads(warning_path.read_text(encoding="utf-8"))
        if warning_path.exists()
        else {}
    )

    return tension_map, pillars, overlays, warnings


def _rebuild() -> None:
    from cte.scoring.engine import build_snapshot

    build_snapshot(persist=True)
    _load.clear()


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Cross-currency regime monitor built around two questions: whether a currency's fundamentals are improving or deteriorating, and whether valuation and policy are cheap or already stretched.

        **How to read it**

        - Structural uses the longer own-history anchor.
        - Regime emphasizes the latest two-year move.
        - Carry is pairwise rather than dollar-centric.
        - Currency Notes surface inflections, policy traps, stress regimes, and fragile carry.

        **Data sources:** FRED, OECD, BIS, Eurostat, Japan e-Stat, UK ONS, CFTC, and Yahoo Finance.
        """
    )

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Display Controls")

    horizon = st.radio(
        "Scoring horizon",
        ["Structural (~10y)", "Regime (~2y)"],
        index=0,
    )
    hz = "struct" if horizon.startswith("Structural") else "regime"

    # Rebuild is admin-only. It requires the raw cache and API credentials,
    # which are intentionally absent from the public deployment.
    can_rebuild = (
        read_cache("macro_backbone") is not None
        and bool(os.environ.get("FRED_API_KEY"))
    )

    if can_rebuild:
        if st.button("Rebuild Snapshot", use_container_width=True):
            with st.spinner("Recomputing snapshot..."):
                _rebuild()
            st.success("Snapshot refreshed.")

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.caption(
        "The scheduled job refreshes the public snapshot once daily. "
        "Methodology and source mapping remain available at the bottom of the page."
    )


# =========================================================
# Header and snapshot load
# =========================================================
st.markdown(f'<div class="hero-title">{TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-caption">{SUBTITLE}</div>', unsafe_allow_html=True)

tm, pillars, overlays, warns = _load()

if tm is None:
    st.warning(
        "No snapshot is available in cache. Run `python -m scripts.backfill` and "
        "then `python -m cte.scoring.engine`, or use Rebuild Snapshot when the "
        "admin controls are available."
    )
    st.stop()

flagged = set(warns)


# =========================================================
# Daily read
# =========================================================
from cte.commentary.narrator import load_commentary

_note, _meta = load_commentary()

with st.container(border=True):
    st.markdown('<div class="section-kicker">Current regime</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Daily Read</div>', unsafe_allow_html=True)
    st.markdown(_note)

    if _meta.get("generated_at"):
        model_name = _meta.get("model", "")
        generated_at = _meta["generated_at"][:16]
        model_suffix = f" · {model_name}" if model_name else ""
        st.caption(
            f"Generated {generated_at}Z{model_suffix} · refreshed once daily by "
            "the scheduled job, not per user."
        )


# =========================================================
# Tension map and currency notes
# =========================================================
st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)

left, right = st.columns([3.15, 1.85], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">Cross-currency map</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Fundamentals vs Stretch</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">The horizontal axis captures fundamental trajectory; '
            'the vertical axis captures valuation and policy stretch. Flagged currencies are '
            'linked to objective notes in the adjacent panel.</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(tension_map_fig(tm, hz, flagged), width="stretch")

with right:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">Signal review</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Currency Notes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Twin-signal reads, horizon divergence, fragile '
            'carry, one-legged positions, overlay effects, and missing-data caveats.</div>',
            unsafe_allow_html=True,
        )

        if not warns:
            st.info("No notable reads in the current snapshot.")
        else:
            for ccy, notes in warns.items():
                with st.expander(
                    f"{ccy}  ·  {len(notes)} note{'s' if len(notes) != 1 else ''}",
                    expanded=(ccy == "GBP"),
                ):
                    for note in notes:
                        st.markdown(f"- {note}")


# =========================================================
# Analytical tabs
# =========================================================
st.markdown("---")

t1, t2, t3, t4 = st.tabs(
    ["Pillar Scores", "Carry Grid", "Overlays", "Currency Detail"]
)

with t1:
    st.markdown('<div class="section-kicker">Component structure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pillar Scores</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Signed z-scores ordered by fundamental score. '
        'Positive values point toward the positive pole of each axis.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.pyplot(pillar_heatmap_fig(pillars, tm, hz), width="stretch")

    st.caption(
        "Inflation and Fiscal are overlay-adjusted for policy-trap and "
        "reward-versus-stress regimes. Any material signal adjustment is surfaced "
        "in Currency Notes."
    )

with t2:
    header_left, header_right = st.columns([3.2, 1.0], gap="large")

    with header_left:
        st.markdown('<div class="section-kicker">Relative rates</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pairwise Carry Grid</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">The dollar is one leg of eight rather than '
            'the organizing hub. Positive values indicate carry earned by the base '
            'currency over the quote currency.</div>',
            unsafe_allow_html=True,
        )

    with header_right:
        basis = st.radio(
            "Carry basis",
            ["2Y real", "2Y nominal"],
            horizontal=True,
        )

    grid_name = "carry_grid_real" if basis == "2Y real" else "carry_grid_nominal"
    grid = _grid(grid_name)
    label = f"{basis.title()} Carry · Base − Quote (%)"

    if grid is not None:
        with st.container(border=True):
            st.pyplot(carry_heatmap_fig(grid, label), width="stretch")
    else:
        st.info(f"No cached {basis.lower()} carry grid is available.")

    st.caption(
        "Nominal shows the raw two-year rate differential. Real carry removes the "
        "inflation tax and is the more useful read on carry quality and fragility."
    )

with t3:
    st.markdown('<div class="section-kicker">Signal modifiers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Objective Overlays</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These diagnostics determine when an apparently '
        'supportive yield or inflation signal should be discounted, neutralized, or '
        'reversed.</div>',
        unsafe_allow_html=True,
    )

    if overlays is not None and not overlays.empty:
        st.dataframe(
            overlays.set_index("ccy"),
            width="stretch",
            height=min(620, 82 + 35 * len(overlays)),
        )
    else:
        st.info("No overlay data is available in the current snapshot.")

    st.caption(
        "yld_fx_corr / yld_regime tests whether a currency is rewarded or punished "
        "for higher yields. feasibility / infl_mult measures hike room against the "
        "inflation trap. ctv_pctile captures carry-to-vol crowding."
    )

with t4:
    st.markdown('<div class="section-kicker">Single-currency drilldown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Currency Detail</div>', unsafe_allow_html=True)

    ccy = st.selectbox("Currency", list(tm["ccy"]))
    row = tm.set_index("ccy").loc[ccy]

    metric_1, metric_2 = st.columns(2, gap="large")
    metric_1.metric(
        f"{ccy} fundamental · {hz}",
        f"{row[f'axis1_fundamental_{hz}']:+.2f}",
    )
    metric_2.metric(
        f"{ccy} stretch · {hz}",
        f"{row[f'axis2_stretch_{hz}']:+.2f}",
    )

    st.markdown("#### Pillar Detail")
    if pillars is not None and not pillars.empty:
        st.dataframe(
            pillars.set_index("ccy").loc[[ccy]],
            width="stretch",
            hide_index=False,
        )
    else:
        st.info("No pillar detail is available in the current snapshot.")

    if ccy in warns:
        st.markdown("#### Active Warnings")
        with st.container(border=True):
            for note in warns[ccy]:
                st.markdown(f"- {note}")


# =========================================================
# Methodology
# =========================================================
st.markdown("---")

with st.expander("Methodology · pillars, carry, and overlays", expanded=False):
    st.markdown(
        """
Every currency is scored against **its own history** on the two lookbacks selected in the sidebar. **Structural (~10y)** is the longer-run level anchor; **Regime (~2y)** captures the more recent move. Each raw input is standardized into a z-score, signed so that a positive reading points toward the positive pole of the relevant axis, then aggregated into pillars and the two headline axes.

### Axis 1 · Fundamental trajectory  
*deteriorating ↔ improving*

- **Growth (A)** uses business-confidence levels and three-month slopes, real GDP growth, and the three-month change in unemployment. Sources include OECD, FRED, Eurostat, and ONS, with the freshest defensible source used for each currency.
- **Inflation (B)** combines the headline CPI gap versus the central bank target with three-month inflation momentum. Above-target, accelerating inflation initially reads as hawkish support. The hike-feasibility overlay can fade or reverse that support when growth is weak and policy is already restrictive.
- **External (C)** combines the current-account balance as a share of GDP with the net international investment position. The current account captures immediate funding pressure; NIIP distinguishes creditor from debtor currencies.
- **Real 10Y (D)** uses the ten-year real yield. For the US this is derived from TIPS; elsewhere it is approximated as the nominal ten-year yield less CPI. The reward-versus-stress overlay controls the sign: rising yields help when the currency is rewarded for them and hurt when the market treats them as fiscal or policy stress.

### Axis 2 · Valuation and policy stretch  
*cheap ↔ maxed-out*

- **Valuation (G)** uses the BIS real effective exchange rate to judge how rich or cheap a currency is against its own history. Valuation carries roughly twice the weight of policy in Axis 2, limiting false stretch signals in early-cycle currencies.
- **Policy (E)** combines the real policy rate with the priced path, measured as the two-year yield minus the policy rate. A high and restrictive stance with substantial tightening already embedded is closer to the stretched pole.

### Carry  
*pairwise rather than dollar-centric*

Carry is inherently relative, so it is shown in the **8 × 8 grid** rather than treated as a per-currency pillar. The nominal grid displays the two-year rate differential between every base and quote currency. The real grid removes inflation and is the cleaner measure of carry quality and vulnerability.

### Overlays  
*objective modifiers surfaced in Currency Notes*

- **Yield reward versus stress** uses the rolling relationship between FX changes and yield changes to determine whether a higher yield supports or damages the currency.
- **Hike feasibility** compares growth health with existing policy restrictiveness. Above-target inflation loses its positive interpretation when the central bank appears trapped.
- **Carry-to-vol** divides real carry by realized FX volatility and percentiles the result. Extreme readings flag carry that may be crowded and fragile to a volatility shock.

Whenever an overlay materially changes a signal, the adjustment is stated in **Currency Notes**, along with relevant horizon conflicts, strong-dollar confounds, one-legged positions, or missing-source caveats.

*Sources: FRED · OECD · BIS · Eurostat · Japan e-Stat · UK ONS · CFTC · Yahoo Finance · national debt offices for yield curves. Full source mapping remains in `docs/DATA_SOURCES.md`.*
        """
    )
