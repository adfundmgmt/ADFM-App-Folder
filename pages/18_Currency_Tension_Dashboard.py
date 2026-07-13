"""Currency Tension Engine — Streamlit dashboard (spec §14).

Reads the persisted engine snapshot from cache and renders the two-axis tension map,
inflection warnings, pillar breakdown, carry grid, positioning, overlays, and
currency-level detail.

Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from cte.adapters.base import read_cache
from cte.config import CACHE_DIR
from cte.dashboard.plots import (
    carry_heatmap_fig,
    pillar_heatmap_fig,
    positioning_fig,
    tension_map_fig,
)


# ============================================================
# Page and visual system
# ============================================================

st.set_page_config(
    page_title="Currency Tension Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --adfm-ink: #2d3142;
        --adfm-muted: #6f7b8a;
        --adfm-soft: #f5f7fb;
        --adfm-line: #e0e6ef;
        --adfm-sidebar: #f3f6fa;
        --adfm-accent: #4c78a8;
        --adfm-positive: #3f8f72;
        --adfm-negative: #c85d5d;
    }

    .block-container {
        padding-top: 2.35rem;
        padding-bottom: 3rem;
        max-width: 1580px;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--adfm-sidebar);
        border-right: 1px solid var(--adfm-line);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.15rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--adfm-ink);
        font-weight: 750;
        letter-spacing: -0.015em;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: #3f4a5a;
        font-size: 0.93rem;
        line-height: 1.52;
    }

    .hero-kicker {
        color: var(--adfm-accent);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        color: var(--adfm-ink);
        font-size: 2.75rem;
        line-height: 1.04;
        font-weight: 800;
        letter-spacing: -0.045em;
        margin: 0;
    }

    .hero-caption {
        color: var(--adfm-muted);
        font-size: 1.02rem;
        line-height: 1.55;
        margin-top: 0.55rem;
        margin-bottom: 1.65rem;
        max-width: 1050px;
    }

    .section-title {
        color: var(--adfm-ink);
        font-size: 1.15rem;
        font-weight: 760;
        letter-spacing: -0.015em;
        margin-bottom: 0.15rem;
    }

    .section-caption {
        color: var(--adfm-muted);
        font-size: 0.91rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    .tool-divider {
        border-top: 1px solid #d8dee8;
        margin: 1.25rem 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--adfm-line);
        border-radius: 14px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(31, 45, 61, 0.03);
    }

    div[data-testid="stMetric"] label {
        color: var(--adfm-muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: var(--adfm-ink);
        font-weight: 760;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        border-color: var(--adfm-line);
        box-shadow: 0 1px 2px rgba(31, 45, 61, 0.025);
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stToggle"] label,
    div[data-testid="stNumberInput"] label {
        color: #536171;
        font-size: 0.75rem;
        font-weight: 760;
        letter-spacing: 0.055em;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        border-radius: 11px;
        background-color: #ffffff;
        border-color: var(--adfm-line);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--adfm-line);
        padding-bottom: 0.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.55rem;
        padding: 0 0.95rem;
        border-radius: 10px 10px 0 0;
        color: var(--adfm-muted);
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: var(--adfm-ink) !important;
        background-color: var(--adfm-soft);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--adfm-line);
        font-weight: 700;
    }

    .stDataFrame {
        border: 1px solid var(--adfm-line);
        border-radius: 12px;
        overflow: hidden;
    }

    .quiet-note {
        color: var(--adfm-muted);
        font-size: 0.83rem;
        line-height: 1.45;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Data helpers
# ============================================================


def _grid(name: str) -> Optional[pd.DataFrame]:
    grid = read_cache(name)
    if grid is None or grid.empty:
        return None
    return grid.set_index(grid.columns[0])


@st.cache_data(show_spinner=False, ttl=600)
def _load():
    """Load the persisted engine snapshot with a short cache TTL.

    The scheduled data workflow can update committed cache files inside the same
    running container. A ten-minute TTL lets the UI surface those updates without
    requiring a public user to force a rebuild.
    """
    tension_map = read_cache("tension_map")
    pillar_scores = read_cache("pillar_scores")
    overlay_data = read_cache("overlays")
    warnings_path = CACHE_DIR / "warnings.json"
    warning_data = (
        json.loads(warnings_path.read_text()) if warnings_path.exists() else {}
    )
    return tension_map, pillar_scores, overlay_data, warning_data


def _rebuild() -> None:
    from cte.scoring.engine import build_snapshot

    build_snapshot(persist=True)
    _load.clear()


def _snapshot_label(frame: Optional[pd.DataFrame]) -> str:
    if frame is None or frame.empty or "date" not in frame.columns:
        return "Latest"

    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return "Latest"
    return dates.max().strftime("%b %d, %Y")


def _quadrant_label(fundamental: Any, stretch: Any) -> str:
    if pd.isna(fundamental) or pd.isna(stretch):
        return "Insufficient data"

    if fundamental >= 0 and stretch < 0:
        return "Improving / Cheap"
    if fundamental >= 0 and stretch >= 0:
        return "Improving / Stretched"
    if fundamental < 0 and stretch < 0:
        return "Deteriorating / Cheap"
    return "Deteriorating / Stretched"


def _section_intro(title: str, caption: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">{caption}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Load persisted state
# ============================================================

tm, pillars, overlays, warns = _load()

if tm is None or tm.empty:
    st.warning(
        "No snapshot is available in cache. Run `python -m scripts.backfill` and "
        "then `python -m cte.scoring.engine`, or use the admin rebuild control."
    )
    st.stop()

_hist = read_cache("snapshot_history")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Cross-sectional FX regime monitor that separates fundamental
        trajectory from valuation and policy stretch.

        **What this page shows**

        - A two-axis map of major currencies against their own histories.
        - Structural, regime, and secular scoring horizons.
        - Month-end trails and historical map reconstruction.
        - Pillar-level drivers, pairwise carry, positioning, and overlays.
        - Objective currency notes when signals diverge, crowd, or become fragile.

        **How to read the map**

        - Rightward movement means fundamentals are improving.
        - Upward movement means the currency is becoming more stretched.
        - The lower-right quadrant is the cleanest cheap-and-improving setup.
        - Rings and notes identify crowding, overlay adjustments, or data caveats.
        """
    )

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Snapshot Controls")

    horizon = st.radio(
        "Scoring horizon",
        ["Regime (~2y)", "Structural (~10y)", "Secular (~15y)"],
        index=1,
        help=(
            "Secular scores each input against roughly 15 years of its own "
            "history. Rate pillars begin later because several source series "
            "start in the early 2000s."
        ),
    )
    hz = {"St": "struct", "Re": "regime", "Se": "secular"}[horizon[:2]]

    st.markdown("### History")
    trail_n = st.slider(
        "Trail length, month-ends",
        min_value=0,
        max_value=12,
        value=6,
        help=(
            "Shows the fading path of each currency's last N month-end "
            "positions, revealing direction and speed."
        ),
    )

    asof_sel = "Live"
    CUSTOM_W = False
    user_w: dict[str, float] = {}

    if _hist is not None and len(_hist):
        from cte.scoring.history import dial_options

        _avail = dial_options(_hist, hz)
        if _avail:
            _mode = st.radio(
                "Map view",
                ["Live", "Historical"],
                horizontal=True,
                help=(
                    "Historical redraws the map at a prior month-end. Daily notes "
                    "and live warning rings remain tied to the current snapshot."
                ),
            )
            if _mode == "Historical":
                _years = sorted({d.year for d in _avail}, reverse=True)
                _yr = st.selectbox("Year", _years, index=0)
                _mopts = [d for d in _avail if d.year == _yr]
                _mo = st.selectbox(
                    "Month",
                    _mopts,
                    index=len(_mopts) - 1,
                    format_func=lambda d: d.strftime("%B"),
                )
                asof_sel = _mo.strftime("%Y-%m-%d")
                st.caption(
                    f"Available range: {_avail[0].strftime('%b %Y')} to "
                    f"{_avail[-1].strftime('%b %Y')} on this horizon."
                )
    else:
        st.caption("Historical month-end snapshots are not yet available.")

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)

    with st.expander("Pillar weights", expanded=False):
        from cte.config import PILLAR_AXIS, PILLAR_DISPLAY, PILLAR_WEIGHT

        st.caption(
            "Re-weight the pillar aggregation into the two map axes. A value of "
            "zero excludes a pillar. Warnings, commentary, and carry retain the "
            "engine's default weights."
        )

        _wp = [pillar for pillar in PILLAR_WEIGHT if pillar != "F_carry"]

        if st.button("Reset pillar weights", use_container_width=True):
            for _pillar in _wp:
                st.session_state[f"pw_{_pillar}"] = float(PILLAR_WEIGHT[_pillar])

        for _axis, _label in (
            ("axis1_fundamental", "Axis 1: Fundamental trajectory"),
            ("axis2_stretch", "Axis 2: Valuation and policy stretch"),
        ):
            st.markdown(f"**{_label}**")
            for _pillar in [
                pillar for pillar in _wp if PILLAR_AXIS.get(pillar) == _axis
            ]:
                user_w[_pillar] = st.slider(
                    PILLAR_DISPLAY[_pillar],
                    min_value=0.0,
                    max_value=3.0,
                    value=float(PILLAR_WEIGHT[_pillar]),
                    step=0.25,
                    key=f"pw_{_pillar}",
                )

        CUSTOM_W = any(
            abs(user_w[pillar] - PILLAR_WEIGHT[pillar]) > 1e-9
            for pillar in _wp
        )

    if read_cache("macro_backbone") is not None and os.environ.get("FRED_API_KEY"):
        st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Admin")
        if st.button("Rebuild snapshot", use_container_width=True):
            with st.spinner("Recomputing the engine snapshot..."):
                _rebuild()
            st.success("Snapshot refreshed.")

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    with st.expander("Data and refresh", expanded=False):
        st.markdown(
            """
            **Sources:** FRED, OECD, BIS, Eurostat, Japan e-Stat, UK ONS,
            CFTC, Yahoo Finance, and national debt-management offices.

            **Refresh:** The public deployment reads a persisted snapshot refreshed
            by the scheduled workflow. The UI cache expires every ten minutes.

            **Documentation:** Full source mapping is maintained in
            `docs/DATA_SOURCES.md`.
            """
        )


# ============================================================
# Schema guard and custom-weight composition
# ============================================================

if f"axis1_fundamental_{hz}" not in tm.columns:
    st.warning(
        f"The committed snapshot does not include **{horizon}** scores yet. "
        "Run the daily workflow once to reseed the histories and snapshot. "
        "Structural scores are shown for now."
    )
    hz, horizon = "struct", "Structural (~10y)"

flagged = set(warns)

if CUSTOM_W:
    _ph_all = read_cache("pillar_history")
    if _ph_all is not None and len(_ph_all):
        from cte.scoring.compositor import axes_from_pillars

        _keys = ("date", "ccy", "kind")
        _recomposed = axes_from_pillars(
            _ph_all,
            "struct",
            user_w,
            keys=_keys,
        ).merge(
            axes_from_pillars(_ph_all, "regime", user_w, keys=_keys),
            on=list(_keys),
            how="outer",
        )

        if len(_recomposed):
            _hist = _recomposed
            _last = _recomposed[_recomposed.date == _recomposed.date.max()]
            tm = _last.drop(columns=["date", "kind"]).reset_index(drop=True)
    else:
        st.warning(
            "Custom weights require pillar history. Run the daily workflow once "
            "to seed it. Default weighting remains active."
        )
        CUSTOM_W = False


# ============================================================
# Historical frames
# ============================================================

HIST_MODE = asof_sel != "Live" and _hist is not None
_d = pd.Timestamp(asof_sel) if HIST_MODE else None

tm_v = tm
pillars_v = pillars
carry_v = None
pos_v = overlays

if HIST_MODE:
    tm_v = (
        _hist[(_hist.date + pd.offsets.MonthEnd(0)) == _d]
        .sort_values("date")
        .groupby("ccy")
        .tail(1)
    )

    _ph = read_cache("pillar_history")
    if _ph is not None:
        _pillar_rows = (
            _ph[(_ph.date + pd.offsets.MonthEnd(0)) == _d]
            .sort_values("date")
            .groupby(["ccy", "pillar"])
            .tail(1)
        )
        if len(_pillar_rows):
            _value_col = hz if hz in _pillar_rows.columns else "struct"
            pillars_v = (
                _pillar_rows.pivot_table(
                    index="ccy",
                    columns="pillar",
                    values=_value_col,
                )
                .round(2)
                .reset_index()
            )

    _ch = read_cache("carry_history")
    if _ch is not None:
        carry_v = (
            _ch[(_ch.date + pd.offsets.MonthEnd(0)) == _d]
            .sort_values("date")
            .groupby("ccy")
            .tail(1)
            .set_index("ccy")
        )

    from cte.flags.positioning import positioning_asof

    pos_v = positioning_asof(_d)

elif hz != "struct":
    _ph2 = read_cache("pillar_history")
    if _ph2 is not None and hz in getattr(_ph2, "columns", []):
        _latest_pillars = _ph2[_ph2.date == _ph2.date.max()]
        if len(_latest_pillars) and _latest_pillars[hz].notna().any():
            pillars_v = (
                _latest_pillars.pivot_table(
                    index="ccy",
                    columns="pillar",
                    values=hz,
                )
                .round(2)
                .reset_index()
            )

crowded: set[str] = set()
if overlays is not None and "pos_label" in overlays.columns:
    crowded = set(
        overlays.loc[
            overlays.pos_label.astype(str).str.startswith("CROWDED"),
            "ccy",
        ]
    )


# ============================================================
# Main page header and status row
# ============================================================

st.markdown('<div class="hero-kicker">ADFM Macro Systems</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Currency Tension Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-caption">
        Cross-sectional FX regime map separating <strong>fundamental trajectory</strong>
        from <strong>valuation and policy stretch</strong>. Each currency is scored
        against its own history, while pairwise carry and positioning remain visible
        as independent risk conditioners.
    </div>
    """,
    unsafe_allow_html=True,
)

status_1, status_2, status_3, status_4 = st.columns(4)
status_1.metric("Snapshot", _snapshot_label(tm))
status_2.metric("Active horizon", horizon.replace(" (~", " · ").replace(")", ""))
status_3.metric("Map view", "Live" if not HIST_MODE else asof_sel)
status_4.metric("Flagged currencies", str(len(flagged)))

if CUSTOM_W:
    st.info(
        "Custom pillar weights are active for the map, trails, and historical dial. "
        "Warnings, narrative commentary, overlays, and carry retain default weights."
    )

# Sidebar export is rendered after the current view is resolved.
with st.sidebar:
    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Export")
    st.download_button(
        "Download current map CSV",
        data=tm_v.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"currency_tension_map_{asof_sel if HIST_MODE else 'live'}_{hz}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Daily commentary
# ============================================================

from cte.commentary.narrator import load_commentary

_note, _meta = load_commentary()

with st.container(border=True):
    _section_intro(
        "Daily Read",
        "Scheduled narrative generated from the latest persisted engine state.",
    )
    st.markdown(_note or "No daily commentary is available for this snapshot.")

    if _meta.get("generated_at"):
        st.caption(
            f"Generated {_meta['generated_at'][:16]}Z · "
            f"{_meta.get('model', '')} · refreshed once daily by the scheduled job."
        )


# ============================================================
# Map and currency notes
# ============================================================

map_col, notes_col = st.columns([3.15, 1.45], gap="large")

with map_col:
    with st.container(border=True):
        _section_intro(
            "Tension Map",
            "Rightward is improving fundamentals. Upward is greater valuation and "
            "policy stretch. Trails show recent month-end movement.",
        )

        if HIST_MODE:
            st.pyplot(
                tension_map_fig(
                    tm_v,
                    hz,
                    None,
                    history=_hist,
                    trail_months=trail_n,
                    asof_label=asof_sel,
                ),
                use_container_width=True,
            )
            st.caption(
                f"Historical view as of {asof_sel}, computed using today's revised "
                "macro data vintage. Notes and warning rings remain live-only."
            )
        else:
            st.pyplot(
                tension_map_fig(
                    tm,
                    hz,
                    flagged,
                    history=_hist,
                    trail_months=trail_n,
                    crowded=crowded,
                ),
                use_container_width=True,
            )

            if CUSTOM_W:
                _custom_weight_text = ", ".join(
                    f"{key.split('_', 1)[1].title()} {value:g}"
                    for key, value in user_w.items()
                )
                st.caption(
                    f"Custom pillar weights: {_custom_weight_text}. The map and "
                    "trails are recomposed from persisted pillar history."
                )

with notes_col:
    with st.container(border=True):
        _section_intro(
            "Currency Notes",
            "Objective flags covering twin-signal alignment, regime divergence, "
            "crowding, fragile carry, one-legged positions, and overlay adjustments.",
        )

        if not warns:
            st.info("No notable reads in the live snapshot.")
        else:
            ordered_currencies = sorted(warns, key=lambda ccy: (-len(warns[ccy]), ccy))
            for ccy in ordered_currencies:
                notes = warns[ccy]
                with st.expander(
                    f"{ccy} · {len(notes)} note{'s' if len(notes) != 1 else ''}",
                    expanded=(ccy == ordered_currencies[0]),
                ):
                    for note in notes:
                        st.markdown(f"- {note}")


# ============================================================
# Analytical tabs
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

t_pillars, t_carry, t_positioning, t_overlays, t_detail = st.tabs(
    [
        "Pillar Scores",
        "Carry Grid",
        "Positioning",
        "Overlays",
        "Currency Detail",
    ]
)

with t_pillars:
    _section_intro(
        "Pillar Scores",
        "Signed z-scores by currency and pillar, ordered by the selected "
        "fundamental horizon.",
    )

    if HIST_MODE:
        st.caption(f"Historical pillar state as of {asof_sel} ({hz}).")

    if pillars_v is None or pillars_v.empty:
        st.info("No pillar-score data is available for this view.")
    else:
        with st.container(border=True):
            st.pyplot(
                pillar_heatmap_fig(pillars_v, tm_v, hz),
                use_container_width=True,
            )

    st.caption(
        "Green marks the positive pole of each axis: supportive for Axis 1 and "
        "more stretched for Axis 2. Inflation and fiscal readings can be adjusted "
        "by policy-trap and reward-versus-stress overlays."
    )

with t_carry:
    _section_intro(
        "Pairwise Carry",
        "Two-year rate differentials across the full currency matrix. The dollar "
        "is one leg of eight rather than the organizing hub.",
    )

    basis = st.radio(
        "Carry basis",
        ["2Y real", "2Y nominal"],
        horizontal=True,
        key="carry_basis",
    )
    _feature = "real_2y" if basis == "2Y real" else "nominal_2y"
    label = f"{basis.title()} Carry · Base minus Quote (%)"

    if HIST_MODE:
        grid = None
        if carry_v is not None and _feature in carry_v.columns:
            from cte.transform.pairwise import grid_from_values

            grid = grid_from_values(carry_v[_feature])
        st.caption(
            f"Historical carry state as of {asof_sel}."
            if grid is not None
            else "No carry history is available at this date."
        )
    else:
        grid_name = (
            "carry_grid_real" if basis == "2Y real" else "carry_grid_nominal"
        )
        grid = _grid(grid_name)

    if grid is not None and len(grid):
        with st.container(border=True):
            st.pyplot(
                carry_heatmap_fig(grid, label),
                use_container_width=True,
            )
    else:
        st.info("No carry grid is available for the selected basis and date.")

    st.caption(
        "Nominal carry shows the raw rate differential. Real carry strips the "
        "inflation tax and is the cleaner test of fragile or crowded carry."
    )

with t_positioning:
    _section_intro(
        "Speculative Positioning",
        "CFTC Traders-in-Financial-Futures positioning, used strictly as a "
        "crowding and unwind-risk conditioner.",
    )

    if HIST_MODE:
        st.caption(
            f"As of {asof_sel}: the last weekly CFTC report on or before that date, "
            "with its own 13-week path."
        )

    if pos_v is not None and "lev_z" in getattr(pos_v, "columns", []):
        with st.container(border=True):
            st.pyplot(positioning_fig(pos_v), use_container_width=True)

        st.caption(
            "Scope: CFTC TFF futures and options combined, covering the listed "
            "derivatives slice only. It excludes OTC forwards and swaps, corporate "
            "hedging, and real-money cash flows. Net positions are expressed as a "
            "share of open interest and z-scored against each currency's own "
            "ten-year history."
        )

        _positioning_columns = [
            "ccy",
            "pos_label",
            "lev_z",
            "lev_z_13w",
            "am_z",
            "am_z_13w",
            "lev_pct_oi",
            "am_pct_oi",
            "pos_date",
        ]
        _positioning_table = pos_v[
            [
                column
                for column in _positioning_columns
                if column in pos_v.columns
            ]
        ].dropna(subset=["lev_z"])

        st.dataframe(
            _positioning_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No positioning data is available. The TFF cache may be absent or the "
            "overlay may not have been rebuilt yet."
        )

with t_overlays:
    _section_intro(
        "Conditioning Overlays",
        "The state variables that bend or qualify raw signals: yield reward versus "
        "stress, hike feasibility, carry-to-vol crowding, and positioning.",
    )

    overlay_table = None

    if HIST_MODE:
        _overlay_history = read_cache("overlay_history")
        if _overlay_history is not None:
            overlay_table = (
                _overlay_history[
                    (_overlay_history.date + pd.offsets.MonthEnd(0)) == _d
                ]
                .sort_values("date")
                .groupby("ccy")
                .tail(1)
                .set_index("ccy")
                .drop(columns=["date", "kind"], errors="ignore")
            )

        if overlay_table is not None and len(overlay_table):
            st.caption(
                f"Historical conditioning state as of {asof_sel}. Carry-to-vol "
                "percentiles use only the history available through that date."
            )

            _historical_positioning_columns = [
                column
                for column in (
                    "lev_pct_oi",
                    "am_pct_oi",
                    "lev_z",
                    "am_z",
                    "pos_label",
                    "pos_date",
                )
                if pos_v is not None and column in pos_v.columns
            ]

            if _historical_positioning_columns:
                overlay_table = overlay_table.join(
                    pos_v.set_index("ccy")[_historical_positioning_columns],
                    how="left",
                )
        else:
            st.info(
                "No overlay history is available at this date. Run the scheduled "
                "workflow once to seed it."
            )
    else:
        if overlays is not None and not overlays.empty:
            overlay_table = overlays.set_index("ccy")

    if overlay_table is not None and len(overlay_table):
        st.dataframe(overlay_table, use_container_width=True)

    st.caption(
        "`yld_fx_corr` and `yld_regime` show whether markets reward or punish a "
        "currency for higher yields. `feasibility` and `infl_mult` capture hike room "
        "versus policy-trap risk. `ctv_pctile` measures carry-to-vol crowding."
    )

with t_detail:
    _section_intro(
        "Currency Detail",
        "Single-currency readout combining the active map coordinates, quadrant, "
        "pillar profile, and live warning stack.",
    )

    currency_options = list(tm_v["ccy"])
    ccy = st.selectbox("Currency", currency_options, key="currency_detail_select")

    if HIST_MODE:
        st.caption(f"Historical state as of {asof_sel}. Warning notes remain live-only.")

    row = tm_v.set_index("ccy").loc[ccy]
    fundamental = row.get(f"axis1_fundamental_{hz}")
    stretch = row.get(f"axis2_stretch_{hz}")

    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
    detail_1.metric(
        f"{ccy} fundamental",
        f"{fundamental:+.2f}" if pd.notna(fundamental) else "n/a",
    )
    detail_2.metric(
        f"{ccy} stretch",
        f"{stretch:+.2f}" if pd.notna(stretch) else "n/a",
    )
    detail_3.metric("Quadrant", _quadrant_label(fundamental, stretch))
    detail_4.metric("Live notes", str(len(warns.get(ccy, []))))

    pillar_col, warning_col = st.columns([2.2, 1], gap="large")

    with pillar_col:
        with st.container(border=True):
            _section_intro(
                "Pillar Profile",
                "Underlying standardized inputs for the selected currency.",
            )

            if pillars_v is not None and not pillars_v.empty:
                _pillar_table = pillars_v.set_index("ccy")
                if ccy in _pillar_table.index:
                    st.dataframe(
                        _pillar_table.loc[[ccy]],
                        use_container_width=True,
                    )
                else:
                    st.info("No pillar row is available for this currency.")
            else:
                st.info("No pillar-score data is available for this view.")

    with warning_col:
        with st.container(border=True):
            _section_intro(
                "Live Notes",
                "Current objective flags and caveats.",
            )

            if ccy in warns:
                for note in warns[ccy]:
                    st.markdown(f"- {note}")
            else:
                st.info("No live warning notes for this currency.")


# ============================================================
# Methodology
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Methodology and source map", expanded=False):
    st.markdown(
        """
Every currency is scored against **its own history** on the lookbacks selected in the
sidebar. **Structural (~10y)** is the level anchor, **Regime (~2y)** captures recent
movement, and **Secular (~15y)** stretches the anchor across a fuller generation of
cycles. Rate pillars begin later because euro-area yield data starts in 2004 and US
TIPS in 2003. Read structural and regime as a pair. Each input becomes a z-score,
is signed so a positive value points toward the positive pole of its axis, and is
then aggregated into pillars and the two map coordinates.

### Axis 1: Fundamental trajectory

- **Growth (A):** OECD business-confidence level and three-month slope, real GDP
  growth, and the three-month change in unemployment using the freshest available
  FRED, Eurostat, ONS, or OECD source for each currency.
- **Inflation (B):** Headline CPI relative to the central-bank target plus
  three-month inflation momentum. Above-target, accelerating inflation reads as
  hawkish support until the hike-feasibility overlay judges the central bank trapped.
- **External (C):** Current-account balance as a share of GDP and net international
  investment position as a share of GDP. The flow captures current pressure; the
  stock identifies creditor and debtor structures.
- **Real 10Y (D):** Ten-year real yield, using TIPS in the US and nominal yields less
  CPI elsewhere. Its sign is conditioned by whether the currency is rewarded or
  punished as yields rise.

### Axis 2: Valuation and policy stretch

- **Valuation (G):** BIS real effective exchange rate versus the currency's own
  history. This is the direct cheap-versus-expensive read and receives more weight
  than policy.
- **Policy (E):** Real policy rate and the priced path, measured through the two-year
  yield relative to the policy rate. High and restrictive readings imply greater
  stretch.

### Pairwise carry

Carry remains outside the per-currency map because it is inherently relative. The
8 by 8 matrix shows two-year rate differentials between every currency pair in both
nominal and real terms. The dollar is one leg rather than the default hub.

### Conditioning overlays

- **Yield reward versus stress:** Rolling FX-yield correlation determines whether
  higher yields support or undermine the currency.
- **Hike feasibility:** Growth health relative to existing policy restriction tests
  whether inflation support is credible or trapped.
- **Carry-to-vol:** Real carry divided by realized FX volatility and expressed as a
  historical percentile to identify fragile crowding.
- **Speculative positioning:** CFTC TFF leveraged-fund and asset-manager net positions
  as a share of open interest, z-scored against each currency's own history. This is
  used only as a crowding and unwind-risk flag.

### Historical reconstruction

Month-end positions are recomputed through the available history using as-of overlay
multipliers and appended to the persisted record. The dial redraws the map at prior
month-ends. Historical views use today's revised macro-data vintage, which is suitable
for trails and pattern recognition but is not a point-in-time backtest without further
publication-lag adjustments.

*Sources: FRED, OECD, BIS, Eurostat, Japan e-Stat, UK ONS, CFTC, Yahoo Finance, and
national debt offices. Full mapping is maintained in `docs/DATA_SOURCES.md`.*
        """
    )

st.markdown(
    '<div class="quiet-note">Currency Tension Engine · persisted snapshot architecture · analytical signals are descriptive, not trade instructions.</div>',
    unsafe_allow_html=True,
)
