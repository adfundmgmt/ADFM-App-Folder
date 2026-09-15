"""Compact Currency Tension Engine Streamlit page.

The scoring engine remains unchanged. This page keeps the decision surface focused on
the cross-sectional tension map and a sortable ranking table, with deeper diagnostics
available on demand.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import streamlit as st

from adfm_core.ui import (
    PageHeader,
    inject_institutional_tool_finish,
    render_footer,
    render_page_header,
    render_sidebar_about,
)
from cte.adapters.base import read_cache
from cte.config import CACHE_DIR
from cte.dashboard.plots import pillar_heatmap_fig, tension_map_fig

st.set_page_config(
    page_title="Currency Tension Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1480px;
    }
    .cte-status {
        margin: -0.25rem 0 0.45rem;
        color: #555555;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.76rem;
        line-height: 1.45;
    }
    .cte-section {
        margin: 0.45rem 0 0.35rem;
        color: #171717;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .cte-note {
        color: #666666;
        font-size: 0.76rem;
        line-height: 1.45;
    }
    div[data-testid="stDataFrame"] {
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_institutional_tool_finish()


@st.cache_data(show_spinner=False, ttl=600)
def _load() -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    dict[str, list[str]],
]:
    tension_map = read_cache("tension_map")
    pillar_scores = read_cache("pillar_scores")
    overlay_data = read_cache("overlays")
    warnings_path = CACHE_DIR / "warnings.json"
    warning_data = (
        json.loads(warnings_path.read_text(encoding="utf-8"))
        if warnings_path.exists()
        else {}
    )
    return tension_map, pillar_scores, overlay_data, warning_data


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


def _snapshot_generated_at() -> Optional[pd.Timestamp]:
    meta_path = CACHE_DIR / "commentary_meta.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = payload.get("generated_at")
            if raw:
                stamp = pd.Timestamp(raw)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                return stamp
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    history = read_cache("snapshot_history")
    if history is not None and not history.empty and "date" in history.columns:
        dates = pd.to_datetime(history["date"], errors="coerce", utc=True).dropna()
        if not dates.empty:
            return dates.max()
    return None


def _snapshot_label() -> str:
    stamp = _snapshot_generated_at()
    if stamp is None:
        return "Update time unavailable"
    eastern = stamp.tz_convert("America/New_York")
    return eastern.strftime("%b %d, %Y · %-I:%M %p ET")


def _historical_overlay(asof: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    if asof is None:
        return overlays
    history = read_cache("overlay_history")
    if history is None or history.empty or "date" not in history.columns:
        return None
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    rows = (
        history[history["date"] <= asof]
        .sort_values("date")
        .groupby("ccy")
        .tail(1)
    )
    return rows if not rows.empty else None


def _positioning_asof(asof: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    if asof is None:
        return overlays
    try:
        from cte.flags.positioning import positioning_asof

        return positioning_asof(asof)
    except Exception:
        return None


def _ranking_table(
    frame: pd.DataFrame,
    horizon_key: str,
    overlay_frame: Optional[pd.DataFrame],
    positioning_frame: Optional[pd.DataFrame],
    warning_map: dict[str, list[str]],
    historical: bool,
) -> pd.DataFrame:
    xcol = f"axis1_fundamental_{horizon_key}"
    ycol = f"axis2_stretch_{horizon_key}"
    table = frame[["ccy", xcol, ycol]].copy()
    table = table.rename(
        columns={
            "ccy": "FX",
            xcol: "Fundamental",
            ycol: "Stretch",
        }
    )
    table["Quadrant"] = [
        _quadrant_label(f, s)
        for f, s in zip(table["Fundamental"], table["Stretch"], strict=False)
    ]

    if overlay_frame is not None and not overlay_frame.empty:
        ov = overlay_frame.copy()
        if "ccy" in ov.columns:
            ov = ov.set_index("ccy")
        if "ctv_pctile" in ov.columns:
            table["Carry/Vol %ile"] = table["FX"].map(ov["ctv_pctile"])

    if positioning_frame is not None and not positioning_frame.empty:
        pos = positioning_frame.copy()
        if "ccy" in pos.columns:
            pos = pos.set_index("ccy")
        if "lev_z" in pos.columns:
            table["Lev Funds Z"] = table["FX"].map(pos["lev_z"])

    table["Flags"] = (
        pd.NA
        if historical
        else table["FX"].map(lambda ccy: len(warning_map.get(ccy, []))).astype("Int64")
    )

    table["Fundamental"] = pd.to_numeric(table["Fundamental"], errors="coerce").round(2)
    table["Stretch"] = pd.to_numeric(table["Stretch"], errors="coerce").round(2)
    if "Carry/Vol %ile" in table.columns:
        table["Carry/Vol %ile"] = pd.to_numeric(
            table["Carry/Vol %ile"], errors="coerce"
        ).round(0)
    if "Lev Funds Z" in table.columns:
        table["Lev Funds Z"] = pd.to_numeric(
            table["Lev Funds Z"], errors="coerce"
        ).round(2)

    return table.sort_values(
        ["Fundamental", "Stretch"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


tm, pillars, overlays, warns = _load()

if tm is None or tm.empty:
    st.warning("No currency snapshot is available in cache.")
    st.stop()

_hist = read_cache("snapshot_history")

with st.sidebar:
    render_sidebar_about("6_Currency_Tension_Engine.py")

    horizon = st.radio(
        "Horizon",
        ["Regime (~2y)", "Structural (~10y)", "Secular (~15y)"],
        index=1,
    )
    hz = {"Re": "regime", "St": "struct", "Se": "secular"}[horizon[:2]]

    asof_sel = "Live"
    available_dates: list[pd.Timestamp] = []
    if _hist is not None and not _hist.empty:
        from cte.scoring.history import dial_options

        available_dates = list(dial_options(_hist, hz))

    view = st.radio(
        "View",
        ["Live", "Historical"],
        horizontal=True,
        disabled=not bool(available_dates),
    )
    if view == "Historical" and available_dates:
        selected_date = st.selectbox(
            "Month-end",
            list(reversed(available_dates)),
            format_func=lambda d: d.strftime("%b %Y"),
        )
        asof_sel = selected_date.strftime("%Y-%m-%d")

    trail_n = 6
    custom_weights = False
    user_w: dict[str, float] = {}

    with st.expander("Advanced", expanded=False):
        trail_n = st.slider(
            "Trail length, month-ends",
            min_value=0,
            max_value=12,
            value=6,
        )

        from cte.config import PILLAR_AXIS, PILLAR_DISPLAY, PILLAR_WEIGHT

        st.caption("Optional map re-weighting. Default engine weights remain the baseline.")
        weight_pillars = [p for p in PILLAR_WEIGHT if p != "F_carry"]

        if st.button("Reset pillar weights", use_container_width=True):
            for pillar in weight_pillars:
                st.session_state[f"pw_{pillar}"] = float(PILLAR_WEIGHT[pillar])

        for axis, label in (
            ("axis1_fundamental", "Fundamental trajectory"),
            ("axis2_stretch", "Valuation + policy stretch"),
        ):
            st.markdown(f"**{label}**")
            for pillar in [p for p in weight_pillars if PILLAR_AXIS.get(p) == axis]:
                user_w[pillar] = st.slider(
                    PILLAR_DISPLAY[pillar],
                    min_value=0.0,
                    max_value=3.0,
                    value=float(PILLAR_WEIGHT[pillar]),
                    step=0.25,
                    key=f"pw_{pillar}",
                )

        custom_weights = any(
            abs(user_w[p] - PILLAR_WEIGHT[p]) > 1e-9 for p in weight_pillars
        )


if f"axis1_fundamental_{hz}" not in tm.columns:
    st.warning(f"{horizon} scores are unavailable; showing Structural (~10y).")
    hz, horizon = "struct", "Structural (~10y)"

flagged = set(warns)

if custom_weights:
    pillar_history = read_cache("pillar_history")
    if pillar_history is not None and not pillar_history.empty:
        from cte.scoring.compositor import axes_from_pillars

        keys = ("date", "ccy", "kind")
        recomposed = None
        for key in ("struct", "regime", "secular"):
            part = axes_from_pillars(
                pillar_history,
                key,
                user_w,
                keys=keys,
            )
            recomposed = (
                part
                if recomposed is None
                else recomposed.merge(part, on=list(keys), how="outer")
            )

        if recomposed is not None and not recomposed.empty:
            _hist = recomposed
            latest = recomposed[recomposed["date"] == recomposed["date"].max()]
            tm = latest.drop(columns=["date", "kind"], errors="ignore").reset_index(
                drop=True
            )
    else:
        st.warning("Custom weights require pillar history; default weights remain active.")
        custom_weights = False


hist_mode = asof_sel != "Live" and _hist is not None and not _hist.empty
asof_date = pd.Timestamp(asof_sel) if hist_mode else None

tm_v = tm
pillars_v = pillars

if hist_mode and asof_date is not None:
    history = _hist.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    tm_v = (
        history[(history["date"] + pd.offsets.MonthEnd(0)) == asof_date]
        .sort_values("date")
        .groupby("ccy")
        .tail(1)
    )

    pillar_history = read_cache("pillar_history")
    if pillar_history is not None and not pillar_history.empty:
        ph = pillar_history.copy()
        ph["date"] = pd.to_datetime(ph["date"], errors="coerce")
        rows = (
            ph[(ph["date"] + pd.offsets.MonthEnd(0)) == asof_date]
            .sort_values("date")
            .groupby(["ccy", "pillar"])
            .tail(1)
        )
        if not rows.empty:
            value_col = hz if hz in rows.columns else "struct"
            pillars_v = (
                rows.pivot_table(
                    index="ccy",
                    columns="pillar",
                    values=value_col,
                )
                .round(2)
                .reset_index()
            )
elif hz != "struct":
    pillar_history = read_cache("pillar_history")
    if (
        pillar_history is not None
        and not pillar_history.empty
        and hz in pillar_history.columns
    ):
        ph = pillar_history.copy()
        ph["date"] = pd.to_datetime(ph["date"], errors="coerce")
        latest = ph[ph["date"] == ph["date"].max()]
        if not latest.empty and latest[hz].notna().any():
            pillars_v = (
                latest.pivot_table(index="ccy", columns="pillar", values=hz)
                .round(2)
                .reset_index()
            )

overlay_v = _historical_overlay(asof_date)
positioning_v = _positioning_asof(asof_date)

crowded: set[str] = set()
if positioning_v is not None and "pos_label" in positioning_v.columns:
    crowded = set(
        positioning_v.loc[
            positioning_v["pos_label"].astype(str).str.startswith("CROWDED"),
            "ccy",
        ]
    )

render_page_header(
    PageHeader(
        title="Currency Tension Engine",
        description="FX regime map: fundamental trajectory versus valuation and policy stretch.",
        eyebrow="ADFM Macro Regime",
    )
)

st.markdown(
    """
    <style>
    .adfm-page-header {
        margin-bottom: .45rem !important;
        padding: .55rem 0 .45rem !important;
    }
    .adfm-page-title {
        font-size: clamp(1.75rem, 2.5vw, 2.2rem) !important;
    }
    .adfm-page-description {
        margin-top: .3rem !important;
        font-size: .78rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

view_label = asof_sel if hist_mode else "Live"
status_bits = [horizon.replace(" (~", " · ").replace(")", ""), view_label]
if not hist_mode:
    status_bits.insert(1, f"Updated {_snapshot_label()}")
if custom_weights:
    status_bits.append("Custom weights")
st.markdown(
    f'<div class="cte-status">{" &nbsp;|&nbsp; ".join(status_bits)}</div>',
    unsafe_allow_html=True,
)

map_frame = tm_v if hist_mode else tm
map_flags = None if hist_mode else flagged
map_history = _hist if _hist is not None else None

fig = tension_map_fig(
    map_frame,
    hz,
    map_flags,
    history=map_history,
    trail_months=trail_n,
    crowded=crowded,
    asof_label=asof_sel if hist_mode else None,
)
fig.set_size_inches(8.4, 5.4, forward=True)
st.pyplot(fig, use_container_width=True)

ranking = _ranking_table(
    tm_v,
    hz,
    overlay_v,
    positioning_v,
    warns,
    hist_mode,
)

st.markdown('<div class="cte-section">Cross-Currency Ranking</div>', unsafe_allow_html=True)
st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True,
    height=316,
    column_config={
        "FX": st.column_config.TextColumn("FX", width="small"),
        "Fundamental": st.column_config.NumberColumn(
            "Fundamental", format="%+.2f", width="small"
        ),
        "Stretch": st.column_config.NumberColumn(
            "Stretch", format="%+.2f", width="small"
        ),
        "Quadrant": st.column_config.TextColumn("Quadrant", width="medium"),
        "Carry/Vol %ile": st.column_config.NumberColumn(
            "Carry/Vol %ile", format="%.0f", width="small"
        ),
        "Lev Funds Z": st.column_config.NumberColumn(
            "Lev Funds Z", format="%+.2f", width="small"
        ),
        "Flags": st.column_config.NumberColumn("Flags", format="%d", width="small"),
    },
)
st.caption(
    "Click any column header to rank. Carry/Vol is the engine's historical percentile; "
    "Leveraged Funds Z is CFTC TFF positioning. Historical views suppress live flag counts."
)

with st.expander("Diagnostics", expanded=False):
    tab_pillars, tab_detail, tab_daily = st.tabs(
        ["Pillar Scores", "Currency Detail", "Daily Read"]
    )

    with tab_pillars:
        if pillars_v is None or pillars_v.empty:
            st.info("No pillar-score data is available for this view.")
        else:
            st.pyplot(
                pillar_heatmap_fig(pillars_v, tm_v, hz),
                use_container_width=True,
            )
        st.caption(
            "Positive values point toward the positive pole of the relevant axis. "
            "The map remains the primary decision surface."
        )

    with tab_detail:
        currency_options = list(tm_v["ccy"])
        ccy = st.selectbox("Currency", currency_options, key="currency_detail_select")
        detail = ranking[ranking["FX"] == ccy]
        if not detail.empty:
            st.dataframe(detail, use_container_width=True, hide_index=True)

        if pillars_v is not None and not pillars_v.empty:
            pillar_table = pillars_v.set_index("ccy")
            if ccy in pillar_table.index:
                profile = pillar_table.loc[[ccy]].T.reset_index()
                profile.columns = ["Pillar", "Score"]
                st.dataframe(profile, use_container_width=True, hide_index=True)

        notes = warns.get(ccy, []) if not hist_mode else []
        if notes:
            st.markdown("**Live flags**")
            for note in notes:
                st.markdown(f"- {note}")
        elif hist_mode:
            st.caption("Live warning notes are intentionally hidden in historical mode.")
        else:
            st.caption("No live warning notes for this currency.")

    with tab_daily:
        from cte.commentary.narrator import load_commentary

        note, meta = load_commentary()
        st.markdown(note or "No daily commentary is available for this snapshot.")
        if meta.get("generated_at"):
            stamp = pd.Timestamp(meta["generated_at"])
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            stamp = stamp.tz_convert("America/New_York")
            st.caption(
                f"Generated {stamp.strftime('%b %d, %Y · %-I:%M %p ET')} "
                f"· {meta.get('model', '')}"
            )

with st.sidebar:
    with st.expander("Export", expanded=False):
        st.download_button(
            "Download current map CSV",
            data=tm_v.to_csv(index=False).encode("utf-8"),
            file_name=f"currency_tension_map_{asof_sel if hist_mode else 'live'}_{hz}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown(
    '<div class="cte-note">Sources: FRED, OECD, BIS, Eurostat, Japan e-Stat, UK ONS, '
    'CFTC, Yahoo Finance and national debt-management offices. Signals are descriptive, '
    'not trade instructions.</div>',
    unsafe_allow_html=True,
)

render_footer()
