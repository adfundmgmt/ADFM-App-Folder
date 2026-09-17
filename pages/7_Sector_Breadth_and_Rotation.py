"""ADFM Sector Breadth and Rotation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.sector_rotation import (
    LABEL_MODES,
    STATE_STREET_SECTOR_ETFS,
    UNIVERSE_SCOPES,
    RotationWindow,
    adaptive_axis_range,
    attach_breadth,
    build_asset_levels,
    build_breadth_member_map,
    build_catalog,
    compute_snapshot,
    download_prices,
    fetch_state_street_holdings,
    price_diagnostics,
    relative_series,
    required_tickers,
    select_catalog,
    trail_for_selected,
)
from adfm_core.ui import (
    PageHeader,
    render_footer,
    render_page_header,
    render_section_header,
    render_sidebar_about,
)
from adfm_sector_rotation_config import (
    SECTOR_GROUP_COLORS,
    STATE_COLORS,
    TRAIL_OPTIONS,
    WINDOW_PRESETS,
)

st.set_page_config(page_title="Sector Breadth and Rotation", layout="wide")
render_page_header(
    PageHeader(
        title="Sector Breadth and Rotation",
        description=(
            "Cross-sectional equity leadership, constituent breadth, and rotation "
            "across sectors, industries, themes, and countries."
        ),
        eyebrow="ADFM Equity Leadership",
    )
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return download_prices(list(tickers))


@st.cache_data(ttl=21600, show_spinner=False)
def cached_state_street_holdings(ticker: str) -> List[str]:
    try:
        return fetch_state_street_holdings(ticker)
    except Exception:
        return []


def load_sector_holdings(tickers: List[str]) -> Dict[str, List[str]]:
    if not tickers:
        return {}
    holdings: Dict[str, List[str]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(tickers))) as pool:
        futures = {pool.submit(cached_state_street_holdings, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                members = future.result()
            except Exception:
                members = []
            if members:
                holdings[ticker] = members
    return holdings


def rotation_window(choice: str) -> RotationWindow:
    cfg = WINDOW_PRESETS[choice]
    return RotationWindow(cfg["short"], cfg["long"], cfg["label_short"], cfg["label_long"])


def make_rs_chart(raw_prices, asset_levels, catalog, selected_key) -> go.Figure:
    row = catalog.loc[catalog["Key"] == selected_key].iloc[0]
    asset = asset_levels[selected_key]
    broad_name = row["Broad Benchmark"]
    rel = relative_series(asset, raw_prices[broad_name]).dropna()
    fig = go.Figure()
    if not rel.empty:
        normalized = rel / rel.iloc[0] * 100.0
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized.values, mode="lines", name=f"vs {broad_name}",
            line=dict(width=2.3, color="#285f9e"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))
    parent_name = row["Parent Benchmark"]
    if parent_name and parent_name in raw_prices.columns:
        parent_rel = relative_series(asset, raw_prices[parent_name]).dropna()
        if not parent_rel.empty:
            parent_norm = parent_rel / parent_rel.iloc[0] * 100.0
            fig.add_trace(go.Scatter(
                x=parent_norm.index, y=parent_norm.values, mode="lines", name=f"vs {parent_name}",
                line=dict(width=1.7, dash="dash", color="#64748b"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ))
    fig.add_hline(y=100, line_width=1, line_dash="dot", opacity=0.35)
    fig.update_layout(
        height=410, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="",
        yaxis_title="Relative strength index (start = 100)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def label_keys(snapshot, mode, top_n, selected_key):
    if mode == "No labels":
        return set()
    if mode == "All tickers":
        return set(snapshot["Key"])
    if mode == "Top ranked only":
        return set(snapshot.nsmallest(top_n, "Rank")["Key"])
    return {selected_key}


def make_rotation_map(snapshot, selected_key, trail, window, label_mode, label_top_n) -> go.Figure:
    fig = go.Figure()
    plot_df = snapshot.dropna(subset=["Map X", "Map Y"]).copy()
    labels = label_keys(plot_df, label_mode, label_top_n, selected_key)
    if not trail.empty:
        selected_row = plot_df.loc[plot_df["Key"] == selected_key]
        trail_color = "#111827"
        if not selected_row.empty:
            trail_color = SECTOR_GROUP_COLORS.get(selected_row.iloc[0]["Sector Group"], trail_color)
        fig.add_trace(go.Scatter(
            x=trail["x"], y=trail["y"], mode="lines+markers",
            line=dict(width=1.8, color=trail_color), marker=dict(size=5, color=trail_color),
            opacity=0.65, name="Selected trail",
            hovertemplate="%{x:.2%}, %{y:.2%}<extra>Selected trail</extra>",
        ))
    for _, row in plot_df.iterrows():
        is_selected = row["Key"] == selected_key
        color = SECTOR_GROUP_COLORS.get(row["Sector Group"], "#94a3b8")
        border = STATE_COLORS.get(row["State"], "#334155")
        symbol = row["Ticker"] if row["Ticker"] else "Basket"
        text = symbol if row["Key"] in labels else ""
        fig.add_trace(go.Scatter(
            x=[row["Map X"]], y=[row["Map Y"]], mode="markers+text" if text else "markers",
            text=[text] if text else None, textposition="top center",
            marker=dict(
                size=20 if is_selected else 13, color=color, opacity=0.95 if is_selected else 0.82,
                line=dict(width=3 if is_selected else 1.5, color=border),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Name']}</b><br>State: {row['State']} ({int(row['Days in State'])}d)<br>"
                f"1M rel: {row['1M Rel']:.2%}<br>3M rel: {row['3M Rel']:.2%}<br>"
                f"5D movement: {row['Movement Speed']:.2%}<br>Angle: {row['Movement Angle']:.0f}°<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_width=1, opacity=0.35)
    fig.add_vline(x=0, line_width=1, opacity=0.35)
    x_range = adaptive_axis_range(plot_df["Map X"], trail["x"] if not trail.empty else None)
    y_range = adaptive_axis_range(plot_df["Map Y"], trail["y"] if not trail.empty else None)
    x_low, x_high = x_range
    y_low, y_high = y_range
    fig.add_annotation(x=x_high, y=y_high, text="Leading", showarrow=False, xanchor="right", yanchor="top")
    fig.add_annotation(x=x_low, y=y_high, text="Improving", showarrow=False, xanchor="left", yanchor="top")
    fig.add_annotation(x=x_low, y=y_low, text="Lagging", showarrow=False, xanchor="left", yanchor="bottom")
    fig.add_annotation(x=x_high, y=y_low, text="Weakening", showarrow=False, xanchor="right", yanchor="bottom")
    fig.update_layout(
        height=610, margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(title=f"{window.long_label} relative rotation", tickformat=".1%", range=list(x_range), zeroline=False),
        yaxis=dict(title=f"{window.short_label} relative rotation", tickformat=".1%", range=list(y_range), zeroline=False),
    )
    return fig


def selected_rows(event) -> List[int]:
    try:
        return list(event.selection.rows)
    except Exception:
        try:
            return list(event.get("selection", {}).get("rows", []))
        except Exception:
            return []


catalog = build_catalog()
with st.sidebar:
    render_sidebar_about("7_Sector_Breadth_and_Rotation.py")
    st.header("Settings")
    universe = st.selectbox(
        "Universe", options=UNIVERSE_SCOPES, index=UNIVERSE_SCOPES.index("Industries"),
        help="Sectors, granular industries, cross-sector themes, country ETFs, or the full map.",
    )
    universe_catalog = select_catalog(catalog, universe)
    group_options = universe_catalog["Sector Group"].drop_duplicates().tolist()
    selected_groups = st.multiselect("Sector group filter", options=group_options, default=group_options)
    window_choice = st.selectbox("Rotation window", options=list(WINDOW_PRESETS.keys()), index=0)
    trail_choice = st.selectbox("Selected rotation trail", options=list(TRAIL_OPTIONS.keys()), index=1)
    label_mode = st.selectbox("Labels on map", options=LABEL_MODES, index=0)
    label_top_n = st.slider("Label top N", 5, 50, 20, 5, disabled=label_mode != "Top ranked only")
    show_diagnostics = st.checkbox("Show data diagnostics", value=False)

selected_catalog = select_catalog(catalog, universe, selected_groups)
if selected_catalog.empty:
    st.warning("No groups selected.")
    st.stop()

window = rotation_window(window_choice)
sector_etfs = [
    ticker for ticker in selected_catalog.loc[selected_catalog["Universe"] == "Sectors", "Ticker"].tolist()
    if ticker in STATE_STREET_SECTOR_ETFS
]
sector_holdings = load_sector_holdings(sector_etfs)
breadth_members = build_breadth_member_map(selected_catalog, sector_holdings)
tickers = required_tickers(selected_catalog, breadth_members)
with st.spinner("Loading market data..."):
    raw_prices = cached_prices(tuple(tickers))
if raw_prices.empty:
    st.error("No market data returned for the selected universe.")
    st.stop()
required_benchmarks = sorted(set(selected_catalog["Broad Benchmark"]) - {""})
missing_benchmarks = [b for b in required_benchmarks if b not in raw_prices or not raw_prices[b].notna().any()]
if missing_benchmarks:
    st.error("Required benchmark data unavailable: " + ", ".join(missing_benchmarks))
    st.stop()

asset_levels = build_asset_levels(raw_prices, selected_catalog)
snapshot = compute_snapshot(raw_prices, asset_levels, selected_catalog, window)
snapshot = attach_breadth(snapshot, raw_prices, breadth_members)
valid_snapshot = snapshot[snapshot["1M Rel"].notna()].copy()
if valid_snapshot.empty:
    st.error("No selected exposures have enough current history for the rotation analysis.")
    st.stop()
valid_snapshot = valid_snapshot.sort_values(["Rank", "Name"]).reset_index(drop=True)
latest_dt = pd.to_datetime(raw_prices.index.max()).date()
breadth_coverage = int(valid_snapshot["% > 50D"].notna().sum())
st.caption(
    f"Data through {latest_dt} | {len(valid_snapshot)}/{len(selected_catalog)} exposures available | "
    f"Constituent breadth available for {breadth_coverage}/{len(valid_snapshot)}"
)

render_section_header(
    "Rotation snapshot",
    "Ranked by 1-month performance relative to each exposure's broad benchmark; click any row for detail.",
)
table_columns = [
    "Key", "Rank", "Ticker", "Industry", "Sector Group", "Kind", "State", "Days in State",
    "1W Rel", "1M Rel", "3M Rel", "Weekly Rel Δ", "Rank Δ", "1M Abs", "Parent 1M Rel",
    "% > 50D", "% > 200D", "Breadth 1M Δ", "vs 50D", "DD from 52W High",
]
table = valid_snapshot[table_columns].copy()
table["Ticker"] = table["Ticker"].replace("", "Basket")
percent_columns = [
    "1W Rel", "1M Rel", "3M Rel", "Weekly Rel Δ", "1M Abs", "Parent 1M Rel",
    "% > 50D", "% > 200D", "Breadth 1M Δ", "vs 50D", "DD from 52W High",
]
for column in percent_columns:
    table[column] = table[column] * 100.0
column_config = {
    "Key": None,
    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
    "Days in State": st.column_config.NumberColumn("Days", format="%d"),
    "Rank Δ": st.column_config.NumberColumn("Rank Δ", format="%+.0f"),
}
for column in percent_columns:
    column_config[column] = st.column_config.NumberColumn(column, format="%+.1f%%")
selection_event = st.dataframe(
    table, use_container_width=True, hide_index=True,
    height=min(780, max(240, 35 * (len(table) + 1))), column_config=column_config,
    on_select="rerun", selection_mode="single-row", key="rotation_snapshot_table",
)
rows = selected_rows(selection_event)
selected_row_index = rows[0] if rows and rows[0] < len(valid_snapshot) else 0
selected_key = str(valid_snapshot.iloc[selected_row_index]["Key"])
selected_meta = selected_catalog.loc[selected_catalog["Key"] == selected_key].iloc[0]

render_section_header(
    "Relative strength",
    f"{selected_meta['Name']} versus {selected_meta['Broad Benchmark']}"
    + (f" and parent {selected_meta['Parent Benchmark']}" if selected_meta["Parent Benchmark"] else ""),
)
st.plotly_chart(
    make_rs_chart(raw_prices, asset_levels, selected_catalog, selected_key),
    use_container_width=True, config={"displayModeBar": False, "responsive": True},
)

render_section_header("Rotation map", f"{window.short_label} versus {window.long_label} benchmark-relative rotation")
selected_trail = trail_for_selected(
    raw_prices, asset_levels, selected_catalog, selected_key, window, TRAIL_OPTIONS[trail_choice]
)
rotation_fig = make_rotation_map(valid_snapshot, selected_key, selected_trail, window, label_mode, label_top_n)
st.plotly_chart(rotation_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

if show_diagnostics:
    with st.expander("Data diagnostics", expanded=True):
        st.dataframe(price_diagnostics(raw_prices, tickers), use_container_width=True, hide_index=True)
        if sector_etfs:
            st.dataframe(pd.DataFrame({
                "Sector ETF": sector_etfs,
                "Constituent holdings": [len(sector_holdings.get(ticker, [])) for ticker in sector_etfs],
            }), use_container_width=True, hide_index=True)

csv = valid_snapshot.drop(columns=["Movement ΔX", "Movement ΔY"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "Download rotation snapshot", data=csv,
    file_name="adfm_sector_rotation_snapshot.csv", mime="text/csv",
)
render_footer()
