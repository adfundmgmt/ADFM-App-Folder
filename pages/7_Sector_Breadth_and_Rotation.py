from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from adfm_core.palette import PASTEL_20
from adfm_core.sector_rotation import (
    build_catalog,
    compute_breadth,
    compute_relative_metrics,
    extension_metrics,
    movement_from_coordinates,
    rotation_history,
    synthetic_equal_weight_level,
)
from adfm_core.sector_rotation_holdings import (
    parse_spdr_holdings_table,
    xlsx_first_sheet_to_frame,
)
from adfm_core.sector_rotation_ui import (
    STATE_EDGE,
    STATE_PALETTE,
    display_name,
    full_extent_axis_range,
    select_auto_labels,
    style_rotation_table,
)
from adfm_core.ui import PageHeader, render_footer, render_page_header, render_section_header, render_sidebar_about
from adfm_sector_rotation_config import BENCHMARKS, DOWNLOAD_CHUNK_SIZE, DOWNLOAD_RETRIES

st.set_page_config(page_title="Sector Breadth and Rotation", layout="wide")

render_page_header(
    PageHeader(
        title="Sector Breadth and Rotation",
        description="Sector, industry, thematic and global equity rotation with transparent breadth and state persistence.",
        eyebrow="ADFM Equity Leadership",
    )
)

WINDOWS = {
    "Fast | 1M vs 3M": (21, 63),
    "Intermediate | 3M vs 6M": (63, 126),
    "Trend | 6M vs 12M": (126, 252),
}
SPDR_SECTOR_ETFS = {"XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"}
SPDR_HOLDINGS_URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"


@st.cache_data(ttl=3600, show_spinner=False)
def _download_batch(tickers: Tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            raw = yf.download(
                list(tickers), period="3y", interval="1d", auto_adjust=False,
                progress=False, group_by="column", threads=True,
            )
            if raw is None or raw.empty:
                raise ValueError("empty download")
            out: Dict[str, pd.Series] = {}
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in tickers:
                    for field in ("Adj Close", "Close"):
                        for key in ((field, ticker), (ticker, field)):
                            if key in raw.columns:
                                series = pd.to_numeric(raw[key], errors="coerce")
                                if series.notna().any():
                                    out[ticker] = series
                                    break
                        if ticker in out:
                            break
            elif len(tickers) == 1:
                ticker = tickers[0]
                for field in ("Adj Close", "Close"):
                    if field in raw.columns:
                        out[ticker] = pd.to_numeric(raw[field], errors="coerce")
                        break
            frame = pd.DataFrame(out)
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            return frame.sort_index()
        except Exception:
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...]) -> pd.DataFrame:
    unique = list(dict.fromkeys(ticker for ticker in tickers if ticker and not ticker.startswith("BASKET_")))
    pieces: List[pd.DataFrame] = []
    for i in range(0, len(unique), DOWNLOAD_CHUNK_SIZE):
        part = _download_batch(tuple(unique[i:i + DOWNLOAD_CHUNK_SIZE]))
        if not part.empty:
            pieces.append(part)
    prices = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    missing_tickers = [ticker for ticker in unique if ticker not in prices.columns or not prices[ticker].notna().any()]
    fallback_tickers = [ticker for ticker in missing_tickers if ticker in BENCHMARKS]
    for ticker in fallback_tickers:
        part = _download_batch((ticker,))
        if not part.empty:
            prices = pd.concat([prices, part], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    return prices.sort_index() if not prices.empty else prices


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_spdr_holdings(ticker: str) -> List[str]:
    if ticker not in SPDR_SECTOR_ETFS:
        return []
    try:
        response = requests.get(
            SPDR_HOLDINGS_URL.format(ticker=ticker.lower()),
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        frame = xlsx_first_sheet_to_frame(response.content)
        return parse_spdr_holdings_table(frame)
    except Exception:
        return []


def asset_series(row: pd.Series, prices: pd.DataFrame) -> pd.Series:
    members = row["Members"]
    if isinstance(members, tuple):
        cols = [ticker for ticker in members if ticker in prices.columns]
        return synthetic_equal_weight_level(prices[cols]) if cols else pd.Series(dtype=float)
    ticker = row["Ticker"]
    return pd.to_numeric(prices[ticker], errors="coerce") if ticker in prices.columns else pd.Series(dtype=float)


def rel_return_series(asset: pd.Series, benchmark: pd.Series, periods: int) -> pd.Series:
    pair = pd.concat([asset.rename("a"), benchmark.rename("b")], axis=1).dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    ratio = pair["a"] / pair["b"]
    return ratio.pct_change(periods, fill_method=None)


def _rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def _selection_rows(key: str, row_count: int) -> List[int]:
    state = st.session_state.get(key)
    rows: List[int] = []
    try:
        if hasattr(state, "selection"):
            rows = list(state.selection.rows)
        elif isinstance(state, dict):
            selection = state.get("selection", {})
            rows = list(selection.get("rows", [])) if isinstance(selection, dict) else []
    except Exception:
        rows = []
    return [int(index) for index in rows if isinstance(index, (int, np.integer)) and 0 <= int(index) < row_count]


catalog = build_catalog()
with st.sidebar:
    render_sidebar_about("7_Sector_Breadth_and_Rotation.py")
    st.header("Settings")
    scope = st.selectbox("Universe", ["Sectors", "Industries", "Themes", "Countries", "All"], index=1)
    window_label = st.selectbox("Rotation window", list(WINDOWS), index=0)
    trail_sessions = st.selectbox("Tail length", [3, 5, 8, 12], index=1, format_func=lambda value: f"{value} sessions")

if scope != "All":
    view = catalog[catalog["Universe"] == scope].copy()
else:
    view = catalog.copy()

with st.sidebar:
    group_options = view["Sector Group"].drop_duplicates().tolist()
    selected_groups = st.multiselect("Group filter", group_options, default=group_options)
view = view[view["Sector Group"].isin(selected_groups)].copy()

if view.empty:
    st.info("No groups selected.")
    st.stop()

market_tickers = set(view.loc[view["Kind"] == "ETF", "Ticker"])
market_tickers.update(view["Benchmark"])
market_tickers.update(view["Parent"])
for members in view["Members"].dropna():
    market_tickers.update(members)

with st.spinner("Loading market data..."):
    prices = fetch_prices(tuple(sorted(market_tickers)))

if prices.empty:
    st.error("No market data returned.")
    st.stop()

sector_members: Dict[str, List[str]] = {}
sector_member_tickers: set[str] = set()
if scope in {"Sectors", "All"}:
    for ticker in view.loc[view["Ticker"].isin(SPDR_SECTOR_ETFS), "Ticker"]:
        members = fetch_spdr_holdings(ticker)
        if members:
            sector_members[ticker] = members
            sector_member_tickers.update(members)
if sector_member_tickers:
    member_prices = fetch_prices(tuple(sorted(sector_member_tickers)))
    for col in member_prices.columns:
        if col not in prices.columns:
            prices[col] = member_prices[col]

short_window, long_window = WINDOWS[window_label]
series_by_id: Dict[str, pd.Series] = {}
rows: List[Dict[str, object]] = []
rank_hist: Dict[str, pd.Series] = {}
rotation_by_id: Dict[str, pd.DataFrame] = {}

for _, meta in view.iterrows():
    asset = asset_series(meta, prices).dropna()
    benchmark_ticker = meta["Benchmark"]
    if asset.empty or benchmark_ticker not in prices.columns:
        continue
    benchmark = pd.to_numeric(prices[benchmark_ticker], errors="coerce")
    relative = compute_relative_metrics(asset, benchmark)
    extension = extension_metrics(asset)
    rotation = rotation_history(asset, benchmark, short_window=short_window, long_window=long_window)
    if rotation.dropna(subset=["x", "y"]).empty:
        continue

    parent_rel = np.nan
    parent = meta["Parent"]
    if parent in prices.columns and parent != meta["Ticker"]:
        parent_rel = compute_relative_metrics(asset, pd.to_numeric(prices[parent], errors="coerce"))["rel_1m"]

    breadth = {"above_50d": np.nan, "above_200d": np.nan, "breadth_1m_change": np.nan, "coverage": 0}
    members = meta["Members"]
    if isinstance(members, tuple):
        cols = [ticker for ticker in members if ticker in prices.columns]
        if cols:
            breadth = compute_breadth(prices[cols])
    elif meta["Ticker"] in sector_members:
        cols = [ticker for ticker in sector_members[meta["Ticker"]] if ticker in prices.columns]
        if cols:
            breadth = compute_breadth(prices[cols])

    current = rotation.dropna(subset=["x", "y"]).iloc[-1]
    movement = movement_from_coordinates(rotation["x"], rotation["y"])
    series_by_id[meta["Id"]] = asset
    rotation_by_id[meta["Id"]] = rotation
    rank_hist[meta["Id"]] = rel_return_series(asset, benchmark, 21)

    rows.append({
        "Id": meta["Id"], "ETF": meta["Ticker"], "Industry": meta["Name"],
        "Group": meta["Sector Group"], "Kind": meta["Kind"], "State": current["state"],
        "Days in State": int(current["days_in_state"]), "1W Rel": relative["rel_1w"],
        "1M Rel": relative["rel_1m"], "3M Rel": relative["rel_3m"],
        "Weekly Rel Change": relative["rel_1w_change"], "1M Abs": extension["abs_1m"],
        "vs Parent 1M": parent_rel, "Dist. 50D": extension["dist_50d"],
        "52W Drawdown": extension["drawdown_52w"], "Above 50D": breadth["above_50d"],
        "Above 200D": breadth["above_200d"], "Breadth 1M Chg": breadth["breadth_1m_change"],
        "Breadth N": int(breadth["coverage"]), "Map X": float(current["x"]),
        "Map Y": float(current["y"]), "5D Speed": movement["speed"], "5D Angle": movement["angle"],
    })

snapshot = pd.DataFrame(rows)
if snapshot.empty:
    st.warning("No selected series have enough clean history for this rotation window.")
    st.stop()

rank_frame = pd.concat(rank_hist, axis=1).sort_index()
if len(rank_frame) > 5:
    current_rank = rank_frame.iloc[-1].rank(ascending=False, method="min")
    prior_rank = rank_frame.iloc[-6].rank(ascending=False, method="min")
    rank_change = prior_rank - current_rank
    snapshot["Weekly Rank Change"] = snapshot["Id"].map(rank_change)
else:
    snapshot["Weekly Rank Change"] = np.nan

snapshot = snapshot.sort_values(["1M Rel", "3M Rel"], ascending=False, na_position="last").reset_index(drop=True)
latest_date = pd.to_datetime(prices.index.max()).date()
coverage = len(snapshot)
st.caption(
    f"Data through {latest_date:%b %d, %Y} | {coverage} of {len(view)} selected exposures have sufficient history | "
    "Breadth appears only where constituent coverage is verified."
)

table_key = f"sector_rotation_table_{scope.lower().replace(' ', '_')}"
selected_idx = _selection_rows(table_key, len(snapshot))
if selected_idx:
    selected_ids = snapshot.iloc[selected_idx]["Id"].tolist()[:8]
else:
    selected_ids = snapshot.head(min(3, len(snapshot)))["Id"].tolist()

render_section_header(
    "Rotation map",
    "Pastel quadrants show the current state. Every exposure carries a short tail; selected markers are emphasized.",
)
tail_by_id: Dict[str, pd.DataFrame] = {}
axis_x = snapshot["Map X"].tolist()
axis_y = snapshot["Map Y"].tolist()
for item_id in snapshot["Id"]:
    hist = rotation_by_id[item_id].dropna(subset=["x", "y"]).tail(trail_sessions)
    tail_by_id[item_id] = hist
    if not hist.empty:
        axis_x.extend(hist["x"].tolist())
        axis_y.extend(hist["y"].tolist())

x_range = full_extent_axis_range(axis_x)
y_range = full_extent_axis_range(axis_y)
plot_snapshot = snapshot.copy()
label_ids = select_auto_labels(plot_snapshot, selected_ids=selected_ids, max_labels=14)

map_fig = go.Figure()
quadrants = [
    (x_range[0], 0, 0, y_range[1], STATE_PALETTE["Improving"]),
    (0, x_range[1], 0, y_range[1], STATE_PALETTE["Leading"]),
    (x_range[0], 0, y_range[0], 0, STATE_PALETTE["Lagging"]),
    (0, x_range[1], y_range[0], 0, STATE_PALETTE["Weakening"]),
]
for x0, x1, y0, y1, fill in quadrants:
    map_fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
        fillcolor=fill, opacity=0.13, line=dict(width=0), layer="below",
    )

for item_id in snapshot["Id"]:
    row = plot_snapshot.loc[plot_snapshot["Id"] == item_id].iloc[0]
    hist = tail_by_id[item_id]
    if len(hist) < 2:
        continue
    edge = STATE_EDGE.get(str(row["State"]), "#7B8791")
    map_fig.add_trace(go.Scatter(
        x=hist["x"],
        y=hist["y"],
        mode="lines",
        line=dict(color=_rgba(edge, 0.22), width=0.9),
        hoverinfo="skip",
        showlegend=False,
    ))

for state, group in plot_snapshot.groupby("State", dropna=False):
    labels = [
        display_name(ticker, industry) if item_id in label_ids else ""
        for ticker, industry, item_id in zip(group["ETF"], group["Industry"], group["Id"], strict=True)
    ]
    sizes = [13 if item_id in selected_ids else 10 for item_id in group["Id"]]
    widths = [2.4 if item_id in selected_ids else 1.1 for item_id in group["Id"]]
    map_fig.add_trace(go.Scatter(
        x=group["Map X"],
        y=group["Map Y"],
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color="#4B5563"),
        marker=dict(
            size=sizes,
            color=STATE_PALETTE.get(str(state), STATE_PALETTE["Neutral"]),
            line=dict(width=widths, color=STATE_EDGE.get(str(state), "#7B8791")),
        ),
        name=str(state),
        customdata=np.stack([
            group["Industry"], group["Map X"], group["Map Y"], group["1M Rel"], group["3M Rel"], group["Days in State"],
        ], axis=-1),
        hovertemplate=(
            "%{customdata[0]}<br>Long-window rel %{customdata[1]:.1%}<br>Short-window rel %{customdata[2]:.1%}"
            "<br>1M rel %{customdata[3]:.1%}<br>3M rel %{customdata[4]:.1%}<br>Days in state %{customdata[5]:.0f}<extra></extra>"
        ),
    ))

map_fig.add_vline(x=0, line_width=1, line_color="#7C8794")
map_fig.add_hline(y=0, line_width=1, line_color="#7C8794")
map_fig.update_xaxes(
    range=x_range,
    tickformat=".1%",
    title=f"Long-window relative return ({long_window} sessions)",
    gridcolor="#E8EDF2",
    zeroline=False,
)
map_fig.update_yaxes(
    range=y_range,
    tickformat=".1%",
    title=f"Short-window relative return ({short_window} sessions)",
    gridcolor="#E8EDF2",
    zeroline=False,
)
map_fig.update_layout(
    height=650,
    margin=dict(l=30, r=30, t=10, b=60),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0),
)
st.plotly_chart(map_fig, width="stretch", config={"displayModeBar": False, "responsive": True})

render_section_header(
    "Relative strength",
    "Selected exposures versus their broad benchmark, rebased to 100. Table selections update this chart and the selected map markers.",
)
rs_fig = go.Figure()
for line_index, item_id in enumerate(selected_ids):
    meta = view.loc[view["Id"] == item_id].iloc[0]
    asset = series_by_id[item_id]
    bench = pd.to_numeric(prices[meta["Benchmark"]], errors="coerce")
    pair = pd.concat([asset.rename("a"), bench.rename("b")], axis=1).dropna().tail(126)
    if pair.empty:
        continue
    ratio = pair["a"] / pair["b"]
    rebased = ratio / ratio.iloc[0] * 100.0
    label = display_name(str(meta["Ticker"]), str(meta["Name"]))
    text = [""] * len(rebased)
    text[-1] = f"  {label}"
    rs_fig.add_trace(go.Scatter(
        x=rebased.index,
        y=rebased,
        mode="lines+text",
        text=text,
        textposition="middle right",
        textfont=dict(size=10),
        line=dict(color=PASTEL_20[line_index % len(PASTEL_20)], width=2.4),
        name=label,
        hovertemplate=f"{label}<br>%{{x|%b %d, %Y}}<br>%{{y:.1f}}<extra></extra>",
    ))
rs_fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="#8A949E")
rs_fig.update_xaxes(gridcolor="#EEF1F4", showline=False)
rs_fig.update_yaxes(gridcolor="#E4E9EE", title="Relative strength, 100=start", showline=False)
rs_fig.update_layout(
    height=380,
    margin=dict(l=30, r=95, t=10, b=30),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
)
st.plotly_chart(rs_fig, width="stretch", config={"displayModeBar": False, "responsive": True})

render_section_header(
    "Rotation table",
    "Sortable leadership, transition, extension and breadth measures. Select rows to highlight their markers and relative-strength series above.",
)
table_columns = [
    "ETF", "Industry", "Group", "State", "Days in State", "1W Rel", "1M Rel", "3M Rel",
    "Weekly Rel Change", "Weekly Rank Change", "1M Abs", "vs Parent 1M", "Dist. 50D",
    "52W Drawdown", "Above 50D", "Above 200D", "Breadth 1M Chg",
]
table_display = snapshot[table_columns].copy()
table_display["ETF"] = [
    display_name(ticker, industry)
    for ticker, industry in zip(table_display["ETF"], table_display["Industry"], strict=True)
]
table_display = table_display.rename(columns={
    "ETF": "Exposure",
    "Days in State": "Days",
    "Weekly Rel Change": "1W Δ Rel",
    "Weekly Rank Change": "Rank Δ",
    "vs Parent 1M": "vs Parent",
    "Dist. 50D": "vs 50D",
    "52W Drawdown": "52W DD",
    "Above 50D": ">50D",
    "Above 200D": ">200D",
    "Breadth 1M Chg": "Breadth Δ",
})
styled_table = style_rotation_table(table_display)
event = st.dataframe(
    styled_table,
    key=table_key,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    height=min(720, 38 + 35 * min(len(table_display), 19)),
    column_config={
        "Exposure": st.column_config.TextColumn("Exposure", width="medium"),
        "Industry": st.column_config.TextColumn("Industry", width="large"),
        "Group": st.column_config.TextColumn("Group", width="medium"),
        "State": st.column_config.TextColumn("State", width="small"),
        "Days": st.column_config.NumberColumn("Days", format="%d", width="small"),
    },
)
_ = event

excluded_tickers = sorted(set(view["Ticker"]) - set(snapshot["ETF"]))
if excluded_tickers:
    st.caption(
        f"Dropped {len(excluded_tickers)} ticker(s) for insufficient or stale price history: "
        f"{', '.join(excluded_tickers[:18])}{'…' if len(excluded_tickers) > 18 else ''}"
    )

render_footer(
    data_note="Market prices: Yahoo Finance. Major-sector constituent breadth: State Street daily holdings when available. Transparent baskets: equal-weight member prices."
)