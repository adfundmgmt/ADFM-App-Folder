from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from adfm_core.sector_rotation import (
    adaptive_axis_range,
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
STATE_COLORS = {
    "Leading": "#111111",
    "Improving": "#666666",
    "Weakening": "#9a9a9a",
    "Lagging": "#c2c2c2",
    "Neutral": "#e0e0e0",
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
                                s = pd.to_numeric(raw[key], errors="coerce")
                                if s.notna().any():
                                    out[ticker] = s
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
    unique = list(dict.fromkeys(t for t in tickers if t and not t.startswith("BASKET_")))
    pieces: List[pd.DataFrame] = []
    for i in range(0, len(unique), DOWNLOAD_CHUNK_SIZE):
        part = _download_batch(tuple(unique[i:i + DOWNLOAD_CHUNK_SIZE]))
        if not part.empty:
            pieces.append(part)
    prices = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    missing_tickers = [t for t in unique if t not in prices.columns or not prices[t].notna().any()]
    fallback_tickers = [
        ticker for ticker in missing_tickers if ticker in BENCHMARKS
    ]
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
        cols = [t for t in members if t in prices.columns]
        return synthetic_equal_weight_level(prices[cols]) if cols else pd.Series(dtype=float)
    ticker = row["Ticker"]
    return pd.to_numeric(prices[ticker], errors="coerce") if ticker in prices.columns else pd.Series(dtype=float)


def rel_return_series(asset: pd.Series, benchmark: pd.Series, periods: int) -> pd.Series:
    pair = pd.concat([asset.rename("a"), benchmark.rename("b")], axis=1).dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    ratio = pair["a"] / pair["b"]
    return ratio.pct_change(periods, fill_method=None)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.1%}"


def pts(value: float) -> str:
    return "" if pd.isna(value) else f"{value:+.1f}"


catalog = build_catalog()
with st.sidebar:
    render_sidebar_about("7_Sector_Breadth_and_Rotation.py")
    st.header("Settings")
    scope = st.selectbox("Universe", ["Sectors", "Industries", "Themes", "Countries", "All"], index=1)
    window_label = st.selectbox("Rotation window", list(WINDOWS), index=0)
    trail_sessions = st.selectbox("Selected trails", [0, 20, 40, 60], index=1, format_func=lambda x: "None" if x == 0 else f"{x} sessions")

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

# Major SPDR sectors get full constituent breadth. Transparent stock baskets use
# their explicit members. Other ETFs remain blank instead of using top-holdings
# proxies that would overstate participation.
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
        cols = [t for t in members if t in prices.columns]
        if cols:
            breadth = compute_breadth(prices[cols])
    elif meta["Ticker"] in sector_members:
        cols = [t for t in sector_members[meta["Ticker"]] if t in prices.columns]
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
st.caption(f"Data through {latest_date:%b %d, %Y} | {coverage} of {len(view)} selected exposures have sufficient history | Breadth appears only where constituent coverage is verified.")

render_section_header("Rotation table", "Sortable leadership, transition, extension and breadth measures. Select rows for the detail chart and map trails.")
columns = [
    "ETF", "Industry", "Group", "State", "Days in State", "1W Rel", "1M Rel", "3M Rel",
    "Weekly Rel Change", "Weekly Rank Change", "1M Abs", "vs Parent 1M", "Dist. 50D",
    "52W Drawdown", "Above 50D", "Above 200D", "Breadth 1M Chg", "Breadth N",
]
display = snapshot[columns].copy()
percent_cols = ["1W Rel", "1M Rel", "3M Rel", "Weekly Rel Change", "1M Abs", "vs Parent 1M", "Dist. 50D", "52W Drawdown"]
for col in percent_cols:
    display[col] = display[col].map(pct)
for col in ["Above 50D", "Above 200D", "Breadth 1M Chg"]:
    display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.0f}%")
display["Weekly Rank Change"] = display["Weekly Rank Change"].map(pts)

event = st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    height=min(720, 38 + 35 * min(len(display), 19)),
)
selected_idx = list(event.selection.rows) if hasattr(event, "selection") else []
if not selected_idx:
    selected_idx = [0]
selected_ids = snapshot.iloc[selected_idx]["Id"].tolist()

render_section_header("Selected relative strength", "Selected exposures versus their broad benchmark, rebased to 100.")
rs_fig = go.Figure()
for item_id in selected_ids[:8]:
    meta = view.loc[view["Id"] == item_id].iloc[0]
    asset = series_by_id[item_id]
    bench = pd.to_numeric(prices[meta["Benchmark"]], errors="coerce")
    pair = pd.concat([asset.rename("a"), bench.rename("b")], axis=1).dropna().tail(126)
    if pair.empty:
        continue
    ratio = pair["a"] / pair["b"]
    rebased = ratio / ratio.iloc[0] * 100.0
    rs_fig.add_trace(go.Scatter(x=rebased.index, y=rebased, mode="lines", name=str(meta["Name"])))
rs_fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"), yaxis_title="Relative strength, 100=start")
st.plotly_chart(rs_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

render_section_header("Rotation map", "Current positions for the full selected universe; trails are drawn only for selected rows.")
map_fig = go.Figure()
for state, group in snapshot.groupby("State", dropna=False):
    map_fig.add_trace(go.Scatter(
        x=group["Map X"], y=group["Map Y"], mode="markers+text",
        text=group["ETF"], textposition="top center",
        marker=dict(size=9, color=STATE_COLORS.get(state, "#aaaaaa"), line=dict(width=1, color="#000000")),
        name=str(state),
        customdata=np.stack([group["Industry"], group["1M Rel"], group["3M Rel"]], axis=-1),
        hovertemplate="%{customdata[0]}<br>1M rel %{customdata[1]:.1%}<br>3M rel %{customdata[2]:.1%}<extra></extra>",
    ))

axis_x = snapshot["Map X"].tolist()
axis_y = snapshot["Map Y"].tolist()
if trail_sessions:
    for item_id in selected_ids[:8]:
        hist = rotation_by_id[item_id].dropna(subset=["x", "y"]).tail(trail_sessions)
        if hist.empty:
            continue
        label = snapshot.loc[snapshot["Id"] == item_id, "ETF"].iloc[0]
        map_fig.add_trace(go.Scatter(x=hist["x"], y=hist["y"], mode="lines", line=dict(width=2), name=f"{label} trail", showlegend=False))
        axis_x.extend(hist["x"].tolist())
        axis_y.extend(hist["y"].tolist())

x_range = adaptive_axis_range(axis_x)
y_range = adaptive_axis_range(axis_y)
map_fig.add_vline(x=0, line_width=1, line_color="#777777")
map_fig.add_hline(y=0, line_width=1, line_color="#777777")
map_fig.update_xaxes(range=x_range, tickformat=".1%", title=f"Long-window relative return ({long_window} sessions)")
map_fig.update_yaxes(range=y_range, tickformat=".1%", title=f"Short-window relative return ({short_window} sessions)")
map_fig.update_layout(height=620, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h"))
st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

excluded_tickers = sorted(set(view["Ticker"]) - set(snapshot["ETF"]))
if excluded_tickers:
    st.caption(f"Dropped {len(excluded_tickers)} ticker(s) for insufficient or stale price history: {', '.join(excluded_tickers[:18])}{'…' if len(excluded_tickers) > 18 else ''}")

render_footer(data_note="Market prices: Yahoo Finance. Major-sector constituent breadth: State Street daily holdings when available. Transparent baskets: equal-weight member prices.")
