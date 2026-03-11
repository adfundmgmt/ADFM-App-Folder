from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="Cross-Asset Volume Dashboard", layout="wide")
plt.style.use("default")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.05rem;
            padding-bottom: 1rem;
            max-width: 1580px;
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.12rem;
        }
        .sub-title {
            font-size: 1rem;
            color: #5f6368;
            margin-bottom: 1.1rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
        }
        .insight-card {
            padding: 0.9rem 1rem;
            border: 1px solid #E8E8E8;
            border-radius: 12px;
            background: #FAFAFA;
            margin-bottom: 0.65rem;
        }
        .muted {
            color: #6b7280;
            font-size: 0.86rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #ECECEC;
            border-radius: 12px;
            padding: 0.65rem 0.75rem 0.6rem 0.75rem;
            background: #FBFBFB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSET_MAP: Dict[str, List[Tuple[str, str]]] = {
    "Equities": [
        ("SPY", "S&P 500"),
        ("QQQ", "Nasdaq 100"),
        ("IWM", "Russell 2000"),
        ("SMH", "Semiconductors"),
        ("XLF", "Financials"),
        ("XLK", "Technology"),
        ("XLE", "Energy"),
        ("XLI", "Industrials"),
        ("ARKK", "High Beta Growth"),
    ],
    "Credit": [
        ("LQD", "IG Credit"),
        ("HYG", "High Yield"),
        ("JNK", "HY Alt"),
        ("TLT", "Long Duration"),
        ("IEF", "7-10Y Treasuries"),
        ("EMB", "EM Sovereign"),
        ("BKLN", "Senior Loans"),
        ("MBB", "Agency MBS"),
    ],
    "Commodities": [
        ("GLD", "Gold"),
        ("SLV", "Silver"),
        ("USO", "WTI Oil"),
        ("UNG", "Nat Gas"),
        ("CPER", "Copper"),
        ("DBA", "Agriculture"),
        ("DBC", "Broad Basket"),
    ],
    "Currencies": [
        ("UUP", "Dollar Index Proxy"),
        ("FXE", "Euro"),
        ("FXY", "Japanese Yen"),
        ("FXB", "British Pound"),
        ("FXA", "Australian Dollar"),
        ("FXC", "Canadian Dollar"),
        ("CEW", "EM FX Basket"),
        ("CYB", "Chinese Yuan"),
    ],
}


@dataclass
class SignalRow:
    asset_class: str
    ticker: str
    label: str
    close: float
    daily_pct: float
    vol_multiple: float
    vol_z: float
    dollar_vol_vs_avg: float
    breakout_20d: str
    trend_50d: str
    trend_200d: str
    atr_pct: float
    score: float
    action: str


def human_num(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"


def safe_zscore(series: pd.Series, lookback: int = 60) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std().replace(0, np.nan)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def minmax_scale(series: pd.Series, low: float = 1.0, high: float = 5.0) -> pd.Series:
    s = series.copy().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(low, index=series.index)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series((low + high) / 2, index=series.index)
    out = low + (s - smin) * (high - low) / (smax - smin)
    return out.clip(low, high)


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_data(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    out: Dict[str, pd.DataFrame] = {}

    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        tmp = raw.copy()
        tmp.columns = [str(c) for c in tmp.columns]
        out[tickers[0]] = tmp.dropna(how="all")
        return out

    for ticker in tickers:
        if ticker in raw.columns.get_level_values(0):
            tmp = raw[ticker].copy()
            tmp.columns = [str(c) for c in tmp.columns]
            out[ticker] = tmp.dropna(how="all")
    return out


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    d["ret_1d"] = d["Close"].pct_change() * 100
    d["vol_avg_20"] = d["Volume"].rolling(20).mean()
    d["vol_multiple"] = d["Volume"] / d["vol_avg_20"]
    d["vol_z"] = safe_zscore(d["Volume"], 60)
    d["dollar_volume"] = d["Close"] * d["Volume"]
    d["dollar_vol_avg_20"] = d["dollar_volume"].rolling(20).mean()
    d["dollar_vol_vs_avg"] = d["dollar_volume"] / d["dollar_vol_avg_20"]
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_50"] = d["Close"].rolling(50).mean()
    d["ma_200"] = d["Close"].rolling(200).mean()
    d["hh_20_prev"] = d["High"].shift(1).rolling(20).max()
    d["ll_20_prev"] = d["Low"].shift(1).rolling(20).min()

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr_14"] = d["tr"].rolling(14).mean()
    d["atr_pct"] = d["atr_14"] / d["Close"] * 100

    d["line_weight"] = minmax_scale(d["vol_multiple"].fillna(1.0), 1.25, 5.75)
    d["line_color_score"] = d["vol_multiple"].clip(lower=0.5, upper=4.0)
    return d


def compute_signal_table(raw: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[SignalRow] = []
    enriched: Dict[str, pd.DataFrame] = {}

    for asset_class, items in ASSET_MAP.items():
        for ticker, label in items:
            df = raw.get(ticker)
            if df is None or df.empty:
                continue
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue

            d = enrich_df(df)
            if len(d) < 220:
                continue

            latest = d.iloc[-1]
            if pd.isna(latest["vol_multiple"]) or pd.isna(latest["ma_50"]) or pd.isna(latest["ma_200"]):
                continue

            breakout = "Inside"
            if latest["Close"] > latest["hh_20_prev"]:
                breakout = "20D Up"
            elif latest["Close"] < latest["ll_20_prev"]:
                breakout = "20D Down"

            trend_50 = "Above" if latest["Close"] > latest["ma_50"] else "Below"
            trend_200 = "Above" if latest["Close"] > latest["ma_200"] else "Below"

            score = 0.0
            score += min(float(latest["vol_multiple"]), 4.0) * 1.35
            score += min(float(latest["dollar_vol_vs_avg"]), 4.0) * 0.95
            score += max(min(float(latest["vol_z"]), 4.0), -2.0) * 0.8
            score += 1.25 if breakout in ["20D Up", "20D Down"] else 0.0
            score += 0.75 if trend_50 == "Above" else 0.0
            score += 0.75 if trend_200 == "Above" else 0.0
            score += min(abs(float(latest["ret_1d"])) / 2.5, 2.0)

            if breakout == "20D Up" and latest["vol_multiple"] >= 1.5 and trend_50 == "Above":
                action = "Bullish expansion"
            elif breakout == "20D Down" and latest["vol_multiple"] >= 1.5 and trend_50 == "Below":
                action = "Bearish expansion"
            elif latest["vol_multiple"] >= 1.8 and abs(latest["ret_1d"]) <= 0.6:
                action = "Absorption / churn"
            elif latest["vol_multiple"] >= 1.4 and trend_50 != trend_200:
                action = "Transition regime"
            else:
                action = "Watch"

            rows.append(
                SignalRow(
                    asset_class=asset_class,
                    ticker=ticker,
                    label=label,
                    close=float(latest["Close"]),
                    daily_pct=float(latest["ret_1d"]),
                    vol_multiple=float(latest["vol_multiple"]),
                    vol_z=float(latest["vol_z"]),
                    dollar_vol_vs_avg=float(latest["dollar_vol_vs_avg"]),
                    breakout_20d=breakout,
                    trend_50d=trend_50,
                    trend_200d=trend_200,
                    atr_pct=float(latest["atr_pct"]),
                    score=float(score),
                    action=action,
                )
            )
            enriched[ticker] = d

    table = pd.DataFrame([r.__dict__ for r in rows])
    if table.empty:
        return pd.DataFrame(), enriched
    table = table.sort_values(["score", "vol_multiple"], ascending=[False, False]).reset_index(drop=True)
    return table, enriched


def summarize_market(table: pd.DataFrame) -> Dict[str, str]:
    if table.empty:
        return {}

    bullish = int((table["action"] == "Bullish expansion").sum())
    bearish = int((table["action"] == "Bearish expansion").sum())
    churn = int((table["action"] == "Absorption / churn").sum())
    transition = int((table["action"] == "Transition regime").sum())
    lead = table.iloc[0]

    def leader_line(asset_class: str) -> str:
        sub = table[table["asset_class"] == asset_class].sort_values("score", ascending=False).head(1)
        if sub.empty:
            return f"No notable {asset_class.lower()} signal."
        r = sub.iloc[0]
        return (
            f"{r['ticker']} leads on {r['vol_multiple']:.2f}x volume with {r['daily_pct']:+.2f}% "
            f"price action and a {r['breakout_20d']} state."
        )

    return {
        "headline": (
            f"{lead['ticker']} is the highest-urgency setup, trading {lead['daily_pct']:+.2f}% "
            f"on {lead['vol_multiple']:.2f}x volume with a {lead['action'].lower()} profile."
        ),
        "breadth": f"Bullish expansions {bullish} | Bearish expansions {bearish} | Churn {churn} | Transitions {transition}",
        "equities": leader_line("Equities"),
        "credit": leader_line("Credit"),
        "commodities": leader_line("Commodities"),
        "currencies": leader_line("Currencies"),
    }


def prep_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close"] = out["close"].map(lambda x: f"{x:,.2f}")
    out["1D %"] = out["daily_pct"].map(lambda x: f"{x:+.2f}%")
    out["Vol / 20D"] = out["vol_multiple"].map(lambda x: f"{x:.2f}x")
    out["Vol Z"] = out["vol_z"].map(lambda x: f"{x:.2f}")
    out["$Vol / 20D"] = out["dollar_vol_vs_avg"].map(lambda x: f"{x:.2f}x")
    out["ATR %"] = out["atr_pct"].map(lambda x: f"{x:.2f}%")
    out["20D"] = out["breakout_20d"]
    out["50D"] = out["trend_50d"]
    out["200D"] = out["trend_200d"]
    out["Signal"] = out["action"]
    out["Score"] = out["score"].map(lambda x: f"{x:.2f}")
    out = out[
        [
            "asset_class",
            "ticker",
            "label",
            "Close",
            "1D %",
            "Vol / 20D",
            "Vol Z",
            "$Vol / 20D",
            "ATR %",
            "20D",
            "50D",
            "200D",
            "Signal",
            "Score",
        ]
    ]
    out.columns = [
        "Asset",
        "Ticker",
        "Name",
        "Close",
        "1D %",
        "Vol / 20D",
        "Vol Z",
        "$Vol / 20D",
        "ATR %",
        "20D State",
        "50D",
        "200D",
        "Signal",
        "Urgency",
    ]
    return out


def signal_bg(action: str) -> str:
    return {
        "Bullish expansion": "#dff5e7",
        "Bearish expansion": "#fde7e7",
        "Absorption / churn": "#fff4d7",
        "Transition regime": "#e8f0ff",
    }.get(action, "#f4f4f5")


def plot_volume_weighted_panel(
    ticker: str,
    label: str,
    df: pd.DataFrame,
    lookback: int = 120,
    show_bars: bool = True,
    normalize_price: bool = False,
) -> None:
    view = df.tail(lookback).copy()

    if normalize_price:
        base = view["Close"].iloc[0]
        price_series = 100 * view["Close"] / base
        ma20 = 100 * view["ma_20"] / base
        ma50 = 100 * view["ma_50"] / base
        y_label = "Indexed Price"
    else:
        price_series = view["Close"]
        ma20 = view["ma_20"]
        ma50 = view["ma_50"]
        y_label = "Price"

    x = mdates.date2num(view.index.to_pydatetime())
    y = price_series.values.astype(float)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    vol_for_segments = view["line_color_score"].iloc[1:].fillna(1.0).values
    width_for_segments = view["line_weight"].iloc[1:].fillna(2.0).values

    norm = Normalize(vmin=max(0.75, np.nanmin(vol_for_segments)), vmax=max(1.5, np.nanmax(vol_for_segments)))
    cmap = plt.cm.viridis

    fig, ax1 = plt.subplots(figsize=(12.2, 4.9))
    ax2 = ax1.twinx()

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(vol_for_segments)
    lc.set_linewidth(width_for_segments)
    ax1.add_collection(lc)

    ax1.plot(view.index, ma20, linewidth=1.0, alpha=0.9, linestyle="--", label="20D MA")
    ax1.plot(view.index, ma50, linewidth=1.0, alpha=0.9, linestyle="-.", label="50D MA")
    ax1.scatter(view.index[-1], price_series.iloc[-1], s=32, zorder=5)

    if show_bars:
        ax2.bar(view.index, view["Volume"], alpha=0.15, width=1.8, label="Volume")
        ax2.plot(view.index, view["vol_avg_20"], linewidth=1.0, alpha=0.85, label="20D Avg Volume")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda val, _: human_num(val)))
        ax2.set_ylabel("Volume")

    ax1.set_title(f"{ticker} | {label} | Price Path Weighted by Relative Volume")
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.18)
    ax1.set_xlim(view.index.min(), view.index.max())
    ax1.set_ylim(np.nanmin(y) * 0.98, np.nanmax(y) * 1.02)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.01, fraction=0.04)
    cbar.set_label("Relative Volume vs 20D")

    handles1, labels1 = ax1.get_legend_handles_labels()
    if show_bars:
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, ncol=2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def signal_cards(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if df.empty:
        st.info("No setups matched this filter.")
        return

    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="insight-card" style="background:{signal_bg(row['action'])};">
                <div style="font-weight:700;">{row['ticker']} | {row['label']}</div>
                <div class="muted">{row['asset_class']} | {row['action']}</div>
                <div style="margin-top:0.35rem;">
                    {row['daily_pct']:+.2f}% on {row['vol_multiple']:.2f}x volume. Dollar volume is running at {row['dollar_vol_vs_avg']:.2f}x the 20-day average.
                    20-day state is {row['breakout_20d']}, while trend stack is {row['trend_50d']} the 50-day and {row['trend_200d']} the 200-day.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        """
        This dashboard treats volume as part of the line itself rather than a separate afterthought.

        The main chart uses a volume-weighted price path:
        thicker and more intense line segments mean the instrument was moving on heavier relative volume,
        while thinner segments mean the move had less participation behind it.

        That helps separate noisy drift from real tape events. A clean breakout with a visibly thicker line
        is more actionable than the same price pattern on weak turnover.

        Because public consolidated spot volume does not exist for most FX pairs, currencies are tracked
        with listed ETF proxies.
        """
    )

    st.markdown("### Controls")
    selected_classes = st.multiselect("Asset classes", options=list(ASSET_MAP.keys()), default=list(ASSET_MAP.keys()))
    min_vol_multiple = st.slider("Minimum relative volume", 1.0, 4.0, 1.2, 0.1)
    min_abs_move = st.slider("Minimum absolute 1D move %", 0.0, 5.0, 0.25, 0.05)
    top_n = st.slider("Rows to show", 5, 40, 20, 1)
    lookback = st.slider("Chart lookback", 60, 180, 120, 10)
    normalize_price = st.checkbox("Normalize price to 100", value=False)
    show_volume_bars = st.checkbox("Show volume bars under line", value=True)

    selected_tickers: List[str] = []
    for asset_class in selected_classes:
        selected_tickers.extend([t for t, _ in ASSET_MAP[asset_class]])

st.markdown("<div class='main-title'>Cross-Asset Volume Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Volume-led cross-asset monitor where price lines themselves reflect participation.</div>",
    unsafe_allow_html=True,
)

if not selected_tickers:
    st.warning("Select at least one asset class.")
    st.stop()

with st.spinner("Loading market data..."):
    raw = fetch_data(selected_tickers, period="2y")
    signal_table, enriched = compute_signal_table(raw)

if signal_table.empty:
    st.error("No valid market data returned for the selected universe.")
    st.stop()

signal_table = signal_table[
    signal_table["asset_class"].isin(selected_classes)
    & (signal_table["vol_multiple"] >= min_vol_multiple)
    & (signal_table["daily_pct"].abs() >= min_abs_move)
].copy()

if signal_table.empty:
    st.warning("No instruments match the current filters.")
    st.stop()

summary = summarize_market(signal_table)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Signals", f"{len(signal_table):,}")
m2.metric("Bullish expansions", f"{(signal_table['action'] == 'Bullish expansion').sum():,}")
m3.metric("Bearish expansions", f"{(signal_table['action'] == 'Bearish expansion').sum():,}")
m4.metric("Median vol / 20D", f"{signal_table['vol_multiple'].median():.2f}x")

st.markdown("<div class='section-title'>Executive read</div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="insight-card">
        <div><strong>Headline:</strong> {summary['headline']}</div>
        <div style="margin-top:0.3rem;"><strong>Breadth:</strong> {summary['breadth']}</div>
        <div style="margin-top:0.3rem;"><strong>Equities:</strong> {summary['equities']}</div>
        <div style="margin-top:0.3rem;"><strong>Credit:</strong> {summary['credit']}</div>
        <div style="margin-top:0.3rem;"><strong>Commodities:</strong> {summary['commodities']}</div>
        <div style="margin-top:0.3rem;"><strong>Currencies:</strong> {summary['currencies']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    signal_cards(
        signal_table[signal_table["action"].isin(["Bullish expansion", "Bearish expansion"])].head(6),
        "Highest-conviction setups",
    )
with right:
    signal_cards(
        signal_table[signal_table["action"].isin(["Absorption / churn", "Transition regime"])].head(6),
        "Secondary setups",
    )

st.markdown("<div class='section-title'>Ranked monitor</div>", unsafe_allow_html=True)
st.dataframe(prep_display(signal_table.head(top_n)), use_container_width=True, hide_index=True)

st.markdown("<div class='section-title'>Drilldown</div>", unsafe_allow_html=True)
col1, col2 = st.columns([1.0, 2.2])

with col1:
    drill_ticker = st.selectbox("Select instrument", signal_table["ticker"].tolist(), index=0)
    row = signal_table.loc[signal_table["ticker"] == drill_ticker].iloc[0]

    st.markdown(
        f"""
        <div class="insight-card" style="background:{signal_bg(row['action'])};">
            <div style="font-weight:700; font-size:1.03rem;">{row['ticker']} | {row['label']}</div>
            <div class="muted">{row['asset_class']} | {row['action']}</div>
            <div style="margin-top:0.35rem;">Close: {row['close']:.2f}</div>
            <div>1D move: {row['daily_pct']:+.2f}%</div>
            <div>Relative volume: {row['vol_multiple']:.2f}x</div>
            <div>Dollar volume / 20D: {row['dollar_vol_vs_avg']:.2f}x</div>
            <div>Breakout state: {row['breakout_20d']}</div>
            <div>50D trend: {row['trend_50d']}</div>
            <div>200D trend: {row['trend_200d']}</div>
            <div>ATR: {row['atr_pct']:.2f}%</div>
            <div>Urgency: {row['score']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if row["action"] == "Bullish expansion":
        st.markdown(
            "Read: follow-through is more likely when upside is being carried by visibly heavier participation."
        )
    elif row["action"] == "Bearish expansion":
        st.markdown(
            "Read: this is the kind of downside move that tends to force repositioning rather than invite lazy dip-buying."
        )
    elif row["action"] == "Absorption / churn":
        st.markdown(
            "Read: turnover is elevated but price has not resolved yet, which often means ownership transfer before the real move."
        )
    elif row["action"] == "Transition regime":
        st.markdown("Read: the tape is trying to change character, but trend confirmation is still incomplete.")
    else:
        st.markdown(
            "Read: elevated attention candidate, but there is not enough confirmation to treat it as a real event yet."
        )

with col2:
    plot_volume_weighted_panel(
        drill_ticker,
        row["label"],
        enriched[drill_ticker],
        lookback=lookback,
        show_bars=show_volume_bars,
        normalize_price=normalize_price,
    )

st.markdown("<div class='section-title'>Asset class internals</div>", unsafe_allow_html=True)
tabs = st.tabs(selected_classes)

for tab, asset_class in zip(tabs, selected_classes):
    with tab:
        sub = signal_table[signal_table["asset_class"] == asset_class].copy()
        if sub.empty:
            st.info("No setups under current filters.")
        else:
            st.dataframe(prep_display(sub.head(12)), use_container_width=True, hide_index=True)

st.caption(
    "Data source: Yahoo Finance via yfinance. Currencies are represented by listed ETF proxies because "
    "spot FX lacks a single consolidated public volume tape."
)
