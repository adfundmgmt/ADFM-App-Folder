# streamlit_cross_asset_page.py
# Run: streamlit run streamlit_cross_asset_page.py

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Weekly Cross-Asset Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px 16px; border-radius: 14px;}
        .small-note {font-size: 0.84rem; opacity: 0.75;}
        .card {background: rgba(128,128,128,0.08); padding: 16px; border-radius: 14px; margin-bottom: 10px;}
        .signal-good {font-weight: 700;}
        .signal-bad {font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Universe
# -----------------------------
ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    # Equities
    "SPY": {"name": "S&P 500", "sleeve": "Equities"},
    "QQQ": {"name": "Nasdaq 100", "sleeve": "Equities"},
    "IWM": {"name": "Russell 2000", "sleeve": "Equities"},
    "RSP": {"name": "Equal-Weight S&P", "sleeve": "Equities"},
    "EFA": {"name": "DM ex-US", "sleeve": "Equities"},
    "EEM": {"name": "Emerging Markets", "sleeve": "Equities"},
    "KWEB": {"name": "China Internet", "sleeve": "Equities"},
    "SMH": {"name": "Semiconductors", "sleeve": "Equities"},
    "XLF": {"name": "Financials", "sleeve": "Equities"},
    "XLE": {"name": "Energy Equities", "sleeve": "Equities"},
    "XLU": {"name": "Utilities", "sleeve": "Equities"},
    # Rates / bonds
    "SHY": {"name": "1-3Y Treasuries", "sleeve": "Rates"},
    "IEF": {"name": "7-10Y Treasuries", "sleeve": "Rates"},
    "TLT": {"name": "20Y+ Treasuries", "sleeve": "Rates"},
    "TIP": {"name": "TIPS", "sleeve": "Rates"},
    # Credit
    "LQD": {"name": "Investment Grade Credit", "sleeve": "Credit"},
    "HYG": {"name": "High Yield Credit", "sleeve": "Credit"},
    "JNK": {"name": "High Yield Credit 2", "sleeve": "Credit"},
    "EMB": {"name": "EM Sovereign Debt", "sleeve": "Credit"},
    # FX
    "UUP": {"name": "US Dollar", "sleeve": "FX"},
    "FXE": {"name": "Euro", "sleeve": "FX"},
    "FXY": {"name": "Japanese Yen", "sleeve": "FX"},
    "FXB": {"name": "British Pound", "sleeve": "FX"},
    "CYB": {"name": "Chinese Yuan", "sleeve": "FX"},
    # Commodities
    "GLD": {"name": "Gold", "sleeve": "Commodities"},
    "SLV": {"name": "Silver", "sleeve": "Commodities"},
    "USO": {"name": "Oil", "sleeve": "Commodities"},
    "UNG": {"name": "Natural Gas", "sleeve": "Commodities"},
    "DBA": {"name": "Agriculture", "sleeve": "Commodities"},
    "DBC": {"name": "Broad Commodities", "sleeve": "Commodities"},
    "COPX": {"name": "Copper Miners", "sleeve": "Commodities"},
    # Vol / alternatives
    "VIXY": {"name": "VIX Short-Term Futures", "sleeve": "Volatility"},
    "VXZ": {"name": "VIX Mid-Term Futures", "sleeve": "Volatility"},
    "BTC-USD": {"name": "Bitcoin", "sleeve": "Crypto"},
    "ETH-USD": {"name": "Ethereum", "sleeve": "Crypto"},
}

DEFAULT_TICKERS = list(ASSET_UNIVERSE.keys())
CORE_CHART_TICKERS = ["SPY", "QQQ", "TLT", "HYG", "UUP", "GLD", "USO", "VIXY", "BTC-USD"]


# -----------------------------
# Helpers
# -----------------------------
def fmt_pct(x: float | int | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%"


def fmt_num(x: float | int | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x:.{decimals}f}"


def safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if std is None or pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / std


def pct_change_over(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-1 - periods] - 1)


def get_close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0).unique())
        level1 = list(raw.columns.get_level_values(1).unique())

        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Adj Close" in level0:
            close = raw["Adj Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", level=1, axis=1).copy()
        elif "Adj Close" in level1:
            close = raw.xs("Adj Close", level=1, axis=1).copy()
        else:
            raise ValueError("Could not locate close prices in downloaded data.")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[col]].copy()
        close.columns = tickers[:1]

    close = close.loc[:, [c for c in close.columns if c in tickers]]
    close = close.dropna(how="all").sort_index()
    return close


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_prices(tickers: List[str], period: str = "3y") -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = get_close_prices(raw, tickers)
    return close


@st.cache_data(show_spinner=False)
def build_weekly_frame(close: pd.DataFrame) -> pd.DataFrame:
    weekly = close.resample("W-FRI").last().dropna(how="all")
    weekly = weekly.loc[:, weekly.notna().sum() >= 26]
    return weekly


@st.cache_data(show_spinner=False)
def build_features(weekly: pd.DataFrame, universe: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    rows = []
    weekly_rets = weekly.pct_change()
    current_year = weekly.index[-1].year if len(weekly) else datetime.now().year

    for ticker in weekly.columns:
        s = weekly[ticker].dropna()
        r = weekly_rets[ticker].dropna()
        if len(s) < 26:
            continue

        first_current_year = s[s.index.year == current_year]
        ytd = np.nan
        if len(first_current_year) > 0:
            ytd = float(s.iloc[-1] / first_current_year.iloc[0] - 1)

        ma13 = s.rolling(13).mean().iloc[-1]
        ma26 = s.rolling(26).mean().iloc[-1]
        mean52 = s.rolling(52).mean().iloc[-1] if len(s) >= 52 else np.nan
        std52 = s.rolling(52).std().iloc[-1] if len(s) >= 52 else np.nan
        high52 = s.rolling(52).max().iloc[-1] if len(s) >= 52 else s.max()

        rows.append(
            {
                "ticker": ticker,
                "asset": universe.get(ticker, {}).get("name", ticker),
                "sleeve": universe.get(ticker, {}).get("sleeve", "Other"),
                "last": float(s.iloc[-1]),
                "ret_1w": pct_change_over(s, 1),
                "ret_4w": pct_change_over(s, 4),
                "ret_13w": pct_change_over(s, 13),
                "ret_26w": pct_change_over(s, 26),
                "ret_ytd": ytd,
                "vol_13w": float(r.tail(13).std() * math.sqrt(52)) if len(r) >= 5 else np.nan,
                "trend_13w": float(s.iloc[-1] / ma13 - 1) if pd.notna(ma13) and ma13 != 0 else np.nan,
                "trend_26w": float(s.iloc[-1] / ma26 - 1) if pd.notna(ma26) and ma26 != 0 else np.nan,
                "z_52w": float((s.iloc[-1] - mean52) / std52) if pd.notna(std52) and std52 != 0 else np.nan,
                "dd_52w": float(s.iloc[-1] / high52 - 1) if pd.notna(high52) and high52 != 0 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["score"] = (
        0.35 * safe_zscore(df["ret_4w"])
        + 0.30 * safe_zscore(df["ret_13w"])
        + 0.25 * safe_zscore(df["trend_26w"])
        - 0.10 * safe_zscore(df["vol_13w"])
    )

    conditions = [
        (df["score"] >= 0.75) & (df["trend_13w"] > 0) & (df["ret_4w"] > 0),
        (df["score"] <= -0.75) & (df["trend_13w"] < 0) & (df["ret_4w"] < 0),
        df["z_52w"].abs() >= 2.0,
    ]
    choices = ["Long / add candidate", "Short / avoid candidate", "Stretched / watch reversal"]
    df["action"] = np.select(conditions, choices, default="Neutral")
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def get_metric(features: pd.DataFrame, ticker: str, col: str) -> float:
    row = features.loc[features["ticker"] == ticker, col]
    if row.empty:
        return np.nan
    return float(row.iloc[0])


def regime_label(features: pd.DataFrame) -> tuple[str, str]:
    equity_4w = np.nanmean([get_metric(features, t, "ret_4w") for t in ["SPY", "QQQ", "IWM", "RSP"]])
    credit_impulse = get_metric(features, "HYG", "ret_4w") - get_metric(features, "LQD", "ret_4w")
    duration_4w = get_metric(features, "TLT", "ret_4w")
    dollar_4w = get_metric(features, "UUP", "ret_4w")
    vol_4w = get_metric(features, "VIXY", "ret_4w")
    oil_4w = get_metric(features, "USO", "ret_4w")

    risk_score = 0
    risk_score += 1 if equity_4w > 0 else -1
    risk_score += 1 if credit_impulse > 0 else -1
    risk_score += 1 if vol_4w < 0 else -1
    risk_score += 1 if dollar_4w < 0 else -1
    risk_score += 1 if duration_4w > 0 else -1

    if risk_score >= 3:
        label = "Risk-on with easing pressure"
        note = "Equities, credit, vol, dollar, and duration are broadly consistent with looser financial conditions. Respect upside momentum, but watch crowded growth and rate-sensitive beta."
    elif risk_score <= -3:
        label = "Risk-off / tightening pressure"
        note = "The tape is showing pressure through equities, credit, volatility, dollar strength, or duration weakness. Prioritize liquidity, gross discipline, and thesis invalidation."
    else:
        label = "Mixed regime / dispersion tape"
        note = "Cross-asset confirmation is weak. This is a selection market: relative-value, pair trades, and smaller gross are cleaner than broad beta."

    if oil_4w > 0.08:
        note += " Oil is also adding inflation pressure."
    elif oil_4w < -0.08:
        note += " Oil is reinforcing the disinflation impulse."

    return label, note


def pressure_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["Equity beta", np.nanmean([get_metric(features, t, "ret_4w") for t in ["SPY", "QQQ", "IWM", "RSP"]])],
        ["Credit risk appetite", get_metric(features, "HYG", "ret_4w") - get_metric(features, "LQD", "ret_4w")],
        ["Duration impulse", get_metric(features, "TLT", "ret_4w")],
        ["Dollar pressure", get_metric(features, "UUP", "ret_4w")],
        ["Oil impulse", get_metric(features, "USO", "ret_4w")],
        ["Volatility pressure", get_metric(features, "VIXY", "ret_4w")],
    ]
    df = pd.DataFrame(rows, columns=["pressure_point", "four_week_move"])
    df["read"] = np.select(
        [df["four_week_move"] > 0.02, df["four_week_move"] < -0.02],
        ["Rising", "Falling"],
        default="Flat",
    )
    return df


def make_return_heatmap(features: pd.DataFrame) -> go.Figure:
    data = features.copy()
    cols = ["ret_1w", "ret_4w", "ret_13w", "ret_ytd"]
    data = data.sort_values(["sleeve", "ret_4w"], ascending=[True, False])
    z = data[cols].to_numpy() * 100
    text = data[cols].applymap(lambda x: "" if pd.isna(x) else f"{x * 100:+.1f}%")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["1W", "4W", "13W", "YTD"],
            y=data["ticker"] + "  " + data["asset"],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            coloraxis="coloraxis",
        )
    )
    fig.update_layout(
        height=max(520, 24 * len(data)),
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis=dict(colorscale="RdYlGn", cmid=0),
    )
    return fig


def make_normalized_chart(weekly: pd.DataFrame, tickers: List[str]) -> go.Figure:
    selected = [t for t in tickers if t in weekly.columns]
    data = weekly[selected].dropna(how="all").tail(78)
    norm = data / data.iloc[0] * 100

    fig = px.line(norm, x=norm.index, y=norm.columns, labels={"value": "Indexed to 100", "index": "Date", "variable": "Asset"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def make_scatter(features: pd.DataFrame) -> go.Figure:
    data = features.copy()
    data["ret_13w_pct"] = data["ret_13w"] * 100
    data["trend_26w_pct"] = data["trend_26w"] * 100
    data["vol_13w_pct"] = data["vol_13w"] * 100

    fig = px.scatter(
        data,
        x="trend_26w_pct",
        y="ret_13w_pct",
        size="vol_13w_pct",
        hover_name="ticker",
        hover_data={"asset": True, "sleeve": True, "z_52w": ":.2f", "score": ":.2f", "vol_13w_pct": ":.1f"},
        labels={"trend_26w_pct": "Distance from 26W MA (%)", "ret_13w_pct": "13W return (%)", "vol_13w_pct": "13W annualized vol"},
    )
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def make_corr_heatmap(weekly: pd.DataFrame, tickers: List[str]) -> go.Figure:
    selected = [t for t in tickers if t in weekly.columns]
    corr = weekly[selected].pct_change().tail(26).corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def make_drawdown_chart(weekly: pd.DataFrame, tickers: List[str]) -> go.Figure:
    selected = [t for t in tickers if t in weekly.columns]
    data = weekly[selected].dropna(how="all").tail(104)
    dd = data / data.cummax() - 1
    fig = px.line(dd * 100, x=dd.index, y=dd.columns, labels={"value": "Drawdown (%)", "index": "Date", "variable": "Asset"})
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def styled_action_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["ticker", "asset", "sleeve", "action", "score", "ret_1w", "ret_4w", "ret_13w", "ret_ytd", "z_52w", "dd_52w", "vol_13w"]].copy()
    pct_cols = ["ret_1w", "ret_4w", "ret_13w", "ret_ytd", "dd_52w", "vol_13w"]
    for c in pct_cols:
        out[c] = out[c].map(lambda x: fmt_pct(x))
    out["score"] = out["score"].map(lambda x: fmt_num(x))
    out["z_52w"] = out["z_52w"].map(lambda x: fmt_num(x))
    return out


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
    sleeves = sorted(set(v["sleeve"] for v in ASSET_UNIVERSE.values()))
    selected_sleeves = st.multiselect("Sleeves", sleeves, default=sleeves)
    selected_tickers = [t for t, meta in ASSET_UNIVERSE.items() if meta["sleeve"] in selected_sleeves]
    selected_tickers = st.multiselect("Tickers", list(ASSET_UNIVERSE.keys()), default=selected_tickers)
    chart_tickers = st.multiselect(
        "Chart focus",
        selected_tickers,
        default=[t for t in CORE_CHART_TICKERS if t in selected_tickers],
    )
    st.caption("Data source: Yahoo Finance through yfinance. Use this for weekly triage, not official marks.")


# -----------------------------
# Data load
# -----------------------------
st.title("Weekly Cross-Asset Cockpit")
st.caption("Fast weekly regime read across equities, rates, credit, FX, commodities, volatility, and crypto.")

if not selected_tickers:
    st.warning("Select at least one ticker.")
    st.stop()

with st.spinner("Loading weekly cross-asset data..."):
    close = load_prices(selected_tickers, period=period)
    weekly = build_weekly_frame(close)
    features = build_features(weekly, ASSET_UNIVERSE)

if weekly.empty or features.empty:
    st.error("No usable price history returned. Check the selected tickers or data connection.")
    st.stop()

as_of = weekly.index[-1].strftime("%b %d, %Y")
label, note = regime_label(features)


# -----------------------------
# Header metrics
# -----------------------------
spy_1w = get_metric(features, "SPY", "ret_1w")
qqq_1w = get_metric(features, "QQQ", "ret_1w")
tlt_1w = get_metric(features, "TLT", "ret_1w")
hyg_lqd_4w = get_metric(features, "HYG", "ret_4w") - get_metric(features, "LQD", "ret_4w")
uup_4w = get_metric(features, "UUP", "ret_4w")
uso_4w = get_metric(features, "USO", "ret_4w")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("S&P 500 1W", fmt_pct(spy_1w))
c2.metric("Nasdaq 1W", fmt_pct(qqq_1w))
c3.metric("TLT 1W", fmt_pct(tlt_1w))
c4.metric("HY vs IG 4W", fmt_pct(hyg_lqd_4w))
c5.metric("Dollar 4W", fmt_pct(uup_4w))
c6.metric("Oil 4W", fmt_pct(uso_4w))

st.markdown(f"### Regime: {label}")
st.write(note)
st.caption(f"Weekly close through {as_of}. Signals are rules-based and should be treated as a triage layer, not a trading system.")


# -----------------------------
# Action deck
# -----------------------------
st.markdown("## Action deck")

longs = features[features["action"] == "Long / add candidate"].head(8)
shorts = features[features["action"] == "Short / avoid candidate"].sort_values("score").head(8)
stretched = features[features["action"] == "Stretched / watch reversal"].sort_values("z_52w", key=lambda s: s.abs(), ascending=False).head(8)
movers = features.reindex(features["ret_1w"].abs().sort_values(ascending=False).index).head(8)

ac1, ac2, ac3, ac4 = st.columns(4)
with ac1:
    st.markdown("**Long / add candidates**")
    st.dataframe(styled_action_table(longs), use_container_width=True, hide_index=True)
with ac2:
    st.markdown("**Short / avoid candidates**")
    st.dataframe(styled_action_table(shorts), use_container_width=True, hide_index=True)
with ac3:
    st.markdown("**Stretched / reversal watch**")
    st.dataframe(styled_action_table(stretched), use_container_width=True, hide_index=True)
with ac4:
    st.markdown("**Largest 1W movers**")
    st.dataframe(styled_action_table(movers), use_container_width=True, hide_index=True)


# -----------------------------
# Pressure map
# -----------------------------
st.markdown("## Macro pressure points")
pt = pressure_table(features)
pt_display = pt.copy()
pt_display["four_week_move"] = pt_display["four_week_move"].map(fmt_pct)
st.dataframe(pt_display, use_container_width=True, hide_index=True)


# -----------------------------
# Visuals
# -----------------------------
st.markdown("## Visuals")

left, right = st.columns([1.05, 0.95])
with left:
    st.markdown("**Cross-asset tape, indexed to 100**")
    st.plotly_chart(make_normalized_chart(weekly, chart_tickers), use_container_width=True)
with right:
    st.markdown("**Trend vs 13-week return**")
    st.plotly_chart(make_scatter(features), use_container_width=True)

st.markdown("**Return heatmap**")
st.plotly_chart(make_return_heatmap(features), use_container_width=True)

left2, right2 = st.columns(2)
with left2:
    st.markdown("**26-week correlation map**")
    st.plotly_chart(make_corr_heatmap(weekly, chart_tickers), use_container_width=True)
with right2:
    st.markdown("**Two-year drawdown map**")
    st.plotly_chart(make_drawdown_chart(weekly, chart_tickers), use_container_width=True)


# -----------------------------
# Full signal table
# -----------------------------
st.markdown("## Full signal table")
full = styled_action_table(features)
st.dataframe(full, use_container_width=True, hide_index=True)

st.markdown(
    """
    <p class="small-note">
    Interpretation: score combines 4-week momentum, 13-week momentum, distance from 26-week trend, and recent volatility. 
    A high score is trend confirmation. A low score is downside confirmation. A high absolute 52-week z-score is a stretch warning.
    </p>
    """,
    unsafe_allow_html=True,
)
