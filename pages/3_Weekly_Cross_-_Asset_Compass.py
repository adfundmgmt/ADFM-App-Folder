from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Weekly Cross-Asset Compass",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px;}
        .stMetric {
            background: #f7f8fa;
            border: 1px solid rgba(49,51,63,0.08);
            padding: 14px 16px;
            border-radius: 16px;
        }
        .callout {
            background: #f7f8fa;
            border: 1px solid rgba(49,51,63,0.08);
            border-radius: 16px;
            padding: 14px 16px;
            min-height: 140px;
        }
        .mini {
            font-size: 0.85rem;
            color: rgba(49,51,63,0.72);
        }
        .bigline {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

UNIVERSE: Dict[str, Dict[str, str]] = {
    # Equities
    "SPY": {"name": "S&P 500", "sleeve": "Equities"},
    "QQQ": {"name": "Nasdaq 100", "sleeve": "Equities"},
    "IWM": {"name": "Russell 2000", "sleeve": "Equities"},
    "RSP": {"name": "Equal Weight S&P", "sleeve": "Equities"},
    "EFA": {"name": "DM ex-US", "sleeve": "Equities"},
    "EEM": {"name": "Emerging Markets", "sleeve": "Equities"},
    "KWEB": {"name": "China Internet", "sleeve": "Equities"},
    "SMH": {"name": "Semiconductors", "sleeve": "Equities"},
    "XLF": {"name": "Financials", "sleeve": "Equities"},
    "XLE": {"name": "Energy Equities", "sleeve": "Equities"},
    # Rates / credit
    "SHY": {"name": "1-3Y Treasuries", "sleeve": "Rates"},
    "IEF": {"name": "7-10Y Treasuries", "sleeve": "Rates"},
    "TLT": {"name": "20Y+ Treasuries", "sleeve": "Rates"},
    "TIP": {"name": "TIPS", "sleeve": "Rates"},
    "LQD": {"name": "Investment Grade Credit", "sleeve": "Credit"},
    "HYG": {"name": "High Yield Credit", "sleeve": "Credit"},
    # FX / commodities / vol / crypto
    "UUP": {"name": "US Dollar", "sleeve": "FX"},
    "FXE": {"name": "Euro", "sleeve": "FX"},
    "FXY": {"name": "Japanese Yen", "sleeve": "FX"},
    "GLD": {"name": "Gold", "sleeve": "Commodities"},
    "SLV": {"name": "Silver", "sleeve": "Commodities"},
    "USO": {"name": "Oil", "sleeve": "Commodities"},
    "UNG": {"name": "Natural Gas", "sleeve": "Commodities"},
    "DBC": {"name": "Broad Commodities", "sleeve": "Commodities"},
    "VIXY": {"name": "VIX Short-Term Futures", "sleeve": "Volatility"},
    "BTC-USD": {"name": "Bitcoin", "sleeve": "Crypto"},
    "ETH-USD": {"name": "Ethereum", "sleeve": "Crypto"},
}

CORE_TAPE = ["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP", "GLD", "USO", "VIXY", "BTC-USD"]


@dataclass
class AnchorDates:
    latest: pd.Timestamp
    prior: pd.Timestamp
    last_friday: pd.Timestamp
    previous_friday: pd.Timestamp | None


def pct_fmt(x: float | int | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%"


def num_fmt(x: float | int | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x) or np.isinf(x):
        return "n/a"
    return f"{x:.{decimals}f}"


def get_close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", level=1, axis=1).copy()
        else:
            close = raw.xs("Adj Close", level=1, axis=1).copy()
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[col]].copy()
        close.columns = tickers[:1]

    close = close.loc[:, [c for c in close.columns if c in tickers]]
    close = close.sort_index().dropna(how="all")
    return close


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_prices(tickers: List[str], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = get_close_prices(raw, tickers)
    return close.ffill()


def get_anchor_dates(close: pd.DataFrame) -> AnchorDates:
    idx = close.dropna(how="all").index
    latest = idx[-1]
    prior = idx[-2] if len(idx) >= 2 else idx[-1]
    friday_candidates = idx[idx.weekday == 4]
    if len(friday_candidates) == 0:
        last_friday = latest
        previous_friday = idx[-6] if len(idx) >= 6 else None
    else:
        last_friday = friday_candidates[friday_candidates <= latest][-1]
        earlier_fridays = friday_candidates[friday_candidates < last_friday]
        previous_friday = earlier_fridays[-1] if len(earlier_fridays) else None
    return AnchorDates(latest=latest, prior=prior, last_friday=last_friday, previous_friday=previous_friday)


def safe_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-1 - periods] - 1)


def since_date_change(series: pd.Series, dt: pd.Timestamp) -> float:
    s = series.dropna()
    if s.empty or dt not in s.index:
        return np.nan
    base = s.loc[dt]
    last = s.iloc[-1]
    if pd.isna(base) or base == 0:
        return np.nan
    return float(last / base - 1)


def build_feature_table(close: pd.DataFrame, anchors: AnchorDates) -> pd.DataFrame:
    rows = []
    current_year = anchors.latest.year
    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < 60:
            continue
        ret_1d = safe_change(s, 1)
        ret_1w = safe_change(s, 5)
        ret_1m = safe_change(s, 21)
        ret_3m = safe_change(s, 63)
        ret_6m = safe_change(s, 126)
        ret_since_friday = since_date_change(s, anchors.last_friday)

        ytd_slice = s[s.index.year == current_year]
        ret_ytd = float(s.iloc[-1] / ytd_slice.iloc[0] - 1) if not ytd_slice.empty else np.nan

        ma20 = s.rolling(20).mean().iloc[-1]
        ma50 = s.rolling(50).mean().iloc[-1]
        ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
        high_252 = s.rolling(252).max().iloc[-1] if len(s) >= 252 else s.max()
        low_252 = s.rolling(252).min().iloc[-1] if len(s) >= 252 else s.min()
        vol_1m = s.pct_change().tail(21).std() * np.sqrt(252)

        dist_20 = float(s.iloc[-1] / ma20 - 1) if pd.notna(ma20) and ma20 != 0 else np.nan
        dist_50 = float(s.iloc[-1] / ma50 - 1) if pd.notna(ma50) and ma50 != 0 else np.nan
        dist_200 = float(s.iloc[-1] / ma200 - 1) if pd.notna(ma200) and ma200 != 0 else np.nan
        dd = float(s.iloc[-1] / high_252 - 1) if pd.notna(high_252) and high_252 != 0 else np.nan
        pct_from_low = float(s.iloc[-1] / low_252 - 1) if pd.notna(low_252) and low_252 != 0 else np.nan

        rows.append(
            {
                "ticker": ticker,
                "asset": UNIVERSE[ticker]["name"],
                "sleeve": UNIVERSE[ticker]["sleeve"],
                "ret_1d": ret_1d,
                "ret_since_friday": ret_since_friday,
                "ret_1w": ret_1w,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_ytd": ret_ytd,
                "dist_20": dist_20,
                "dist_50": dist_50,
                "dist_200": dist_200,
                "drawdown": dd,
                "pct_from_52w_low": pct_from_low,
                "vol_1m": float(vol_1m) if pd.notna(vol_1m) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    score = (
        0.35 * zscore(df["ret_since_friday"])
        + 0.25 * zscore(df["ret_1m"])
        + 0.20 * zscore(df["dist_50"])
        + 0.10 * zscore(df["dist_200"])
        - 0.10 * zscore(df["vol_1m"])
    )
    df["score"] = score

    stretched_up = (df["dist_20"] > 0.06) & (df["ret_1d"] > 0.015)
    stretched_down = (df["dist_20"] < -0.06) & (df["ret_1d"] < -0.015)

    df["action"] = np.select(
        [
            (df["score"] > 0.7) & (df["dist_50"] > 0),
            (df["score"] < -0.7) & (df["dist_50"] < 0),
            stretched_up | stretched_down,
        ],
        ["Leadership", "Pressure", "Stretch"],
        default="Neutral",
    )

    return df.sort_values(["score", "ret_since_friday"], ascending=[False, False]).reset_index(drop=True)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / std


def get_value(df: pd.DataFrame, ticker: str, col: str) -> float:
    row = df.loc[df["ticker"] == ticker, col]
    if row.empty:
        return np.nan
    return float(row.iloc[0])


def regime_summary(features: pd.DataFrame) -> Tuple[str, str]:
    eq = np.nanmean([get_value(features, x, "ret_since_friday") for x in ["SPY", "QQQ", "IWM", "RSP"]])
    rates = get_value(features, "TLT", "ret_since_friday")
    credit = get_value(features, "HYG", "ret_since_friday") - get_value(features, "LQD", "ret_since_friday")
    dollar = get_value(features, "UUP", "ret_since_friday")
    vol = get_value(features, "VIXY", "ret_since_friday")
    oil = get_value(features, "USO", "ret_since_friday")

    score = 0
    score += 1 if eq > 0 else -1
    score += 1 if credit > 0 else -1
    score += 1 if rates > 0 else -1
    score += 1 if dollar < 0 else -1
    score += 1 if vol < 0 else -1

    if score >= 3:
        label = "Risk-on follow-through"
        note = "Equities, credit, duration and vol are moving in a direction consistent with easier conditions. Broad beta can work, but watch for crowded growth leadership."
    elif score <= -3:
        label = "Risk-off pressure"
        note = "The post-Friday tape is being confirmed by tightening signals. Respect dollar strength, credit slippage, and equity weakness before pressing gross."
    else:
        label = "Mixed tape / dispersion"
        note = "The tape is not giving clean cross-asset confirmation. Focus on relative-value, keep gross honest, and let leadership versus laggards do the work."

    if oil > 0.04:
        note += " Oil is adding inflation pressure."
    elif oil < -0.04:
        note += " Oil is reinforcing the disinflation impulse."

    return label, note


def make_tape_chart(close: pd.DataFrame, tickers: List[str], anchors: AnchorDates, lookback: int) -> go.Figure:
    tickers = [t for t in tickers if t in close.columns]
    data = close[tickers].dropna(how="all").tail(lookback)
    norm = data.div(data.iloc[0]).mul(100)

    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[ticker],
                mode="lines",
                name=ticker,
                line=dict(width=2),
                hovertemplate=f"{ticker}<br>%{{x|%b %d, %Y}}<br>Indexed: %{{y:.1f}}<extra></extra>",
            )
        )

    # Friday and latest annotations
    if anchors.last_friday in norm.index:
        fig.add_vline(x=anchors.last_friday, line_width=1.5, line_dash="dash")
        fig.add_annotation(
            x=anchors.last_friday,
            y=1.02,
            yref="paper",
            text=f"Friday<br>{anchors.last_friday.strftime('%b %d')}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
        )
    if anchors.latest in norm.index:
        fig.add_vline(x=anchors.latest, line_width=1.5)
        fig.add_annotation(
            x=anchors.latest,
            y=1.02,
            yref="paper",
            text=f"Today<br>{anchors.latest.strftime('%b %d')}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        yaxis_title="Indexed to 100",
        xaxis_title="",
        hovermode="x unified",
    )
    return fig


def make_heatmap(features: pd.DataFrame) -> go.Figure:
    cols = ["ret_1d", "ret_since_friday", "ret_1w", "ret_1m", "ret_ytd"]
    labels = ["Today", "Since Friday", "1W", "1M", "YTD"]
    data = features.sort_values(["sleeve", "ret_since_friday"], ascending=[True, False]).copy()
    z = data[cols].to_numpy() * 100
    text = np.empty_like(z, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            text[i, j] = "" if pd.isna(val) else f"{val:+.1f}%"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=[f"{t}  {a}" for t, a in zip(data["ticker"], data["asset"])],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            colorscale="RdYlGn",
            zmid=0,
            colorbar_title="%",
        )
    )
    fig.update_layout(height=max(520, 24 * len(data)), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def make_since_friday_bar(features: pd.DataFrame) -> go.Figure:
    data = features.sort_values("ret_since_friday", ascending=True).copy()
    data["since_friday_pct"] = data["ret_since_friday"] * 100
    fig = px.bar(
        data,
        x="since_friday_pct",
        y="ticker",
        color="sleeve",
        orientation="h",
        hover_data={"asset": True, "since_friday_pct": ":.1f", "ret_1d": ":.2%", "ret_1m": ":.2%"},
        labels={"since_friday_pct": "Since Friday (%)", "ticker": ""},
    )
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def make_scatter(features: pd.DataFrame) -> go.Figure:
    data = features.copy()
    data["today_pct"] = data["ret_1d"] * 100
    data["since_friday_pct"] = data["ret_since_friday"] * 100
    data["dist_50_pct"] = data["dist_50"] * 100
    data["vol_1m_pct"] = data["vol_1m"] * 100
    fig = px.scatter(
        data,
        x="dist_50_pct",
        y="since_friday_pct",
        color="sleeve",
        size="vol_1m_pct",
        hover_name="ticker",
        hover_data={
            "asset": True,
            "today_pct": ":.1f",
            "since_friday_pct": ":.1f",
            "ret_1m": ":.1%",
            "score": ":.2f",
            "vol_1m_pct": False,
        },
        labels={"dist_50_pct": "Distance from 50D MA (%)", "since_friday_pct": "Move since Friday (%)"},
    )
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def make_sleeve_bars(features: pd.DataFrame) -> go.Figure:
    grouped = (
        features.groupby("sleeve", as_index=False)[["ret_1d", "ret_since_friday", "ret_1m"]]
        .mean(numeric_only=True)
        .melt(id_vars="sleeve", var_name="period", value_name="ret")
    )
    mapping = {"ret_1d": "Today", "ret_since_friday": "Since Friday", "ret_1m": "1M"}
    grouped["period"] = grouped["period"].map(mapping)
    grouped["ret_pct"] = grouped["ret"] * 100
    fig = px.bar(
        grouped,
        x="sleeve",
        y="ret_pct",
        color="period",
        barmode="group",
        labels={"ret_pct": "Return (%)", "sleeve": ""},
    )
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_takeaways(features: pd.DataFrame) -> dict:
    leadership = features.sort_values(["ret_since_friday", "ret_1d"], ascending=False).head(5)
    pressure = features.sort_values(["ret_since_friday", "ret_1d"], ascending=True).head(5)
    reversals = features.reindex(features["ret_1d"].abs().sort_values(ascending=False).index).head(5)

    def format_lines(df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            lines.append(f"• {row['ticker']} {pct_fmt(row['ret_1d'])} today, {pct_fmt(row['ret_since_friday'])} since Friday")
        return "\n".join(lines)

    return {
        "leadership": format_lines(leadership),
        "pressure": format_lines(pressure),
        "reversals": format_lines(reversals),
    }


def make_table(features: pd.DataFrame) -> pd.DataFrame:
    out = features[[
        "ticker",
        "asset",
        "sleeve",
        "action",
        "ret_1d",
        "ret_since_friday",
        "ret_1w",
        "ret_1m",
        "ret_ytd",
        "dist_50",
        "dist_200",
        "drawdown",
        "score",
    ]].copy()
    for c in ["ret_1d", "ret_since_friday", "ret_1w", "ret_1m", "ret_ytd", "dist_50", "dist_200", "drawdown"]:
        out[c] = out[c].map(pct_fmt)
    out["score"] = out["score"].map(num_fmt)
    return out


st.title("Weekly Cross-Asset Compass")
st.caption("Visual-first weekly macro page. The point is to surface what changed from Friday to today, where confirmation is broad, and where pressure is building.")

with st.expander("Controls", expanded=False):
    period = st.selectbox("History", ["1y", "18mo", "2y", "3y"], index=1)
    include = st.multiselect(
        "Assets",
        list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys()),
    )
    tape_focus = st.multiselect(
        "Tape chart focus",
        include,
        default=[t for t in CORE_TAPE if t in include],
    )
    lookback = st.slider("Tape lookback (trading days)", min_value=60, max_value=252, value=126, step=21)
    st.caption("Data source: Yahoo Finance via yfinance. This is for weekly triage and should sit on top of your real data stack, not replace it.")

if not include:
    st.warning("Select at least one asset.")
    st.stop()

with st.spinner("Loading data..."):
    close = load_prices(include, period)

if close.empty:
    st.error("No price data returned.")
    st.stop()

anchors = get_anchor_dates(close)
features = build_feature_table(close, anchors)
if features.empty:
    st.error("Not enough data to build signals.")
    st.stop()

regime, note = regime_summary(features)
takeaways = build_takeaways(features)

# Snapshot row
snap1, snap2, snap3, snap4, snap5, snap6 = st.columns(6)
snap1.metric("S&P 500 today", pct_fmt(get_value(features, "SPY", "ret_1d")), pct_fmt(get_value(features, "SPY", "ret_since_friday")))
snap2.metric("Nasdaq today", pct_fmt(get_value(features, "QQQ", "ret_1d")), pct_fmt(get_value(features, "QQQ", "ret_since_friday")))
snap3.metric("TLT today", pct_fmt(get_value(features, "TLT", "ret_1d")), pct_fmt(get_value(features, "TLT", "ret_since_friday")))
snap4.metric("HY minus IG", pct_fmt(get_value(features, "HYG", "ret_since_friday") - get_value(features, "LQD", "ret_since_friday")))
snap5.metric("Dollar since Friday", pct_fmt(get_value(features, "UUP", "ret_since_friday")))
snap6.metric("Oil since Friday", pct_fmt(get_value(features, "USO", "ret_since_friday")))

st.markdown(f"### Regime: {regime}")
st.write(note)
st.caption(
    f"Latest available close: {anchors.latest.strftime('%A, %b %d, %Y')}. "
    f"Reference Friday: {anchors.last_friday.strftime('%A, %b %d, %Y')}. "
    f"Metric deltas in the snapshot row are moves since Friday."
)

st.markdown("## Friday to today")
left, right = st.columns([1.05, 0.95])
with left:
    st.plotly_chart(make_heatmap(features), use_container_width=True)
with right:
    st.plotly_chart(make_since_friday_bar(features), use_container_width=True)

st.markdown("## Cross-asset tape")
st.plotly_chart(make_tape_chart(close, tape_focus, anchors, lookback), use_container_width=True)

st.markdown("## Regime map")
left2, right2 = st.columns(2)
with left2:
    st.plotly_chart(make_scatter(features), use_container_width=True)
with right2:
    st.plotly_chart(make_sleeve_bars(features), use_container_width=True)

st.markdown("## Actionable takeaways")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="callout"><div class="bigline">Leadership</div><div class="mini">What is being bid both today and since Friday.</div><br><pre style="white-space: pre-wrap; font-family: inherit;">' + takeaways["leadership"] + '</pre></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="callout"><div class="bigline">Pressure</div><div class="mini">What is getting sold and where the tape is weak.</div><br><pre style="white-space: pre-wrap; font-family: inherit;">' + takeaways["pressure"] + '</pre></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="callout"><div class="bigline">Attention</div><div class="mini">Largest daily swings. These are the names to inspect first.</div><br><pre style="white-space: pre-wrap; font-family: inherit;">' + takeaways["reversals"] + '</pre></div>', unsafe_allow_html=True)

st.markdown("## Full cross-asset table")
st.dataframe(make_table(features), use_container_width=True, hide_index=True)

st.caption(
    "Leadership means positive tape confirmation through today and since Friday. "
    "Pressure means the opposite. Stretch flags are based on short-term extension versus the 20-day trend combined with a large daily move."
)
