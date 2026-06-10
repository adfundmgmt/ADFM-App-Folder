from __future__ import annotations

from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# ============================================================
# Page Config
# ============================================================

TITLE = "Credit Conditions Monitor"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (3, 8)

FRED_SERIES: Dict[str, str] = {
    "BAMLH0A0HYM2": "HY OAS",
    "BAMLC0A0CM": "IG OAS",
    "BAMLH0A3HYC": "CCC OAS",
    "NFCI": "Chicago Fed NFCI",
}

MARKET_TICKERS: Tuple[str, ...] = (
    "HYG",
    "JNK",
    "LQD",
    "BKLN",
    "EMB",
    "KRE",
    "SPY",
    "TLT",
    "IEF",
    "SHY",
)

COLORS = {
    "green": "#2f9e44",
    "red": "#d9480f",
    "amber": "#f59f00",
    "blue": "#2563eb",
    "purple": "#7c3aed",
    "grey": "#6b7280",
    "light_grey": "#f3f4f6",
    "dark": "#111827",
}

LOOKBACKS = {
    "1W": 7,
    "1M": 30,
    "3M": 91,
    "6M": 182,
    "YTD": "YTD",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2.0rem;
            max-width: 1550px;
        }

        .adfm-title {
            font-size: 1.95rem;
            line-height: 1.15;
            font-weight: 780;
            margin: 0 0 0.25rem 0;
            color: #111827;
        }

        .adfm-subtitle {
            font-size: 0.96rem;
            color: #6b7280;
            margin-bottom: 1.1rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
            border: 1px solid #e5e7eb;
            border-radius: 13px;
            padding: 13px 15px 11px 15px;
            min-height: 98px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        .metric-label {
            font-size: 0.72rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            margin-bottom: 0.42rem;
        }

        .metric-value {
            font-size: 1.26rem;
            font-weight: 760;
            color: #111827;
            line-height: 1.18;
        }

        .metric-footnote {
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.42rem;
            line-height: 1.35;
        }

        .section-title {
            font-size: 1.04rem;
            font-weight: 760;
            color: #111827;
            margin-top: 0.85rem;
            margin-bottom: 0.45rem;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px 14px;
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def safe_last_date(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    try:
        return pd.to_datetime(df.dropna(how="all").index[-1]).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def fmt_num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 1, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "N/A"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def fmt_bps(x: float, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "N/A"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.0f} bps"


def signal_color(score: float) -> str:
    if score <= -0.35:
        return COLORS["green"]
    if score >= 0.35:
        return COLORS["red"]
    return COLORS["amber"]


def classify_credit(score: float) -> str:
    if score <= -0.60:
        return "Easy / Risk-On"
    if score <= -0.20:
        return "Constructive"
    if score < 0.25:
        return "Neutral / Watch"
    if score < 0.60:
        return "Tightening"
    return "Stress / De-risk"


def asof_value(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan

    clean = clean.sort_index()
    eligible = clean.loc[clean.index <= target]

    if eligible.empty:
        return np.nan

    return float(eligible.iloc[-1])


def absolute_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan

    now = latest(clean)
    then = asof_value(clean, target)

    if not np.isfinite(now) or not np.isfinite(then):
        return np.nan

    return float(now - then)


def pct_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan

    now = latest(clean)
    then = asof_value(clean, target)

    if not np.isfinite(now) or not np.isfinite(then) or then == 0:
        return np.nan

    return float(now / then - 1.0)


def drawdown(series: pd.Series, window: int = 126) -> float:
    clean = series.dropna().tail(window)
    if clean.empty:
        return np.nan
    peak = clean.cummax().iloc[-1]
    if peak == 0:
        return np.nan
    return float(clean.iloc[-1] / peak - 1.0)


def zscore(series: pd.Series, window: int = 252) -> float:
    clean = series.dropna().tail(window)
    if len(clean) < 30:
        return np.nan

    std = clean.std()
    if not np.isfinite(std) or std == 0:
        return np.nan

    return float((clean.iloc[-1] - clean.mean()) / std)


def rolling_percentile_last(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    clean = series.astype(float).copy()

    def pct_rank(x: np.ndarray) -> float:
        s = pd.Series(x).dropna()
        if len(s) < min_periods:
            return np.nan
        return float(s.rank(pct=True).iloc[-1])

    return clean.rolling(window=window, min_periods=min_periods).apply(pct_rank, raw=True)


def rebase(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.dropna(how="all").copy()
    if clean.empty:
        return clean

    clean = clean.ffill().bfill()
    first = clean.iloc[0].replace(0, np.nan)
    return clean.divide(first).multiply(100)


def lookback_target(label: str) -> pd.Timestamp:
    today = pd.Timestamp(date.today())

    if label == "YTD":
        return pd.Timestamp(date(today.year, 1, 1))

    days = LOOKBACKS.get(label, 30)
    if isinstance(days, int):
        return today - pd.Timedelta(days=days)

    return today - pd.Timedelta(days=30)


def source_status(fred: pd.DataFrame, market: pd.DataFrame) -> str:
    fred_date = safe_last_date(fred)
    market_date = safe_last_date(market)

    if not fred.empty and not market.empty:
        return f"FRED latest: {fred_date} | Market latest: {market_date}"
    if not fred.empty:
        return f"FRED latest: {fred_date} | Market proxies unavailable"
    if not market.empty:
        return f"FRED unavailable | Market latest: {market_date}"
    return "No source data loaded"


# ============================================================
# Data Fetching
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_ids: Tuple[str, ...], start_iso: str) -> pd.DataFrame:
    frames: List[pd.Series] = []

    for series_id in series_ids:
        try:
            response = requests.get(
                FRED_URL,
                params={"id": series_id, "cosd": start_iso},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            raw = pd.read_csv(StringIO(response.text))
            if raw.empty or len(raw.columns) < 2:
                continue

            raw.columns = ["date", series_id]
            values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")
            dates = pd.to_datetime(raw["date"], errors="coerce")

            series = pd.Series(values.values, index=dates, name=series_id)
            series = series[~series.index.isna()]
            frames.append(series)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.ffill()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    try:
        data = yf.download(
            list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()
            else:
                return pd.DataFrame()
        else:
            close = data[["Close"]].copy()
            close.columns = [tickers[0]]

        close = close.dropna(how="all").ffill()
        close.index = pd.to_datetime(close.index)
        close = close[~close.index.duplicated(keep="last")]
        return close

    except Exception:
        return pd.DataFrame()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Tracks credit stress across cash spreads and liquid market proxies.

        The goal is to identify whether credit is confirming risk appetite or exposing pressure through HY, CCC, banks, loans, EM debt, and duration-sensitive credit.
        """
    )

    st.divider()

    st.header("Controls")
    fred_years = st.selectbox("Spread history", [3, 5, 10, 20], index=2)
    market_period = st.selectbox("Market proxy history", ["1y", "2y", "3y", "5y"], index=2)
    focus_window = st.selectbox("Focus window", ["1W", "1M", "3M", "6M", "YTD"], index=1)
    stress_window = st.selectbox("Stress percentile window", [126, 252, 504], index=1, format_func=lambda x: f"{x} trading days")
    show_raw = st.checkbox("Show raw data", value=False)


# ============================================================
# Load Data
# ============================================================

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Visual-first credit regime, spread pressure, bank stress, loan risk appetite, and HY market confirmation.</div>",
    unsafe_allow_html=True,
)

start = date.today() - timedelta(days=int(fred_years * 365.25) + 120)

fred = fetch_fred(tuple(FRED_SERIES.keys()), start.isoformat())
market = fetch_market(tuple(MARKET_TICKERS), market_period)

if fred.empty and market.empty:
    st.error("No credit or market data loaded. Check FRED/Yahoo connectivity.")
    st.stop()


# ============================================================
# Derived Proxies
# ============================================================

proxy = pd.DataFrame()

if not market.empty:
    proxy = pd.DataFrame(index=market.index)

    if {"HYG", "LQD"}.issubset(market.columns):
        proxy["HYG/LQD"] = market["HYG"] / market["LQD"]

    if {"JNK", "LQD"}.issubset(market.columns):
        proxy["JNK/LQD"] = market["JNK"] / market["LQD"]

    if {"KRE", "SPY"}.issubset(market.columns):
        proxy["KRE/SPY"] = market["KRE"] / market["SPY"]

    if {"BKLN", "LQD"}.issubset(market.columns):
        proxy["BKLN/LQD"] = market["BKLN"] / market["LQD"]

    if {"EMB", "LQD"}.issubset(market.columns):
        proxy["EMB/LQD"] = market["EMB"] / market["LQD"]

    if {"TLT", "SHY"}.issubset(market.columns):
        proxy["TLT/SHY"] = market["TLT"] / market["SHY"]


# ============================================================
# Composite Credit Stress
# Higher = tighter / worse
# Lower = easier / better
# ============================================================

stress_components = pd.DataFrame()

for series_id, label in FRED_SERIES.items():
    if series_id in fred.columns:
        pct = rolling_percentile_last(fred[series_id], window=stress_window)
        stress_components[label] = pct * 2.0 - 1.0

if "HYG/LQD" in proxy.columns:
    pct = rolling_percentile_last(proxy["HYG/LQD"], window=stress_window)
    stress_components["HYG/LQD Stress"] = -(pct * 2.0 - 1.0)

if "JNK/LQD" in proxy.columns:
    pct = rolling_percentile_last(proxy["JNK/LQD"], window=stress_window)
    stress_components["JNK/LQD Stress"] = -(pct * 2.0 - 1.0)

if "KRE/SPY" in proxy.columns:
    pct = rolling_percentile_last(proxy["KRE/SPY"], window=stress_window)
    stress_components["KRE/SPY Stress"] = -(pct * 2.0 - 1.0)

if "EMB/LQD" in proxy.columns:
    pct = rolling_percentile_last(proxy["EMB/LQD"], window=stress_window)
    stress_components["EMB/LQD Stress"] = -(pct * 2.0 - 1.0)

if not market.empty and "HYG" in market.columns:
    hyg_dd = market["HYG"] / market["HYG"].rolling(126, min_periods=30).max() - 1.0
    stress_components["HYG Drawdown Stress"] = np.clip(-hyg_dd * 10.0, -1.0, 1.0)

if stress_components.empty:
    credit_score_ts = pd.Series(dtype=float)
    credit_score = 0.0
else:
    stress_components = stress_components.sort_index().ffill()
    credit_score_ts = stress_components.mean(axis=1).dropna()
    credit_score = latest(credit_score_ts) if not credit_score_ts.empty else 0.0

credit_regime = classify_credit(credit_score)


# ============================================================
# Key Metrics
# ============================================================

target = lookback_target(focus_window)

hy_oas = latest(fred["BAMLH0A0HYM2"]) if "BAMLH0A0HYM2" in fred.columns else np.nan
ig_oas = latest(fred["BAMLC0A0CM"]) if "BAMLC0A0CM" in fred.columns else np.nan
ccc_oas = latest(fred["BAMLH0A3HYC"]) if "BAMLH0A3HYC" in fred.columns else np.nan
nfci_latest = latest(fred["NFCI"]) if "NFCI" in fred.columns else np.nan

hy_change = absolute_change_since(fred["BAMLH0A0HYM2"] * 100, target) if "BAMLH0A0HYM2" in fred.columns else np.nan
ccc_change = absolute_change_since(fred["BAMLH0A3HYC"] * 100, target) if "BAMLH0A3HYC" in fred.columns else np.nan

hyg_lqd_ret = pct_change_since(proxy["HYG/LQD"], target) * 100 if "HYG/LQD" in proxy.columns else np.nan
kre_spy_ret = pct_change_since(proxy["KRE/SPY"], target) * 100 if "KRE/SPY" in proxy.columns else np.nan
emb_lqd_ret = pct_change_since(proxy["EMB/LQD"], target) * 100 if "EMB/LQD" in proxy.columns else np.nan
hyg_drawdown = drawdown(market["HYG"], 126) * 100 if not market.empty and "HYG" in market.columns else np.nan

cards = [
    (
        "Credit Regime",
        credit_regime,
        f"Composite {credit_score:+.2f}",
        signal_color(credit_score),
    ),
    (
        "HY OAS",
        fmt_bps(hy_oas * 100) if np.isfinite(hy_oas) else "N/A",
        f"{focus_window} {fmt_bps(hy_change, signed=True)}",
        COLORS["red"] if np.isfinite(hy_change) and hy_change > 20 else COLORS["blue"],
    ),
    (
        "CCC OAS",
        fmt_bps(ccc_oas * 100) if np.isfinite(ccc_oas) else "N/A",
        f"{focus_window} {fmt_bps(ccc_change, signed=True)}",
        COLORS["red"] if np.isfinite(ccc_change) and ccc_change > 50 else COLORS["purple"],
    ),
    (
        "HYG/LQD",
        fmt_pct(hyg_lqd_ret, signed=True),
        f"{focus_window} relative",
        COLORS["green"] if np.isfinite(hyg_lqd_ret) and hyg_lqd_ret > 0 else COLORS["red"],
    ),
    (
        "KRE/SPY",
        fmt_pct(kre_spy_ret, signed=True),
        f"{focus_window} relative",
        COLORS["green"] if np.isfinite(kre_spy_ret) and kre_spy_ret > 0 else COLORS["red"],
    ),
    (
        "HYG Drawdown",
        fmt_pct(hyg_drawdown),
        "6M drawdown",
        COLORS["red"] if np.isfinite(hyg_drawdown) and hyg_drawdown < -3 else COLORS["amber"],
    ),
]

for col, (label, value, footnote, color) in zip(st.columns(6), cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
                <div class="metric-footnote">{footnote}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Visuals First
# ============================================================

top_left, top_right = st.columns([1.2, 0.8])

with top_left:
    st.markdown("<div class='section-title'>Credit Stress Pulse</div>", unsafe_allow_html=True)

    if credit_score_ts.empty:
        st.info("Composite stress could not be calculated with the available data.")
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=credit_score_ts.index,
                y=credit_score_ts,
                mode="lines",
                name="Composite Credit Stress",
                line=dict(color=signal_color(credit_score), width=2.6),
            )
        )

        fig.add_hline(y=0.60, line_width=1, line_dash="dot", line_color=COLORS["red"])
        fig.add_hline(y=0.25, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        fig.add_hline(y=-0.20, line_width=1, line_dash="dot", line_color=COLORS["green"])
        fig.add_hline(y=0, line_width=1, line_color="#d1d5db")

        fig.update_layout(
            height=385,
            margin=dict(l=12, r=12, t=12, b=12),
            yaxis_title="Stress score",
            xaxis_title=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.markdown("<div class='section-title'>Pressure Map</div>", unsafe_allow_html=True)

    pressure_rows: List[List[str]] = []

    def add_pressure_row(name: str, value: float, z: float, read: str, inverse: bool = False) -> None:
        if not np.isfinite(z):
            regime = "N/A"
        else:
            adjusted = -z if inverse else z
            if adjusted >= 1.0:
                regime = "Stress"
            elif adjusted <= -1.0:
                regime = "Easy"
            else:
                regime = "Neutral"

        pressure_rows.append([name, value, z, regime, read])

    if "BAMLH0A0HYM2" in fred.columns:
        add_pressure_row("HY OAS", hy_oas * 100, zscore(fred["BAMLH0A0HYM2"], 252), "Cash HY spread")

    if "BAMLH0A3HYC" in fred.columns:
        add_pressure_row("CCC OAS", ccc_oas * 100, zscore(fred["BAMLH0A3HYC"], 252), "Lower-quality credit")

    if "NFCI" in fred.columns:
        add_pressure_row("NFCI", nfci_latest, zscore(fred["NFCI"], 252), "Financial conditions")

    if "HYG/LQD" in proxy.columns:
        add_pressure_row("HYG/LQD", latest(proxy["HYG/LQD"]), zscore(proxy["HYG/LQD"], 252), "HY beta vs IG", inverse=True)

    if "KRE/SPY" in proxy.columns:
        add_pressure_row("KRE/SPY", latest(proxy["KRE/SPY"]), zscore(proxy["KRE/SPY"], 252), "Bank equity stress", inverse=True)

    if "EMB/LQD" in proxy.columns:
        add_pressure_row("EMB/LQD", latest(proxy["EMB/LQD"]), zscore(proxy["EMB/LQD"], 252), "EM credit pressure", inverse=True)

    if pressure_rows:
        pressure_df = pd.DataFrame(
            pressure_rows,
            columns=["Signal", "Latest", "Z-Score", "Regime", "Read-through"],
        )

        pressure_df["Latest"] = pressure_df["Latest"].apply(lambda x: fmt_num(x, 2))
        pressure_df["Z-Score"] = pressure_df["Z-Score"].apply(lambda x: fmt_num(x, 2))

        st.dataframe(
            pressure_df,
            use_container_width=True,
            hide_index=True,
            height=385,
        )
    else:
        st.info("Pressure map unavailable.")


# ============================================================
# Spread and Proxy Charts
# ============================================================

left, right = st.columns([1.05, 0.95])

with left:
    st.markdown("<div class='section-title'>Cash Credit Spreads</div>", unsafe_allow_html=True)

    spread_cols = [c for c in ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC"] if c in fred.columns]

    if not spread_cols:
        st.info("FRED credit-spread series did not load.")
    else:
        spread_fig = make_subplots(specs=[[{"secondary_y": True}]])

        if "BAMLH0A0HYM2" in fred.columns:
            spread_fig.add_trace(
                go.Scatter(
                    x=fred.index,
                    y=fred["BAMLH0A0HYM2"] * 100,
                    name="HY OAS",
                    line=dict(color=COLORS["red"], width=2),
                ),
                secondary_y=False,
            )

        if "BAMLC0A0CM" in fred.columns:
            spread_fig.add_trace(
                go.Scatter(
                    x=fred.index,
                    y=fred["BAMLC0A0CM"] * 100,
                    name="IG OAS",
                    line=dict(color=COLORS["blue"], width=2),
                ),
                secondary_y=False,
            )

        if "BAMLH0A3HYC" in fred.columns:
            spread_fig.add_trace(
                go.Scatter(
                    x=fred.index,
                    y=fred["BAMLH0A3HYC"] * 100,
                    name="CCC OAS",
                    line=dict(color=COLORS["purple"], width=2),
                ),
                secondary_y=True,
            )

        spread_fig.update_yaxes(title_text="HY / IG OAS bps", secondary_y=False)
        spread_fig.update_yaxes(title_text="CCC OAS bps", secondary_y=True)

        spread_fig.update_layout(
            height=400,
            margin=dict(l=12, r=12, t=12, b=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )

        st.plotly_chart(spread_fig, use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Market Credit Proxies</div>", unsafe_allow_html=True)

    if proxy.empty:
        st.info("Market proxy data unavailable.")
    else:
        proxy_plot_cols = [c for c in ["HYG/LQD", "JNK/LQD", "KRE/SPY", "BKLN/LQD", "EMB/LQD"] if c in proxy.columns]
        proxy_rebased = rebase(proxy[proxy_plot_cols])

        proxy_fig = go.Figure()

        for col_name in proxy_rebased.columns:
            proxy_fig.add_trace(
                go.Scatter(
                    x=proxy_rebased.index,
                    y=proxy_rebased[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(width=2),
                )
            )

        proxy_fig.add_hline(y=100, line_width=1, line_color="#d1d5db")

        proxy_fig.update_layout(
            height=400,
            margin=dict(l=12, r=12, t=12, b=12),
            yaxis_title="Rebased to 100",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )

        st.plotly_chart(proxy_fig, use_container_width=True)


# ============================================================
# Decision Table
# ============================================================

st.markdown("<div class='section-title'>Credit Tape</div>", unsafe_allow_html=True)

tape_rows: List[Dict[str, str]] = []


def add_spread_tape(name: str, series: pd.Series, read: str) -> None:
    tape_rows.append(
        {
            "Signal": name,
            "Latest": fmt_bps(latest(series) * 100),
            "1W": fmt_bps(absolute_change_since(series * 100, lookback_target("1W")), signed=True),
            "1M": fmt_bps(absolute_change_since(series * 100, lookback_target("1M")), signed=True),
            "3M": fmt_bps(absolute_change_since(series * 100, lookback_target("3M")), signed=True),
            "YTD": fmt_bps(absolute_change_since(series * 100, lookback_target("YTD")), signed=True),
            "Z": fmt_num(zscore(series, 252), 2),
            "Read-through": read,
        }
    )


def add_ratio_tape(name: str, series: pd.Series, read: str) -> None:
    tape_rows.append(
        {
            "Signal": name,
            "Latest": fmt_num(latest(series), 3),
            "1W": fmt_pct(pct_change_since(series, lookback_target("1W")) * 100, signed=True),
            "1M": fmt_pct(pct_change_since(series, lookback_target("1M")) * 100, signed=True),
            "3M": fmt_pct(pct_change_since(series, lookback_target("3M")) * 100, signed=True),
            "YTD": fmt_pct(pct_change_since(series, lookback_target("YTD")) * 100, signed=True),
            "Z": fmt_num(zscore(series, 252), 2),
            "Read-through": read,
        }
    )


if "BAMLH0A0HYM2" in fred.columns:
    add_spread_tape("HY OAS", fred["BAMLH0A0HYM2"], "Wider means tighter credit")

if "BAMLC0A0CM" in fred.columns:
    add_spread_tape("IG OAS", fred["BAMLC0A0CM"], "Investment-grade funding stress")

if "BAMLH0A3HYC" in fred.columns:
    add_spread_tape("CCC OAS", fred["BAMLH0A3HYC"], "Lower-quality default risk")

if "NFCI" in fred.columns:
    tape_rows.append(
        {
            "Signal": "NFCI",
            "Latest": fmt_num(latest(fred["NFCI"]), 2),
            "1W": fmt_num(absolute_change_since(fred["NFCI"], lookback_target("1W")), 2),
            "1M": fmt_num(absolute_change_since(fred["NFCI"], lookback_target("1M")), 2),
            "3M": fmt_num(absolute_change_since(fred["NFCI"], lookback_target("3M")), 2),
            "YTD": fmt_num(absolute_change_since(fred["NFCI"], lookback_target("YTD")), 2),
            "Z": fmt_num(zscore(fred["NFCI"], 252), 2),
            "Read-through": "Higher means tighter financial conditions",
        }
    )

if "HYG/LQD" in proxy.columns:
    add_ratio_tape("HYG/LQD", proxy["HYG/LQD"], "HY beta relative to IG")

if "JNK/LQD" in proxy.columns:
    add_ratio_tape("JNK/LQD", proxy["JNK/LQD"], "High-yield ETF confirmation")

if "KRE/SPY" in proxy.columns:
    add_ratio_tape("KRE/SPY", proxy["KRE/SPY"], "Regional bank stress proxy")

if "BKLN/LQD" in proxy.columns:
    add_ratio_tape("BKLN/LQD", proxy["BKLN/LQD"], "Loan risk appetite")

if "EMB/LQD" in proxy.columns:
    add_ratio_tape("EMB/LQD", proxy["EMB/LQD"], "EM credit pressure")

if tape_rows:
    tape = pd.DataFrame(tape_rows)
    st.dataframe(tape, use_container_width=True, hide_index=True, height=360)
else:
    st.info("No tape signals loaded.")


# ============================================================
# Interpretation Box
# ============================================================

if credit_score >= 0.60:
    readthrough = (
        "Credit is in stress mode. The dashboard is saying de-risking pressure is no longer isolated; "
        "cash spreads, credit ETFs, or bank proxies are moving in the same direction."
    )
elif credit_score >= 0.25:
    readthrough = (
        "Credit is tightening. This is the zone where equity rallies need confirmation from HYG/LQD and KRE/SPY. "
        "If those ratios fail while spreads widen, the tape is losing credit sponsorship."
    )
elif credit_score <= -0.20:
    readthrough = (
        "Credit remains supportive. Spreads and proxies are not signaling funding stress, which keeps the burden of proof on equity bears."
    )
else:
    readthrough = (
        "Credit is neutral. The signal is not strong enough by itself. Watch CCC OAS, KRE/SPY, and HYG/LQD for the next directional break."
    )

st.markdown(
    f"""
    <div class='note-box'>
        <b>Read-through:</b> {readthrough}<br>
        <b>Sources:</b> {source_status(fred, market)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Raw Data
# ============================================================

if show_raw:
    with st.expander("Raw FRED Data", expanded=False):
        if fred.empty:
            st.info("No FRED data loaded.")
        else:
            renamed = fred.rename(columns=FRED_SERIES)
            st.dataframe(renamed.tail(300), use_container_width=True)

    with st.expander("Raw Market Proxy Data", expanded=False):
        if proxy.empty:
            st.info("No market proxy data loaded.")
        else:
            st.dataframe(proxy.tail(300), use_container_width=True)
