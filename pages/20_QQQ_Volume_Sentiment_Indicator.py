import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="QQQ Volume Sentiment Indicator",
    layout="wide",
)

# =============================================================================
# STYLING
# =============================================================================
CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1650px;
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
    }
    h1, h2, h3 {
        letter-spacing: 0.1px;
        font-weight: 650;
    }
    .adfm-card {
        background: #fafafa;
        border: 1px solid #e9e9e9;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .adfm-small {
        color: #666;
        font-size: 0.92rem;
    }
    .adfm-metric-label {
        color: #666;
        font-size: 0.88rem;
        margin-bottom: 2px;
    }
    .adfm-metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #202223;
        line-height: 1.1;
    }
    .adfm-metric-sub {
        color: #666;
        font-size: 0.84rem;
        margin-top: 4px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #ececec;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "QQQ Volume as an Inverse Sentiment Indicator"
APP_SUBTITLE = (
    "High volume tends to align with fear and capitulation, while unusually quiet volume often "
    "shows complacency. The framework is inspired by McClellan-style sentiment work."
)
SYMBOL = "QQQ"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =============================================================================
# HELPERS
# =============================================================================
def metric_card(label: str, value: str, subtext: str = "") -> str:
    return f"""
    <div class="adfm-card">
        <div class="adfm-metric-label">{label}</div>
        <div class="adfm-metric-value">{value}</div>
        <div class="adfm-metric-sub">{subtext}</div>
    </div>
    """

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def format_billions(x: float) -> str:
    return f"{x / 1_000_000_000:.2f}B"

def format_millions(x: float) -> str:
    return f"{x / 1_000_000:.2f}M"

def today_utc_ts() -> pd.Timestamp:
    return pd.Timestamp.utcnow().normalize()

def _request_text(url: str, timeout: int = 20, retries: int = 3, sleep_s: float = 1.0) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(sleep_s * (attempt + 1))
    raise last_err

def fetch_from_stooq(symbol: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Stooq is a solid free fallback for OHLCV and often avoids Yahoo throttling.
    Data comes back in descending order.
    """
    stooq_symbol = symbol.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    text = _request_text(url, timeout=20, retries=3, sleep_s=1.0)
    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns:
        raise ValueError("Stooq returned no usable data.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.rename(columns=str.title)
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stooq data missing columns: {missing}")

    for col in expected:
        df[col] = safe_numeric(df[col])

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    df = df.sort_values("Date")
    df = df[df["Date"] >= start_date].copy()
    df = df.set_index("Date")
    return df

def fetch_from_yahoo_chart_api(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Public Yahoo chart endpoint. Lighter than repeated yfinance Ticker().history() usage.
    """
    period1 = int(start_date.timestamp())
    period2 = int((end_date + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&includePrePost=false&events=div%2Csplits"
    )
    resp_text = _request_text(url, timeout=20, retries=3, sleep_s=1.0)
    data = requests.models.complexjson.loads(resp_text)

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", None)

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s"),
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": adjclose if adjclose is not None else quote["close"],
        "Volume": quote["volume"],
    })

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = safe_numeric(df[col])

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
    df = df.set_index("Date").sort_index()
    return df

def fetch_from_yfinance_download(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Final fallback.
    """
    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[needed].copy()
    for col in needed:
        df[col] = safe_numeric(df[col])
    df = df.dropna(subset=needed)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_qqq_ohlcv(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    errors = []

    try:
        df = fetch_from_stooq(SYMBOL, start_date)
        if not df.empty:
            return df, "Stooq"
    except Exception as e:
        errors.append(f"Stooq: {e}")

    try:
        df = fetch_from_yahoo_chart_api(SYMBOL, start_date, end_date)
        if not df.empty:
            return df, "Yahoo Chart API"
    except Exception as e:
        errors.append(f"Yahoo Chart API: {e}")

    try:
        df = fetch_from_yfinance_download(SYMBOL, start_date, end_date)
        if not df.empty:
            return df, "yfinance"
    except Exception as e:
        errors.append(f"yfinance: {e}")

    raise RuntimeError(" | ".join(errors))

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_shares_outstanding(symbol: str) -> Optional[float]:
    """
    Optional enhancement only. If it fails, the app continues on raw volume.
    """
    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            shares = fast_info.get("shares")
            if shares and np.isfinite(shares):
                return float(shares)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        shares = info.get("sharesOutstanding")
        if shares and np.isfinite(shares):
            return float(shares)
    except Exception:
        pass

    return None

def compute_signal_table(
    df: pd.DataFrame,
    ma_period: int,
    bb_period: int,
    extreme_high_z: float,
    extreme_low_z: float,
    normalize_volume: bool,
    shares_outstanding: Optional[float],
    exclude_holiday_noise: bool,
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()

    if normalize_volume and shares_outstanding and shares_outstanding > 0:
        out["Vol_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Volume (% of Shares Outstanding)"
    else:
        out["Vol_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (Shares)"

    out["Vol_MA"] = out["Vol_Display"].rolling(ma_period, min_periods=ma_period).mean()
    out["Vol_Base"] = out["Vol_Display"].rolling(bb_period, min_periods=bb_period).mean()
    out["Vol_Std"] = out["Vol_Display"].rolling(bb_period, min_periods=bb_period).std(ddof=0)
    out["Vol_Z"] = (out["Vol_Display"] - out["Vol_Base"]) / out["Vol_Std"].replace(0, np.nan)

    out["Vol_Median_20"] = out["Vol_Display"].rolling(20, min_periods=10).median()
    out["Likely_Thin_Session"] = out["Vol_Display"] < 0.55 * out["Vol_Median_20"]

    out["Is_High"] = out["Vol_Z"] >= extreme_high_z
    out["Is_Low"] = out["Vol_Z"] <= extreme_low_z

    if exclude_holiday_noise:
        out.loc[out["Likely_Thin_Session"], "Is_Low"] = False

    return out, vol_label

def build_chart(
    df: pd.DataFrame,
    vol_label: str,
    ma_period: int,
    highlight_extremes: bool,
    vol_bar_opacity: float,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="QQQ",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
        ),
        row=1,
        col=1,
    )

    if highlight_extremes:
        highs = df[df["Is_High"]].copy()
        lows = df[df["Is_Low"]].copy()

        if not highs.empty:
            fig.add_trace(
                go.Scatter(
                    x=highs.index,
                    y=highs["Low"] * 0.992,
                    mode="markers",
                    name="High Volume Signal",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color="rgba(220, 38, 38, 0.95)",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Vol Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([highs["Close"], highs["Vol_Z"]]),
                ),
                row=1,
                col=1,
            )

        if not lows.empty:
            fig.add_trace(
                go.Scatter(
                    x=lows.index,
                    y=lows["High"] * 1.008,
                    mode="markers",
                    name="Low Volume Signal",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="rgba(245, 158, 11, 0.95)",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Vol Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([lows["Close"], lows["Vol_Z"]]),
                ),
                row=1,
                col=1,
            )

    bar_colors = np.where(
        df["Is_High"],
        "rgba(220, 38, 38, 0.90)",
        np.where(
            df["Is_Low"],
            "rgba(245, 158, 11, 0.90)",
            f"rgba(59, 130, 246, {vol_bar_opacity})",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            name="Volume",
            marker_color=bar_colors,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Vol_MA"],
            mode="lines",
            name=f"Volume {ma_period}D MA",
            line=dict(color="rgba(30, 64, 175, 0.9)", width=2.2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>MA: %{y:,.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=860,
        template="plotly_white",
        margin=dict(l=50, r=35, t=60, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
        ),
        title=dict(
            text="QQQ Price and Volume Sentiment Overlay",
            x=0.01,
            xanchor="left",
            font=dict(size=16),
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
        tickformat="%b %Y",
    )

    fig.update_yaxes(
        title_text="Price ($)",
        row=1,
        col=1,
        showgrid=True,
        gridcolor="rgba(120,120,120,0.15)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=vol_label,
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(120,120,120,0.12)",
        zeroline=False,
    )

    return fig

def style_extremes_table(df: pd.DataFrame, vol_label: str, normalize_volume: bool):
    fmt = {
        "Close": "${:.2f}",
        "Z-Score": "{:.2f}",
        "1D Return %": "{:.2f}",
        "5D Return %": "{:.2f}",
    }
    if normalize_volume:
        fmt[vol_label] = "{:.4f}"
    else:
        fmt[vol_label] = "{:,.0f}"
    return df.style.format(fmt)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    **Purpose**

    Track QQQ volume as a sentiment input. The framework assumes heavy ETF volume often appears
    during fear, hedging, and liquidation, while quiet tape can reflect complacency.

    **What it does**
    - Pulls QQQ OHLCV from multiple public-data paths with fallbacks
    - Computes volume z-scores against a rolling baseline
    - Flags unusually high-volume and low-volume sessions
    - Optionally normalizes volume by shares outstanding when available

    **Interpretation**
    - High-volume spikes often align with stress and local washouts
    - Low-volume readings can align with complacency, but they are noisier

    **Data handling**
    - Primary OHLCV sources: Stooq, Yahoo Chart API, yfinance fallback
    - Shares outstanding is optional and fetched separately
    """
)

st.sidebar.header("Settings")

lookback_months = st.sidebar.slider(
    "Lookback (months)",
    min_value=6,
    max_value=60,
    value=24,
    step=3,
)

ma_period = st.sidebar.slider(
    "Volume Moving Average Period",
    min_value=5,
    max_value=50,
    value=10,
    step=1,
)

normalize_volume = st.sidebar.checkbox(
    "Normalize Volume by Shares Outstanding",
    value=True,
    help="Uses shares outstanding when available. If it fails, the app automatically falls back to raw volume.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Detection")

bb_period = st.sidebar.slider(
    "Volume Baseline Lookback (days)",
    min_value=20,
    max_value=100,
    value=50,
    step=5,
)

highlight_extremes = st.sidebar.checkbox(
    "Highlight Extreme Volume Days",
    value=True,
)

extreme_high_z = st.sidebar.slider(
    "Bottom Signal Threshold (Z-score)",
    min_value=1.0,
    max_value=4.0,
    value=2.5,
    step=0.25,
)

extreme_low_z = st.sidebar.slider(
    "Top Signal Threshold (Z-score)",
    min_value=-4.0,
    max_value=-0.5,
    value=-2.0,
    step=0.25,
)

exclude_holiday_noise = st.sidebar.checkbox(
    "Filter Obvious Thin-Session Noise",
    value=True,
    help="Removes likely half-days and holiday-distorted low-volume readings from bearish low-volume flags.",
)

vol_bar_opacity = st.sidebar.slider(
    "Volume Bar Opacity",
    min_value=0.10,
    max_value=0.80,
    value=0.30,
    step=0.05,
)

# =============================================================================
# HEADER
# =============================================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# =============================================================================
# DATA FETCH
# =============================================================================
end_date = today_utc_ts()
warmup_days = max(bb_period * 3, 180)
start_date = end_date - pd.Timedelta(days=lookback_months * 31 + warmup_days)

with st.spinner("Loading QQQ data..."):
    try:
        raw_df, data_source = fetch_qqq_ohlcv(start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch QQQ data from all sources. Details: {e}")
        st.stop()

shares_outstanding = None
if normalize_volume:
    with st.spinner("Fetching shares outstanding..."):
        shares_outstanding = fetch_shares_outstanding(SYMBOL)

if raw_df.empty:
    st.warning("No data returned for the selected window.")
    st.stop()

# Trim to requested visible window after warmup
visible_start = end_date - pd.Timedelta(days=lookback_months * 31)
raw_df = raw_df[raw_df.index >= (visible_start - pd.Timedelta(days=warmup_days))].copy()

df, vol_label = compute_signal_table(
    raw_df,
    ma_period=ma_period,
    bb_period=bb_period,
    extreme_high_z=extreme_high_z,
    extreme_low_z=extreme_low_z,
    normalize_volume=normalize_volume,
    shares_outstanding=shares_outstanding,
    exclude_holiday_noise=exclude_holiday_noise,
)

df = df[df.index >= visible_start].copy()

if df.empty:
    st.warning("No usable data remained after processing.")
    st.stop()

# =============================================================================
# SUMMARY METRICS
# =============================================================================
latest = df.iloc[-1]
latest_close = float(latest["Close"])
latest_vol_z = float(latest["Vol_Z"]) if pd.notna(latest["Vol_Z"]) else np.nan
high_count = int(df["Is_High"].sum())
low_count = int(df["Is_Low"].sum())
latest_volume = float(latest["Vol_Display"])

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        metric_card(
            "Latest Close",
            f"${latest_close:,.2f}",
            f"Source: {data_source}",
        ),
        unsafe_allow_html=True,
    )

with c2:
    z_text = f"{latest_vol_z:.2f}" if np.isfinite(latest_vol_z) else "N/A"
    regime_text = (
        "Fear / washout zone"
        if np.isfinite(latest_vol_z) and latest_vol_z >= extreme_high_z
        else "Complacency / quiet tape"
        if np.isfinite(latest_vol_z) and latest_vol_z <= extreme_low_z
        else "Within normal range"
    )
    st.markdown(
        metric_card(
            "Latest Volume Z-Score",
            z_text,
            regime_text,
        ),
        unsafe_allow_html=True,
    )

with c3:
    vol_value = f"{latest_volume:.4f}" if "Shares Outstanding" in vol_label else f"{latest_volume:,.0f}"
    sub = (
        "Normalized volume"
        if "Shares Outstanding" in vol_label
        else "Raw shares volume"
    )
    st.markdown(
        metric_card(
            "Latest Volume Reading",
            vol_value,
            sub,
        ),
        unsafe_allow_html=True,
    )

with c4:
    shares_text = (
        format_billions(shares_outstanding)
        if shares_outstanding and np.isfinite(shares_outstanding)
        else "Unavailable"
    )
    st.markdown(
        metric_card(
            "Shares Outstanding",
            shares_text,
            "Used only when normalization is enabled and available",
        ),
        unsafe_allow_html=True,
    )

# =============================================================================
# CHART
# =============================================================================
fig = build_chart(
    df=df,
    vol_label=vol_label,
    ma_period=ma_period,
    highlight_extremes=highlight_extremes,
    vol_bar_opacity=vol_bar_opacity,
)
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# INTERPRETATION BLOCK
# =============================================================================
recent_highs_n = int(df["Is_High"].tail(63).sum())  # about 3 months
recent_lows_n = int(df["Is_Low"].tail(63).sum())

st.markdown("### Read on the Tape")
interp_left, interp_right = st.columns([1.35, 1])

with interp_left:
    st.markdown(
        f"""
        <div class="adfm-card">
            <div><b>Current read</b></div>
            <div class="adfm-small" style="margin-top:8px;">
                Over the visible window, the model flagged <b>{high_count}</b> high-volume stress signals and
                <b>{low_count}</b> low-volume complacency signals. Over roughly the last 3 months, that narrows to
                <b>{recent_highs_n}</b> and <b>{recent_lows_n}</b>, respectively.
                The latest reading sits at a volume z-score of <b>{z_text}</b>, which places current activity in
                <b>{regime_text.lower()}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with interp_right:
    source_note = (
        "Normalization active"
        if "Shares Outstanding" in vol_label
        else "Normalization inactive or unavailable"
    )
    st.markdown(
        f"""
        <div class="adfm-card">
            <div><b>Data quality note</b></div>
            <div class="adfm-small" style="margin-top:8px;">
                OHLCV source used this run: <b>{data_source}</b>.<br><br>
                Volume mode: <b>{source_note}</b>.<br><br>
                Thin-session filter: <b>{"On" if exclude_holiday_noise else "Off"}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# EXTREME TABLES
# =============================================================================
st.markdown("### Recent Extreme Volume Days")

df["1D Return %"] = df["Close"].pct_change(1) * 100.0
df["5D Return %"] = df["Close"].pct_change(5).shift(-5) * 100.0

table_cols = ["Close", "Vol_Display", "Vol_Z", "1D Return %", "5D Return %"]

recent_highs = (
    df[df["Is_High"]][table_cols]
    .rename(columns={"Vol_Display": vol_label, "Vol_Z": "Z-Score"})
    .tail(12)
    .sort_index(ascending=False)
)

recent_lows = (
    df[df["Is_Low"]][table_cols]
    .rename(columns={"Vol_Display": vol_label, "Vol_Z": "Z-Score"})
    .tail(12)
    .sort_index(ascending=False)
)

left, right = st.columns(2)

with left:
    st.markdown("#### High Volume Days")
    st.caption("These are usually the more useful signals. They often show up during stress, deleveraging, and panic hedging.")
    if not recent_highs.empty:
        st.dataframe(
            style_extremes_table(recent_highs, vol_label, "Shares Outstanding" in vol_label),
            use_container_width=True,
            height=430,
        )
    else:
        st.info("No high-volume extremes in the selected window.")

with right:
    st.markdown("#### Low Volume Days")
    st.caption("These can map to complacency, but they are inherently noisier and more vulnerable to calendar distortions.")
    if not recent_lows.empty:
        st.dataframe(
            style_extremes_table(recent_lows, vol_label, "Shares Outstanding" in vol_label),
            use_container_width=True,
            height=430,
        )
    else:
        st.info("No low-volume extremes in the selected window.")

# =============================================================================
# METHODOLOGY
# =============================================================================
with st.expander("Methodology"):
    st.markdown(
        """
**Framework**

For a very liquid ETF like QQQ, volume can function as a sentiment proxy. Heavy ETF turnover often appears
when participants are rushing to hedge, reduce risk, or exit exposure. Quiet tape often aligns with confidence
or indifference.

**Signal construction**

The app computes a rolling average and rolling standard deviation of volume over the selected baseline window.
Each day’s volume is then translated into a z-score relative to that rolling baseline:

`Volume Z-Score = (Today Volume - Rolling Mean) / Rolling Std Dev`

A high positive z-score means the market is trading far above its recent “normal” pace. A deeply negative
z-score means the tape is unusually quiet.

**Why normalization matters**

Raw ETF volume is imperfect for long-range comparison because the share count changes over time through
creations and redemptions. When shares outstanding is available, the app expresses volume as a % of shares
outstanding. That makes a 2026 reading more comparable to an older reading.

**Why low-volume signals need caution**

Quiet sessions can reflect complacency, but they can also reflect half-days, holidays, summer lulls, or a
dead news cycle. That is why the app includes an optional thin-session filter for low-volume flags.

**What changed in this build**

The app no longer depends on `Ticker.info` or a single Yahoo path just to function. OHLCV is fetched from
multiple public sources with fallbacks, and the normalization layer is optional rather than hard-required.
That is the main reason this version is more stable.
        """
    )
