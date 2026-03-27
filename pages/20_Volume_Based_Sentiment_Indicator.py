import time
import json
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
    page_title="Volume Based Sentiment Indicator",
    layout="wide",
)

# =============================================================================
# STYLING
# =============================================================================
CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1720px;
        padding-top: 1.6rem;
        padding-bottom: 0.75rem;
    }
    h1, h2, h3 {
        letter-spacing: 0.05px;
        font-weight: 700;
    }
    .page-title-wrap {
        padding-top: 0.15rem;
        margin-bottom: 0.45rem;
    }
    .page-title-main {
        font-size: 54px;
        font-weight: 760;
        color: #1f2937;
        line-height: 1.02;
        margin: 0 0 8px 0;
        padding: 0;
    }
    .page-title-sub {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.45;
        margin-bottom: 18px;
        max-width: 1200px;
    }
    .adfm-card {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 13px 16px;
        margin-bottom: 10px;
        min-height: 112px;
    }
    .adfm-label {
        color: #6b7280;
        font-size: 0.86rem;
        margin-bottom: 4px;
    }
    .adfm-value {
        font-size: 1.50rem;
        font-weight: 760;
        color: #111827;
        line-height: 1.08;
        margin-bottom: 6px;
    }
    .adfm-sub {
        color: #6b7280;
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .sidebar-note {
        color: #666;
        font-size: 0.90rem;
        line-height: 1.45;
    }
    .chart-shell {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px 8px 2px 8px;
        margin-top: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "Volume Based Sentiment Indicator"
APP_SUBTITLE = (
    "Volume is treated as a sentiment input. Heavy turnover often aligns with fear, stress, and forced repositioning, "
    "while quiet tape often reflects complacency. The goal is to place current activity in the context of the instrument's own history."
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "TLT", "GLD", "HYG", "SMH", "NVDA", "TSLA", "META"]

# =============================================================================
# HELPERS
# =============================================================================
def metric_card(label: str, value: str, subtext: str = "") -> str:
    return f"""
    <div class="adfm-card">
        <div class="adfm-label">{label}</div>
        <div class="adfm-value">{value}</div>
        <div class="adfm-sub">{subtext}</div>
    </div>
    """

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def format_billions(x: float) -> str:
    return f"{x / 1_000_000_000:.2f}B"

def today_utc_ts() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()

def normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out.index = out.index.normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()
    return out

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
    return normalize_dt_index(df)

def fetch_from_yahoo_chart_api(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    period1 = int(start_date.timestamp())
    period2 = int((end_date + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&includePrePost=false&events=div%2Csplits"
    )

    resp_text = _request_text(url, timeout=20, retries=3, sleep_s=1.0)
    data = json.loads(resp_text)

    chart = data.get("chart", {})
    result_list = chart.get("result", [])
    if not result_list:
        err = chart.get("error")
        raise ValueError(f"Yahoo Chart API returned no result. Error: {err}")

    result = result_list[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")

    if not timestamps:
        raise ValueError("Yahoo Chart API returned empty timestamps.")

    close_vals = adjclose if adjclose is not None else quote.get("close")

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s", utc=True),
        "Open": quote.get("open"),
        "High": quote.get("high"),
        "Low": quote.get("low"),
        "Close": close_vals,
        "Volume": quote.get("volume"),
    })

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = safe_numeric(df[col])

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
    df = df.set_index("Date")
    return normalize_dt_index(df)

def fetch_from_yfinance_download(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
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

    df = df.dropna(subset=needed).copy()
    return normalize_dt_index(df)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    errors = []

    try:
        df = fetch_from_stooq(symbol, start_date)
        if not df.empty:
            return df, "Stooq"
    except Exception as e:
        errors.append(f"Stooq: {e}")

    try:
        df = fetch_from_yahoo_chart_api(symbol, start_date, end_date)
        if not df.empty:
            return df, "Yahoo Chart API"
    except Exception as e:
        errors.append(f"Yahoo Chart API: {e}")

    try:
        df = fetch_from_yfinance_download(symbol, start_date, end_date)
        if not df.empty:
            return df, "yfinance"
    except Exception as e:
        errors.append(f"yfinance: {e}")

    raise RuntimeError(" | ".join(errors))

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_shares_outstanding(symbol: str) -> Optional[float]:
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
    baseline_period: int,
    high_z: float,
    low_z: float,
    normalize_volume: bool,
    shares_outstanding: Optional[float],
    exclude_thin_sessions: bool,
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()

    if normalize_volume and shares_outstanding and shares_outstanding > 0:
        out["Vol_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Volume (% of Shares Outstanding)"
    else:
        out["Vol_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (Shares)"

    out["Vol_MA"] = out["Vol_Display"].rolling(ma_period, min_periods=ma_period).mean()
    out["Vol_Base"] = out["Vol_Display"].rolling(baseline_period, min_periods=baseline_period).mean()
    out["Vol_Std"] = out["Vol_Display"].rolling(baseline_period, min_periods=baseline_period).std(ddof=0)
    out["Vol_Z"] = (out["Vol_Display"] - out["Vol_Base"]) / out["Vol_Std"].replace(0, np.nan)

    out["Vol_Median_20"] = out["Vol_Display"].rolling(20, min_periods=10).median()
    out["Likely_Thin_Session"] = out["Vol_Display"] < 0.55 * out["Vol_Median_20"]

    if exclude_thin_sessions:
        mask = out["Likely_Thin_Session"] & (out["Vol_Z"] < low_z)
        out.loc[mask, "Vol_Z"] = np.nan

    out["Is_High"] = out["Vol_Z"] >= high_z
    out["Is_Low"] = out["Vol_Z"] <= low_z

    out["Vol_PctRank"] = (
        out["Vol_Display"]
        .rolling(252, min_periods=60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )

    return out, vol_label

def build_plotly_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    ma_period: int,
    show_last_price: bool,
    vol_opacity: float,
):
    chart_df = df.copy()

    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(height=760)
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.80, 0.20],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            increasing_line_color="#16a34a",
            increasing_fillcolor="#16a34a",
            decreasing_line_color="#dc2626",
            decreasing_fillcolor="#dc2626",
            line_width=1,
            name=symbol,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    high_vol_df = chart_df[chart_df["Is_High"].fillna(False)].copy()
    if not high_vol_df.empty:
        low_anchor = (chart_df["High"] - chart_df["Low"]).replace(0, np.nan)
        typical_range = float(low_anchor.median()) if low_anchor.notna().any() else 0.0
        if not np.isfinite(typical_range) or typical_range <= 0:
            typical_range = float(chart_df["Close"].median()) * 0.01

        marker_y = high_vol_df["Low"] - typical_range * 0.18

        fig.add_trace(
            go.Scatter(
                x=high_vol_df.index,
                y=marker_y,
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=8,
                    color="#16a34a",
                    line=dict(width=0),
                ),
                name="High Vol",
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "High-volume signal<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    base_rgba = f"rgba(148,163,184,{vol_opacity:.2f})"
    high_rgba = "rgba(22,163,74,0.45)"
    vol_colors = np.where(chart_df["Is_High"].fillna(False), high_rgba, base_rgba)

    volume_hover_label = "Volume (%)" if "Shares Outstanding" in vol_label else "Volume"

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Vol_Display"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
            hovertemplate=f"<b>%{{x|%b %d, %Y}}</b><br>{volume_hover_label}: %{{y:,.4f}}<extra></extra>"
            if "Shares Outstanding" in vol_label
            else "<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Vol_MA"],
            mode="lines",
            line=dict(color="#2563eb", width=1.6),
            name=f"{ma_period}D Vol MA",
            hovertemplate=f"<b>%{{x|%b %d, %Y}}</b><br>{ma_period}D Vol MA: %{{y:,.4f}}<extra></extra>"
            if "Shares Outstanding" in vol_label
            else f"<b>%{{x|%b %d, %Y}}</b><br>{ma_period}D Vol MA: %{{y:,.0f}}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if show_last_price and len(chart_df) > 0:
        last_close = float(chart_df["Close"].iloc[-1])
        last_x = chart_df.index[-1]

        fig.add_hline(
            y=last_close,
            line_width=1,
            line_dash="dot",
            line_color="#111827",
            row=1,
            col=1,
        )

        fig.add_annotation(
            x=last_x,
            y=last_close,
            text=f"{last_close:,.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=8,
            font=dict(size=11, color="#111827"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="rgba(17,24,39,0.15)",
            borderwidth=1,
            row=1,
            col=1,
        )

    fig.update_layout(
        height=780,
        margin=dict(l=18, r=18, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        bargap=0.0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color="#6b7280"),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#6b7280"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#6b7280"),
        title_text="Price",
        title_font=dict(size=11, color="#6b7280"),
        fixedrange=False,
    )

    fig.update_yaxes(
        row=2,
        col=1,
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color="#6b7280"),
        title_text=vol_label,
        title_font=dict(size=11, color="#6b7280"),
        fixedrange=False,
    )

    fig.update_annotations(font=dict(color="#111827"))

    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
<div class="sidebar-note">
This tool treats volume as a sentiment input for any liquid instrument.

What it shows:
<br>• Price path
<br>• Daily volume and volume trend
<br>• Current activity versus its rolling baseline
<br>• Stress and complacency zones based on volume z-scores

How to read it:
<br>• Heavy volume usually shows fear, stress, hedging, or forced repositioning
<br>• Very quiet tape can reflect complacency, but those signals are weaker
<br>• The chart is intentionally simple: price on top, participation underneath
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Instrument")
symbol = st.sidebar.text_input("Ticker", value="QQQ").upper().strip()

st.sidebar.caption("Quick ideas")
quick_cols = st.sidebar.columns(2)
for i, sym in enumerate(DEFAULT_SYMBOLS[:6]):
    with quick_cols[i % 2]:
        if st.button(sym, use_container_width=True):
            symbol = sym

st.sidebar.markdown("---")
st.sidebar.header("Window and Signal Setup")

lookback_months = st.sidebar.slider(
    "Lookback (months)",
    min_value=6,
    max_value=60,
    value=24,
    step=3,
)

ma_period = st.sidebar.slider(
    "Volume MA (days)",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
)

baseline_period = st.sidebar.slider(
    "Volume Baseline Lookback (days)",
    min_value=20,
    max_value=120,
    value=60,
    step=5,
)

high_z = st.sidebar.slider(
    "Stress Threshold (Z-score)",
    min_value=1.0,
    max_value=4.0,
    value=2.5,
    step=0.25,
)

low_z = st.sidebar.slider(
    "Complacency Threshold (Z-score)",
    min_value=-4.0,
    max_value=-0.5,
    value=-2.0,
    step=0.25,
)

normalize_volume = st.sidebar.checkbox(
    "Normalize by Shares Outstanding",
    value=True,
    help="For equities and ETFs, this improves long-range comparability when share count data is available.",
)

exclude_thin_sessions = st.sidebar.checkbox(
    "Filter Thin Sessions",
    value=True,
    help="Removes obvious holiday and half-day distortions from low-volume signals.",
)

show_last_price = st.sidebar.checkbox(
    "Show Last Price Line",
    value=False,
)

vol_opacity = st.sidebar.slider(
    "Base Volume Bar Opacity",
    min_value=0.08,
    max_value=0.40,
    value=0.18,
    step=0.02,
)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f"""
    <div class="page-title-wrap">
        <div class="page-title-main">{APP_TITLE}</div>
        <div class="page-title-sub">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DATA FETCH
# =============================================================================
if not symbol:
    st.warning("Enter a ticker to continue.")
    st.stop()

end_date = today_utc_ts()
warmup_days = max(baseline_period * 3, 220)
start_date = end_date - pd.Timedelta(days=lookback_months * 31 + warmup_days)
visible_start = end_date - pd.Timedelta(days=lookback_months * 31)

with st.spinner(f"Loading {symbol} data..."):
    try:
        raw_df, data_source = fetch_ohlcv(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch {symbol} data from all sources. Details: {e}")
        st.stop()

shares_outstanding = None
if normalize_volume:
    with st.spinner("Fetching shares outstanding..."):
        shares_outstanding = fetch_shares_outstanding(symbol)

if raw_df.empty:
    st.warning("No data returned for the selected instrument.")
    st.stop()

raw_df = normalize_dt_index(raw_df)
raw_df = raw_df.loc[raw_df.index >= (visible_start - pd.Timedelta(days=warmup_days))].copy()

df, vol_label = compute_signal_table(
    raw_df,
    ma_period=ma_period,
    baseline_period=baseline_period,
    high_z=high_z,
    low_z=low_z,
    normalize_volume=normalize_volume,
    shares_outstanding=shares_outstanding,
    exclude_thin_sessions=exclude_thin_sessions,
)

df = normalize_dt_index(df)
df = df.loc[df.index >= visible_start].copy()

if df.empty:
    st.warning("No usable data remained after processing.")
    st.stop()

# =============================================================================
# TOP METRICS
# =============================================================================
latest = df.iloc[-1]
latest_close = float(latest["Close"])
latest_vol_z = float(latest["Vol_Z"]) if pd.notna(latest["Vol_Z"]) else np.nan
latest_vol_pct = float(latest["Vol_PctRank"]) if pd.notna(latest["Vol_PctRank"]) else np.nan
latest_volume = float(latest["Vol_Display"])

if np.isfinite(latest_vol_z) and latest_vol_z >= high_z:
    regime_text = "Stress / capitulation"
elif np.isfinite(latest_vol_z) and latest_vol_z <= low_z:
    regime_text = "Complacency / quiet tape"
else:
    regime_text = "Normal volume regime"

percentile_text = f"{latest_vol_pct:.0f}th percentile" if np.isfinite(latest_vol_pct) else "N/A"

if "Shares Outstanding" in vol_label:
    volume_text = f"{latest_volume:.4f}"
    volume_sub = "Normalized volume"
else:
    volume_text = f"{latest_volume:,.0f}"
    volume_sub = "Raw shares volume"

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        metric_card("Instrument", symbol, f"Source: {data_source}"),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        metric_card("Latest Close", f"${latest_close:,.2f}", "Adjusted daily close"),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        metric_card(
            "Volume Z-Score",
            f"{latest_vol_z:.2f}" if np.isfinite(latest_vol_z) else "N/A",
            regime_text,
        ),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        metric_card("Volume Percentile", percentile_text, "Versus rolling 1-year history"),
        unsafe_allow_html=True,
    )

with c5:
    shares_text = (
        format_billions(shares_outstanding)
        if shares_outstanding and np.isfinite(shares_outstanding)
        else "Unavailable"
    )
    st.markdown(
        metric_card(
            "Volume Mode",
            volume_text,
            f"{volume_sub} | Shares out: {shares_text}",
        ),
        unsafe_allow_html=True,
    )

# =============================================================================
# CHART
# =============================================================================
plotly_fig = build_plotly_chart(
    df=df,
    symbol=symbol,
    vol_label=vol_label,
    ma_period=ma_period,
    show_last_price=show_last_price,
    vol_opacity=vol_opacity,
)

st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
st.plotly_chart(
    plotly_fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": True,
        "responsive": True,
    },
)
st.markdown("</div>", unsafe_allow_html=True)
