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
        max-width: 1680px;
        padding-top: 1.65rem;
        padding-bottom: 0.85rem;
    }
    h1, h2, h3 {
        letter-spacing: 0.05px;
        font-weight: 700;
    }
    .page-title-wrap {
        padding-top: 0.15rem;
        margin-bottom: 0.35rem;
    }
    .page-title-main {
        font-size: 56px;
        font-weight: 750;
        color: #303445;
        line-height: 1.02;
        margin: 0 0 10px 0;
        padding: 0;
    }
    .page-title-sub {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.4;
        margin-bottom: 20px;
    }
    .adfm-card {
        background: #fafafa;
        border: 1px solid #e8e8e8;
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
        font-weight: 750;
        color: #1f2937;
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

COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "vol_base": "rgba(41, 98, 255, 0.14)",
    "vol_stress": "rgba(239, 83, 80, 0.72)",
    "vol_quiet": "rgba(251, 140, 0, 0.62)",
    "vol_ma": "#2962FF",
    "zscore": "#111827",
    "stress_line": "rgba(239, 83, 80, 0.90)",
    "quiet_line": "rgba(251, 140, 0, 0.90)",
    "zero_line": "rgba(120, 120, 120, 0.55)",
    "stress_fill": "rgba(239, 83, 80, 0.06)",
    "quiet_fill": "rgba(251, 140, 0, 0.045)",
    "grid": "rgba(120,120,120,0.14)",
    "text": "#222222",
    "subtle": "#6b7280",
    "last": "#1565C0",
    "bg": "white",
}

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

def build_rangebreaks(index: pd.DatetimeIndex) -> list[dict]:
    if len(index) == 0:
        return []
    normalized = pd.DatetimeIndex(index).normalize().unique().sort_values()
    full_bdays = pd.date_range(start=normalized.min(), end=normalized.max(), freq="B")
    missing_bdays = full_bdays.difference(normalized)

    rangebreaks = [dict(bounds=["sat", "mon"])]
    if len(missing_bdays) > 0:
        rangebreaks.append(dict(values=missing_bdays.strftime("%Y-%m-%d").tolist()))
    return rangebreaks

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

def sparse_signal_points(df: pd.DataFrame, signal_col: str, gap_days: int = 18) -> pd.DataFrame:
    signal_df = df[df[signal_col]].copy()
    if signal_df.empty:
        return signal_df

    kept_rows = []
    last_kept = None
    for idx, row in signal_df.iterrows():
        if last_kept is None or (idx - last_kept).days >= gap_days:
            kept_rows.append((idx, row))
            last_kept = idx

    if not kept_rows:
        return signal_df.iloc[0:0].copy()

    kept_index = [x[0] for x in kept_rows]
    return signal_df.loc[kept_index].copy()

def build_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    ma_period: int,
    high_z: float,
    low_z: float,
    show_markers: bool,
    show_last_price: bool,
    vol_opacity: float,
) -> go.Figure:
    rangebreaks = build_rangebreaks(df.index)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=[0.68, 0.18, 0.14],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]],
    )

    stress_windows = df[df["Vol_Z"] >= high_z].copy()
    quiet_windows = df[df["Vol_Z"] <= low_z].copy()

    for idx in stress_windows.index:
        fig.add_vrect(
            x0=idx - pd.Timedelta(hours=12),
            x1=idx + pd.Timedelta(hours=12),
            fillcolor=COLORS["stress_fill"],
            line_width=0,
            row="all",
            col=1,
        )

    for idx in quiet_windows.index:
        fig.add_vrect(
            x0=idx - pd.Timedelta(hours=12),
            x1=idx + pd.Timedelta(hours=12),
            fillcolor=COLORS["quiet_fill"],
            line_width=0,
            row="all",
            col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLORS["up"],
            increasing_fillcolor=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            decreasing_fillcolor=COLORS["down"],
            name=symbol,
            showlegend=False,
            whiskerwidth=0.5,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if show_markers:
        highs_sparse = sparse_signal_points(df, "Is_High", gap_days=18)
        lows_sparse = sparse_signal_points(df, "Is_Low", gap_days=18)

        if not highs_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=highs_sparse.index,
                    y=highs_sparse["Low"] * 0.992,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color="rgba(239,83,80,0.92)",
                        line=dict(width=1, color="white"),
                    ),
                    name="Stress Signal",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([highs_sparse["Close"], highs_sparse["Vol_Z"]]),
                ),
                row=1,
                col=1,
            )

        if not lows_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=lows_sparse.index,
                    y=lows_sparse["High"] * 1.008,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=8,
                        color="rgba(251,140,0,0.90)",
                        line=dict(width=1, color="white"),
                    ),
                    name="Complacency Signal",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([lows_sparse["Close"], lows_sparse["Vol_Z"]]),
                ),
                row=1,
                col=1,
            )

    if show_last_price and len(df) > 0:
        last_close = float(df["Close"].iloc[-1])
        fig.add_hline(
            y=last_close,
            line_width=1,
            line_dash="dot",
            line_color=COLORS["last"],
            row=1,
            col=1,
            annotation_text=f"Last {last_close:,.2f}",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["last"], size=11),
        )

    bar_colors = np.where(
        df["Vol_Z"] >= high_z,
        COLORS["vol_stress"],
        np.where(
            df["Vol_Z"] <= low_z,
            COLORS["vol_quiet"],
            f"rgba(41, 98, 255, {vol_opacity})",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            marker_color=bar_colors,
            name="Volume",
            showlegend=True,
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
            line=dict(color=COLORS["vol_ma"], width=2.1),
            name=f"Volume {ma_period}D MA",
            showlegend=True,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume MA: %{y:,.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Vol_Z"],
            mode="lines",
            line=dict(color=COLORS["zscore"], width=1.9),
            name="Volume Z-Score",
            showlegend=True,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume Z: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=high_z,
        line=dict(color=COLORS["stress_line"], width=1.2, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0,
        line=dict(color=COLORS["zero_line"], width=1.0, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=low_z,
        line=dict(color=COLORS["quiet_line"], width=1.2, dash="dot"),
        row=3,
        col=1,
    )

    for r in [1, 2, 3]:
        fig.update_yaxes(
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            fixedrange=False,
            automargin=True,
            row=r,
            col=1,
        )

    fig.update_xaxes(
        type="date",
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        rangebreaks=rangebreaks,
        tickformat="%b %d\n%Y",
    )

    fig.update_yaxes(title_text="Price", nticks=10, row=1, col=1)
    fig.update_yaxes(title_text="Vol", nticks=6, row=2, col=1)
    fig.update_yaxes(title_text="Z", nticks=6, row=3, col=1)

    fig.update_layout(
        height=940,
        title=None,
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        hovermode="x unified",
        margin=dict(l=42, r=18, t=34, b=10),
        font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12, color="#4b5563"),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            traceorder="normal",
        ),
        bargap=0.06,
    )

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
<br>• The bottom panel matters most because it tells you where current activity sits versus normal
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

show_markers = st.sidebar.checkbox(
    "Show Sparse Signal Markers",
    value=True,
)

show_last_price = st.sidebar.checkbox(
    "Show Last Price Line",
    value=False,
)

vol_opacity = st.sidebar.slider(
    "Base Volume Bar Opacity",
    min_value=0.08,
    max_value=0.40,
    value=0.14,
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
fig = build_chart(
    df=df,
    symbol=symbol,
    vol_label=vol_label,
    ma_period=ma_period,
    high_z=high_z,
    low_z=low_z,
    show_markers=show_markers,
    show_last_price=show_last_price,
    vol_opacity=vol_opacity,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False},
)
