import math
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
    page_title="Volume Sentiment",
    layout="wide",
)

# =============================================================================
# STYLING
# =============================================================================
CUSTOM_CSS = """
<style>
    .stApp {
        background: #f6f7fb;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 1.1rem;
        padding-bottom: 1.25rem;
    }

    div[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e7eaf2;
    }

    .app-shell {
        padding: 0;
    }

    .hero-wrap {
        background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
        border: 1px solid #e7eaf2;
        border-radius: 18px;
        padding: 22px 24px 18px 24px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        margin-bottom: 14px;
    }

    .eyebrow {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
    }

    .hero-title {
        font-size: 34px;
        line-height: 1.05;
        font-weight: 780;
        color: #101828;
        margin: 0;
    }

    .hero-sub {
        margin-top: 10px;
        color: #475467;
        font-size: 14px;
        line-height: 1.55;
        max-width: 1000px;
    }

    .ticker-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 13px;
        font-weight: 700;
        white-space: nowrap;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e7eaf2;
        border-radius: 16px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        min-height: 102px;
    }

    .stat-label {
        color: #667085;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.01em;
    }

    .stat-value {
        color: #101828;
        font-size: 28px;
        font-weight: 780;
        line-height: 1.0;
        margin-bottom: 8px;
    }

    .stat-sub {
        color: #475467;
        font-size: 12px;
        line-height: 1.45;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e7eaf2;
        border-radius: 18px;
        padding: 10px 12px 8px 12px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        margin-bottom: 14px;
    }

    .section-caption {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 4px 2px 12px 2px;
    }

    .micro-grid {
        display: grid;
        grid-template-columns: 1.3fr 1fr 1fr;
        gap: 12px;
        margin-top: 2px;
    }

    .micro-card {
        background: #ffffff;
        border: 1px solid #e7eaf2;
        border-radius: 16px;
        padding: 14px 15px;
        min-height: 88px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }

    .micro-title {
        color: #101828;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .micro-text {
        color: #475467;
        font-size: 12px;
        line-height: 1.55;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 7px 10px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid transparent;
    }

    .badge-calm {
        color: #0f766e;
        background: #ecfdf3;
        border-color: #b7ebd0;
    }

    .badge-normal {
        color: #1d4ed8;
        background: #eff6ff;
        border-color: #c7ddff;
    }

    .badge-stress {
        color: #b42318;
        background: #fef3f2;
        border-color: #f7c2bd;
    }

    .sidebar-head {
        color: #101828;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 4px;
        margin-top: 6px;
    }

    .sidebar-copy {
        color: #475467;
        font-size: 12px;
        line-height: 1.55;
    }

    @media (max-width: 1150px) {
        .stats-grid, .micro-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 760px) {
        .stats-grid, .micro-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 28px;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "Volume Sentiment"
APP_SUBTITLE = (
    "A cleaner read on participation. The goal is to show whether the tape is quiet, normal, "
    "or stressed relative to the instrument's own recent history, without burying the signal in unnecessary decoration."
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "TLT", "GLD", "HYG", "SMH", "NVDA", "META", "TSLA"]


# =============================================================================
# HELPERS
# =============================================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_billions(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x / 1_000_000_000:.2f}B"


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:,.{digits}f}"


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
    stress_z: float,
    calm_z: float,
    normalize_volume: bool,
    shares_outstanding: Optional[float],
    exclude_thin_sessions: bool,
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()

    if normalize_volume and shares_outstanding and shares_outstanding > 0:
        out["Vol_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Volume (% of shares outstanding)"
    else:
        out["Vol_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (shares)"

    out["Vol_MA"] = out["Vol_Display"].rolling(ma_period, min_periods=ma_period).mean()
    out["Vol_Base"] = out["Vol_Display"].rolling(baseline_period, min_periods=baseline_period).mean()
    out["Vol_Std"] = out["Vol_Display"].rolling(baseline_period, min_periods=baseline_period).std(ddof=0)
    out["Vol_Z"] = (out["Vol_Display"] - out["Vol_Base"]) / out["Vol_Std"].replace(0, np.nan)

    out["Vol_PctRank"] = (
        out["Vol_Display"]
        .rolling(252, min_periods=80)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )

    out["Vol_Median_20"] = out["Vol_Display"].rolling(20, min_periods=10).median()
    out["Thin_Session"] = out["Vol_Display"] < 0.55 * out["Vol_Median_20"]

    if exclude_thin_sessions:
        thin_mask = out["Thin_Session"] & (out["Vol_Z"] <= calm_z)
        out.loc[thin_mask, "Vol_Z"] = np.nan
        out.loc[thin_mask, "Vol_PctRank"] = np.nan

    out["Stress"] = out["Vol_Z"] >= stress_z
    out["Calm"] = out["Vol_Z"] <= calm_z

    out["Ret_1D"] = out["Close"].pct_change() * 100.0
    out["Ret_20D"] = out["Close"].pct_change(20) * 100.0

    return out, vol_label


def classify_regime(vol_z: Optional[float], stress_z: float, calm_z: float) -> Tuple[str, str]:
    if vol_z is None or not np.isfinite(vol_z):
        return "Signal unavailable", "normal"

    if vol_z >= stress_z:
        return "Stress regime", "stress"

    if vol_z <= calm_z:
        return "Quiet regime", "calm"

    return "Normal regime", "normal"


def stat_card(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>
    """


def badge_html(text: str, kind: str) -> str:
    css_class = {
        "stress": "badge badge-stress",
        "calm": "badge badge-calm",
        "normal": "badge badge-normal",
    }.get(kind, "badge badge-normal")
    return f'<span class="{css_class}">{text}</span>'


def describe_signal(latest_vol_z: Optional[float], latest_pct: Optional[float], stress_z: float, calm_z: float) -> str:
    regime, _ = classify_regime(latest_vol_z, stress_z, calm_z)

    if latest_vol_z is None or not np.isfinite(latest_vol_z):
        return "The current signal is not reliable yet because the rolling baseline is still forming."

    if regime == "Stress regime":
        return (
            f"Participation is elevated versus its own trailing baseline. "
            f"The latest reading sits at {latest_vol_z:.2f} standard deviations above normal, "
            f"which usually lines up with fear, forced repositioning, hedging demand, or event-driven attention."
        )

    if regime == "Quiet regime":
        return (
            f"Participation is materially below its recent norm. "
            f"The latest reading sits at {latest_vol_z:.2f} standard deviations below baseline, "
            f"which usually points to calm tape, lower urgency, or simple lack of interest."
        )

    pct_text = f"{latest_pct:.0f}th percentile" if latest_pct is not None and np.isfinite(latest_pct) else "its normal range"
    return (
        f"Participation looks ordinary right now. "
        f"Volume is sitting around {pct_text} of the trailing 1-year distribution, "
        f"so the tape is giving you price information without much confirmation from turnover."
    )


def build_clean_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    ma_period: int,
    show_last_price_line: bool,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    price_line_color = "#0f172a"
    volume_bar_color = "rgba(148, 163, 184, 0.38)"
    stress_dot_color = "#dc2626"
    calm_dot_color = "#16a34a"
    ma_color = "#2563eb"

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color=price_line_color, width=2.2),
            name=f"{symbol} Price",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    ret_colors = np.where(df["Ret_1D"] >= 0, "rgba(22, 163, 74, 0.12)", "rgba(220, 38, 38, 0.12)")
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            marker_color=volume_bar_color,
            name="Volume",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.4f}<extra></extra>"
            if "% of shares outstanding" in vol_label.lower()
            else "<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Vol_MA"],
            mode="lines",
            line=dict(color=ma_color, width=2.0),
            name=f"{ma_period}D Volume MA",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>MA: %{y:,.4f}<extra></extra>"
            if "% of shares outstanding" in vol_label.lower()
            else "<b>%{x|%b %d, %Y}</b><br>MA: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    stress_df = df[df["Stress"].fillna(False)].copy()
    if not stress_df.empty:
        fig.add_trace(
            go.Scatter(
                x=stress_df.index,
                y=stress_df["Close"],
                mode="markers",
                marker=dict(size=7, color=stress_dot_color, line=dict(width=0)),
                name="Stress days",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Stress day<extra></extra>",
            ),
            row=1,
            col=1,
        )

    calm_df = df[df["Calm"].fillna(False)].copy()
    if not calm_df.empty:
        fig.add_trace(
            go.Scatter(
                x=calm_df.index,
                y=calm_df["Close"],
                mode="markers",
                marker=dict(size=6, color=calm_dot_color, line=dict(width=0)),
                name="Quiet days",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Quiet day<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if show_last_price_line and len(df) > 0:
        last_close = float(df["Close"].iloc[-1])
        fig.add_hline(
            y=last_close,
            line_width=1,
            line_dash="dot",
            line_color="rgba(15,23,42,0.35)",
            row=1,
            col=1,
        )

    fig.update_layout(
        height=760,
        margin=dict(l=18, r=18, t=10, b=8),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.75)",
            borderwidth=0,
            font=dict(size=11, color="#475467"),
        ),
        bargap=0.15,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#667085"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        title_text="Price",
        title_font=dict(size=11, color="#667085"),
        tickfont=dict(size=11, color="#667085"),
        showgrid=True,
        gridcolor="#e9edf5",
        zeroline=False,
        showline=False,
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text=vol_label,
        title_font=dict(size=11, color="#667085"),
        tickfont=dict(size=10, color="#667085"),
        showgrid=True,
        gridcolor="#f0f4f8",
        zeroline=False,
        showline=False,
    )

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown('<div class="sidebar-head">About This Tool</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-copy">A cleaner participation dashboard for ETFs and liquid equities. '
    'It compares current turnover against the instrument’s own trailing behavior and flags unusually stressed or unusually quiet sessions.</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

symbol = st.sidebar.text_input("Ticker", value="QQQ").upper().strip()

st.sidebar.caption("Quick picks")
qp1, qp2 = st.sidebar.columns(2)
for i, sym in enumerate(DEFAULT_SYMBOLS):
    target_col = qp1 if i % 2 == 0 else qp2
    with target_col:
        if st.button(sym, use_container_width=True):
            symbol = sym

st.sidebar.markdown("---")

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
    value=20,
    step=1,
)

baseline_period = st.sidebar.slider(
    "Baseline lookback (days)",
    min_value=20,
    max_value=120,
    value=60,
    step=5,
)

stress_z = st.sidebar.slider(
    "Stress threshold (z-score)",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.25,
)

calm_z = st.sidebar.slider(
    "Quiet threshold (z-score)",
    min_value=-4.0,
    max_value=-0.5,
    value=-1.5,
    step=0.25,
)

normalize_volume = st.sidebar.checkbox(
    "Normalize by shares outstanding",
    value=True,
)

exclude_thin_sessions = st.sidebar.checkbox(
    "Filter thin sessions",
    value=True,
)

show_last_price_line = st.sidebar.checkbox(
    "Show last price guide",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-copy">Suggested default: normalize volume for ETFs and single stocks, keep thin-session filtering on, and avoid clutter. The chart is strongest when it only surfaces genuinely unusual participation.</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# VALIDATION
# =============================================================================
if not symbol:
    st.warning("Enter a ticker to continue.")
    st.stop()

# =============================================================================
# DATA FETCH
# =============================================================================
end_date = today_utc_ts()
warmup_days = max(baseline_period * 3, 252)
start_date = end_date - pd.Timedelta(days=lookback_months * 31 + warmup_days)
visible_start = end_date - pd.Timedelta(days=lookback_months * 31)

with st.spinner(f"Loading {symbol}..."):
    try:
        raw_df, data_source = fetch_ohlcv(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch {symbol} data. Details: {e}")
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
    stress_z=stress_z,
    calm_z=calm_z,
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
# LATEST METRICS
# =============================================================================
latest = df.iloc[-1]
latest_close = float(latest["Close"]) if pd.notna(latest["Close"]) else np.nan
latest_vol_z = float(latest["Vol_Z"]) if pd.notna(latest["Vol_Z"]) else np.nan
latest_vol_pct = float(latest["Vol_PctRank"]) if pd.notna(latest["Vol_PctRank"]) else np.nan
latest_vol = float(latest["Vol_Display"]) if pd.notna(latest["Vol_Display"]) else np.nan
latest_ret_1d = float(latest["Ret_1D"]) if pd.notna(latest["Ret_1D"]) else np.nan
latest_ret_20d = float(latest["Ret_20D"]) if pd.notna(latest["Ret_20D"]) else np.nan

regime_label, regime_kind = classify_regime(latest_vol_z, stress_z, calm_z)
signal_text = describe_signal(latest_vol_z, latest_vol_pct, stress_z, calm_z)

if "% of shares outstanding" in vol_label.lower():
    vol_display_text = f"{latest_vol:.4f}"
    vol_display_sub = "Current normalized turnover"
else:
    vol_display_text = f"{latest_vol:,.0f}"
    vol_display_sub = "Current raw share volume"

pct_rank_text = f"{latest_vol_pct:.0f}th pct" if np.isfinite(latest_vol_pct) else "N/A"
z_text = f"{latest_vol_z:.2f}" if np.isfinite(latest_vol_z) else "N/A"

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="eyebrow">Participation Dashboard</div>
        <div class="hero-title-row">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="ticker-pill">{symbol} · Source: {data_source}</div>
        </div>
        <div style="margin-top: 12px;">{badge_html(regime_label, regime_kind)}</div>
        <div class="hero-sub">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# STATS
# =============================================================================
stats_html = f"""
<div class="stats-grid">
    {stat_card("Latest Close", f"${latest_close:,.2f}", "Adjusted daily close")}
    {stat_card("Volume Z-Score", z_text, "Distance from trailing baseline")}
    {stat_card("Volume Percentile", pct_rank_text, "Position within trailing 1-year range")}
    {stat_card("Current Turnover", vol_display_text, vol_display_sub + f" · Shares out: {format_billions(shares_outstanding)}")}
</div>
"""
st.markdown(stats_html, unsafe_allow_html=True)

# =============================================================================
# MAIN CHART
# =============================================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Price and participation</div>', unsafe_allow_html=True)

fig = build_clean_chart(
    df=df,
    symbol=symbol,
    vol_label=vol_label,
    ma_period=ma_period,
    show_last_price_line=show_last_price_line,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": True,
        "responsive": True,
    },
)
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# COMMENTARY / MICRO PANELS
# =============================================================================
context_html = f"""
<div class="micro-grid">
    <div class="micro-card">
        <div class="micro-title">Read on the tape</div>
        <div class="micro-text">{signal_text}</div>
    </div>
    <div class="micro-card">
        <div class="micro-title">Short-term price move</div>
        <div class="micro-text">
            1D return: <b>{fmt_pct(latest_ret_1d)}</b><br>
            20D return: <b>{fmt_pct(latest_ret_20d)}</b><br><br>
            Volume matters most when it confirms or sharply contradicts the recent move.
        </div>
    </div>
    <div class="micro-card">
        <div class="micro-title">How to use it</div>
        <div class="micro-text">
            Stress readings usually matter most near inflection points, breakouts, breakdowns, or macro shocks.
            Quiet readings matter more when price is extended and participation refuses to expand.
        </div>
    </div>
</div>
"""
st.markdown(context_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
