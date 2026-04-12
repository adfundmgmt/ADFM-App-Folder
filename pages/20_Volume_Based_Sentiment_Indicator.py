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
st.set_page_config(page_title="Volume Sentiment Explorer", layout="wide")


# =============================================================================
# STYLING
# =============================================================================
st.markdown(
    """
    <style>
    .adfm-card-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 0.35rem 0 0.85rem 0;
    }
    .adfm-card {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        background: #fafaf8;
        padding: 12px 14px;
    }
    .adfm-card-label {
        font-size: 0.78rem;
        color: #666666;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    .adfm-card-value {
        font-size: 1.12rem;
        font-weight: 700;
        color: #111111;
        line-height: 1.15;
    }
    .adfm-card-sub {
        font-size: 0.80rem;
        color: #666666;
        margin-top: 6px;
        line-height: 1.25;
    }
    .adfm-note {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        background: #fcfcfb;
        padding: 14px 16px;
        margin: 0.35rem 0 1rem 0;
        line-height: 1.55;
        color: #161616;
    }
    .adfm-note-title {
        font-size: 0.80rem;
        color: #666666;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 7px;
    }
    .adfm-divider {
        margin-top: 0.1rem;
        margin-bottom: 0.85rem;
    }
    @media (max-width: 1100px) {
        .adfm-card-row {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 700px) {
        .adfm-card-row {
            grid-template-columns: repeat(1, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================
APP_TITLE = "Volume Sentiment Explorer"
APP_SUBTITLE = "Frame current participation versus the instrument's own recent history."

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "TLT", "GLD", "HYG", "SMH", "NVDA", "META", "TSLA"]


# =============================================================================
# UI HELPERS
# =============================================================================
def card_html(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="adfm-card">
        <div class="adfm-card-label">{label}</div>
        <div class="adfm-card-value">{value}</div>
        <div class="adfm-card-sub">{sub}</div>
    </div>
    """


def render_cards(cards: list[tuple[str, str, str]]) -> None:
    html = '<div class="adfm-card-row">'
    for label, value, sub in cards:
        html += card_html(label, value, sub)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="adfm-note">
            <div class="adfm-note-title">{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA HELPERS
# =============================================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_billions(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "Unavailable"
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
        vol_label = "Volume (% of shares outstanding)"
    else:
        out["Vol_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (shares)"

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

    out["Ret_1D"] = out["Close"].pct_change() * 100.0
    out["Ret_20D"] = out["Close"].pct_change(20) * 100.0

    return out, vol_label


def build_plotly_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    ma_period: int,
    show_last_price: bool,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="#111111", width=2.0),
            name=symbol,
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    stress_df = df[df["Is_High"].fillna(False)].copy()
    if not stress_df.empty:
        fig.add_trace(
            go.Scatter(
                x=stress_df.index,
                y=stress_df["Close"],
                mode="markers",
                marker=dict(size=7, color="#c2410c"),
                name="Stress",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Stress day<extra></extra>",
            ),
            row=1,
            col=1,
        )

    quiet_df = df[df["Is_Low"].fillna(False)].copy()
    if not quiet_df.empty:
        fig.add_trace(
            go.Scatter(
                x=quiet_df.index,
                y=quiet_df["Close"],
                mode="markers",
                marker=dict(size=6, color="#166534"),
                name="Quiet",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Quiet day<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            marker_color="rgba(140, 140, 140, 0.30)",
            name="Volume",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.4f}<extra></extra>"
            if "% of shares" in vol_label.lower()
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
            line=dict(color="#2563eb", width=1.8),
            name=f"{ma_period}D MA",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>MA: %{y:,.4f}<extra></extra>"
            if "% of shares" in vol_label.lower()
            else "<b>%{x|%b %d, %Y}</b><br>MA: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if show_last_price and len(df) > 0:
        last_close = float(df["Close"].iloc[-1])
        fig.add_hline(
            y=last_close,
            line_width=1,
            line_dash="dot",
            line_color="rgba(17,17,17,0.35)",
            row=1,
            col=1,
        )

    fig.update_layout(
        height=740,
        margin=dict(l=16, r=16, t=6, b=6),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#555555"),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#efefef",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#666666"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        title_text="Price",
        title_font=dict(size=11, color="#666666"),
        tickfont=dict(size=11, color="#666666"),
        showgrid=True,
        gridcolor="#e8e8e8",
        zeroline=False,
        showline=False,
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text=vol_label,
        title_font=dict(size=11, color="#666666"),
        tickfont=dict(size=10, color="#666666"),
        showgrid=True,
        gridcolor="#f0f0f0",
        zeroline=False,
        showline=False,
    )

    return fig


def describe_regime(vol_z: float, high_z: float, low_z: float) -> str:
    if not np.isfinite(vol_z):
        return "Signal unavailable. There is not enough stable history yet to classify current participation cleanly."
    if vol_z >= high_z:
        return "Participation is elevated versus its own trailing baseline. That usually lines up with stress, forced repositioning, event-driven interest, or active hedging."
    if vol_z <= low_z:
        return "Participation is unusually quiet versus its own trailing baseline. That usually lines up with complacency, low urgency, or a simple lack of sponsorship."
    return "Participation is sitting in a normal range versus its own trailing baseline. Price is moving without a major volume confirmation signal right now."


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Purpose: Volume sentiment explorer for ETFs and liquid equities.

What it does

• Compares current turnover against the instrument's own recent history  
• Flags unusually stressed and unusually quiet sessions  
• Normalizes by shares outstanding when available  
• Keeps the chart simple so the participation signal is easy to read

How to read it

• Heavy volume often reflects fear, stress, or forced repositioning  
• Quiet volume often reflects complacency or lack of urgency  
• The signal is most useful around inflection points, breakouts, and breakdowns
        """
    )

    st.markdown("---")
    symbol = st.text_input("Ticker", value="QQQ").upper().strip()

    st.caption("Quick ideas")
    q1, q2 = st.columns(2)
    for i, sym in enumerate(DEFAULT_SYMBOLS):
        with (q1 if i % 2 == 0 else q2):
            if st.button(sym, use_container_width=True):
                symbol = sym

    st.markdown("---")
    st.subheader("Signal Setup")

    lookback_months = st.slider(
        "Lookback (months)",
        min_value=6,
        max_value=60,
        value=24,
        step=3,
    )

    ma_period = st.slider(
        "Volume MA (days)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
    )

    baseline_period = st.slider(
        "Volume baseline lookback (days)",
        min_value=20,
        max_value=120,
        value=60,
        step=5,
    )

    high_z = st.slider(
        "Stress threshold (z-score)",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.25,
    )

    low_z = st.slider(
        "Quiet threshold (z-score)",
        min_value=-4.0,
        max_value=-0.5,
        value=-1.5,
        step=0.25,
    )

    normalize_volume = st.checkbox(
        "Normalize by shares outstanding",
        value=True,
    )

    exclude_thin_sessions = st.checkbox(
        "Filter thin sessions",
        value=True,
    )

    show_last_price = st.checkbox(
        "Show last price guide",
        value=False,
    )


# =============================================================================
# HEADER
# =============================================================================
st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)
st.markdown("<hr class='adfm-divider'>", unsafe_allow_html=True)

if not symbol:
    st.warning("Enter a ticker to continue.")
    st.stop()


# =============================================================================
# FETCH + PREP
# =============================================================================
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
# METRICS
# =============================================================================
latest = df.iloc[-1]
latest_close = float(latest["Close"]) if pd.notna(latest["Close"]) else np.nan
latest_vol_z = float(latest["Vol_Z"]) if pd.notna(latest["Vol_Z"]) else np.nan
latest_vol_pct = float(latest["Vol_PctRank"]) if pd.notna(latest["Vol_PctRank"]) else np.nan
latest_volume = float(latest["Vol_Display"]) if pd.notna(latest["Vol_Display"]) else np.nan
ret_1d = float(latest["Ret_1D"]) if pd.notna(latest["Ret_1D"]) else np.nan
ret_20d = float(latest["Ret_20D"]) if pd.notna(latest["Ret_20D"]) else np.nan

regime_text = describe_regime(latest_vol_z, high_z, low_z)
percentile_text = f"{latest_vol_pct:.0f}th percentile" if np.isfinite(latest_vol_pct) else "N/A"

if "% of shares" in vol_label.lower():
    volume_text = f"{latest_volume:.4f}"
    volume_sub = f"Normalized turnover | Shares out: {format_billions(shares_outstanding)}"
else:
    volume_text = f"{latest_volume:,.0f}"
    volume_sub = f"Raw share volume | Shares out: {format_billions(shares_outstanding)}"

ret_1d_text = f"{ret_1d:+.2f}%" if np.isfinite(ret_1d) else "N/A"
ret_20d_text = f"{ret_20d:+.2f}%" if np.isfinite(ret_20d) else "N/A"
vol_z_text = f"{latest_vol_z:.2f}" if np.isfinite(latest_vol_z) else "N/A"

render_cards([
    ("Latest close", f"${latest_close:,.2f}", f"Adjusted daily close | Source: {data_source}"),
    ("Volume z-score", vol_z_text, "Distance from trailing baseline"),
    ("Volume percentile", percentile_text, "Versus rolling 1-year history"),
    ("Current turnover", volume_text, volume_sub),
])

render_note(
    "PARTICIPATION READ",
    f"{regime_text} The instrument is up {ret_1d_text} over 1 day and {ret_20d_text} over 20 days, so the key question is whether price is being confirmed by real sponsorship or just drifting on ordinary turnover."
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
)

st.plotly_chart(
    plotly_fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": True,
        "responsive": True,
    },
)

st.caption("© 2026 AD Fund Management LP")
