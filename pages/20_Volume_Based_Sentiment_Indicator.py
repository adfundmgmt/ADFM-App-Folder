import json
import time
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
    page_title="Volume Regime Explorer",
    layout="wide",
)

st.title("Volume Regime Explorer")
st.caption("Compare current participation to the instrument's own trailing history.")


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_SYMBOLS = ["QQQ", "SPY", "IWM", "TLT", "GLD", "HYG", "SMH", "NVDA", "META", "TSLA"]

US_MARKET_HOLIDAYS = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def today_utc_ts() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out[~out.index.isna()].copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()
    return out


def fmt_large_number(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:,.0f}"


def request_text(url: str, timeout: int = 20, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    raise last_err


# =============================================================================
# DATA FETCH
# =============================================================================
def fetch_from_yahoo_chart_api(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    period1 = int(start_date.timestamp())
    period2 = int((end_date + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&includePrePost=false&events=div%2Csplits"
    )

    text = request_text(url)
    payload = json.loads(text)
    chart = payload.get("chart", {})
    result = chart.get("result", [])

    if not result:
        raise ValueError(f"No chart data returned. Error: {chart.get('error')}")

    result0 = result[0]
    ts = result0.get("timestamp", [])
    quote = result0.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result0.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    closes = adjclose if adjclose is not None else quote.get("close")

    if not ts:
        raise ValueError("Yahoo chart API returned no timestamps.")

    df = pd.DataFrame({
        "Date": pd.to_datetime(ts, unit="s", utc=True),
        "Open": quote.get("open"),
        "High": quote.get("high"),
        "Low": quote.get("low"),
        "Close": closes,
        "Volume": quote.get("volume"),
    })

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = safe_numeric(df[col])

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
    df = df.set_index("Date")
    return normalize_dt_index(df)


def fetch_from_yfinance(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
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
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[needed].copy()
    for col in needed:
        df[col] = safe_numeric(df[col])

    df = df.dropna(subset=needed).copy()
    return normalize_dt_index(df)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, str]:
    errors = []

    try:
        df = fetch_from_yahoo_chart_api(symbol, start_date, end_date)
        if not df.empty:
            return df, "Yahoo Chart API"
    except Exception as e:
        errors.append(f"Yahoo Chart API: {e}")

    try:
        df = fetch_from_yfinance(symbol, start_date, end_date)
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


# =============================================================================
# SIGNAL ENGINE
# =============================================================================
def compute_percentile_of_last(values: pd.Series) -> float:
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)


def classify_event(pctl: float, high_cutoff: float, low_cutoff: float) -> str:
    if not np.isfinite(pctl):
        return "Unavailable"
    if pctl >= high_cutoff:
        return "Heavy"
    if pctl <= low_cutoff:
        return "Quiet"
    return "Normal"


def compute_volume_framework(
    df: pd.DataFrame,
    volume_mode: str,
    shares_outstanding: Optional[float],
    percentile_window: int,
    smooth_window: int,
    high_cutoff: float,
    low_cutoff: float,
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()

    if volume_mode == "Turnover %" and shares_outstanding and shares_outstanding > 0:
        out["Volume_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Turnover (% of shares outstanding)"
    else:
        out["Volume_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (shares)"

    out["Volume_Baseline"] = out["Volume_Display"].rolling(
        smooth_window, min_periods=max(10, smooth_window // 2)
    ).median()

    out["Volume_Ratio"] = out["Volume_Display"] / out["Volume_Baseline"].replace(0, np.nan)

    out["Volume_Pctl"] = out["Volume_Display"].rolling(
        percentile_window, min_periods=max(40, percentile_window // 3)
    ).apply(compute_percentile_of_last, raw=False)

    out["State"] = out["Volume_Pctl"].apply(lambda x: classify_event(x, high_cutoff, low_cutoff))
    out["Is_Heavy"] = out["State"].eq("Heavy")
    out["Is_Quiet"] = out["State"].eq("Quiet")

    out["Ret_1D"] = out["Close"].pct_change() * 100.0
    out["Ret_5D"] = out["Close"].pct_change(5) * 100.0
    out["Ret_20D"] = out["Close"].pct_change(20) * 100.0

    out["Price_MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["Price_MA50"] = out["Close"].rolling(50, min_periods=50).mean()

    return out, vol_label


def build_recent_events_table(
    df: pd.DataFrame,
    vol_label: str,
    event_filter: str,
    max_rows: int = 12,
) -> pd.DataFrame:
    events = df[df["State"].isin(["Heavy", "Quiet"])].copy()

    if event_filter == "Heavy only":
        events = events[events["State"] == "Heavy"].copy()
    elif event_filter == "Quiet only":
        events = events[events["State"] == "Quiet"].copy()

    if events.empty:
        return pd.DataFrame()

    events = events.tail(max_rows).copy()
    events["Date"] = events.index.strftime("%Y-%m-%d")

    if "Turnover" in vol_label:
        volume_col = events["Volume_Display"].map(lambda x: f"{x:.2f}%")
        baseline_col = events["Volume_Baseline"].map(lambda x: f"{x:.2f}%")
    else:
        volume_col = events["Volume_Display"].map(fmt_large_number)
        baseline_col = events["Volume_Baseline"].map(fmt_large_number)

    out = pd.DataFrame({
        "Date": events["Date"],
        "State": events["State"],
        "Volume": volume_col,
        "Baseline": baseline_col,
        "Ratio": events["Volume_Ratio"].map(lambda x: f"{x:.2f}x" if np.isfinite(x) else "N/A"),
        "Percentile": events["Volume_Pctl"].map(lambda x: f"{x:.0f}" if np.isfinite(x) else "N/A"),
        "1D Return": events["Ret_1D"].map(lambda x: f"{x:+.2f}%" if np.isfinite(x) else "N/A"),
    })

    return out.iloc[::-1].reset_index(drop=True)


# =============================================================================
# CHARTING
# =============================================================================
def build_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    show_price_mas: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=symbol,
            line=dict(width=2.2, color="#111111"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if show_price_mas:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA20"],
                mode="lines",
                name="20D MA",
                line=dict(width=1.1, color="rgba(59,130,246,0.72)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>20D MA: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA50"],
                mode="lines",
                name="50D MA",
                line=dict(width=1.1, color="rgba(107,114,128,0.78)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>50D MA: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    heavy = df[df["Is_Heavy"]].copy()
    quiet = df[df["Is_Quiet"]].copy()

    if not heavy.empty:
        fig.add_trace(
            go.Scatter(
                x=heavy.index,
                y=heavy["Close"],
                mode="markers",
                name="Heavy",
                marker=dict(size=7, color="#b45309"),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Heavy volume session"
                    "<br>Percentile: %{customdata[0]:.0f}"
                    "<br>Ratio: %{customdata[1]:.2f}x"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([heavy["Volume_Pctl"], heavy["Volume_Ratio"]]),
            ),
            row=1,
            col=1,
        )

    if not quiet.empty:
        fig.add_trace(
            go.Scatter(
                x=quiet.index,
                y=quiet["Close"],
                mode="markers",
                name="Quiet",
                marker=dict(size=6, color="#166534"),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Quiet volume session"
                    "<br>Percentile: %{customdata[0]:.0f}"
                    "<br>Ratio: %{customdata[1]:.2f}x"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([quiet["Volume_Pctl"], quiet["Volume_Ratio"]]),
            ),
            row=1,
            col=1,
        )

    bar_colors = np.where(
        df["State"].eq("Heavy"),
        "rgba(180,83,9,0.62)",
        np.where(df["State"].eq("Quiet"), "rgba(22,101,52,0.58)", "rgba(140,140,140,0.24)")
    )

    if "Turnover" in vol_label:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Turnover: %{y:.2f}%"
            "<br>Percentile: %{customdata[0]:.0f}"
            "<br>Ratio: %{customdata[1]:.2f}x"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:.2f}%<extra></extra>"
    else:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Volume: %{y:,.0f}"
            "<br>Percentile: %{customdata[0]:.0f}"
            "<br>Ratio: %{customdata[1]:.2f}x"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:,.0f}<extra></extra>"

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume_Display"],
            name="Volume",
            marker_color=bar_colors,
            width=0.78 * 24 * 60 * 60 * 1000,
            customdata=np.column_stack([df["Volume_Pctl"], df["Volume_Ratio"]]),
            hovertemplate=bar_hover,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume_Baseline"],
            mode="lines",
            name="Baseline",
            line=dict(width=1.8, color="#2563eb"),
            hovertemplate=baseline_hover,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        margin=dict(l=12, r=12, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        bargap=0.05,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#444444"),
        ),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=US_MARKET_HOLIDAYS),
        ],
        showgrid=True,
        gridcolor="#f1f3f5",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#666666"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        title_text="Price",
        showgrid=True,
        gridcolor="#ececec",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#666666"),
        title_font=dict(size=11, color="#666666"),
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text=vol_label,
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color="#666666"),
        title_font=dict(size=11, color="#666666"),
    )

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Volume-based sentiment monitor comparing daily participation versus trailing history.

        **What this tab shows**
        - Unusually heavy or quiet participation relative to each instrument's own baseline.
        - A cleaner read on conviction behind recent price movement.
        - Focused diagnostics for tape strength and exhaustion risk.

        **Data source**
        - Public market price and volume history feeds used throughout the app.
        """
    )

    st.markdown("---")

    symbol_input = st.text_input("Ticker", value="QQQ")
    symbol = symbol_input.upper().strip()

    st.caption("Quick tickers")
    c1, c2 = st.columns(2)
    selected_symbol = None
    for i, sym in enumerate(DEFAULT_SYMBOLS):
        with (c1 if i % 2 == 0 else c2):
            if st.button(sym, use_container_width=True):
                selected_symbol = sym

    if selected_symbol:
        symbol = selected_symbol

    st.markdown("---")
    st.subheader("Lookback")

    lookback_months = st.slider(
        "Visible history (months)",
        min_value=6,
        max_value=36,
        value=18,
        step=3,
    )

    percentile_window = st.slider(
        "Percentile window (days)",
        min_value=60,
        max_value=252,
        value=126,
        step=21,
    )

    smooth_window = st.slider(
        "Baseline window (days)",
        min_value=10,
        max_value=60,
        value=20,
        step=5,
    )

    st.markdown("---")
    st.subheader("Detection")

    volume_mode = st.selectbox(
        "Volume mode",
        options=["Raw volume", "Turnover %"],
        index=1,
    )

    high_cutoff = st.slider(
        "Heavy threshold percentile",
        min_value=75,
        max_value=99,
        value=90,
        step=1,
    )

    low_cutoff = st.slider(
        "Quiet threshold percentile",
        min_value=1,
        max_value=25,
        value=10,
        step=1,
    )

    show_price_mas = st.checkbox("Show 20D and 50D price averages", value=False)
    show_event_table = st.checkbox("Show recent event table", value=True)
    event_filter = st.selectbox(
        "Event table filter",
        options=["All extremes", "Heavy only", "Quiet only"],
        index=0,
    )


# =============================================================================
# DATA PREP
# =============================================================================
if not symbol:
    st.warning("Enter a ticker to continue.")
    st.stop()

end_date = today_utc_ts()
warmup_days = max(percentile_window + 40, 220)
start_date = end_date - pd.Timedelta(days=lookback_months * 31 + warmup_days)
visible_start = end_date - pd.Timedelta(days=lookback_months * 31)

with st.spinner(f"Loading {symbol}..."):
    try:
        raw_df, data_source = fetch_ohlcv(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}. Details: {e}")
        st.stop()

if raw_df.empty:
    st.warning("No data returned for this ticker.")
    st.stop()

shares_outstanding = None
if volume_mode == "Turnover %":
    with st.spinner("Fetching shares outstanding..."):
        shares_outstanding = fetch_shares_outstanding(symbol)

raw_df = normalize_dt_index(raw_df)

df, vol_label = compute_volume_framework(
    raw_df,
    volume_mode=volume_mode,
    shares_outstanding=shares_outstanding,
    percentile_window=percentile_window,
    smooth_window=smooth_window,
    high_cutoff=high_cutoff,
    low_cutoff=low_cutoff,
)

df = df.loc[df.index >= visible_start].copy()

if df.empty:
    st.warning("No usable data remained after processing.")
    st.stop()


# =============================================================================
# LATEST SNAPSHOT
# =============================================================================
latest = df.iloc[-1]

latest_close = float(latest["Close"]) if pd.notna(latest["Close"]) else np.nan
latest_vol = float(latest["Volume_Display"]) if pd.notna(latest["Volume_Display"]) else np.nan
latest_base = float(latest["Volume_Baseline"]) if pd.notna(latest["Volume_Baseline"]) else np.nan
latest_ratio = float(latest["Volume_Ratio"]) if pd.notna(latest["Volume_Ratio"]) else np.nan
latest_pctl = float(latest["Volume_Pctl"]) if pd.notna(latest["Volume_Pctl"]) else np.nan
latest_ret_1d = float(latest["Ret_1D"]) if pd.notna(latest["Ret_1D"]) else np.nan
latest_state = latest["State"] if pd.notna(latest["State"]) else "Unavailable"

left, right = st.columns([0.70, 0.30], gap="large")

with right:
    st.markdown("### Current read")
    st.metric("State", latest_state)
    st.metric("Volume percentile", f"{latest_pctl:.0f}" if np.isfinite(latest_pctl) else "N/A")
    st.metric("Vs baseline", f"{latest_ratio:.2f}x" if np.isfinite(latest_ratio) else "N/A")
    st.metric("Last close", f"${latest_close:,.2f}" if np.isfinite(latest_close) else "N/A")
    st.metric("1D move", f"{latest_ret_1d:+.2f}%" if np.isfinite(latest_ret_1d) else "N/A")

    if "Turnover" in vol_label:
        st.caption(
            f"Latest turnover {latest_vol:.2f}% versus {latest_base:.2f}% baseline"
            if np.isfinite(latest_vol) and np.isfinite(latest_base)
            else "Turnover data unavailable"
        )
    else:
        st.caption(
            f"Latest volume {fmt_large_number(latest_vol)} versus {fmt_large_number(latest_base)} baseline"
            if np.isfinite(latest_vol) and np.isfinite(latest_base)
            else "Volume data unavailable"
        )

    with st.expander("Data notes", expanded=False):
        st.write(f"Source: {data_source}")
        st.write(f"Mode: {vol_label}")
        st.write(f"Shares outstanding: {fmt_large_number(shares_outstanding)}")
        st.write(f"Visible window: last {lookback_months} months")
        st.write(f"Percentile window: {percentile_window} trading days")
        st.write(f"Baseline window: {smooth_window} trading days")
        st.write("X-axis compresses weekends and listed US market holidays.")

with left:
    fig = build_chart(
        df=df,
        symbol=symbol,
        vol_label=vol_label,
        show_price_mas=show_price_mas,
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


# =============================================================================
# RECENT EXTREMES TABLE
# =============================================================================
if show_event_table:
    events = build_recent_events_table(
        df=df,
        vol_label=vol_label,
        event_filter=event_filter,
        max_rows=12,
    )

    st.markdown("### Recent extremes")
    if events.empty:
        st.info("No extreme sessions found in the visible window.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)

st.caption("© 2026 AD Fund Management LP")
