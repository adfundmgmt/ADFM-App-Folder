# app.py

import io
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Cboe Total Put/Call Ratio",
    layout="wide",
)

CBOE_CSV_URLS = [
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/pcratioarchive.csv",
]

CBOE_DAILY_URL = "https://www.cboe.com/us/options/market_statistics/daily/?dt={date}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def clean_column_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .upper()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("  ", " ")
    )


def find_header_row(raw_text: str) -> int:
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        cleaned = line.upper()
        if "DATE" in cleaned and ("P/C" in cleaned or "PUT/CALL" in cleaned or "RATIO" in cleaned):
            return i
    return 0


def parse_cboe_csv_text(raw_text: str) -> pd.DataFrame:
    header_row = find_header_row(raw_text)

    df = pd.read_csv(io.StringIO(raw_text), skiprows=header_row)
    df.columns = [clean_column_name(c) for c in df.columns]

    date_col = None
    for col in df.columns:
        if col == "DATE" or "DATE" in col:
            date_col = col
            break

    if date_col is None:
        raise ValueError("Could not find a date column in Cboe CSV.")

    ratio_candidates = [
        col for col in df.columns
        if (
            "P/C" in col
            or "PUT/CALL" in col
            or "RATIO" in col
        )
        and "TOTAL" in col
        and col != date_col
    ]

    if not ratio_candidates:
        ratio_candidates = [
            col for col in df.columns
            if ("P/C" in col or "PUT/CALL" in col or "RATIO" in col)
            and col != date_col
        ]

    if not ratio_candidates:
        raise ValueError("Could not find a put/call ratio column in Cboe CSV.")

    ratio_col = ratio_candidates[0]

    out = df[[date_col, ratio_col]].copy()
    out.columns = ["Date", "Total Put/Call Ratio"]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Total Put/Call Ratio"] = pd.to_numeric(
        out["Total Put/Call Ratio"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    out = out.dropna(subset=["Date", "Total Put/Call Ratio"])
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.sort_values("Date").reset_index(drop=True)

    return out


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_cboe_csv_history() -> pd.DataFrame:
    frames = []

    for url in CBOE_CSV_URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()

            parsed = parse_cboe_csv_text(response.text)
            if not parsed.empty:
                frames.append(parsed)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["Date", "Total Put/Call Ratio"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Date", "Total Put/Call Ratio"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)

    return combined


def parse_daily_total_put_call_ratio(html: str) -> float | None:
    match = re.search(r"TOTAL\s+PUT/CALL\s+RATIO\s+([0-9]+(?:\.[0-9]+)?)", html, re.IGNORECASE)
    if match:
        return float(match.group(1))

    try:
        tables = pd.read_html(io.StringIO(html))
        for table in tables:
            table_str = table.astype(str).to_string(index=False).upper()
            if "TOTAL PUT/CALL RATIO" in table_str:
                flat = table.astype(str).values.flatten().tolist()
                for i, item in enumerate(flat):
                    if "TOTAL PUT/CALL RATIO" in item.upper():
                        for candidate in flat[i + 1:i + 4]:
                            candidate = str(candidate).replace(",", "").strip()
                            if re.match(r"^[0-9]+(?:\.[0-9]+)?$", candidate):
                                return float(candidate)
    except Exception:
        pass

    return None


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_cboe_daily_range(start_date: date, end_date: date) -> pd.DataFrame:
    rows = []

    business_days = pd.bdate_range(start=start_date, end=end_date)

    for dt in business_days:
        dt_str = dt.strftime("%Y-%m-%d")
        url = CBOE_DAILY_URL.format(date=dt_str)

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            ratio = parse_daily_total_put_call_ratio(response.text)
            if ratio is not None and np.isfinite(ratio):
                rows.append(
                    {
                        "Date": pd.Timestamp(dt.date()),
                        "Total Put/Call Ratio": ratio,
                    }
                )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Date", "Total Put/Call Ratio"])

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Date"], keep="last")
    out = out.sort_values("Date").reset_index(drop=True)

    return out


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_spx(start_date: date, end_date: date) -> pd.DataFrame:
    df = yf.download(
        "^GSPC",
        start=start_date - timedelta(days=7),
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df.empty:
        return pd.DataFrame(columns=["Date", "SPX"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    out = df[[close_col]].copy()
    out.columns = ["SPX"]
    out = out.reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.dropna(subset=["SPX"])

    return out


def load_put_call_data(start_date: date, end_date: date) -> pd.DataFrame:
    csv_history = load_cboe_csv_history()

    if csv_history.empty:
        daily = fetch_cboe_daily_range(start_date, end_date)
        return daily

    csv_history["Date"] = pd.to_datetime(csv_history["Date"]).dt.tz_localize(None)
    csv_history = csv_history[
        (csv_history["Date"] >= pd.Timestamp(start_date))
        & (csv_history["Date"] <= pd.Timestamp(end_date))
    ].copy()

    latest_csv_date = csv_history["Date"].max() if not csv_history.empty else None

    if latest_csv_date is None:
        daily_start = start_date
    else:
        daily_start = max(start_date, (latest_csv_date + pd.Timedelta(days=1)).date())

    needs_daily_backfill = daily_start <= end_date

    if needs_daily_backfill:
        daily = fetch_cboe_daily_range(daily_start, end_date)
        combined = pd.concat([csv_history, daily], ignore_index=True)
    else:
        combined = csv_history

    combined = combined.dropna(subset=["Date", "Total Put/Call Ratio"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)

    return combined


def make_chart(
    df: pd.DataFrame,
    ma_window: int,
    show_raw: bool,
    show_thresholds: bool,
    upper_threshold: float,
    lower_threshold: float,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.38],
    )

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Total Put/Call Ratio"],
                mode="lines",
                name="$CPC Raw",
                line=dict(width=1),
                opacity=0.35,
                hovertemplate="%{x|%b %d, %Y}<br>$CPC Raw: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["CPC MA"],
            mode="lines",
            name=f"$CPC MA({ma_window})",
            line=dict(width=2),
            hovertemplate="%{x|%b %d, %Y}<br>$CPC MA: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if show_thresholds:
        fig.add_hline(
            y=upper_threshold,
            row=1,
            col=1,
            line_width=1,
            line_dash="solid",
            annotation_text=f"Upper threshold {upper_threshold:.2f}",
            annotation_position="top left",
        )

        fig.add_hline(
            y=lower_threshold,
            row=1,
            col=1,
            line_width=1,
            line_dash="dot",
            annotation_text=f"Lower threshold {lower_threshold:.2f}",
            annotation_position="bottom left",
        )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SPX"],
            mode="lines",
            name="$SPX",
            line=dict(width=2),
            hovertemplate="%{x|%b %d, %Y}<br>SPX: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        margin=dict(l=40, r=40, t=45, b=35),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        title=dict(
            text="Cboe Total Put/Call Ratio vs. S&P 500",
            x=0,
            xanchor="left",
        ),
    )

    fig.update_yaxes(title_text="Total Put/Call Ratio", row=1, col=1)
    fig.update_yaxes(title_text="S&P 500", row=2, col=1)

    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showgrid=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showgrid=True,
        row=2,
        col=1,
    )

    return fig


st.title("Cboe Total Put/Call Ratio")
st.caption("Replicates a StockCharts-style $CPC panel with the S&P 500 underneath.")

with st.sidebar:
    st.markdown("### Controls")

    today = date.today()
    default_start = today - timedelta(days=730)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)

    ma_window = st.slider("Put/call moving average", min_value=5, max_value=200, value=63, step=1)

    show_raw = st.checkbox("Show raw daily $CPC", value=False)
    show_thresholds = st.checkbox("Show threshold lines", value=True)

    upper_threshold = st.number_input("Upper threshold", value=0.95, step=0.01, format="%.2f")
    lower_threshold = st.number_input("Lower threshold", value=0.80, step=0.01, format="%.2f")

    st.markdown("### About This Tool")
    st.write(
        "The put/call ratio comes from Cboe market statistics. "
        "The S&P 500 series is pulled with yfinance using ^GSPC. "
        "The main line is a moving average of the Cboe Total Put/Call Ratio."
    )

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Loading Cboe put/call data and S&P 500 prices..."):
    pcr = load_put_call_data(start_date, end_date)
    spx = load_spx(start_date, end_date)

if pcr.empty:
    st.error("Could not load Cboe put/call ratio data.")
    st.stop()

if spx.empty:
    st.error("Could not load S&P 500 data.")
    st.stop()

df = pd.merge(pcr, spx, on="Date", how="inner")
df = df.sort_values("Date").reset_index(drop=True)

df["CPC MA"] = df["Total Put/Call Ratio"].rolling(ma_window, min_periods=max(3, ma_window // 4)).mean()

df = df.dropna(subset=["CPC MA", "SPX"])

if df.empty:
    st.error("No overlapping data found for the selected date range.")
    st.stop()

fig = make_chart(
    df=df,
    ma_window=ma_window,
    show_raw=show_raw,
    show_thresholds=show_thresholds,
    upper_threshold=upper_threshold,
    lower_threshold=lower_threshold,
)

st.plotly_chart(fig, use_container_width=True)

latest = df.iloc[-1]

st.caption(
    f"Latest available: {latest['Date'].strftime('%Y-%m-%d')} | "
    f"$CPC raw: {latest['Total Put/Call Ratio']:.2f} | "
    f"$CPC MA({ma_window}): {latest['CPC MA']:.2f} | "
    f"SPX: {latest['SPX']:,.0f}"
)
