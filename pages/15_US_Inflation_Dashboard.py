import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="US CPI Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FRED_SERIES = {
    "headline": "CPIAUCNS",
    "core": "CPILFESL",
    "recession": "USREC",
}

START_DATE_FULL = "1990-01-01"
APP_TITLE = "US Inflation Dashboard"

THEME = {
    "headline": "#2563eb",
    "core": "#f59e0b",
    "recession_fill": "rgba(60, 60, 60, 0.16)",
    "text": "#111827",
    "subtle_text": "#6b7280",
    "grid": "rgba(120, 120, 120, 0.16)",
    "zero": "rgba(30, 41, 59, 0.35)",
    "paper_bg": "white",
    "plot_bg": "white",
    "border": "rgba(31, 41, 55, 0.10)",
    "card_bg": "#ffffff",
}

PERIOD_OPTIONS = ["1M", "3M", "6M", "9M", "1Y", "3Y", "5Y", "All"]

CACHE_DIR = Path(".fred_cache")
CACHE_DIR.mkdir(exist_ok=True)

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
REQUEST_TIMEOUT = (10, 60)  # connect timeout, read timeout
MAX_RETRIES = 4

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.8rem;
        max-width: 1650px;
    }
    h1, h2, h3 {
        letter-spacing: 0.1px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(31, 41, 55, 0.10);
        border-radius: 14px;
        padding: 10px 12px;
    }
    .adfm-card {
        background: #ffffff;
        border: 1px solid rgba(31, 41, 55, 0.10);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 86px;
    }
    .adfm-card-label {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .adfm-card-value {
        font-size: 22px;
        font-weight: 600;
        color: #111827;
        line-height: 1.2;
    }
    .adfm-caption {
        color: #6b7280;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: US CPI dashboard for inflation trend, momentum, and regime classification.

        What it covers  
        • Headline and core CPI across level, YoY, MoM, and 3M annualised views  
        • Short-horizon inflation momentum versus slower-moving YoY base effects  
        • Recession overlays for historical context and regime interpretation

        Data source  
        • Public FRED CSV endpoints from the St. Louis Fed
        """
    )
    st.markdown("---")
    st.subheader("Time Range")
    period = st.selectbox("Select period:", PERIOD_OPTIONS, index=7)
    use_custom_range = st.checkbox("Custom date range", value=False)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=1.25,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Cache-Control": "no-cache",
        }
    )
    return session


def apply_base_layout(fig: go.Figure, title: str | None = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=THEME["paper_bg"],
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text"], size=13),
        margin=dict(l=24, r=24, t=70 if title else 24, b=24),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor=THEME["border"],
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=THEME["grid"],
        showline=True,
        linecolor=THEME["border"],
        rangeslider=dict(visible=False),
        tickformat="%Y-%m",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=THEME["grid"],
        zeroline=True,
        zerolinecolor=THEME["zero"],
        showline=True,
        linecolor=THEME["border"],
    )
    return fig


def fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}%"


def safe_delta_text(curr: float, prev: float) -> str:
    if pd.isna(curr) or pd.isna(prev):
        return "vs prior print: N/A"
    return f"vs prior print: {curr - prev:+.2f} pp"


def classify_inflation_regime(core_yoy: float, core_3m_ann: float) -> str:
    if pd.isna(core_yoy) or pd.isna(core_3m_ann):
        return "Insufficient data"
    gap = core_3m_ann - core_yoy
    if gap <= -0.35:
        return "Disinflationary"
    if gap >= 0.35:
        return "Reheating"
    return "Sticky"


def month_start(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="start")


def period_to_start_date(period_label: str, end_date: pd.Timestamp) -> pd.Timestamp:
    end_date = month_start(end_date)

    if period_label == "All":
        return month_start(pd.Timestamp(START_DATE_FULL))

    mapping = {
        "1M": 3,
        "3M": 6,
        "6M": 9,
        "9M": 12,
        "1Y": 18,
        "3Y": 42,
        "5Y": 66,
    }
    months_back = mapping.get(period_label)
    if months_back is None:
        raise ValueError(f"Invalid period selected: {period_label}")
    return month_start(end_date - pd.DateOffset(months=months_back))


def build_custom_month_range(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    default_start: pd.Timestamp,
    default_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_index = pd.period_range(min_date, max_date, freq="M")
    month_labels = [str(p) for p in month_index]

    default_start_period = pd.Period(default_start, freq="M")
    default_end_period = pd.Period(default_end, freq="M")

    try:
        default_value = (
            month_labels.index(str(default_start_period)),
            month_labels.index(str(default_end_period)),
        )
    except ValueError:
        default_value = (0, len(month_labels) - 1)

    sel = st.sidebar.select_slider(
        "Select month range:",
        options=month_labels,
        value=(month_labels[default_value[0]], month_labels[default_value[1]]),
    )
    start = pd.Period(sel[0], freq="M").to_timestamp(how="start")
    end = pd.Period(sel[1], freq="M").to_timestamp(how="start")
    return start, end


def detect_fred_error_payload(text: str, series_id: str) -> None:
    preview = text[:800].lower()
    if "<html" in preview or "<!doctype html" in preview:
        raise RuntimeError(
            f"FRED returned HTML instead of CSV for {series_id}. "
            f"This is usually a temporary upstream issue or rate-limit page."
        )
    if "error" in preview and "date" not in preview:
        raise RuntimeError(f"FRED returned an error payload for {series_id}.")


def normalize_fred_columns(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    original_cols = list(df.columns)
    normalized_map = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=normalized_map)

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    date_col = None
    for candidate in ["date", "observation_date"]:
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break

    value_col = None
    for candidate in [series_id.lower(), "value", "observation_value", "cpi", "usrec"]:
        if candidate in lower_map:
            value_col = lower_map[candidate]
            break

    if value_col is None:
        non_date_cols = [c for c in df.columns if c != date_col]
        if len(non_date_cols) == 1:
            value_col = non_date_cols[0]

    if date_col is None or value_col is None:
        raise ValueError(
            f"Unexpected FRED response format for {series_id}. Columns received: {original_cols}"
        )

    out = df[[date_col, value_col]].copy()
    out.columns = ["DATE", series_id]
    return out


def cache_path_for_series(series_id: str) -> Path:
    return CACHE_DIR / f"{series_id}.csv"


def write_series_cache(series_id: str, text: str) -> None:
    cache_path_for_series(series_id).write_text(text, encoding="utf-8")


def read_series_cache(series_id: str) -> str | None:
    p = cache_path_for_series(series_id)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def fetch_fred_csv_text(series_id: str) -> str:
    url = FRED_BASE_URL.format(series_id=series_id)
    session = get_requests_session()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            text = response.text
            detect_fred_error_payload(text, series_id)
            if not text.strip():
                raise RuntimeError(f"Empty CSV payload for {series_id}")
            write_series_cache(series_id, text)
            return text
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES + 1:
                time.sleep(min(1.5 * attempt, 5))

    cached = read_series_cache(series_id)
    if cached:
        return cached

    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_error}")


def fetch_fred_csv(series_id: str) -> pd.DataFrame:
    text = fetch_fred_csv_text(series_id)
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise RuntimeError(f"Empty dataframe returned for {series_id}")
    df = normalize_fred_columns(df, series_id)
    return df


def load_fred_series(series_id: str, start: str) -> pd.Series:
    df = fetch_fred_csv(series_id)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    df = df.dropna(subset=["DATE"]).copy()
    df = df[df["DATE"] >= pd.Timestamp(start)].copy()

    s = df.set_index("DATE")[series_id].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = s.index.to_period("M").to_timestamp(how="start")
    s.name = series_id
    return s


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_fred_dataset(start: str = START_DATE_FULL) -> pd.DataFrame:
    series_map = {}
    for key, series_id in FRED_SERIES.items():
        series_map[key] = load_fred_series(series_id, start)

    df = pd.concat(series_map.values(), axis=1)
    df.columns = list(series_map.keys())
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.resample("MS").last()
    return df


def prepare_cpi_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["headline_yoy"] = df["headline"].pct_change(12) * 100
    df["core_yoy"] = df["core"].pct_change(12) * 100
    df["headline_mom"] = df["headline"].pct_change(1) * 100
    df["core_mom"] = df["core"].pct_change(1) * 100
    df["core_3m_ann"] = ((df["core"] / df["core"].shift(3)) ** 4 - 1) * 100
    df["headline_3m_ann"] = ((df["headline"] / df["headline"].shift(3)) ** 4 - 1) * 100
    df["core_3m_vs_yoy_gap"] = df["core_3m_ann"] - df["core_yoy"]

    valid_hist = df["core_3m_ann"].dropna()
    if not valid_hist.empty:
        df["core_3m_ann_percentile"] = df["core_3m_ann"].apply(
            lambda x: np.nan if pd.isna(x) else valid_hist.le(x).mean() * 100
        )
        mean = valid_hist.mean()
        std = valid_hist.std(ddof=0)
        if pd.notna(std) and not np.isclose(std, 0):
            df["core_3m_ann_zscore"] = (df["core_3m_ann"] - mean) / std
        else:
            df["core_3m_ann_zscore"] = np.nan
    else:
        df["core_3m_ann_percentile"] = np.nan
        df["core_3m_ann_zscore"] = np.nan

    return df


def filter_window(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start_date = month_start(start_date)
    end_date = month_start(end_date)
    return df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()


def get_recession_periods(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    f = flag.fillna(0).astype(int)
    if f.empty:
        return []

    starts = f.index[(f.eq(1)) & (f.shift(1, fill_value=0).eq(0))]
    ends = f.index[(f.eq(0)) & (f.shift(1, fill_value=0).eq(1))]

    if not f.empty and f.iloc[-1] == 1:
        ends = ends.append(pd.Index([f.index[-1]]))

    periods = []
    for s, e in zip(starts, ends):
        if e >= s:
            periods.append((pd.Timestamp(s), pd.Timestamp(e)))
    return periods


def trim_recession_periods(
    periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for s, e in periods:
        if e < start_date or s > end_date:
            continue
        out.append((max(s, start_date), min(e, end_date)))
    return out


def add_recession_shapes(fig: go.Figure, periods: list[tuple[pd.Timestamp, pd.Timestamp]], rows: list[int]) -> None:
    for row in rows:
        xref = "x" if row == 1 else f"x{row}"
        yref = "y domain" if row == 1 else f"y{row} domain"
        for s, e in periods:
            fig.add_shape(
                type="rect",
                xref=xref,
                yref=yref,
                x0=s,
                x1=e,
                y0=0,
                y1=1,
                fillcolor=THEME["recession_fill"],
                line_width=0,
                layer="below",
            )


def latest_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float("nan") if s.empty else float(s.iloc[-1])


def prev_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float("nan") if len(s) < 2 else float(s.iloc[-2])


def build_latest_table(df: pd.DataFrame, rows: int = 12) -> pd.DataFrame:
    cols = [
        "headline",
        "core",
        "headline_yoy",
        "core_yoy",
        "headline_mom",
        "core_mom",
        "core_3m_ann",
        "core_3m_vs_yoy_gap",
    ]
    tbl = df[cols].dropna(how="all").tail(rows).copy()
    tbl.index.name = "Date"
    tbl.columns = [
        "Headline CPI",
        "Core CPI",
        "Headline YoY (%)",
        "Core YoY (%)",
        "Headline MoM (%)",
        "Core MoM (%)",
        "Core 3M Ann. (%)",
        "Core 3M Ann. - YoY Gap (pp)",
    ]
    return tbl


def render_info_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="adfm-card">
            <div class="adfm-card-label">{label}</div>
            <div class="adfm-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
def plot_yoy(df: pd.DataFrame, recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["headline_yoy"],
            name="Headline CPI YoY",
            mode="lines",
            line=dict(color=THEME["headline"], width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Headline YoY: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["core_yoy"],
            name="Core CPI YoY",
            mode="lines",
            line=dict(color=THEME["core"], width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Core YoY: %{y:.2f}%<extra></extra>",
        )
    )

    add_recession_shapes(fig, recession_periods, rows=[1])
    fig.update_yaxes(title_text="% YoY")
    apply_base_layout(fig, title="US CPI YoY | Headline vs Core", height=430)
    return fig


def plot_mom(df: pd.DataFrame, recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Headline CPI MoM", "Core CPI MoM"),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["headline_mom"],
            name="Headline MoM",
            marker_color=THEME["headline"],
            hovertemplate="%{x|%Y-%m}<br>Headline MoM: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["core_mom"],
            name="Core MoM",
            marker_color=THEME["core"],
            hovertemplate="%{x|%Y-%m}<br>Core MoM: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    add_recession_shapes(fig, recession_periods, rows=[1, 2])
    fig.update_yaxes(title_text="% MoM", row=1, col=1)
    fig.update_yaxes(title_text="% MoM", row=2, col=1)
    fig.update_layout(showlegend=False)
    apply_base_layout(fig, title="US CPI MoM Prints", height=560)
    return fig


def plot_core_panel(df: pd.DataFrame, recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Core CPI Index", "Core CPI 3M Annualised"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["core"],
            name="Core CPI Index",
            mode="lines",
            line=dict(color=THEME["core"], width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Core Index: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["core_3m_ann"],
            name="Core 3M Annualised",
            mode="lines",
            line=dict(color=THEME["headline"], width=2.6),
            hovertemplate="%{x|%Y-%m}<br>Core 3M Ann.: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    add_recession_shapes(fig, recession_periods, rows=[1, 2])
    fig.update_yaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="% Ann.", row=2, col=1)
    apply_base_layout(fig, title="Core CPI Level and Short-Horizon Inflation Signal", height=560)
    return fig

# -----------------------------------------------------------------------------
# Data load and prep
# -----------------------------------------------------------------------------
try:
    raw_df = fetch_fred_dataset(START_DATE_FULL)
except Exception as e:
    st.error(f"Failed to load FRED data: {e}")
    st.stop()

if raw_df.empty:
    st.error("Critical FRED data failed to load. Please refresh or check your connection.")
    st.stop()

df = prepare_cpi_dataset(raw_df)

required_cols = ["headline", "core", "recession", "headline_yoy", "core_yoy", "headline_mom", "core_mom"]
if df[required_cols].dropna(how="all").empty:
    st.error("The dataset loaded, but the required CPI fields are missing or empty.")
    st.stop()

# -----------------------------------------------------------------------------
# Date window handling
# -----------------------------------------------------------------------------
latest_data_date = month_start(df.index.max())
default_start_date = period_to_start_date(period, latest_data_date)
default_end_date = latest_data_date

if use_custom_range:
    start_date, end_date = build_custom_month_range(
        min_date=month_start(df.index.min()),
        max_date=latest_data_date,
        default_start=default_start_date,
        default_end=default_end_date,
    )
else:
    start_date, end_date = default_start_date, default_end_date

window_df = filter_window(df, start_date, end_date)

if window_df.empty:
    st.warning("No data is available for the selected range.")
    st.stop()

if window_df[["headline_yoy", "core_yoy", "headline_mom", "core_mom"]].dropna(how="all").empty:
    st.warning("The selected window does not contain enough observations to compute inflation rates.")
    st.stop()

recession_periods = get_recession_periods(df["recession"])
recession_periods_window = trim_recession_periods(recession_periods, start_date, end_date)

# -----------------------------------------------------------------------------
# Header and regime strip
# -----------------------------------------------------------------------------
st.title(APP_TITLE)

latest_row = df.dropna(subset=["headline", "core"], how="all").iloc[-1]
latest_print_date = latest_row.name

headline_yoy_latest = latest_valid(window_df["headline_yoy"])
headline_yoy_prev = prev_valid(window_df["headline_yoy"])

core_yoy_latest = latest_valid(window_df["core_yoy"])
core_yoy_prev = prev_valid(window_df["core_yoy"])

core_3m_latest = latest_valid(window_df["core_3m_ann"])
core_3m_prev = prev_valid(window_df["core_3m_ann"])

headline_3m_latest = latest_valid(window_df["headline_3m_ann"])
gap_latest = latest_valid(window_df["core_3m_vs_yoy_gap"])
percentile_latest = latest_valid(window_df["core_3m_ann_percentile"])

regime_label = classify_inflation_regime(core_yoy_latest, core_3m_latest)

st.markdown(
    f"""
    <div class="adfm-caption">
        Latest available print: {latest_print_date.strftime('%B %Y')} |
        Window: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Headline CPI YoY", fmt_pct(headline_yoy_latest), safe_delta_text(headline_yoy_latest, headline_yoy_prev))
m2.metric("Core CPI YoY", fmt_pct(core_yoy_latest), safe_delta_text(core_yoy_latest, core_yoy_prev))
m3.metric("Core 3M Ann.", fmt_pct(core_3m_latest), safe_delta_text(core_3m_latest, core_3m_prev))
m4.metric("Headline 3M Ann.", fmt_pct(headline_3m_latest))
m5.metric("Core 3M vs YoY Gap", "N/A" if pd.isna(gap_latest) else f"{gap_latest:+.2f} pp", regime_label)

c1, c2, c3 = st.columns(3)
with c1:
    render_info_card("Inflation regime", regime_label)
with c2:
    zscore_latest = latest_valid(window_df["core_3m_ann_zscore"])
    zscore_text = "N/A" if pd.isna(zscore_latest) else f"{zscore_latest:.2f}σ"
    render_info_card("Core 3M annualised z-score", zscore_text)
with c3:
    pct_text = "N/A" if pd.isna(percentile_latest) else f"{percentile_latest:.1f}th pct"
    render_info_card("Core 3M annualised historical percentile", pct_text)

st.markdown("")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
st.plotly_chart(plot_yoy(window_df, recession_periods_window), use_container_width=True)
st.plotly_chart(plot_mom(window_df, recession_periods_window), use_container_width=True)
st.plotly_chart(plot_core_panel(window_df, recession_periods_window), use_container_width=True)

# -----------------------------------------------------------------------------
# Latest prints table
# -----------------------------------------------------------------------------
st.subheader("Latest Prints")
latest_tbl = build_latest_table(window_df, rows=12)

fmt_cols = {
    "Headline CPI": "{:.2f}",
    "Core CPI": "{:.2f}",
    "Headline YoY (%)": "{:.2f}",
    "Core YoY (%)": "{:.2f}",
    "Headline MoM (%)": "{:.2f}",
    "Core MoM (%)": "{:.2f}",
    "Core 3M Ann. (%)": "{:.2f}",
    "Core 3M Ann. - YoY Gap (pp)": "{:.2f}",
}

st.dataframe(
    latest_tbl.style.format(fmt_cols),
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
with st.expander("Download Data"):
    include_recession_flag = st.checkbox("Include recession flag", value=False)

    export_cols = [
        "headline",
        "core",
        "headline_yoy",
        "core_yoy",
        "headline_mom",
        "core_mom",
        "headline_3m_ann",
        "core_3m_ann",
        "core_3m_vs_yoy_gap",
        "core_3m_ann_percentile",
        "core_3m_ann_zscore",
    ]
    if include_recession_flag:
        export_cols.append("recession")

    export_df = window_df[export_cols].copy()
    export_df.index.name = "Date"

    export_names = [
        "Headline CPI",
        "Core CPI",
        "Headline YoY (%)",
        "Core YoY (%)",
        "Headline MoM (%)",
        "Core MoM (%)",
        "Headline 3M Ann. (%)",
        "Core 3M Ann. (%)",
        "Core 3M Ann. - YoY Gap (pp)",
        "Core 3M Ann. Percentile",
        "Core 3M Ann. Z-Score",
    ]
    if include_recession_flag:
        export_names.append("Recession Flag")

    export_df.columns = export_names

    st.download_button(
        "Download CSV",
        export_df.to_csv(index=True, index_label="Date"),
        file_name="us_cpi_dashboard_data.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Methodology
# -----------------------------------------------------------------------------
with st.expander("Methodology & Sources", expanded=False):
    st.markdown(
        """
        **Data source**  
        FRED direct CSV endpoint from the St. Louis Fed

        **Series used**  
        • Headline CPI: `CPIAUCNS`  
        • Core CPI: `CPILFESL`  
        • Recession indicator: `USREC`

        **Transformations**  
        • YoY = `(CPI_t / CPI_{t-12} - 1) × 100`  
        • MoM = `(CPI_t / CPI_{t-1} - 1) × 100`  
        • 3M annualised = `((CPI_t / CPI_{t-3})^4 - 1) × 100`  
        • Core 3M vs YoY gap = `Core 3M annualised - Core YoY`

        **Interpretation layer**  
        The dashboard labels the current inflation regime using the gap between core 3M annualised inflation and core YoY inflation:  
        • materially below YoY = disinflationary  
        • near YoY = sticky  
        • materially above YoY = reheating

        **Reliability changes in this version**  
        • Replaced `urllib` with `requests.Session()`  
        • Added retry and backoff logic for transient FRED/network issues  
        • Increased read timeout materially  
        • Added last-good local CSV cache fallback per series

        **Notes**  
        All series are standardised to month-start timestamps and aligned into one monthly dataframe before calculation, plotting, and export.
        """
    )

st.caption("© 2026 AD Fund Management LP")
