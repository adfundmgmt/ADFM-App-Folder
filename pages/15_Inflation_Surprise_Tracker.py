import os
import time
import threading
from io import StringIO
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- Config ----------------
TITLE = "Inflation Surprise Tracker"
START_DATE = "2010-01-01"
CACHE_DIR = "macro_cache"

FRED_SERIES = {
    "headline_cpi": "CPIAUCNS",
    "core_cpi": "CPILFESL",
    "recession": "USREC",
}

ASSET_MAP = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "TLT": "TLT",
    "DXY": "DX-Y.NYB",
    "Gold": "GLD",
    "Crude": "CL=F",
}

SURPRISE_METHODS = [
    "Consensus upload",
    "Vs prior month",
    "Vs 6M rolling median",
]

WINDOW_OPTIONS = ["3Y", "5Y", "10Y", "All"]

st.set_page_config(page_title=TITLE, layout="wide")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------- Professional UI ----------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: 0.1px;
    }

    .adfm-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: -0.15rem;
        color: #111827;
    }

    .adfm-subtitle {
        font-size: 0.98rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        min-height: 92px;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }

    .metric-footnote {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 0.4rem;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }

    .sidebar-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 1rem;
    }

    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>CPI event study dashboard measuring inflation surprises and cross-asset reaction around each release.</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### About This Tool")
    st.markdown(
        """
This dashboard tracks CPI event risk and how markets actually reacted.

**What it shows**
- Headline and core CPI MoM and YoY
- Surprise versus uploaded consensus or fallback proxy
- Same-day, 1D, and 5D moves in SPY, QQQ, TLT, DXY, Gold, and Crude
- Sensitivity of each asset to inflation shocks

**How to use it**
- Uploaded consensus is the best version
- Proxy mode still helps when you want directionality and reaction mapping
- 0D is the cleanest release-day shock
- 1D and 5D help you see whether the tape confirmed or faded the initial move
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    history_window = st.selectbox("History window", WINDOW_OPTIONS, index=1)
    surprise_method = st.selectbox("Surprise method", SURPRISE_METHODS, index=0)
    reaction_horizon = st.selectbox("Reaction horizon", ["0D", "1D", "5D"], index=2)
    use_core_for_summary = st.checkbox("Use core surprise in summary", value=True)
    show_loader_diagnostics = st.checkbox("Show loader diagnostics", value=False)
    st.caption("FRED source: public CSV endpoint")
    st.caption("No FRED API key required")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### Optional Consensus Upload")
    uploaded_consensus = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "Columns supported: release_date, cpi_month, headline_mom_consensus, "
            "core_mom_consensus"
        ),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Clear cached data"):
        st.cache_data.clear()
        for filename in os.listdir(CACHE_DIR):
            path = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(path):
                os.remove(path)
        st.success("Cache cleared. Rerun the app.")

# ---------------- Helpers ----------------
def cache_path(name: str) -> str:
    safe_name = name.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.csv")


def save_df_cache(name: str, df: pd.DataFrame) -> None:
    df.to_csv(cache_path(name))


def load_df_cache(name: str) -> Optional[pd.DataFrame]:
    path = cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def fmt_pct(x: float, digits: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:.{digits}f}%"


def fmt_pp(x: float, digits: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:+.{digits}f} pp"


def fmt_num(x: float, digits: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:.{digits}f}"


def metric_card(label: str, value: str, footnote: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def month_start(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="start")


def build_request_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.75,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Connection": "keep-alive",
        }
    )
    return session


# ---------------- FRED loaders: same approach as your working example ----------------
def fetch_fred_csv(series_id: str, start: pd.Timestamp, end: pd.Timestamp, timeout: int = 20) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start.strftime('%Y-%m-%d')}"
        f"&coed={end.strftime('%Y-%m-%d')}"
    )

    session = build_request_session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    text = r.text
    preview = text[:1000].lower()
    if not text.strip():
        raise ValueError(f"Empty payload returned for {series_id}")
    if "<html" in preview or "<!doctype html" in preview:
        raise ValueError(f"HTML returned instead of CSV for {series_id}")

    df = pd.read_csv(StringIO(text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable data returned for {series_id}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    s = pd.Series(df[value_col].values, index=df[date_col], name=series_id)
    s = s[~s.index.duplicated(keep="last")]
    return s


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_all_fred_series(start: pd.Timestamp, end: pd.Timestamp):
    out = {}
    errors = {}
    diagnostics = []

    for key, series_id in FRED_SERIES.items():
        t0 = time.time()
        cache_name = f"fred_{series_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"

        try:
            s = fetch_fred_csv(series_id, start, end, timeout=20)
            out[key] = s
            save_df_cache(cache_name, s.to_frame(name=series_id))
            diagnostics.append(f"{series_id}: live OK in {time.time() - t0:.1f}s")
        except Exception as e:
            cached = load_df_cache(cache_name)
            if cached is not None and not cached.empty:
                cached.index = pd.to_datetime(cached.index)
                out[key] = cached.iloc[:, 0].rename(series_id)
                errors[series_id] = f"live failed, used cache: {e}"
                diagnostics.append(f"{series_id}: cache fallback after {time.time() - t0:.1f}s")
            else:
                out[key] = pd.Series(dtype="float64", name=series_id)
                errors[series_id] = str(e)
                diagnostics.append(f"{series_id}: failed in {time.time() - t0:.1f}s")

    return out, errors, diagnostics


# ---------------- Yahoo loaders with hard timeout ----------------
def _download_one_ticker_worker(ticker: str, start: str, result: dict) -> None:
    try:
        end_date = (pd.Timestamp.today() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        if df is None or df.empty:
            raise RuntimeError("Empty dataframe")

        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if col not in df.columns:
            raise RuntimeError("Missing Close/Adj Close")

        px = pd.to_numeric(df[col], errors="coerce").dropna().to_frame(name=ticker)
        if px.empty:
            raise RuntimeError("No usable prices after cleaning")

        result["data"] = px
    except Exception as e:
        result["error"] = str(e)


def download_one_ticker_with_timeout(ticker: str, start: str, timeout_sec: int = 18) -> pd.DataFrame:
    result: Dict[str, object] = {}
    thread = threading.Thread(target=_download_one_ticker_worker, args=(ticker, start, result), daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError(f"{ticker} timed out after {timeout_sec}s")
    if "error" in result:
        raise RuntimeError(str(result["error"]))

    return result["data"]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_asset_prices(start: str = START_DATE):
    frames = []
    failures = {}
    diagnostics = []

    for label, ticker in ASSET_MAP.items():
        t0 = time.time()
        cache_name = f"asset_{label}_{ticker}"

        try:
            px = download_one_ticker_with_timeout(ticker=ticker, start=start, timeout_sec=18)
            px.columns = [label]
            px.index = pd.to_datetime(px.index).normalize()
            px = px[~px.index.duplicated(keep="last")].sort_index()
            save_df_cache(cache_name, px)
            frames.append(px)
            diagnostics.append(f"{label}: live OK in {time.time() - t0:.1f}s")
        except Exception as e:
            cached = load_df_cache(cache_name)
            if cached is not None and not cached.empty:
                cached.columns = [label]
                cached.index = pd.to_datetime(cached.index).normalize()
                frames.append(cached)
                failures[label] = f"live failed, used cache: {e}"
                diagnostics.append(f"{label}: cache fallback after {time.time() - t0:.1f}s")
            else:
                failures[label] = str(e)
                diagnostics.append(f"{label}: failed in {time.time() - t0:.1f}s")

    if not frames:
        raise RuntimeError("All market assets failed to load")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices, failures, diagnostics


# ---------------- CPI prep ----------------
def prepare_cpi_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["headline_yoy"] = df["headline_cpi"].pct_change(12) * 100
    df["core_yoy"] = df["core_cpi"].pct_change(12) * 100
    df["headline_mom"] = df["headline_cpi"].pct_change(1) * 100
    df["core_mom"] = df["core_cpi"].pct_change(1) * 100
    df["cpi_month"] = df.index

    approx_release = []
    for dt in df.index:
        next_month = (dt + pd.offsets.MonthBegin(1)) + pd.DateOffset(days=12)
        if next_month.weekday() >= 5:
            next_month = next_month + pd.offsets.BDay(1)
        approx_release.append(pd.Timestamp(next_month).normalize())

    df["release_date_est"] = pd.to_datetime(approx_release)
    df["headline_mom_prior"] = df["headline_mom"].shift(1)
    df["core_mom_prior"] = df["core_mom"].shift(1)
    df["headline_mom_6m_median"] = df["headline_mom"].rolling(6).median().shift(1)
    df["core_mom_6m_median"] = df["core_mom"].rolling(6).median().shift(1)

    return df.reset_index(drop=True)


def parse_consensus_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]

    required_min = {"release_date", "cpi_month"}
    if not required_min.issubset(df.columns):
        raise ValueError(
            "Consensus CSV must include at least release_date and cpi_month"
        )

    if "headline_mom_consensus" not in df.columns and "core_mom_consensus" not in df.columns:
        raise ValueError(
            "Consensus CSV must include headline_mom_consensus or core_mom_consensus"
        )

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.normalize()
    df["cpi_month"] = pd.to_datetime(df["cpi_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    for col in ["headline_mom_consensus", "core_mom_consensus"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["release_date", "cpi_month"]).copy()
    return df


def merge_consensus(cpi_df: pd.DataFrame, uploaded_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = cpi_df.copy()

    if uploaded_df is None:
        df["release_date"] = df["release_date_est"]
        df["headline_mom_consensus"] = np.nan
        df["core_mom_consensus"] = np.nan
        return df

    merged = df.merge(
        uploaded_df[
            [c for c in ["release_date", "cpi_month", "headline_mom_consensus", "core_mom_consensus"] if c in uploaded_df.columns]
        ],
        on="cpi_month",
        how="left",
        suffixes=("", "_uploaded"),
    )

    if "release_date_uploaded" in merged.columns:
        merged["release_date"] = merged["release_date_uploaded"].combine_first(merged["release_date_est"])
        merged = merged.drop(columns=["release_date_uploaded"])
    else:
        merged["release_date"] = merged["release_date_est"]

    return merged


def compute_surprises(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()

    if method == "Consensus upload":
        out["headline_surprise"] = out["headline_mom"] - out["headline_mom_consensus"]
        out["core_surprise"] = out["core_mom"] - out["core_mom_consensus"]
    elif method == "Vs prior month":
        out["headline_surprise"] = out["headline_mom"] - out["headline_mom_prior"]
        out["core_surprise"] = out["core_mom"] - out["core_mom_prior"]
    elif method == "Vs 6M rolling median":
        out["headline_surprise"] = out["headline_mom"] - out["headline_mom_6m_median"]
        out["core_surprise"] = out["core_mom"] - out["core_mom_6m_median"]
    else:
        raise ValueError(f"Unsupported surprise method: {method}")

    return out


# ---------------- Event study ----------------
def next_trading_day_index(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = idx.searchsorted(pd.Timestamp(dt))
    if pos >= len(idx):
        return None
    return idx[pos]


def prev_trading_day_index(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = idx.searchsorted(pd.Timestamp(dt), side="left") - 1
    if pos < 0:
        return None
    return idx[pos]


def compute_event_reactions(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy()
    px.index = pd.to_datetime(px.index).normalize()
    trading_days = px.index

    rows = []
    for _, row in events.iterrows():
        release_date = pd.to_datetime(row["release_date"]).normalize()
        event_day = next_trading_day_index(trading_days, release_date)
        prev_day = prev_trading_day_index(trading_days, event_day) if event_day is not None else None

        if event_day is None or prev_day is None:
            continue

        result = row.to_dict()
        result["event_day"] = event_day
        result["prev_day"] = prev_day

        for asset in prices.columns:
            result[f"{asset}_0d"] = np.nan
            result[f"{asset}_1d"] = np.nan
            result[f"{asset}_5d"] = np.nan

            try:
                p_prev = px.loc[prev_day, asset]
                p_0 = px.loc[event_day, asset]

                pos = trading_days.get_loc(event_day)
                p_1 = px.iloc[pos + 1][asset] if pos + 1 < len(px) else np.nan
                p_5 = px.iloc[pos + 5][asset] if pos + 5 < len(px) else np.nan

                if pd.notna(p_prev) and pd.notna(p_0):
                    result[f"{asset}_0d"] = (p_0 / p_prev - 1.0) * 100
                if pd.notna(p_0) and pd.notna(p_1):
                    result[f"{asset}_1d"] = (p_1 / p_0 - 1.0) * 100
                if pd.notna(p_0) and pd.notna(p_5):
                    result[f"{asset}_5d"] = (p_5 / p_0 - 1.0) * 100
            except Exception:
                pass

        rows.append(result)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("event_day")


def filter_history_window(df: pd.DataFrame, window_label: str, date_col: str) -> pd.DataFrame:
    if df.empty or window_label == "All":
        return df.copy()

    latest = pd.to_datetime(df[date_col]).max()
    mapping = {"3Y": 3, "5Y": 5, "10Y": 10}
    years = mapping.get(window_label, 5)
    cutoff = latest - pd.DateOffset(years=years)
    return df[pd.to_datetime(df[date_col]) >= cutoff].copy()


def pick_reaction_col(asset: str, horizon: str) -> str:
    return f"{asset}_{horizon.lower()}"


def get_summary_surprise_col(use_core: bool) -> str:
    return "core_surprise" if use_core else "headline_surprise"


def correlation_safe(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 6:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1])


def beta_safe(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 6:
        return np.nan
    xv = tmp.iloc[:, 0]
    yv = tmp.iloc[:, 1]
    var = xv.var(ddof=0)
    if np.isclose(var, 0):
        return np.nan
    cov = np.cov(xv, yv, ddof=0)[0, 1]
    return cov / var


def make_sensitivity_table(events: pd.DataFrame, horizon: str, use_core: bool) -> pd.DataFrame:
    surprise_col = get_summary_surprise_col(use_core)
    rows = []

    for asset in ASSET_MAP.keys():
        reaction_col = pick_reaction_col(asset, horizon)
        if reaction_col not in events.columns:
            continue

        tmp = events[[surprise_col, reaction_col]].dropna()
        if tmp.empty:
            continue

        rows.append(
            {
                "Asset": asset,
                "Obs": len(tmp),
                "Avg Return on Positive Surprise (%)": tmp.loc[tmp[surprise_col] > 0, reaction_col].mean(),
                "Avg Return on Negative Surprise (%)": tmp.loc[tmp[surprise_col] < 0, reaction_col].mean(),
                "Correlation to Surprise": correlation_safe(tmp[surprise_col], tmp[reaction_col]),
                "Reaction Beta": beta_safe(tmp[surprise_col], tmp[reaction_col]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Reaction Beta", ascending=False)
    return out


def build_latest_event_table(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cpi_month",
        "release_date",
        "headline_mom",
        "headline_mom_consensus",
        "headline_surprise",
        "core_mom",
        "core_mom_consensus",
        "core_surprise",
    ]
    keep_cols = [c for c in cols if c in events.columns]

    reaction_cols = []
    for asset in ASSET_MAP.keys():
        for h in ["0d", "1d", "5d"]:
            col = f"{asset}_{h}"
            if col in events.columns:
                reaction_cols.append(col)

    return events[keep_cols + reaction_cols].copy().tail(15)


# ---------------- Chart helpers ----------------
def apply_base_layout(fig: go.Figure, title: Optional[str] = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=55 if title else 20, b=30),
        legend=dict(
            orientation="h",
            x=0,
            y=1.03,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eceff3", zeroline=False)
    return fig


def plot_surprise_history(events: pd.DataFrame, use_core: bool) -> go.Figure:
    s_col = get_summary_surprise_col(use_core)
    title_series = "Core CPI MoM Surprise" if use_core else "Headline CPI MoM Surprise"
    colors = np.where(events[s_col] >= 0, "#16a34a", "#dc2626")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=events["event_day"],
            y=events[s_col],
            name=title_series,
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>Surprise: %{y:.2f} pp<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af")
    fig.update_yaxes(title_text="pp")
    return apply_base_layout(fig, title=f"{title_series} Over Time", height=380)


def plot_asset_reaction_bars(events: pd.DataFrame, horizon: str, use_core: bool) -> go.Figure:
    sensitivity = make_sensitivity_table(events, horizon, use_core)
    fig = go.Figure()

    if not sensitivity.empty:
        fig.add_trace(
            go.Bar(
                x=sensitivity["Asset"],
                y=sensitivity["Reaction Beta"],
                name="Reaction Beta",
                hovertemplate="%{x}<br>Beta: %{y:.2f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af")
    fig.update_yaxes(title_text="Beta")
    return apply_base_layout(fig, title=f"Asset Sensitivity to Inflation Surprise | {horizon}", height=360)


def plot_scatter(events: pd.DataFrame, asset: str, horizon: str, use_core: bool) -> go.Figure:
    s_col = get_summary_surprise_col(use_core)
    r_col = pick_reaction_col(asset, horizon)
    tmp = events[[s_col, r_col, "event_day"]].dropna().copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tmp[s_col],
            y=tmp[r_col],
            mode="markers",
            name=asset,
            marker=dict(size=9),
            text=tmp["event_day"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                "Date: %{text}<br>"
                "Surprise: %{x:.2f} pp<br>"
                f"{asset} {horizon}: "
                "%{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#9ca3af")
    fig.update_xaxes(title_text="Surprise (pp)")
    fig.update_yaxes(title_text=f"{asset} return (%)")
    return apply_base_layout(fig, title=f"{asset} Reaction vs Inflation Surprise", height=400)


# ---------------- Data ----------------
today = pd.Timestamp.today().normalize()
start = pd.Timestamp(START_DATE)

status_box = st.empty()
status_box.info("Loading CPI data from FRED...")

with st.spinner("Loading CPI data from FRED..."):
    raw_series, fred_errors, fred_diagnostics = load_all_fred_series(start, today)

required = ["headline_cpi", "core_cpi"]
missing_required = [k for k in required if raw_series[k].empty]

if missing_required:
    st.error(f"Required CPI series failed to load: {', '.join(missing_required)}")
    if fred_errors:
        with st.expander("FRED error details", expanded=True):
            for k, v in fred_errors.items():
                st.write(f"{k}: {v}")
    st.stop()

macro_raw = pd.concat(
    [
        raw_series["headline_cpi"].rename("headline_cpi"),
        raw_series["core_cpi"].rename("core_cpi"),
        raw_series["recession"].rename("recession"),
    ],
    axis=1,
).sort_index().ffill()

macro_raw = macro_raw[macro_raw.index >= start].copy()
macro_raw = macro_raw.dropna(subset=["headline_cpi", "core_cpi"])

if macro_raw.empty:
    st.error("Macro dataframe is empty after cleaning.")
    st.stop()

status_box.info("CPI loaded. Loading market prices asset by asset...")

with st.spinner("Loading market prices..."):
    try:
        prices, market_failures, market_diagnostics = fetch_asset_prices(START_DATE)
    except Exception as e:
        st.error(f"Market price load failed: {e}")
        st.stop()

status_box.success("Data loaded.")

if fred_errors:
    noncritical_fred = {k: v for k, v in fred_errors.items() if k not in ["headline_cpi", "core_cpi"]}
    if noncritical_fred:
        st.warning("Some non-critical FRED series failed to load.")
        with st.expander("FRED non-critical series errors"):
            for k, v in noncritical_fred.items():
                st.write(f"{k}: {v}")

if market_failures:
    st.warning("Some market assets failed to load.")
    with st.expander("Market asset errors"):
        for k, v in market_failures.items():
            st.write(f"{k}: {v}")

if show_loader_diagnostics:
    with st.expander("Loader diagnostics", expanded=True):
        st.markdown("**FRED**")
        for line in fred_diagnostics:
            st.write(line)
        st.markdown("**Market data**")
        for line in market_diagnostics:
            st.write(line)

# ---------------- Build event dataset ----------------
cpi = prepare_cpi_table(macro_raw)

consensus_df = None
if uploaded_consensus is not None:
    try:
        consensus_df = parse_consensus_upload(uploaded_consensus)
    except Exception as e:
        st.error(f"Consensus upload failed validation: {e}")
        st.stop()

events = merge_consensus(cpi, consensus_df)
events = compute_surprises(events, surprise_method)
events = compute_event_reactions(events, prices)

if events.empty:
    st.error("No event dataset could be built.")
    st.stop()

events = filter_history_window(events, history_window, "event_day")

if events.empty:
    st.warning("No events available in the selected window.")
    st.stop()

# ---------------- Snapshot ----------------
latest = events.dropna(subset=["headline_mom", "core_mom"], how="all").iloc[-1]
latest_surprise_col = get_summary_surprise_col(use_core_for_summary)
latest_surprise = latest.get(latest_surprise_col, np.nan)

st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Latest CPI event, latest surprise, and selected cross-asset reaction.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Event Date", pd.to_datetime(latest["event_day"]).strftime("%Y-%m-%d"), "Market event day")
with c2:
    metric_card("Headline MoM", fmt_pct(latest.get("headline_mom", np.nan)), "Actual print")
with c3:
    metric_card("Core MoM", fmt_pct(latest.get("core_mom", np.nan)), "Actual print")
with c4:
    metric_card("Headline Surprise", fmt_pp(latest.get("headline_surprise", np.nan)), "Vs selected method")
with c5:
    metric_card("Core Surprise", fmt_pp(latest.get("core_surprise", np.nan)), "Vs selected method")
with c6:
    metric_card(
        f"Selected Surprise",
        fmt_pp(latest_surprise),
        "Core" if use_core_for_summary else "Headline"
    )

st.caption(
    f"CPI month: {pd.to_datetime(latest['cpi_month']).strftime('%Y-%m')} | "
    f"Method: {surprise_method} | "
    f"Window: {history_window}"
)

# ---------------- Quick Read ----------------
st.markdown("<div class='section-title'>Quick Read</div>", unsafe_allow_html=True)

quick_left, quick_right = st.columns([1.35, 1])

with quick_left:
    mode_text = "uploaded consensus" if uploaded_consensus is not None else "proxy surprise method"
    st.markdown(
        f"""
        <div class="info-box">
        <b>Framework</b><br><br>
        This dashboard treats each CPI release as an event and then measures how major assets behaved around that shock. The cleanest version uses <b>{mode_text}</b>. When no consensus file is uploaded, the app falls back to internal baselines like prior-month MoM or the prior 6-month rolling median.
        <br><br>
        That still gives you a useful read on whether the latest print ran hot or cool versus recent inflation behavior, and whether the market confirmed or faded that impulse.
        </div>
        """,
        unsafe_allow_html=True,
    )

with quick_right:
    spy_col = pick_reaction_col("SPY", reaction_horizon)
    tlt_col = pick_reaction_col("TLT", reaction_horizon)
    spy_move = latest.get(spy_col, np.nan)
    tlt_move = latest.get(tlt_col, np.nan)

    st.markdown(
        f"""
        <div class="info-box">
        <b>Latest reaction</b><br><br>
        SPY moved <b>{fmt_pct(spy_move)}</b> over the selected <b>{reaction_horizon}</b> horizon and TLT moved <b>{fmt_pct(tlt_move)}</b>.
        <br><br>
        The selected summary series is <b>{"core" if use_core_for_summary else "headline"}</b>, which is useful because core usually carries more policy weight.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Chartbook ----------------
st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Surprise history, sensitivity ranking, and cross-asset scatter plots.</div>",
    unsafe_allow_html=True,
)

st.plotly_chart(plot_surprise_history(events, use_core_for_summary), use_container_width=True)

left_top, right_top = st.columns([1.0, 1.0])
with left_top:
    st.plotly_chart(plot_asset_reaction_bars(events, reaction_horizon, use_core_for_summary), use_container_width=True)
with right_top:
    sensitivity_tbl = make_sensitivity_table(events, reaction_horizon, use_core_for_summary)
    if sensitivity_tbl.empty:
        st.info("No sensitivity table available for the selected window.")
    else:
        st.dataframe(
            sensitivity_tbl.style.format(
                {
                    "Avg Return on Positive Surprise (%)": "{:.2f}",
                    "Avg Return on Negative Surprise (%)": "{:.2f}",
                    "Correlation to Surprise": "{:.2f}",
                    "Reaction Beta": "{:.2f}",
                }
            ),
            use_container_width=True,
            height=360,
        )

scatter_row_1 = st.columns(3)
for slot, asset in zip(scatter_row_1, ["SPY", "TLT", "DXY"]):
    with slot:
        st.plotly_chart(
            plot_scatter(events, asset, reaction_horizon, use_core_for_summary),
            use_container_width=True,
        )

scatter_row_2 = st.columns(3)
for slot, asset in zip(scatter_row_2, ["QQQ", "Gold", "Crude"]):
    with slot:
        st.plotly_chart(
            plot_scatter(events, asset, reaction_horizon, use_core_for_summary),
            use_container_width=True,
        )

# ---------------- Lower Panels ----------------
left, right = st.columns([1.2, 0.8])

with left:
    st.markdown("<div class='section-title'>Recent CPI Events</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Latest event rows used in the event study.</div>",
        unsafe_allow_html=True,
    )

    latest_events_tbl = build_latest_event_table(events)
    if latest_events_tbl.empty:
        st.info("No recent events table available.")
    else:
        fmt_map = {}
        for col in latest_events_tbl.columns:
            if "date" in col or "month" in col:
                continue
            if latest_events_tbl[col].dtype.kind in "fi":
                fmt_map[col] = "{:.2f}"

        st.dataframe(
            latest_events_tbl.style.format(fmt_map),
            use_container_width=True,
            height=420,
        )

with right:
    st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the event dataset used in the dashboard.</div>",
        unsafe_allow_html=True,
    )

    export_df = events.copy()
    for col in ["cpi_month", "release_date", "event_day", "prev_day"]:
        if col in export_df.columns:
            export_df[col] = pd.to_datetime(export_df[col]).dt.strftime("%Y-%m-%d")

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="inflation_surprise_tracker.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
        <b>FRED loading approach</b><br>
        This version uses the same direct public CSV endpoint structure as your working liquidity tracker:
        <br><br>
        <code>fredgraph.csv?id=SERIES&cosd=START&coed=END</code>
        <br><br>
        <b>Surprise definitions</b><br>
        • Consensus upload: actual MoM minus uploaded consensus<br>
        • Vs prior month: actual MoM minus prior-month MoM<br>
        • Vs 6M rolling median: actual MoM minus prior 6-month rolling median
        <br><br>
        <b>Asset reaction definitions</b><br>
        • 0D: event-day close versus prior trading day close<br>
        • 1D: next trading day close versus event-day close<br>
        • 5D: five trading days after event-day close versus event-day close
        <br><br>
        <b>Release timing</b><br>
        If no consensus file is uploaded, CPI release dates are estimated mechanically as a fallback.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("© 2026 AD Fund Management LP")
