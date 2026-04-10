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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inflation Surprise Tracker", layout="wide")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
APP_TITLE = "Inflation Surprise Tracker"
START_DATE = "2005-01-01"
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

THEME = {
    "headline": "#2563eb",
    "core": "#f59e0b",
    "pos": "#16a34a",
    "neg": "#dc2626",
    "neutral": "#64748b",
    "text": "#111827",
    "subtle_text": "#6b7280",
    "grid": "rgba(120,120,120,0.16)",
    "zero": "rgba(30,41,59,0.30)",
    "paper_bg": "white",
    "plot_bg": "white",
    "border": "rgba(31,41,55,0.10)",
}

SURPRISE_METHODS = [
    "Consensus upload",
    "Vs prior month",
    "Vs 6M rolling median",
]

WINDOW_OPTIONS = ["3Y", "5Y", "10Y", "All"]

os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Track CPI surprises and measure how markets reacted around each release.

        What it does
        • Computes headline/core CPI MoM and YoY
        • Measures surprise versus uploaded consensus or fallback proxy
        • Tracks same-day, 1D, and 5D asset reactions
        • Shows which assets are most sensitive to inflation shocks

        Data
        • FRED public CSV endpoints for CPI
        • Yahoo Finance for market prices

        Notes
        • Best mode is uploaded consensus with exact release dates
        • Fallback modes are useful but are still proxies
        """
    )
    st.markdown("---")

    selected_window = st.selectbox("History window", WINDOW_OPTIONS, index=1)
    surprise_method = st.selectbox("Surprise method", SURPRISE_METHODS, index=0)
    event_horizon = st.selectbox("Reaction horizon", ["0D", "1D", "5D"], index=2)
    use_core_for_summary = st.checkbox("Use core CPI surprise in summary", value=True)
    show_debug = st.checkbox("Show loader diagnostics", value=False)

    st.markdown("---")
    st.subheader("Optional consensus upload")
    uploaded_consensus = st.file_uploader(
        "Upload CSV with release dates and consensus",
        type=["csv"],
        help=(
            "Columns supported: release_date, cpi_month, headline_mom_consensus, "
            "core_mom_consensus. Release date should be the actual CPI release date."
        ),
    )

    st.markdown("---")
    if st.button("Clear cached data"):
        st.cache_data.clear()
        for filename in os.listdir(CACHE_DIR):
            path = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(path):
                os.remove(path)
        st.success("Cache cleared. Rerun the app.")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def apply_base_layout(fig: go.Figure, title: Optional[str] = None, height: int = 450) -> go.Figure:
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
            bgcolor="rgba(255,255,255,0.75)",
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


def fmt_pct(x: float, digits: int = 2) -> str:
    return "N/A" if pd.isna(x) else f"{x:.{digits}f}%"


def month_start(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="start")


def cache_path(name: str) -> str:
    safe_name = name.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.csv")


def save_text_cache(name: str, text: str) -> None:
    with open(cache_path(name), "w", encoding="utf-8") as f:
        f.write(text)


def load_text_cache(name: str) -> Optional[str]:
    path = cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_df_cache(name: str, df: pd.DataFrame) -> None:
    path = cache_path(name)
    df.to_csv(path)


def load_df_cache(name: str) -> Optional[pd.DataFrame]:
    path = cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.25,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://fred.stlouisfed.org/",
        }
    )
    return session


def detect_bad_payload(text: str, label: str) -> None:
    preview = text[:1500].lower()
    if not text.strip():
        raise RuntimeError(f"Empty payload for {label}")
    if "<html" in preview or "<!doctype html" in preview:
        raise RuntimeError(f"HTML returned instead of CSV for {label}")


def normalize_fred_columns(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    original_cols = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    date_col = None
    for candidate in ["date", "observation_date"]:
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break

    value_col = None
    for candidate in [series_id.lower(), "value", "observation_value"]:
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


def fetch_fred_series_csv(series_id: str) -> pd.DataFrame:
    session = build_session()
    urls = [
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?cosd={START_DATE}&id={series_id}",
    ]

    last_error = None
    for url in urls:
        try:
            resp = session.get(url, timeout=(10, 90))
            resp.raise_for_status()
            text = resp.text
            detect_bad_payload(text, series_id)
            df = pd.read_csv(StringIO(text))
            if df.empty:
                raise RuntimeError(f"Empty CSV returned for {series_id}")
            df = normalize_fred_columns(df, series_id)
            save_text_cache(series_id, text)
            return df
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(0.75)

    cached = load_text_cache(series_id)
    if cached is not None:
        df = pd.read_csv(StringIO(cached))
        df = normalize_fred_columns(df, series_id)
        return df

    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_error}")


def load_fred_series(series_id: str, start: str) -> pd.Series:
    df = fetch_fred_series_csv(series_id)
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
def fetch_macro_data(start: str = START_DATE) -> pd.DataFrame:
    series_map = {}
    failures = []

    for key, series_id in FRED_SERIES.items():
        try:
            series_map[key] = load_fred_series(series_id, start)
        except Exception as e:
            failures.append(f"{key} ({series_id}): {e}")

    if not series_map:
        raise RuntimeError("All FRED series failed. " + " | ".join(failures))

    df = pd.concat(series_map.values(), axis=1)
    df.columns = list(series_map.keys())
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.resample("MS").last()
    df.attrs["failures"] = failures
    return df


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


def download_one_ticker_with_timeout(ticker: str, start: str, timeout_sec: int = 20) -> pd.DataFrame:
    result: Dict[str, object] = {}
    thread = threading.Thread(target=_download_one_ticker_worker, args=(ticker, start, result), daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError(f"{ticker} timed out after {timeout_sec}s")

    if "error" in result:
        raise RuntimeError(str(result["error"]))

    return result["data"]


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_asset_prices(start: str = START_DATE) -> pd.DataFrame:
    frames = []
    failures = []
    diagnostics = []

    for label, ticker in ASSET_MAP.items():
        cache_name = f"asset_{label}_{ticker}"
        t0 = time.time()

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
                failures.append(f"{label} live failed, used cache: {e}")
                diagnostics.append(f"{label}: cache fallback after {time.time() - t0:.1f}s")
            else:
                failures.append(f"{label} ({ticker}): {e}")
                diagnostics.append(f"{label}: failed in {time.time() - t0:.1f}s")

    if not frames:
        raise RuntimeError("All market assets failed. " + " | ".join(failures))

    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.attrs["failures"] = failures
    out.attrs["diagnostics"] = diagnostics
    return out


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

    if not {"release_date", "cpi_month"}.issubset(set(df.columns)):
        raise ValueError(
            "Consensus CSV must include at least: release_date, cpi_month, "
            "headline_mom_consensus, core_mom_consensus"
        )

    if "headline_mom_consensus" not in df.columns and "core_mom_consensus" not in df.columns:
        raise ValueError("Consensus CSV must include headline_mom_consensus or core_mom_consensus")

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


def plot_surprise_history(events: pd.DataFrame, use_core: bool) -> go.Figure:
    s_col = get_summary_surprise_col(use_core)
    title_series = "Core CPI MoM Surprise" if use_core else "Headline CPI MoM Surprise"

    colors = np.where(events[s_col] >= 0, THEME["pos"], THEME["neg"])

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
    fig.add_hline(y=0, line_color=THEME["zero"], line_width=1)
    fig.update_yaxes(title_text="Surprise (pp)")
    apply_base_layout(fig, title=f"{title_series} Over Time", height=380)
    return fig


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

    fig.add_hline(y=0, line_color=THEME["zero"], line_width=1)
    fig.update_yaxes(title_text="Beta of return vs surprise")
    apply_base_layout(fig, title=f"Asset Sensitivity to Inflation Surprise | {horizon}", height=380)
    return fig


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
    fig.add_hline(y=0, line_color=THEME["zero"], line_width=1)
    fig.add_vline(x=0, line_color=THEME["zero"], line_width=1)
    fig.update_xaxes(title_text="Inflation Surprise (pp)")
    fig.update_yaxes(title_text=f"{asset} {horizon} Return (%)")
    apply_base_layout(fig, title=f"{asset} Reaction vs Inflation Surprise", height=410)
    return fig


# -----------------------------------------------------------------------------
# Load data with visible progress
# -----------------------------------------------------------------------------
st.title(APP_TITLE)

status_box = st.empty()
status_box.info("Loading CPI data from FRED...")

try:
    macro_raw = fetch_macro_data(START_DATE)
except Exception as e:
    st.error(f"Failed to load CPI data: {e}")
    st.stop()

macro_failures = macro_raw.attrs.get("failures", [])
if macro_failures:
    st.warning("Some FRED series failed to load: " + " | ".join(macro_failures))

status_box.info("CPI loaded. Loading market prices asset by asset...")

try:
    prices = fetch_asset_prices(START_DATE)
except Exception as e:
    st.error(f"Failed to load market prices: {e}")
    st.stop()

market_failures = prices.attrs.get("failures", [])
market_diagnostics = prices.attrs.get("diagnostics", [])

if market_failures:
    st.warning("Some market assets failed to load: " + " | ".join(market_failures))

status_box.success("Data loaded.")

if show_debug and market_diagnostics:
    with st.expander("Loader diagnostics", expanded=True):
        for line in market_diagnostics:
            st.write(line)

if macro_raw.empty or prices.empty:
    st.error("Required macro data or market data is empty.")
    st.stop()

# -----------------------------------------------------------------------------
# Prepare dataset
# -----------------------------------------------------------------------------
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
    st.error("No event dataset could be built. Try a different date range or check data connectivity.")
    st.stop()

events = filter_history_window(events, selected_window, "event_day")

if events.empty:
    st.warning("No events available in the selected window.")
    st.stop()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
latest = events.dropna(subset=["headline_mom", "core_mom"], how="all").iloc[-1]
latest_surprise_col = get_summary_surprise_col(use_core_for_summary)
latest_surprise = latest.get(latest_surprise_col, np.nan)

st.caption(
    f"Latest event: {pd.to_datetime(latest['event_day']).strftime('%Y-%m-%d')} | "
    f"CPI month: {pd.to_datetime(latest['cpi_month']).strftime('%Y-%m')} | "
    f"Method: {surprise_method}"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Headline CPI MoM", fmt_pct(latest.get("headline_mom", np.nan)))
m2.metric("Core CPI MoM", fmt_pct(latest.get("core_mom", np.nan)))
m3.metric("Headline Surprise", "N/A" if pd.isna(latest.get("headline_surprise", np.nan)) else f"{latest.get('headline_surprise'):+.2f} pp")
m4.metric("Core Surprise", "N/A" if pd.isna(latest.get("core_surprise", np.nan)) else f"{latest.get('core_surprise'):+.2f} pp")
m5.metric(f"Selected Surprise ({'Core' if use_core_for_summary else 'Headline'})", "N/A" if pd.isna(latest_surprise) else f"{latest_surprise:+.2f} pp")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
st.plotly_chart(plot_surprise_history(events, use_core_for_summary), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_asset_reaction_bars(events, event_horizon, use_core_for_summary), use_container_width=True)
with col_b:
    sensitivity_tbl = make_sensitivity_table(events, event_horizon, use_core_for_summary)
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
        )

scatter_cols = st.columns(3)
for slot, asset in zip(scatter_cols, ["SPY", "TLT", "DXY"]):
    with slot:
        st.plotly_chart(plot_scatter(events, asset, event_horizon, use_core_for_summary), use_container_width=True)

scatter_cols2 = st.columns(3)
for slot, asset in zip(scatter_cols2, ["QQQ", "Gold", "Crude"]):
    with slot:
        st.plotly_chart(plot_scatter(events, asset, event_horizon, use_core_for_summary), use_container_width=True)

# -----------------------------------------------------------------------------
# Recent events
# -----------------------------------------------------------------------------
st.subheader("Recent CPI Events")
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
    )

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
with st.expander("Download Data"):
    export_df = events.copy()
    for col in ["cpi_month", "release_date", "event_day", "prev_day"]:
        if col in export_df.columns:
            export_df[col] = pd.to_datetime(export_df[col]).dt.strftime("%Y-%m-%d")

    st.download_button(
        "Download event study CSV",
        export_df.to_csv(index=False),
        file_name="inflation_surprise_tracker.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Methodology
# -----------------------------------------------------------------------------
with st.expander("Methodology & Data Notes", expanded=False):
    st.markdown(
        """
        The event dataset starts with monthly headline CPI and core CPI from FRED. It converts those
        index levels into MoM and YoY inflation rates, then assigns a CPI release date. If you upload
        a consensus file, the app uses your exact release dates and consensus estimates. If you do not,
        it uses an estimated release date and fallback surprise proxies.

        Surprise definitions:
        • Consensus upload: actual MoM inflation minus uploaded consensus
        • Vs prior month: actual MoM inflation minus prior-month MoM
        • Vs 6M rolling median: actual MoM inflation minus prior 6-month rolling median of MoM prints

        Asset reaction definitions:
        • 0D: event-day close versus prior trading day close
        • 1D: next trading day close versus event-day close
        • 5D: five trading days after event-day close versus event-day close
        """
    )

st.caption("© 2026 AD Fund Management LP")
