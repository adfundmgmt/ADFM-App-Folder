import io
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ==============================
# Page config
# ==============================
TITLE = "Liquidity & Fed Policy Tracker"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)


# ==============================
# Constants
# ==============================
FRED_SERIES = {
    "WALCL": {
        "code": "WALCL",
        "label": "Fed Balance Sheet",
        "unit": "millions",
        "display_unit": "billions",
        "color": "#111111",
    },
    "RRP": {
        "code": "RRPONTSYD",
        "label": "ON RRP",
        "unit": "billions",
        "display_unit": "billions",
        "color": "#1f77b4",
    },
    "TGA": {
        "code": "WDTGAL",
        "label": "Treasury General Account",
        "unit": "millions",
        "display_unit": "billions",
        "color": "#d62728",
    },
    "EFFR": {
        "code": "EFFR",
        "label": "Effective Fed Funds Rate",
        "unit": "percent",
        "display_unit": "percent",
        "color": "#ff7f0e",
    },
    "NFCI": {
        "code": "NFCI",
        "label": "National Financial Conditions Index",
        "unit": "index",
        "display_unit": "index",
        "color": "#2ca02c",
    },
}

DEFAULT_SMOOTH_DAYS = 5
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0
CACHE_TTL_SECONDS = 12 * 60 * 60

# Local on-disk backup cache for last-good FRED pulls
LOCAL_CACHE_DIR = Path(".adfm_fred_cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Track system liquidity and Fed stance through a simple macro dashboard.

        **Core framework**

        Net Liquidity = WALCL − RRP − TGA

        **What you are looking at**

        • **WALCL**: Federal Reserve total assets  
        • **RRP**: Overnight reverse repo facility usage  
        • **TGA**: Treasury General Account balance  
        • **EFFR**: Effective fed funds rate  
        • **NFCI**: Chicago Fed National Financial Conditions Index  

        **Panels**

        1. Net Liquidity level and short-term change  
        2. Core balance sheet components in absolute terms  
        3. Components rebased to 100 for directional comparison  
        4. Fed policy and financial conditions
        """
    )

    st.markdown("---")
    st.header("Settings")

    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])

    smooth = st.number_input(
        "Smoothing window (days)",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH_DAYS,
        step=1,
    )

    roc_window = st.selectbox(
        "Change window",
        options=[5, 20, 60],
        index=1,
        format_func=lambda x: f"{x} trading days",
    )

    show_raw = st.checkbox("Show raw lines alongside smoothed lines", value=False)
    show_rebased = st.checkbox("Show rebased components panel", value=True)

    st.caption("Data source: FRED public CSV endpoints with local backup cache")


# ==============================
# Formatting helpers
# ==============================
def fmt_b(x):
    return "N/A" if pd.isna(x) else f"{x:,.0f} B"


def fmt_b_1(x):
    return "N/A" if pd.isna(x) else f"{x:,.1f} B"


def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"


def fmt_idx(x):
    return "N/A" if pd.isna(x) else f"{x:.3f}"


def fmt_delta_b(x):
    if pd.isna(x):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:,.0f} B"


def fmt_delta_pct(x):
    if pd.isna(x):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def fmt_delta_idx(x):
    if pd.isna(x):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


# ==============================
# FRED fetch infrastructure
# ==============================
def make_http_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; ADFM-Liquidity-Tracker/1.0; +https://fred.stlouisfed.org/)",
            "Accept": "text/csv,text/plain,application/octet-stream,*/*",
            "Connection": "keep-alive",
        }
    )
    return session


def build_fred_candidate_urls(series_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    urls = []

    # Most reliable general endpoint
    urls.append(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}")

    # Explicit date-bounded variant
    params = urlencode(
        {
            "id": series_code,
            "cosd": start_str,
            "coed": end_str,
        }
    )
    urls.append(f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}")

    # Alternate host pattern used by FRED chart exports
    urls.append(f"https://fred.stlouisfed.org/graph/fredgraph.csv?cosd={start_str}&coed={end_str}&id={series_code}")

    return urls


def parse_fred_csv_text(text: str, series_code: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))

    if "DATE" not in df.columns:
        possible_date_cols = [c for c in df.columns if c.lower() in {"date", "observation_date"}]
        if possible_date_cols:
            df = df.rename(columns={possible_date_cols[0]: "DATE"})

    if series_code not in df.columns:
        other_cols = [c for c in df.columns if c != "DATE"]
        if len(other_cols) == 1:
            df = df.rename(columns={other_cols[0]: series_code})

    if "DATE" not in df.columns or series_code not in df.columns:
        raise ValueError(f"Unexpected FRED response format for {series_code}. Columns: {list(df.columns)}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[series_code] = pd.to_numeric(df[series_code], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")

    s = df.set_index("DATE")[series_code]
    s.name = series_code
    return s


def local_cache_path(series_code: str) -> Path:
    return LOCAL_CACHE_DIR / f"{series_code}.csv"


def save_series_to_local_cache(series: pd.Series, series_code: str) -> None:
    cache_file = local_cache_path(series_code)
    out = series.reset_index()
    out.columns = ["DATE", series_code]
    out.to_csv(cache_file, index=False)


def load_series_from_local_cache(series_code: str) -> pd.Series | None:
    cache_file = local_cache_path(series_code)
    if not cache_file.exists():
        return None

    try:
        df = pd.read_csv(cache_file)
        if "DATE" not in df.columns or series_code not in df.columns:
            return None
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df[series_code] = pd.to_numeric(df[series_code], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        s = df.set_index("DATE")[series_code]
        s.name = series_code
        return s
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_fred_series(series_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Series, str]:
    """
    Returns:
        series, source_label
    source_label is one of:
        'live'
        'local_cache'
    """
    session = make_http_session()
    urls = build_fred_candidate_urls(series_code, start_date, end_date)

    last_error = None

    for url in urls:
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200 and resp.text and "DATE" in resp.text:
                s = parse_fred_csv_text(resp.text, series_code)
                s = s[(s.index >= start_date) & (s.index <= end_date)]
                if not s.dropna().empty:
                    save_series_to_local_cache(s, series_code)
                    return s, "live"
            else:
                last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)

    cached = load_series_from_local_cache(series_code)
    if cached is not None and not cached.dropna().empty:
        cached = cached[(cached.index >= start_date) & (cached.index <= end_date)]
        if not cached.dropna().empty:
            return cached, "local_cache"

    raise RuntimeError(f"Failed to fetch {series_code}. Last error: {last_error}")


# ==============================
# Data prep
# ==============================
def convert_display_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "WALCL" in out.columns:
        out["WALCL"] = out["WALCL"] / 1000.0
    if "TGA" in out.columns:
        out["TGA"] = out["TGA"] / 1000.0

    return out


def rebase(series: pd.Series, base_window: int = REBASE_BASE_WINDOW, min_base: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()

    if s.dropna().empty:
        return pd.Series(index=s.index, data=100.0)

    base_slice = s.dropna().iloc[: max(1, base_window)]
    base = base_slice.median() if not base_slice.empty else s.dropna().iloc[0]

    if min_base is not None:
        base = max(float(base), float(min_base))

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, data=100.0)

    return (s / base) * 100.0


def build_master_dataframe(start_all: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, dict, dict]:
    errors = {}
    sources = {}
    series_map = {}

    for name, meta in FRED_SERIES.items():
        code = meta["code"]
        try:
            s, src = fetch_fred_series(code, start_all, end_date)
            s.name = name
            series_map[name] = s
            sources[name] = src
        except Exception as e:
            errors[name] = str(e)

    if not series_map:
        # As a final fallback, try loading directly from local cache even if cache function failed above.
        for name, meta in FRED_SERIES.items():
            code = meta["code"]
            cached = load_series_from_local_cache(code)
            if cached is not None and not cached.dropna().empty:
                cached = cached[(cached.index >= start_all) & (cached.index <= end_date)]
                if not cached.dropna().empty:
                    cached.name = name
                    series_map[name] = cached
                    sources[name] = "local_cache"

    if not series_map:
        raise RuntimeError("No live FRED data and no local backup cache found yet.")

    full_idx = pd.date_range(start=start_all, end=end_date, freq="D")
    df = pd.concat(series_map.values(), axis=1).reindex(full_idx).sort_index().ffill()
    df.index.name = "Date"

    df = convert_display_units(df)
    return df, errors, sources


def add_derived_series(df: pd.DataFrame, smoothing_days: int, change_window: int) -> pd.DataFrame:
    out = df.copy()

    required = ["WALCL", "RRP", "TGA"]
    present_required = [c for c in required if c in out.columns]

    if len(present_required) < 3:
        raise ValueError(f"Missing required liquidity components. Present: {present_required}")

    out = out.dropna(subset=required).copy()
    out["NetLiq"] = out["WALCL"] - out["RRP"] - out["TGA"]

    for col in ["WALCL", "RRP", "TGA", "NetLiq", "EFFR", "NFCI"]:
        if col in out.columns:
            out[f"{col}_s"] = (
                out[col].rolling(smoothing_days, min_periods=1).mean()
                if smoothing_days > 1
                else out[col]
            )
            out[f"{col}_{change_window}d_chg"] = out[col].diff(change_window)

    out["WALCL_idx"] = rebase(out["WALCL"])
    out["RRP_idx"] = rebase(out["RRP"], min_base=RRP_BASE_FLOOR_B)
    out["TGA_idx"] = rebase(out["TGA"])

    return out


def classify_regime(netliq_roc: float, effr_roc: float, nfci_level: float, nfci_roc: float) -> str:
    liquidity = "improving" if pd.notna(netliq_roc) and netliq_roc > 0 else "deteriorating"
    policy = "easing" if pd.notna(effr_roc) and effr_roc < 0 else "restrictive/stable"
    conditions = "easy" if pd.notna(nfci_level) and nfci_level < 0 else "tight"
    fc_trend = "easing" if pd.notna(nfci_roc) and nfci_roc < 0 else "tightening"
    return f"Liquidity {liquidity} | Policy {policy} | Conditions {conditions} ({fc_trend})"


def series_commentary(latest_row: pd.Series, roc_days: int) -> list[str]:
    comments = []

    netliq_roc = latest_row.get(f"NetLiq_{roc_days}d_chg")
    effr_roc = latest_row.get(f"EFFR_{roc_days}d_chg")
    nfci_roc = latest_row.get(f"NFCI_{roc_days}d_chg")
    nfci = latest_row.get("NFCI")

    if pd.notna(netliq_roc):
        direction = "rose" if netliq_roc > 0 else "fell" if netliq_roc < 0 else "was flat"
        comments.append(f"Net liquidity {direction} over the last {roc_days} trading days by {fmt_delta_b(netliq_roc)}.")

    if pd.notna(effr_roc):
        if abs(effr_roc) < 1e-12:
            comments.append(f"EFFR was unchanged over the last {roc_days} trading days.")
        else:
            direction = "up" if effr_roc > 0 else "down"
            comments.append(f"EFFR moved {direction} by {fmt_delta_pct(effr_roc)} over the last {roc_days} trading days.")

    if pd.notna(nfci):
        level_text = "easier-than-average" if nfci < 0 else "tighter-than-average" if nfci > 0 else "neutral"
        comments.append(f"NFCI currently points to {level_text} financial conditions at {fmt_idx(nfci)}.")

    if pd.notna(nfci_roc):
        direction = "tightened" if nfci_roc > 0 else "eased" if nfci_roc < 0 else "was unchanged"
        comments.append(f"Financial conditions {direction} over the last {roc_days} trading days by {fmt_delta_idx(nfci_roc)}.")

    return comments


# ==============================
# Load data
# ==============================
today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

df = None
load_errors = {}
load_sources = {}

try:
    raw_df, load_errors, load_sources = build_master_dataframe(start_all=start_all, end_date=today)
    df = add_derived_series(raw_df, smoothing_days=smooth, change_window=roc_window)
    df = df[df.index >= start_lb].copy()
except Exception as e:
    # Soft failure mode. Do not hard-stop the app with the old generic fatal error.
    st.warning(
        "Live FRED data is temporarily unavailable and there is no usable local backup cache yet. "
        "The app is loaded, but data cannot be displayed on this first failed run."
    )
    st.caption(f"Diagnostic detail: {e}")

if df is None or df.empty:
    st.info(
        "Once the app gets one successful live pull, it will keep a local backup and future transient FRED outages will fall back automatically."
    )
    st.stop()

# Show source health without scaring the user
live_count = sum(1 for v in load_sources.values() if v == "live")
cache_count = sum(1 for v in load_sources.values() if v == "local_cache")

if cache_count > 0:
    st.caption(f"Source status: {live_count} live FRED series, {cache_count} local backup series.")

if load_errors:
    with st.expander("Series load warnings"):
        for series_name, err in load_errors.items():
            st.warning(f"{series_name}: {err}")


# ==============================
# Top snapshot
# ==============================
latest = df.iloc[-1]
regime_text = classify_regime(
    latest.get(f"NetLiq_{roc_window}d_chg"),
    latest.get(f"EFFR_{roc_window}d_chg"),
    latest.get("NFCI"),
    latest.get(f"NFCI_{roc_window}d_chg"),
)

st.markdown(f"**Regime snapshot:** {regime_text}")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Net Liquidity",
    fmt_b_1(latest.get("NetLiq")),
    delta=fmt_delta_b(latest.get(f"NetLiq_{roc_window}d_chg")),
)
c2.metric(
    "WALCL",
    fmt_b_1(latest.get("WALCL")),
    delta=fmt_delta_b(latest.get(f"WALCL_{roc_window}d_chg")),
)
c3.metric(
    "RRP",
    fmt_b_1(latest.get("RRP")),
    delta=fmt_delta_b(latest.get(f"RRP_{roc_window}d_chg")),
)
c4.metric(
    "TGA",
    fmt_b_1(latest.get("TGA")),
    delta=fmt_delta_b(latest.get(f"TGA_{roc_window}d_chg")),
)
c5.metric(
    "EFFR",
    fmt_pct(latest.get("EFFR")),
    delta=fmt_delta_pct(latest.get(f"EFFR_{roc_window}d_chg")),
)
c6.metric(
    "NFCI",
    fmt_idx(latest.get("NFCI")),
    delta=fmt_delta_idx(latest.get(f"NFCI_{roc_window}d_chg")),
)

with st.expander("Tape read", expanded=True):
    for line in series_commentary(latest, roc_window):
        st.write(f"• {line}")


# ==============================
# Chart
# ==============================
rows = 4 if show_rebased else 3
subplot_titles = [
    "Net Liquidity (Billions) with Short-Term Change",
    "Core Components (Absolute Levels, Billions)",
]
if show_rebased:
    subplot_titles.append("Components Rebased to 100")
subplot_titles.append("EFFR and NFCI")

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=tuple(subplot_titles),
    specs=[[{"secondary_y": False}]] * (rows - 1) + [[{"secondary_y": True}]],
)

# Row 1
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NetLiq_s"],
        name="Net Liquidity",
        line=dict(color="#111111", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Net Liquidity: %{y:,.1f}B<extra></extra>",
    ),
    row=1,
    col=1,
)

if show_raw:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NetLiq"],
            name="Net Liquidity Raw",
            line=dict(color="rgba(17,17,17,0.25)", width=1),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Raw Net Liquidity: %{y:,.1f}B<extra></extra>",
        ),
        row=1,
        col=1,
    )

netliq_change = df[f"NetLiq_{roc_window}d_chg"]
bar_colors = ["rgba(34,139,34,0.35)" if x >= 0 else "rgba(220,20,60,0.35)" for x in netliq_change.fillna(0)]

fig.add_trace(
    go.Bar(
        x=df.index,
        y=netliq_change,
        name=f"{roc_window}d Δ Net Liquidity",
        marker_color=bar_colors,
        opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}<br>Change: %{y:,.1f}B<extra></extra>",
    ),
    row=1,
    col=1,
)

# Row 2
abs_row = 2
for col, name in [("WALCL", "WALCL"), ("RRP", "RRP"), ("TGA", "TGA")]:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{col}_s"],
            name=name,
            line=dict(color=FRED_SERIES[col]["color"], width=2),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: %{{y:,.1f}}B<extra></extra>",
        ),
        row=abs_row,
        col=1,
    )

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=f"{name} Raw",
                line=dict(color="rgba(150,150,150,0.25)", width=1),
                showlegend=False,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name} Raw: %{{y:,.1f}}B<extra></extra>",
            ),
            row=abs_row,
            col=1,
        )

policy_row = 3

# Row 3 optional rebased
if show_rebased:
    rebase_row = 3
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["WALCL_idx"],
            name="WALCL idx",
            line=dict(color=FRED_SERIES["WALCL"]["color"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>WALCL idx: %{y:.1f}<extra></extra>",
        ),
        row=rebase_row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RRP_idx"],
            name="RRP idx",
            line=dict(color=FRED_SERIES["RRP"]["color"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>RRP idx: %{y:.1f}<extra></extra>",
        ),
        row=rebase_row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["TGA_idx"],
            name="TGA idx",
            line=dict(color=FRED_SERIES["TGA"]["color"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>TGA idx: %{y:.1f}<extra></extra>",
        ),
        row=rebase_row,
        col=1,
    )
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="gray", row=rebase_row, col=1)
    policy_row = 4

# Final row
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EFFR_s"],
        name="EFFR",
        line=dict(color=FRED_SERIES["EFFR"]["color"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>EFFR: %{y:.2f}%<extra></extra>",
    ),
    row=policy_row,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NFCI_s"],
        name="NFCI",
        line=dict(color=FRED_SERIES["NFCI"]["color"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
    ),
    row=policy_row,
    col=1,
    secondary_y=True,
)

if show_raw:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR"],
            name="EFFR Raw",
            line=dict(color="rgba(255,127,14,0.25)", width=1),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>EFFR Raw: %{y:.2f}%<extra></extra>",
        ),
        row=policy_row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI"],
            name="NFCI Raw",
            line=dict(color="rgba(44,160,44,0.25)", width=1),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>NFCI Raw: %{y:.3f}<extra></extra>",
        ),
        row=policy_row,
        col=1,
        secondary_y=True,
    )

fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=policy_row, col=1, secondary_y=True)

fig.update_layout(
    template="plotly_white",
    height=1200 if show_rebased else 980,
    margin=dict(l=60, r=60, t=70, b=50),
    legend=dict(
        orientation="h",
        x=0,
        y=1.03,
        xanchor="left",
        yanchor="bottom",
    ),
    barmode="overlay",
    hovermode="x unified",
)

for r in range(1, rows + 1):
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        tickformat="%b-%y",
        row=r,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        row=r,
        col=1,
    )

fig.update_yaxes(title_text="Billions", row=1, col=1)
fig.update_yaxes(title_text="Billions", row=2, col=1)
if show_rebased:
    fig.update_yaxes(title_text="Index", row=3, col=1)
fig.update_yaxes(title_text="EFFR (%)", row=policy_row, col=1, secondary_y=False)
fig.update_yaxes(title_text="NFCI", row=policy_row, col=1, secondary_y=True)
fig.update_xaxes(title_text="Date", row=rows, col=1)

st.plotly_chart(fig, use_container_width=True)


# ==============================
# Export
# ==============================
with st.expander("Download Data"):
    export_cols = [
        "WALCL",
        "RRP",
        "TGA",
        "NetLiq",
        "EFFR",
        "NFCI",
        f"NetLiq_{roc_window}d_chg",
        f"EFFR_{roc_window}d_chg",
        f"NFCI_{roc_window}d_chg",
        "WALCL_idx",
        "RRP_idx",
        "TGA_idx",
    ]
    export_cols = [c for c in export_cols if c in df.columns]

    out = df[export_cols].copy()
    out.columns = [
        "WALCL_B" if c == "WALCL" else
        "RRP_B" if c == "RRP" else
        "TGA_B" if c == "TGA" else
        "NetLiq_B" if c == "NetLiq" else
        f"NetLiq_{roc_window}D_Change_B" if c == f"NetLiq_{roc_window}d_chg" else
        f"EFFR_{roc_window}D_Change_PctPts" if c == f"EFFR_{roc_window}d_chg" else
        f"NFCI_{roc_window}D_Change" if c == f"NFCI_{roc_window}d_chg" else
        c
        for c in out.columns
    ]
    out.index.name = "Date"

    st.dataframe(out.tail(50), use_container_width=True)

    csv_bytes = out.to_csv().encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="liquidity_fed_policy_tracker.csv",
        mime="text/csv",
    )


# ==============================
# Methodology
# ==============================
with st.expander("Methodology"):
    st.markdown(
        f"""
        **Net Liquidity formula**

        Net Liquidity = WALCL − RRP − TGA

        **Unit handling**

        • WALCL and TGA are converted from millions to billions  
        • RRP is treated in billions  
        • EFFR remains in %  
        • NFCI remains as an index level  

        **Alignment**

        Series are fetched individually from FRED public CSV endpoints, aligned onto a daily calendar, and forward-filled to harmonize reporting frequencies.

        **Resilience layer**

        • Multiple FRED URL patterns are tried for each series  
        • HTTP requests use retries with backoff  
        • Every successful pull is stored in a local last-good cache  
        • If live FRED later fails, the app falls back to the local cache automatically

        **Smoothing**

        The dashboard applies a {smooth}-day rolling average to smooth displayed lines.

        **Rebasing**

        Rebased charts use the median of the first {REBASE_BASE_WINDOW} non-null observations in the selected window.  
        RRP uses a minimum base floor of {RRP_BASE_FLOOR_B:.1f}B to avoid distorted index behavior when the starting level is extremely small.
        """
    )

st.caption("© 2026 AD Fund Management LP")
