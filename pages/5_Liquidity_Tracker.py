import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"

FRED = {
    "fed_bs": "WALCL",        # Fed total assets (millions)
    "rrp": "RRPONTSYD",       # ON RRP (billions)
    "tga": "WDTGAL",          # Treasury General Account (millions)
    "effr": "EFFR",           # Effective Fed Funds Rate (%)
    "nfci": "NFCI",           # Chicago Fed National Financial Conditions Index
}

DEFAULT_SMOOTH_DAYS = 5
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0

FRED_TIMEOUT = 20
FRED_RETRIES = 3
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Net-liquidity monitor tying Fed balance-sheet flows to macro risk backdrop.

        What it covers
        • Core signals and summary outputs for this dashboard
        • Key context needed to interpret current regime or setup
        • Practical view designed for quick internal decision support

        Data source
        • Public market and macro data feeds used throughout the app
        """
    )
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])

    smooth_liquidity = st.number_input(
        "Smoothing window for liquidity series (days)",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH_DAYS,
        step=1,
    )

    smooth_policy = st.checkbox("Also smooth EFFR and NFCI", value=False)

    st.caption("Source: FRED")
    st.caption("Optional API key via st.secrets['FRED_API_KEY'] or env var FRED_API_KEY")

# ---------------- Helpers ----------------
def get_fred_api_key() -> Optional[str]:
    try:
        if "FRED_API_KEY" in st.secrets:
            return st.secrets["FRED_API_KEY"]
    except Exception:
        pass
    return os.getenv("FRED_API_KEY")

@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ADFM-Liquidity-Tracker/1.0",
            "Accept": "application/json,text/csv,application/octet-stream",
        }
    )
    return session

def last_good_path(series_id: str) -> Path:
    return CACHE_DIR / f"{series_id}.parquet"

def save_last_good(series_id: str, s: pd.Series) -> None:
    try:
        df = s.rename(series_id).to_frame()
        df.to_parquet(last_good_path(series_id))
    except Exception:
        pass

def load_last_good(series_id: str) -> pd.Series:
    path = last_good_path(series_id)
    if not path.exists():
        return pd.Series(dtype="float64", name=series_id)
    try:
        df = pd.read_parquet(path)
        if series_id in df.columns:
            s = df[series_id]
        else:
            s = df.iloc[:, 0]
            s.name = series_id
        s.index = pd.to_datetime(s.index)
        return pd.to_numeric(s, errors="coerce").sort_index()
    except Exception:
        return pd.Series(dtype="float64", name=series_id)

def parse_fred_json_observations(payload: dict, series_id: str) -> pd.Series:
    obs = payload.get("observations", [])
    if not obs:
        return pd.Series(dtype="float64", name=series_id)

    rows = []
    for item in obs:
        date_val = item.get("date")
        value = item.get("value")
        if date_val is None:
            continue
        rows.append((date_val, None if value in (None, ".", "") else value))

    if not rows:
        return pd.Series(dtype="float64", name=series_id)

    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    s = pd.Series(df["value"].values, index=df["date"], name=series_id)
    return s

def parse_fred_csv_bytes(content: bytes, series_id: str) -> pd.Series:
    # FRED CSV can come back as either a plain CSV or a zipped CSV payload.
    try:
        df = pd.read_csv(BytesIO(content), compression="infer")
    except Exception:
        df = pd.read_csv(BytesIO(content))

    # Normalize common FRED column conventions
    cols = {c.lower(): c for c in df.columns}
    date_col = None
    value_col = None

    for candidate in ["date", "observation_date"]:
        if candidate in cols:
            date_col = cols[candidate]
            break

    if series_id in df.columns:
        value_col = series_id
    else:
        # Fall back to first non-date column
        non_date_cols = [c for c in df.columns if c != date_col]
        if non_date_cols:
            value_col = non_date_cols[0]

    if date_col is None or value_col is None:
        return pd.Series(dtype="float64", name=series_id)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    s = pd.Series(df[value_col].values, index=df[date_col], name=series_id)
    return s

def fetch_fred_json(
    session: requests.Session,
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: Optional[str],
) -> pd.Series:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": start.strftime("%Y-%m-%d"),
        "observation_end": end.strftime("%Y-%m-%d"),
    }
    if api_key:
        params["api_key"] = api_key

    url = "https://api.stlouisfed.org/fred/series/observations"
    r = session.get(url, params=params, timeout=FRED_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    return parse_fred_json_observations(payload, series_id)

def fetch_fred_csv(
    session: requests.Session,
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    # Public CSV endpoint fallback, no API key required.
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start.strftime('%Y-%m-%d')}&coed={end.strftime('%Y-%m-%d')}"
    r = session.get(url, timeout=FRED_TIMEOUT)
    r.raise_for_status()
    return parse_fred_csv_bytes(r.content, series_id)

@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    session = get_http_session()
    api_key = get_fred_api_key()
    last_error = None

    for attempt in range(1, FRED_RETRIES + 1):
        try:
            s = fetch_fred_json(session, series_id, start, end, api_key=api_key)
            if s.empty:
                raise ValueError(f"{series_id}: empty JSON response")
            s = s.sort_index().ffill()
            save_last_good(series_id, s)
            return s
        except Exception as e:
            last_error = e
            time.sleep(0.6 * attempt)

    for attempt in range(1, FRED_RETRIES + 1):
        try:
            s = fetch_fred_csv(session, series_id, start, end)
            if s.empty:
                raise ValueError(f"{series_id}: empty CSV response")
            s = s.sort_index().ffill()
            save_last_good(series_id, s)
            return s
        except Exception as e:
            last_error = e
            time.sleep(0.6 * attempt)

    cached = load_last_good(series_id)
    if not cached.empty:
        return cached[cached.index <= end].ffill()

    raise RuntimeError(f"Failed loading {series_id}: {last_error}")

def clip_to_lookback(s: pd.Series, start_lb: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if s.empty:
        return s
    s = s[(s.index >= start_lb) & (s.index <= end)]
    return s.sort_index().ffill()

def rebase(series: pd.Series, base_window: int = REBASE_BASE_WINDOW, min_base: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    if s.dropna().empty:
        return pd.Series(index=s.index, data=100.0, name=series.name)

    head = s.dropna().iloc[: max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]

    if min_base is not None:
        base = max(float(base), float(min_base))

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, data=100.0, name=series.name)

    out = (s / base) * 100.0
    out.name = series.name
    return out

def apply_smoothing(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1).mean()

def fmt_b(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.0f} B"

def fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def fmt_nfci(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.3f}"

# ---------------- Load data ----------------
today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

series_map = {}
errors: List[str] = []

for key, series_id in FRED.items():
    try:
        series_map[key] = fred_series(series_id, start_all, today)
    except Exception as e:
        series_map[key] = pd.Series(dtype="float64", name=series_id)
        errors.append(str(e))

fed_bs = clip_to_lookback(series_map["fed_bs"], start_lb, today)
rrp = clip_to_lookback(series_map["rrp"], start_lb, today)
tga = clip_to_lookback(series_map["tga"], start_lb, today)
effr = clip_to_lookback(series_map["effr"], start_lb, today)
nfci = clip_to_lookback(series_map["nfci"], start_lb, today)

required = {"WALCL": fed_bs, "RRP": rrp, "TGA": tga}
missing_required = [name for name, s in required.items() if s.empty]

if missing_required:
    st.error(f"Required series failed to load: {', '.join(missing_required)}")
    if errors:
        with st.expander("Details"):
            for e in errors:
                st.write(e)
    st.stop()

df = pd.concat(
    [
        fed_bs.rename("WALCL"),
        rrp.rename("RRP"),
        tga.rename("TGA"),
        effr.rename("EFFR"),
        nfci.rename("NFCI"),
    ],
    axis=1,
).sort_index().ffill()

df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]

if df.empty:
    st.error("No data available for the selected lookback.")
    st.stop()

# ---------------- Unit normalization ----------------
# WALCL and WDTGAL are in millions. RRPONTSYD is already billions.
df["WALCL_b"] = pd.to_numeric(df["WALCL"], errors="coerce") / 1000.0
df["RRP_b"] = pd.to_numeric(df["RRP"], errors="coerce")
df["TGA_b"] = pd.to_numeric(df["TGA"], errors="coerce") / 1000.0
df["EFFR"] = pd.to_numeric(df["EFFR"], errors="coerce")
df["NFCI"] = pd.to_numeric(df["NFCI"], errors="coerce")

df["NetLiq"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# ---------------- Smoothing ----------------
for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq"]:
    df[f"{col}_s"] = apply_smoothing(df[col], smooth_liquidity)

if smooth_policy:
    df["EFFR_s"] = apply_smoothing(df["EFFR"], smooth_liquidity)
    df["NFCI_s"] = apply_smoothing(df["NFCI"], smooth_liquidity)
else:
    df["EFFR_s"] = df["EFFR"]
    df["NFCI_s"] = df["NFCI"]

# ---------------- Rebased panel ----------------
reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"], base_window=REBASE_BASE_WINDOW)
reb["RRP_idx"] = rebase(df["RRP_b"], base_window=REBASE_BASE_WINDOW, min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"] = rebase(df["TGA_b"], base_window=REBASE_BASE_WINDOW)

# ---------------- Metrics ----------------
latest = {
    "netliq": df["NetLiq"].dropna().iloc[-1] if not df["NetLiq"].dropna().empty else float("nan"),
    "walcl": df["WALCL_b"].dropna().iloc[-1] if not df["WALCL_b"].dropna().empty else float("nan"),
    "rrp": df["RRP_b"].dropna().iloc[-1] if not df["RRP_b"].dropna().empty else float("nan"),
    "tga": df["TGA_b"].dropna().iloc[-1] if not df["TGA_b"].dropna().empty else float("nan"),
    "effr": df["EFFR"].dropna().iloc[-1] if not df["EFFR"].dropna().empty else float("nan"),
    "nfci": df["NFCI"].dropna().iloc[-1] if not df["NFCI"].dropna().empty else float("nan"),
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]))
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))
m6.metric("NFCI", fmt_nfci(latest["nfci"]), help="Above 0 = tighter than historical average")

if errors:
    st.warning("Some non-critical fetches failed. The app is using available data and local cache where possible.")
    with st.expander("Fetch details"):
        for e in errors:
            st.write(e)

# ---------------- Charts ----------------
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions (NFCI)",
    ),
)

# Row 1
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NetLiq_s"],
        name="Net Liquidity",
        line=dict(color="#111111", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Net Liquidity: %{y:,.0f} B<extra></extra>",
    ),
    row=1,
    col=1,
)

# Row 2
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["WALCL_idx"],
        name="WALCL idx",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>WALCL idx: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["RRP_idx"],
        name="RRP idx",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>RRP idx: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["TGA_idx"],
        name="TGA idx",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>TGA idx: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

# Row 3
if not df["EFFR_s"].dropna().empty:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR_s"],
            name="EFFR",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>EFFR: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

# Row 4
if not df["NFCI_s"].dropna().empty:
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#888888", row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI_s"],
            name="NFCI",
            line=dict(color="#1f1f1f", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

fig.update_layout(
    template="plotly_white",
    height=1080,
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(
        orientation="h",
        x=0.0,
        y=1.05,
        xanchor="left",
        yanchor="bottom",
    ),
)

fig.update_yaxes(title_text="Billions", row=1, col=1)
fig.update_yaxes(title_text="Index", row=2, col=1)
fig.update_yaxes(title_text="%", row=3, col=1)
fig.update_yaxes(title_text="Index", row=4, col=1)
fig.update_xaxes(tickformat="%b-%y", title_text="Date", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    out = pd.DataFrame(index=df.index)
    out["WALCL_B"] = df["WALCL_b"]
    out["RRP_B"] = df["RRP_b"]
    out["TGA_B"] = df["TGA_b"]
    out["EFFR_%"] = df["EFFR"]
    out["NFCI"] = df["NFCI"]
    out["NetLiq_B"] = df["NetLiq"]
    out.index.name = "Date"

    st.download_button(
        "Download CSV",
        out.to_csv().encode("utf-8"),
        file_name="liquidity_tracker.csv",
        mime="text/csv",
    )

# ---------------- Methodology ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
        Net Liquidity = WALCL − RRP − TGA

        Unit treatment
        • WALCL and WDTGAL are converted from millions to billions
        • RRPONTSYD is used in billions as published

        Interpretation
        • Higher Net Liquidity generally reflects easier reserve conditions
        • EFFR captures the policy rate
        • NFCI above zero implies tighter-than-average financial conditions

        Smoothing
        • Liquidity series smoothing window: {smooth_liquidity} day(s)
        • Policy series smoothing: {"enabled" if smooth_policy else "disabled"}

        Rebase
        • Rebased components use the median of the first {REBASE_BASE_WINDOW} non-null observations inside the displayed lookback window
        • RRP rebasing uses a floor of {RRP_BASE_FLOOR_B:.1f} B to avoid pathological index jumps when the base is near zero
        """
    )

st.caption("© 2026 AD Fund Management LP")
