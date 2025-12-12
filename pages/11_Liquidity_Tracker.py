import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"

FRED = {
    "WALCL": "WALCL",          # Fed total assets (millions)
    "RRP":   "RRPONTSYD",      # ON RRP (billions)
    "TGA":   "WDTGAL",         # Treasury General Account (millions)
    "EFFR":  "EFFR",           # Effective Fed Funds Rate (%)
    "NFCI":  "NFCI",           # Chicago Fed National Financial Conditions Index
}

LOOKBACK_OPTIONS = {
    "1y": 1,
    "2y": 2,
    "3y": 3,
    "5y": 5,
    "10y": 10,
    "25y": 25,
}

DEFAULT_SMOOTH_DAYS = 5

# Robust rebase parameters
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Goal: track Net Liquidity and policy stance.

        Net Liquidity = WALCL − RRP − TGA

        Series
        • WALCL: Fed balance sheet
        • RRPONTSYD: reverse repo
        • TGA: Treasury General Account
        • EFFR: fed funds rate
        • NFCI: Chicago Fed National Financial Conditions Index
          Values > 0 mean tighter-than-average financial conditions.

        Panels
        1) Net Liquidity
        2) Components rebased
        3) EFFR
        4) NFCI (financial conditions)
        """
    )
    st.markdown("---")
    st.header("Settings")

    lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=2)
    years = LOOKBACK_OPTIONS[lookback_label]

    smooth = st.number_input(
        "Smoothing window (days)", min_value=1, max_value=30, value=DEFAULT_SMOOTH_DAYS, step=1
    )

    show_download = st.checkbox("Enable download section", value=True)
    st.caption("Data source: FRED via pandas-datareader")

# ---------------- Data helpers ----------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_fred(series_map: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch all FRED series in one request for robustness and consistency.
    Returns a df with columns named by keys of series_map.
    """
    tickers = list(series_map.values())
    try:
        raw = pdr.DataReader(tickers, "fred", start, end)
    except Exception:
        return pd.DataFrame()

    # Rename columns back to friendly keys
    inv = {v: k for k, v in series_map.items()}
    raw = raw.rename(columns=inv)

    # Standardize index, forward fill gaps
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().ffill()

    return raw

def safe_delta(series: pd.Series, periods: int):
    if series is None or series.empty:
        return None
    if len(series) <= periods:
        return None
    a = series.iloc[-1]
    b = series.iloc[-(periods + 1)]
    if pd.isna(a) or pd.isna(b):
        return None
    return a - b

def fmt_b(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.0f} B"

def fmt_b_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+,.0f} B"

def fmt_pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.2f}%"

def fmt_pct_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+.2f}%"

def fmt_nfci(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.3f}"

def fmt_nfci_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+.3f}"

def rebase(series: pd.Series, base_window: int = REBASE_BASE_WINDOW, min_base=None) -> pd.Series:
    s = series.copy()
    if s.isna().all():
        return pd.Series(index=s.index, data=100.0)

    head = s.dropna().iloc[: max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]

    if min_base is not None:
        base = max(float(base), float(min_base))

    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, data=100.0)

    return (s / base) * 100.0

# ---------------- Build date range ----------------
today = pd.Timestamp.today().normalize()
start_lb = today - pd.DateOffset(years=years)

# Pull a bit extra so rolling windows and rebases have clean heads
start_fetch = start_lb - pd.DateOffset(months=6)

df = fetch_fred(FRED, start_fetch, today)

if df.empty:
    st.error("FRED fetch failed (empty response).")
    st.stop()

missing = [k for k in FRED.keys() if k not in df.columns]
if missing:
    st.warning(f"Missing series from FRED: {', '.join(missing)}")

# Enforce required columns for Net Liquidity
required = ["WALCL", "RRP", "TGA"]
df = df.dropna(subset=[c for c in required if c in df.columns])
df = df[df.index >= start_lb]

if df.empty:
    st.error("No data for selected lookback.")
    st.stop()

# ---------------- Transform units ----------------
# WALCL and TGA are in millions; convert to billions.
df["WALCL_b"] = df["WALCL"] / 1000.0 if "WALCL" in df.columns else pd.NA
df["TGA_b"] = df["TGA"] / 1000.0 if "TGA" in df.columns else pd.NA

# RRPONTSYD is already billions.
df["RRP_b"] = df["RRP"] if "RRP" in df.columns else pd.NA

df["NetLiq"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# ---------------- Smoothing ----------------
smooth_cols = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR", "NFCI"]
for c in smooth_cols:
    if c not in df.columns:
        continue
    if smooth > 1:
        df[f"{c}_s"] = df[c].rolling(smooth, min_periods=1).mean()
    else:
        df[f"{c}_s"] = df[c]

# ---------------- Rebase components ----------------
reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b_s"])
reb["RRP_idx"] = rebase(df["RRP_b_s"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"] = rebase(df["TGA_b_s"])

# ---------------- Headline metrics ----------------
asof = df.index.max()
st.caption(f"As of {asof:%Y-%m-%d}")

netliq = df["NetLiq"].iloc[-1]
walcl = df["WALCL_b"].iloc[-1]
rrp = df["RRP_b"].iloc[-1]
tga = df["TGA_b"].iloc[-1]
effr = df["EFFR"].iloc[-1] if "EFFR" in df.columns else pd.NA
nfci = df["NFCI"].iloc[-1] if "NFCI" in df.columns else pd.NA

# Deltas: use ~1w (5 obs) and ~1m (21 obs) on the available index cadence
d_netliq_1w = safe_delta(df["NetLiq"], 5)
d_effr_1m = safe_delta(df["EFFR"], 21) if "EFFR" in df.columns else None
d_nfci_1m = safe_delta(df["NFCI"], 21) if "NFCI" in df.columns else None

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(netliq), fmt_b_delta(d_netliq_1w))
m2.metric("WALCL", fmt_b(walcl))
m3.metric("RRP", fmt_b(rrp))
m4.metric("TGA", fmt_b(tga))
m5.metric("EFFR", fmt_pct(effr), fmt_pct_delta(d_effr_1m))
m6.metric("NFCI", fmt_nfci(nfci), fmt_nfci_delta(d_nfci_1m), help=">0 = tighter conditions")

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
        hovertemplate="Date=%{x|%Y-%m-%d}<br>NetLiq=%{y:,.0f} B<extra></extra>",
        line=dict(width=2),
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
        hovertemplate="Date=%{x|%Y-%m-%d}<br>WALCL idx=%{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["RRP_idx"],
        name="RRP idx",
        hovertemplate="Date=%{x|%Y-%m-%d}<br>RRP idx=%{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["TGA_idx"],
        name="TGA idx",
        hovertemplate="Date=%{x|%Y-%m-%d}<br>TGA idx=%{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

# Row 3
if "EFFR_s" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR_s"],
            name="EFFR",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>EFFR=%{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

# Row 4
if "NFCI_s" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI_s"],
            name="NFCI",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>NFCI=%{y:.3f}<extra></extra>",
            line=dict(width=2),
        ),
        row=4,
        col=1,
    )

# Axis titles
fig.update_yaxes(title_text="Billions", row=1, col=1)
fig.update_yaxes(title_text="Index", row=2, col=1)
fig.update_yaxes(title_text="%", row=3, col=1)
fig.update_yaxes(title_text="Level", row=4, col=1)

fig.update_layout(
    template="plotly_white",
    height=1080,
    legend=dict(orientation="h", x=0, y=1.08),
    margin=dict(l=60, r=40, t=60, b=60),
)

fig.update_xaxes(tickformat="%b-%y", row=4, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
if show_download:
    with st.expander("Download Data"):
        out = pd.DataFrame(index=df.index)
        out["WALCL_B"] = df["WALCL_b"]
        out["RRP_B"] = df["RRP_b"]
        out["TGA_B"] = df["TGA_b"]
        if "EFFR" in df.columns:
            out["EFFR_%"] = df["EFFR"]
        if "NFCI" in df.columns:
            out["NFCI"] = df["NFCI"]
        out["NetLiq_B"] = df["NetLiq"]
        out.index.name = "Date"

        st.download_button(
            "Download CSV",
            out.to_csv(),
            file_name="liquidity_tracker.csv",
            mime="text/csv",
        )

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
        Net Liquidity = WALCL − RRP − TGA

        NFCI interpretation
        • Combined measure of credit, leverage, and funding markets
        • Values above zero imply tighter financial conditions relative to history
        • Useful complement to net liquidity and EFFR

        Smoothing uses {smooth}-day averages.
        Rebase uses median of first {REBASE_BASE_WINDOW} observations.
        """
    )

st.caption("© 2025 AD Fund Management LP")
