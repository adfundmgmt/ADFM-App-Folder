import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"

SERIES = {
    "WALCL": "Fed Balance Sheet",
    "RRPONTSYD": "ON RRP",
    "WDTGAL": "Treasury General Account",
    "EFFR": "Effective Fed Funds Rate",
    "NFCI": "Chicago Fed NFCI",
}

DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_SMOOTH = 5

st.set_page_config(page_title=TITLE, layout="wide")

# ---------------- Professional UI ----------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: 0.1px;
    }

    .adfm-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
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

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    .sidebar-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Public macro dashboard tracking balance sheet liquidity, funding drains, policy rate, and financial conditions.</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### About This Tool")
    st.markdown(
        """
This dashboard tracks a simple liquidity framework centered on Fed balance sheet expansion versus reserve drains.

**What it shows**
- Net liquidity using WALCL minus ON RRP minus TGA
- A rebased view of the three core balance sheet components
- Effective Fed Funds Rate as the policy anchor
- Chicago Fed NFCI as a clean read on broader financial conditions

**How to use it**
- Rising net liquidity usually signals easier reserve conditions
- Falling net liquidity often reflects tighter background liquidity
- EFFR helps frame the policy stance
- NFCI above zero suggests tighter-than-average financial conditions
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])

    smooth_days = st.number_input(
        "Smoothing window",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH,
        step=1,
    )

    smooth_policy = st.checkbox("Smooth EFFR and NFCI", value=False)
    st.caption("Source: Public FRED CSV endpoints")
    st.caption("No API key required")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Loaders ----------------
def fetch_fred_csv(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start.strftime('%Y-%m-%d')}"
        f"&coed={end.strftime('%Y-%m-%d')}"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable data returned for {series_id}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    return pd.Series(df[value_col].values, index=df[date_col], name=series_id)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_all_series(start: pd.Timestamp, end: pd.Timestamp):
    out = {}
    errors = {}

    for series_id in SERIES:
        try:
            out[series_id] = fetch_fred_csv(series_id, start, end)
        except Exception as e:
            errors[series_id] = str(e)
            out[series_id] = pd.Series(dtype="float64", name=series_id)

    return out, errors

def smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()

def rebase(s: pd.Series, floor: float | None = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(index=s.index, dtype="float64", name=s.name)

    base = valid.iloc[:10].median()
    if floor is not None:
        base = max(base, floor)

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, dtype="float64", name=s.name)

    return (s / base) * 100.0

def fmt_b(x):
    return "N/A" if pd.isna(x) else f"{x:,.0f} B"

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.3f}"

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

# ---------------- Data ----------------
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years)

with st.spinner("Loading macro series..."):
    raw, errors = load_all_series(start, today)

required = ["WALCL", "RRPONTSYD", "WDTGAL"]
missing = [x for x in required if raw[x].empty]

if missing:
    st.error(f"Required series failed to load: {', '.join(missing)}")
    if errors:
        with st.expander("Error details"):
            for k, v in errors.items():
                st.write(f"{k}: {v}")
    st.stop()

df = pd.concat(
    [
        raw["WALCL"].rename("WALCL"),
        raw["RRPONTSYD"].rename("RRP"),
        raw["WDTGAL"].rename("TGA"),
        raw["EFFR"].rename("EFFR"),
        raw["NFCI"].rename("NFCI"),
    ],
    axis=1,
).sort_index().ffill()

df = df[df.index >= start].copy()
df = df.dropna(subset=["WALCL", "RRP", "TGA"])

if df.empty:
    st.error("Dataframe is empty after cleaning.")
    st.stop()

# ---------------- Units ----------------
# WALCL and WDTGAL are in millions. RRPONTSYD is already billions.
df["WALCL_b"] = pd.to_numeric(df["WALCL"], errors="coerce") / 1000.0
df["RRP_b"] = pd.to_numeric(df["RRP"], errors="coerce")
df["TGA_b"] = pd.to_numeric(df["TGA"], errors="coerce") / 1000.0
df["EFFR"] = pd.to_numeric(df["EFFR"], errors="coerce")
df["NFCI"] = pd.to_numeric(df["NFCI"], errors="coerce")

df["NetLiq"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# ---------------- Smoothed ----------------
df["NetLiq_s"] = smooth(df["NetLiq"], smooth_days)
df["WALCL_s"] = smooth(df["WALCL_b"], smooth_days)
df["RRP_s"] = smooth(df["RRP_b"], smooth_days)
df["TGA_s"] = smooth(df["TGA_b"], smooth_days)

if smooth_policy:
    df["EFFR_s"] = smooth(df["EFFR"], smooth_days)
    df["NFCI_s"] = smooth(df["NFCI"], smooth_days)
else:
    df["EFFR_s"] = df["EFFR"]
    df["NFCI_s"] = df["NFCI"]

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"] = rebase(df["RRP_b"], floor=5.0)
reb["TGA_idx"] = rebase(df["TGA_b"])

latest = df.iloc[-1]

# ---------------- Top Summary ----------------
st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Latest available readings across liquidity, policy, and financial conditions.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Net Liquidity", fmt_b(latest["NetLiq"]), "WALCL - RRP - TGA")
with c2:
    metric_card("WALCL", fmt_b(latest["WALCL_b"]), "Fed total assets")
with c3:
    metric_card("RRP", fmt_b(latest["RRP_b"]), "ON reverse repo")
with c4:
    metric_card("TGA", fmt_b(latest["TGA_b"]), "Treasury cash balance")
with c5:
    metric_card("EFFR", fmt_pct(latest["EFFR"]), "Policy rate")
with c6:
    metric_card("NFCI", fmt_num(latest["NFCI"]), "Above 0 = tighter")

# ---------------- Quick Read ----------------
st.markdown("<div class='section-title'>Quick Read</div>", unsafe_allow_html=True)

quick_left, quick_right = st.columns([1.35, 1])

with quick_left:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Net liquidity framework</b><br><br>
        The dashboard computes net liquidity as <b>Fed balance sheet minus ON RRP minus TGA</b>. A higher reading generally points to easier reserve conditions in the system, while a lower reading usually reflects a tighter liquidity backdrop.
        <br><br>
        Over the selected <b>{lookback}</b> window, the chart lets you separate three different forces: balance sheet expansion or contraction, reserve absorption through reverse repo, and Treasury cash accumulation.
        </div>
        """,
        unsafe_allow_html=True,
    )

with quick_right:
    policy_state = "tighter-than-average" if pd.notna(latest["NFCI"]) and latest["NFCI"] > 0 else "easier-than-average"
    st.markdown(
        f"""
        <div class="info-box">
        <b>Current read</b><br><br>
        EFFR is currently at <b>{fmt_pct(latest["EFFR"])}</b> and NFCI is <b>{fmt_num(latest["NFCI"])}</b>, which implies financial conditions are running <b>{policy_state}</b> on the Chicago Fed measure.
        <br><br>
        Smooth window applied: <b>{smooth_days} day(s)</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

if errors:
    noncritical = {k: v for k, v in errors.items() if k not in required}
    if noncritical:
        st.warning("Some non-critical series failed to load.")
        with st.expander("Series error details"):
            for k, v in noncritical.items():
                st.write(f"{k}: {v}")

# ---------------- Main Chart ----------------
st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Liquidity path, underlying components, policy rate, and financial conditions in one view.</div>",
    unsafe_allow_html=True,
)

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Core Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions (NFCI)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NetLiq_s"],
        name="Net Liquidity",
        mode="lines",
        line=dict(width=2.8, color="#111827"),
        hovertemplate="%{x|%Y-%m-%d}<br>Net Liquidity: %{y:,.0f} B<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["WALCL_idx"],
        name="WALCL idx",
        mode="lines",
        line=dict(width=2.2, color="#2563eb"),
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
        mode="lines",
        line=dict(width=2.2, color="#dc2626"),
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
        mode="lines",
        line=dict(width=2.2, color="#059669"),
        hovertemplate="%{x|%Y-%m-%d}<br>TGA idx: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

if df["EFFR_s"].notna().any():
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR_s"],
            name="EFFR",
            mode="lines",
            line=dict(width=2.2, color="#f59e0b"),
            hovertemplate="%{x|%Y-%m-%d}<br>EFFR: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

if df["NFCI_s"].notna().any():
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af", row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI_s"],
            name="NFCI",
            mode="lines",
            line=dict(width=2.2, color="#7c3aed"),
            hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

fig.update_layout(
    template="plotly_white",
    height=1125,
    margin=dict(l=55, r=25, t=55, b=35),
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

fig.update_annotations(font=dict(size=13, color="#111827"))
fig.update_yaxes(title_text="Billions", showgrid=True, gridcolor="#eceff3", zeroline=False, row=1, col=1)
fig.update_yaxes(title_text="Index", showgrid=True, gridcolor="#eceff3", zeroline=False, row=2, col=1)
fig.update_yaxes(title_text="%", showgrid=True, gridcolor="#eceff3", zeroline=False, row=3, col=1)
fig.update_yaxes(title_text="Index", showgrid=True, gridcolor="#eceff3", zeroline=False, row=4, col=1)
fig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Lower Panels ----------------
left, right = st.columns([1.2, 0.8])

with left:
    st.markdown("<div class='section-title'>Raw Data</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Latest 50 observations for the cleaned working dataset.</div>",
        unsafe_allow_html=True,
    )
    show = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].copy()
    show.columns = ["WALCL (B)", "RRP (B)", "TGA (B)", "EFFR (%)", "NFCI", "Net Liquidity (B)"]
    st.dataframe(show.tail(50), use_container_width=True, height=420)

with right:
    st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the cleaned working dataset used in the dashboard.</div>",
        unsafe_allow_html=True,
    )
    csv = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].to_csv().encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="liquidity_tracker.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
        <b>Formula</b><br>
        Net Liquidity = WALCL - RRP - TGA
        <br><br>
        <b>Unit treatment</b><br>
        WALCL and WDTGAL are converted from millions to billions. RRPONTSYD is already in billions.
        <br><br>
        <b>Smoothing</b><br>
        Liquidity series use a <b>{smooth_days}-day</b> rolling average. Policy series smoothing is <b>{"enabled" if smooth_policy else "disabled"}</b>.
        <br><br>
        <b>Rebasing</b><br>
        Component indexes are rebased to 100 using the median of the first 10 valid observations in the displayed window. RRP rebasing uses a 5.0B floor to reduce unstable index jumps when the base is very small.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("© 2026 AD Fund Management LP")
