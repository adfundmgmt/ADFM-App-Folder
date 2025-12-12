import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ============================== Config ==============================
TITLE = "Liquidity & Fed Policy Tracker"

FRED = {
    "fed_bs": "WALCL",        # Fed total assets (millions)
    "rrp": "RRPONTSYD",       # ON RRP (billions)
    "tga": "WDTGAL",          # Treasury General Account (millions)
    "effr": "EFFR",           # Effective Fed Funds Rate (%)
    "nfci": "NFCI",           # Chicago Fed National Financial Conditions Index
}

DEFAULT_SMOOTH_DAYS = 5

# Robust rebase parameters
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0

LOOKBACK_OPTIONS = {
    "1Y": 1,
    "2Y": 2,
    "3Y": 3,
    "5Y": 5,
    "10Y": 10,
}

# Visual palette (kept clean and institutional)
COLORS = {
    "ink": "#0B1220",
    "muted": "#667085",
    "border": "#EAECF0",
    "bg": "#F7F8FA",
    "card": "#FFFFFF",
    "netliq": "#111827",
    "walcl": "#2563EB",
    "rrp": "#F59E0B",
    "tga": "#10B981",
    "effr": "#7C3AED",
    "nfci": "#374151",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

# ============================== CSS ==============================
st.markdown(
    f"""
    <style>
      .stApp {{
        background: {COLORS["bg"]};
      }}

      /* Tighten top padding */
      .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.0rem;
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background: {COLORS["card"]};
        border-right: 1px solid {COLORS["border"]};
      }}

      /* Remove Streamlit chrome */
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      header {{visibility: hidden;}}

      /* Header card */
      .hero {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      }}
      .hero-title {{
        font-size: 28px;
        font-weight: 800;
        color: {COLORS["ink"]};
        line-height: 1.2;
        margin: 0;
      }}
      .hero-sub {{
        margin-top: 6px;
        color: {COLORS["muted"]};
        font-size: 14px;
      }}
      .pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid {COLORS["border"]};
        color: {COLORS["muted"]};
        font-size: 12px;
        margin-left: 8px;
        vertical-align: middle;
      }}

      /* Metric cards */
      .kpi {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 14px 14px 12px 14px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      }}
      .kpi-label {{
        color: {COLORS["muted"]};
        font-size: 12px;
        margin: 0 0 6px 0;
      }}
      .kpi-value {{
        color: {COLORS["ink"]};
        font-size: 20px;
        font-weight: 800;
        margin: 0;
      }}
      .kpi-foot {{
        color: {COLORS["muted"]};
        font-size: 12px;
        margin-top: 6px;
      }}

      /* Section card */
      .card {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        **Goal:** Track Net Liquidity and policy/conditions in one place.

        **Net Liquidity** = WALCL − RRP − TGA

        **Series**
        - WALCL: Fed balance sheet
        - RRPONTSYD: reverse repo
        - WDTGAL: Treasury General Account
        - EFFR: effective fed funds rate
        - NFCI: Chicago Fed financial conditions (above 0 = tighter than average)
        """
    )

    st.markdown("---")
    st.markdown("### Settings")

    lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=2)
    years = LOOKBACK_OPTIONS[lookback_label]

    smooth = st.number_input(
        "Smoothing window (days)",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH_DAYS,
        step=1,
        help="Rolling mean applied to plotted series.",
    )

    show_rangeslider = st.checkbox("Show range slider", value=True)
    st.caption("Data source: FRED via pandas-datareader")

# ============================== Data ==============================
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill()
    except Exception:
        return pd.Series(dtype=float)


today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=25)   # long pull for continuity
start_lb = today - pd.DateOffset(years=years)

fed_bs = fred_series(FRED["fed_bs"], start_all, today)
rrp = fred_series(FRED["rrp"], start_all, today)
tga = fred_series(FRED["tga"], start_all, today)
effr = fred_series(FRED["effr"], start_all, today)
nfci = fred_series(FRED["nfci"], start_all, today)

df = pd.concat(
    [fed_bs, rrp, tga, effr, nfci],
    axis=1,
    keys=["WALCL", "RRP", "TGA", "EFFR", "NFCI"],
).ffill()

df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]

if df.empty:
    st.error("No data returned for the selected lookback window.")
    st.stop()

# Net Liquidity (billions)
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"] = df["RRP"]                 # already in billions on FRED
df["TGA_b"] = df["TGA"] / 1000.0
df["NetLiq"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Apply smoothing
cols = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR", "NFCI"]
for col in cols:
    df[f"{col}_s"] = (
        df[col].rolling(int(smooth), min_periods=1).mean()
        if int(smooth) > 1
        else df[col]
    )

# Rebase
def rebase(series: pd.Series, base_window: int = REBASE_BASE_WINDOW, min_base: float | None = None) -> pd.Series:
    s = series.copy()
    if s.isna().all():
        return s * 0 + 100

    head = s.dropna().iloc[:max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]
    if min_base is not None:
        base = max(float(base), float(min_base))

    if base == 0 or pd.isna(base):
        return s * 0 + 100
    return (s / base) * 100


reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"] = rebase(df["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"] = rebase(df["TGA_b"])

# ============================== Formatting helpers ==============================
def fmt_b(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.0f} B"

def fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def fmt_nfci(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.3f}"


latest = {
    "asof": df.index[-1],
    "netliq": df["NetLiq"].iloc[-1],
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
    "nfci": df["NFCI"].iloc[-1],
}

# ============================== Header ==============================
st.markdown(
    f"""
    <div class="hero">
      <div>
        <span class="hero-title">{TITLE}</span>
        <span class="pill">As of {latest["asof"].strftime("%b %d, %Y")}</span>
      </div>
      <div class="hero-sub">
        Net Liquidity (WALCL − RRP − TGA), policy rate (EFFR), and financial conditions (NFCI) with clean, consistent scaling and smoothing.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ============================== Main layout ==============================
tab_dash, tab_data, tab_method = st.tabs(["Dashboard", "Data", "Methodology"])

with tab_dash:
    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        ("Net Liquidity", fmt_b(latest["netliq"]), "Billions"),
        ("WALCL", fmt_b(latest["walcl"]), "Billions"),
        ("RRP", fmt_b(latest["rrp"]), "Billions"),
        ("TGA", fmt_b(latest["tga"]), "Billions"),
        ("EFFR", fmt_pct(latest["effr"]), "Policy rate"),
        ("NFCI", fmt_nfci(latest["nfci"]), "> 0 = tighter"),
    ]
    for col, (label, value, foot) in zip([c1, c2, c3, c4, c5, c6], kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                  <p class="kpi-label">{label}</p>
                  <p class="kpi-value">{value}</p>
                  <div class="kpi-foot">{foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Charts
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Net Liquidity (Billions, smoothed)",
            "Components Rebased (Index = 100 at start)",
            "Effective Fed Funds Rate (smoothed)",
            "Financial Conditions (NFCI, smoothed)",
        ),
    )

    # Row 1: Net Liquidity
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NetLiq_s"],
            name="Net Liquidity",
            mode="lines",
            line=dict(color=COLORS["netliq"], width=2.6),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Net Liquidity: %{y:,.0f} B<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Row 2: Rebased components
    fig.add_trace(
        go.Scatter(
            x=reb.index,
            y=reb["WALCL_idx"],
            name="WALCL (idx)",
            mode="lines",
            line=dict(color=COLORS["walcl"], width=2),
            hovertemplate="Date: %{x|%b %d, %Y}<br>WALCL idx: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reb.index,
            y=reb["RRP_idx"],
            name="RRP (idx)",
            mode="lines",
            line=dict(color=COLORS["rrp"], width=2),
            hovertemplate="Date: %{x|%b %d, %Y}<br>RRP idx: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reb.index,
            y=reb["TGA_idx"],
            name="TGA (idx)",
            mode="lines",
            line=dict(color=COLORS["tga"], width=2),
            hovertemplate="Date: %{x|%b %d, %Y}<br>TGA idx: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Row 3: EFFR
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR_s"],
            name="EFFR",
            mode="lines",
            line=dict(color=COLORS["effr"], width=2.4),
            hovertemplate="Date: %{x|%b %d, %Y}<br>EFFR: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Row 4: NFCI (with 0 line)
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=COLORS["border"], row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI_s"],
            name="NFCI",
            mode="lines",
            line=dict(color=COLORS["nfci"], width=2.4),
            hovertemplate="Date: %{x|%b %d, %Y}<br>NFCI: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=1100,
        margin=dict(l=60, r=30, t=70, b=55),
        legend=dict(
            orientation="h",
            x=0,
            y=1.08,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor=COLORS["border"],
            borderwidth=1,
        ),
        hovermode="x unified",
        font=dict(color=COLORS["ink"]),
    )

    fig.update_xaxes(
        tickformat="%b-%y",
        showgrid=True,
        gridcolor=COLORS["border"],
        rangeslider=dict(visible=show_rangeslider),
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["border"])

    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    out = pd.DataFrame(index=df.index)
    out.index.name = "Date"
    out["WALCL_B"] = df["WALCL_b"]
    out["RRP_B"] = df["RRP_b"]
    out["TGA_B"] = df["TGA_b"]
    out["NetLiq_B"] = df["NetLiq"]
    out["EFFR_%"] = df["EFFR"]
    out["NFCI"] = df["NFCI"]

    st.markdown("#### Data table")
    st.dataframe(out, use_container_width=True, height=520)

    st.markdown("")

    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="liquidity_tracker.csv",
        mime="text/csv",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with tab_method:
    st.markdown(
        f"""
        **Definitions**

        Net Liquidity = WALCL − RRP − TGA

        - WALCL is in millions, converted to billions
        - TGA is in millions, converted to billions
        - RRPONTSYD is already in billions on FRED

        **NFCI interpretation**
        - Composite measure of credit, leverage, and funding stress
        - Values above 0 imply tighter than average conditions

        **Smoothing**
        - Uses a {int(smooth)} day rolling mean (min periods = 1)

        **Rebasing**
        - Index base is the median of the first {REBASE_BASE_WINDOW} non-null observations
        - RRP base is floored at {RRP_BASE_FLOOR_B:.1f} B to avoid unstable starts when RRP is near zero
        """
    )

st.caption("© 2025 AD Fund Management LP")
