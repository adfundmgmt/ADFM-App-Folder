############################################################
# Market Stress Composite
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# --------------- Config ---------------
TITLE = "Market Stress Composite"
FRED = {
    "vix": "VIXCLS",          # CBOE VIX (close)
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M Treasury spread (pct points)
    "hy_oas": "BAMLH0A0HYM2", # ICE BofA US High Yield OAS (pct points)
    "ted": "TEDRATE",         # TED spread (pct points)
    "spx": "SP500",           # S&P 500 index level
}
DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10  # history window for percentile ranks
REGIME_HI = 70
REGIME_LO = 30

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    A single **0 to 100 stress score** that blends volatility, credit, curve, funding,
    and equity drawdown. Higher = more stress.

    **Inputs**
    • **VIX (VIXCLS)**: equity implied vol  
    • **HY OAS (BAMLH0A0HYM2)**: high-yield credit spread  
    • **T10Y3M**: 10y minus 3m Treasury spread (inverted for stress)  
    • **TEDRATE**: interbank funding stress  
    • **SPX drawdown**: percent off trailing high

    Each input is converted to a **rolling percentile** over a selectable history window,
    then combined using your weights.
    """)
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=["1y","2y","3y","5y","10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 60, DEFAULT_SMOOTH, 1)
    perc_years = st.number_input("Percentile history window (years)", 3, 25, DEFAULT_PERCENTILE_WINDOW_YEARS, 1)
    st.subheader("Weights")
    w_vix   = st.slider("VIX", 0.0, 1.0, 0.25, 0.05)
    w_hy    = st.slider("HY OAS", 0.0, 1.0, 0.25, 0.05)
    w_curve = st.slider("Yield Curve (T10Y3M inverted)", 0.0, 1.0, 0.20, 0.05)
    w_ted   = st.slider("TED Spread", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)
    show_components = st.checkbox("Show raw component panels", value=True)
    st.caption("Data: FRED via pandas-datareader")

# --------------- Data ---------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.asfreq("B").ffill()  # business day, forward fill
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
# fetch wide so percentile history is available even on short lookbacks
start_all = today - pd.DateOffset(years=max(perc_years + 2, years + 2))

vix   = fred_series(FRED["vix"], start_all, today)
hy    = fred_series(FRED["hy_oas"], start_all, today)
yc    = fred_series(FRED["yc_10y_3m"], start_all, today)      # positive = steep, negative = inverted
ted   = fred_series(FRED["ted"], start_all, today)
spx   = fred_series(FRED["spx"], start_all, today)

df_all = pd.concat([vix, hy, yc, ted, spx], axis=1)
df_all.columns = ["VIX", "HY_OAS", "T10Y3M", "TED", "SPX"]
df_all = df_all.dropna(how="any")

# subset to display lookback
start_lb = today - pd.DateOffset(years=years)
df = df_all[df_all.index >= start_lb].copy()
if df.empty:
    st.error("No data for selected lookback.")
    st.stop()

# --------------- Transformations ---------------
# SPX drawdown as positive stress metric
roll_max = df_all["SPX"].cummax()
dd_all = 100.0 * (df_all["SPX"] / roll_max - 1.0)  # 0 or negative
stress_dd_all = -dd_all.clip(upper=0)              # positive when drawdown

# Invert the curve spread for stress: more inversion => higher stress
inv_curve_all = -df_all["T10Y3M"]

# Rolling percentile helper
def rolling_percentile(s: pd.Series, window_days: int) -> pd.Series:
    # window in calendar days approximated to business days
    w = max(60, int(window_days * 252 / 365))  # at least ~3 months of obs
    return s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

window_days = perc_years * 365

# Percentile ranks in 0..1
pct_vix   = rolling_percentile(df_all["VIX"], window_days)
pct_hy    = rolling_percentile(df_all["HY_OAS"], window_days)
pct_curve = rolling_percentile(inv_curve_all, window_days)
pct_ted   = rolling_percentile(df_all["TED"], window_days)
pct_dd    = rolling_percentile(stress_dd_all, window_days)

# align and subset to display window
scores_all = pd.concat(
    [pct_vix, pct_hy, pct_curve, pct_ted, pct_dd], axis=1
).dropna()
scores_all.columns = ["VIX_p", "HY_p", "CurveInv_p", "TED_p", "DD_p"]

scores = scores_all.loc[df.index.min(): df.index.max()].copy().dropna()
df = df.loc[scores.index]  # align

# optional smoothing
if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# normalize weights to sum 1
weights = np.array([w_vix, w_hy, w_curve, w_ted, w_dd], dtype=float)
if weights.sum() == 0:
    weights = np.array([1, 1, 1, 1, 1], dtype=float)
weights = weights / weights.sum()

# composite 0..100
composite = 100.0 * (scores.values @ weights)
comp_s = pd.Series(composite, index=scores.index, name="Composite")

# --------------- Metrics ---------------
def tag_for_level(x: float) -> str:
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

latest = {
    "date": comp_s.index[-1].date(),
    "level": comp_s.iloc[-1],
    "regime": tag_for_level(comp_s.iloc[-1]),
    "vix": df["VIX"].iloc[-1],
    "hy": df["HY_OAS"].iloc[-1],
    "yc": df["T10Y3M"].iloc[-1],
    "ted": df["TED"].iloc[-1],
    "dd": -100.0 * (df["SPX"].iloc[-1] / df["SPX"].cummax().iloc[-1] - 1.0),
}

def fmt2(x): return "N/A" if pd.isna(x) else f"{x:.2f}"
def fmt0(x): return "N/A" if pd.isna(x) else f"{x:.0f}"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stress Composite", fmt0(latest["level"]))
c2.metric("VIX", fmt2(latest["vix"]))
c3.metric("HY OAS (pp)", fmt2(latest["hy"]))
c4.metric("T10Y3M (pp)", fmt2(latest["yc"]))
c5.metric("TED (pp)", fmt2(latest["ted"]))
c6.metric("SPX Drawdown (%)", fmt2(latest["dd"]))

st.info(f"Regime: {latest['regime']}  |  Weights: VIX {weights[0]:.2f}, HY {weights[1]:.2f}, Curve {weights[2]:.2f}, TED {weights[3]:.2f}, Drawdown {weights[4]:.2f}")

# --------------- Charts ---------------
# Main composite + regime shading
fig = make_subplots(
    rows=2 + (2 if show_components else 0), cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Market Stress Composite (0 to 100)",
        "Component Percentile Scores (0 to 100)",
        "Raw Series (Credit, Curve, Funding, VIX)" if show_components else None,
        "SPX and Drawdown" if show_components else None
    )[: 2 + (2 if show_components else 0)]
)

# Row 1: composite
fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
# regime bands
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

# Row 2: component scores
for name, series, color in [
    ("VIX",        100.0 * scores["VIX_p"],       "#1f77b4"),
    ("HY OAS",     100.0 * scores["HY_p"],        "#d62728"),
    ("CurveInv",   100.0 * scores["CurveInv_p"],  "#2ca02c"),
    ("TED",        100.0 * scores["TED_p"],       "#9467bd"),
    ("Drawdown",   100.0 * scores["DD_p"],        "#ff7f0e"),
]:
    fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(width=1.8, color=color)), row=2, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=2, col=1)

row_ptr = 3
if show_components:
    # Row 3: raw series except SPX
    fig.add_trace(go.Scatter(x=df.index, y=df["HY_OAS"], name="HY OAS (pp)", line=dict(color="#d62728")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["T10Y3M"], name="T10Y3M (pp)", line=dict(color="#2ca02c")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["TED"], name="TED (pp)", line=dict(color="#9467bd")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VIX"], name="VIX", line=dict(color="#1f77b4")), row=row_ptr, col=1)
    fig.update_yaxes(title="Raw units", row=row_ptr, col=1)

    # Row 4: SPX and drawdown
    row_ptr += 1
    spx_rebased = df["SPX"] / df["SPX"].iloc[0] * 100
    dd_series = -100.0 * (df["SPX"] / df["SPX"].cummax() - 1.0)
    fig.add_trace(go.Scatter(x=df.index, y=spx_rebased, name="SPX (rebased=100)", line=dict(color="#7f7f7f")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=dd_series, name="Drawdown (%)", line=dict(color="#ff7f0e")), row=row_ptr, col=1)
    fig.update_yaxes(title="Index / %", row=row_ptr, col=1)

fig.update_layout(
    template="plotly_white",
    height=900 if show_components else 560,
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60)
)
fig.update_xaxes(tickformat="%b-%y", row=(2 + (2 if show_components else 0)), col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# --------------- Download ---------------
with st.expander("Download Data"):
    out = pd.concat(
        [df[["VIX","HY_OAS","T10Y3M","TED","SPX"]],
         100.0*scores.rename(columns={"VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","TED_p":"TED_pct","DD_p":"DD_pct"}),
         comp_s.rename("Composite")],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

# --------------- Methodology ---------------
with st.expander("Methodology"):
    st.markdown(f"""
    **Composite** = weighted average of component percentile scores, scaled 0 to 100.

    **Percentile scoring**  
    For each component we compute a rolling percentile of the latest value relative to the prior
    {perc_years} years of history.  
    Higher percentile means more stressed conditions. For the yield curve we use the **negative**
    of T10Y3M so deeper inversion maps to higher stress.  
    SPX drawdown is the percent below the running high, converted to a positive stress value.

    **Smoothing** applies a simple moving average over {smooth} days to the percentile series.

    **Regimes**  
    Low stress: score <= {REGIME_LO}  
    Neutral: between {REGIME_LO} and {REGIME_HI}  
    High stress: score >= {REGIME_HI}

    Notes: HY OAS and TED are not strictly daily; this app forward-fills to business days for alignment.
    """)

st.caption("© 2025 AD Fund Management LP")
