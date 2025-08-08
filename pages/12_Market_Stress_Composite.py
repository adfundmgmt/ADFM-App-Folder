############################################################
# Market Stress Composite – latest-correct and short-lookback-safe
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --------------- Config ---------------
TITLE = "Market Stress Composite"
FRED = {
    "vix": "VIXCLS",          # CBOE VIX (close)
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M Treasury spread (pct points)
    "hy_oas": "BAMLH0A0HYM2", # ICE BofA US High Yield OAS (pct points)
    # Funding stress proxy: 3M AA Financial CP minus 3M T-bill (daily)
    "cp_3m": "DCPF3M",        # 3-Month AA Financial Commercial Paper Rate
    "tbill_3m": "DTB3",       # 3-Month Treasury Bill Secondary Market Rate
    "spx": "SP500",           # S&P 500 index level
}
DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI = 70
REGIME_LO = 30

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
A single **0 to 100 stress score** blending volatility, credit, curve, funding,
and equity drawdown. Higher = more stress.

**Inputs**
• VIX (VIXCLS): equity implied vol  
• HY OAS (BAMLH0A0HYM2): high-yield credit spread  
• T10Y3M: 10y minus 3m Treasury spread (inverted for stress)  
• Funding spread: 3m AA financial CP − 3m T-bill (DCPF3M − DTB3)  
• SPX drawdown: percent off trailing high

**Method**
Each input becomes a rolling percentile over a long history window, then we weight and sum.
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
    w_fund  = st.slider("Funding (CP − T-bill)", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)
    show_components = st.checkbox("Show raw component panels", value=True)
    st.caption("Data: FRED via pandas-datareader")

# --------------- Data ---------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
# fetch wide so percentile window is covered even for short lookbacks
start_all = today - pd.DateOffset(years=max(perc_years + 2, years + 2))

# load raw
vix   = fred_series(FRED["vix"], start_all, today)
hy    = fred_series(FRED["hy_oas"], start_all, today)
yc    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m  = fred_series(FRED["cp_3m"], start_all, today)
tb3m  = fred_series(FRED["tbill_3m"], start_all, today)
spx   = fred_series(FRED["spx"], start_all, today)

# compute funding spread (daily, in pct points)
fund_raw = (cp3m - tb3m)

# determine last valid dates per series
def last_valid_date(s: pd.Series) -> pd.Timestamp:
    try:
        return s.dropna().index.max()
    except Exception:
        return pd.NaT

lasts = [
    last_valid_date(vix),
    last_valid_date(hy),
    last_valid_date(yc),
    last_valid_date(fund_raw),
    last_valid_date(spx),
]
if any(pd.isna(d) for d in lasts):
    st.error("A required series is empty. Check FRED connectivity or mnemonics.")
    st.stop()

last_common_date = min(lasts)

# build a shared business-day index up to the last common valid date
bidx = pd.bdate_range(start_all, last_common_date)

def to_bidx(s: pd.Series) -> pd.Series:
    return s.reindex(bidx).ffill()

vix, hy, yc, fund, spx = map(to_bidx, [vix, hy, yc, fund_raw, spx])

# assemble, then drop rows until all series are present at least once
df_all = pd.concat([vix, hy, yc, fund, spx], axis=1)
df_all.columns = ["VIX", "HY_OAS", "T10Y3M", "FUND", "SPX"]
first_ok = df_all.dropna(how="any").index.min()
if pd.isna(first_ok):
    st.error("No overlapping history across inputs.")
    st.stop()
df_all = df_all.loc[first_ok:].ffill()

# stale-data notice (if last_common_date is old)
bdays_since = np.busday_count(df_all.index[-1].date(), today.date())
if bdays_since >= 3:
    st.warning(f"Latest common data are {bdays_since} business days old (as of {df_all.index[-1].date()}).")

# subset to lookback for display, but keep full df_all for percentiles
start_lb = max(df_all.index.min(), today - pd.DateOffset(years=years))
df = df_all[df_all.index >= start_lb].copy()
if df.empty:
    st.error("No data for selected lookback.")
    st.stop()

# --------------- Transformations ---------------
# Drawdown stress as positive number
roll_max = df_all["SPX"].cummax()
dd_all = 100.0 * (df_all["SPX"] / roll_max - 1.0)  # negative or 0
stress_dd_all = -dd_all.clip(upper=0)

# Invert curve for stress
inv_curve_all = -df_all["T10Y3M"]

# Rolling percentile helper – computed on df_all window
def rolling_percentile(s: pd.Series, window_days: int) -> pd.Series:
    w = max(60, int(window_days * 252 / 365))  # approx business days
    return s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

window_days = int(perc_years * 365)

pct_vix   = rolling_percentile(df_all["VIX"], window_days)
pct_hy    = rolling_percentile(df_all["HY_OAS"], window_days)
pct_curve = rolling_percentile(inv_curve_all, window_days)
pct_fund  = rolling_percentile(df_all["FUND"], window_days)
pct_dd    = rolling_percentile(stress_dd_all, window_days)

scores_all = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
scores_all.columns = ["VIX_p", "HY_p", "CurveInv_p", "Fund_p", "DD_p"]

# align to display window and drop rows where any percentile is nan
scores = scores_all.loc[df.index.min(): df.index.max()].dropna()
# align df to scores to ensure latest present
df = df.loc[scores.index]

# optional smoothing
if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# weights
weights = np.array([w_vix, w_hy, w_curve, w_fund, w_dd], dtype=float)
weights = weights if weights.sum() > 0 else np.ones(5)
weights = weights / weights.sum()

# composite 0..100
comp = 100.0 * (scores.values @ weights)
comp_s = pd.Series(comp, index=scores.index, name="Composite")

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
    "fund": df["FUND"].iloc[-1],
    "dd": -100.0 * (df["SPX"].iloc[-1] / df["SPX"].cummax().iloc[-1] - 1.0),
}

def fmt2(x): return "N/A" if pd.isna(x) else f"{x:.2f}"
def fmt0(x): return "N/A" if pd.isna(x) else f"{x:.0f}"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stress Composite", fmt0(latest["level"]))
c2.metric("VIX", fmt2(latest["vix"]))
c3.metric("HY OAS (pp)", fmt2(latest["hy"]))
c4.metric("T10Y3M (pp)", fmt2(latest["yc"]))
c5.metric("Funding (pp)", fmt2(latest["fund"]))
c6.metric("SPX Drawdown (%)", fmt2(latest["dd"]))

st.info(
    f"As of {latest['date']}  |  Regime: {latest['regime']}  |  "
    f"Weights: VIX {weights[0]:.2f}, HY {weights[1]:.2f}, Curve {weights[2]:.2f}, Funding {weights[3]:.2f}, Drawdown {weights[4]:.2f}"
)

# --------------- Charts ---------------
rows = 2 + (2 if show_components else 0)
fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Market Stress Composite (0 to 100)",
        "Component Percentile Scores (0 to 100)",
        "Raw Series (Credit, Curve, Funding, VIX)" if show_components else None,
        "SPX and Drawdown" if show_components else None
    )[:rows]
)

# Row 1: composite with regime bands
fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

# Row 2: component percentiles
for name, series, color in [
    ("VIX",        100.0 * scores["VIX_p"],       "#1f77b4"),
    ("HY OAS",     100.0 * scores["HY_p"],        "#d62728"),
    ("CurveInv",   100.0 * scores["CurveInv_p"],  "#2ca02c"),
    ("Funding",    100.0 * scores["Fund_p"],      "#9467bd"),
    ("Drawdown",   100.0 * scores["DD_p"],        "#ff7f0e"),
]:
    fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(width=1.8, color=color)), row=2, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=2, col=1)

# Optional raw panels
row_ptr = 3
if show_components:
    fig.add_trace(go.Scatter(x=df.index, y=df["HY_OAS"], name="HY OAS (pp)", line=dict(color="#d62728")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["T10Y3M"], name="T10Y3M (pp)", line=dict(color="#2ca02c")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["FUND"], name="Funding (pp)", line=dict(color="#9467bd")), row=row_ptr, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VIX"], name="VIX", line=dict(color="#1f77b4")), row=row_ptr, col=1)
    fig.update_yaxes(title="Raw units", row=row_ptr, col=1)

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
fig.update_xaxes(tickformat="%b-%y", row=rows, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# --------------- Download ---------------
with st.expander("Download Data"):
    out = pd.concat(
        [df[["VIX","HY_OAS","T10Y3M","FUND","SPX"]],
         100.0*scores.rename(columns={"VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"}),
         comp_s.rename("Composite")],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

# --------------- Methodology ---------------
with st.expander("Methodology"):
    st.markdown(f"""
Composite = weighted average of component percentile scores, scaled 0 to 100.

Percentiles use a rolling window of {perc_years} years over the full history,
so short lookbacks still have valid tail data.

Curve is inverted so deeper inversion reads as higher stress.
Funding is 3m AA financial CP minus 3m T-bill (daily). Higher = tighter funding.
SPX drawdown converts percent off high to a positive stress input.

Smoothing applies a {smooth}-day moving average to percentile series.
Regimes: Low <= {REGIME_LO}, High >= {REGIME_HI}.
""")
st.caption("© 2025 AD Fund Management LP")
