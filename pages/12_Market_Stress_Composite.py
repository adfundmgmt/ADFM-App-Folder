############################################################
# Market Stress Composite – daily-safe with SPX/Drawdown panel
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
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M Treasury spread
    "hy_oas": "BAMLH0A0HYM2", # ICE BofA US High Yield OAS
    "cp_3m": "DCPF3M",        # 3M AA Financial CP rate
    "tbill_3m": "DTB3",       # 3M Treasury bill
    "spx": "SP500",           # S&P 500 index
}
DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI = 70
REGIME_LO = 30
MAX_STALE_BDAYS = 1   # components older than this are zero-weighted

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
Daily **0–100 stress score** from VIX, HY OAS, inverted 10y−3m,
funding spread (3m AA CP − 3m T-bill), and SPX drawdown.
Stale inputs get zero weight for that day. Others renormalize.
""")
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y","2y","3y","5y","10y"],
                            index=["1y","2y","3y","5y","10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 60, DEFAULT_SMOOTH, 1)
    perc_years = st.number_input("Percentile history window (years)", 3, 25,
                                 DEFAULT_PERCENTILE_WINDOW_YEARS, 1)
    st.subheader("Weights")
    w_vix   = st.slider("VIX", 0.0, 1.0, 0.25, 0.05)
    w_hy    = st.slider("HY OAS", 0.0, 1.0, 0.25, 0.05)
    w_curve = st.slider("Yield Curve (inverted)", 0.0, 1.0, 0.20, 0.05)
    w_fund  = st.slider("Funding (CP − T-bill)", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)
    st.caption("Source: FRED via pandas-datareader")

# --------------- Data ---------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=max(perc_years + 2, years + 2))

vix   = fred_series(FRED["vix"], start_all, today)
hy    = fred_series(FRED["hy_oas"], start_all, today)
yc    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m  = fred_series(FRED["cp_3m"], start_all, today)
tb3m  = fred_series(FRED["tbill_3m"], start_all, today)
spx   = fred_series(FRED["spx"], start_all, today)

fund_raw = (cp3m - tb3m)

# business-day grid through today
bidx = pd.bdate_range(min(x.dropna().index.min() for x in [vix, hy, yc, fund_raw, spx]), today)

def to_bidx(s: pd.Series) -> pd.Series:
    return s.reindex(bidx).ffill()

vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix, hy, yc, fund_raw, spx])

# freshness mask: 1 if print age ≤ MAX_STALE_BDAYS
def fresh_mask(original: pd.Series, bindex: pd.DatetimeIndex, max_stale: int) -> pd.Series:
    obs = original.reindex(bindex)
    pos = np.arange(len(bindex))
    has = obs.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=bindex).ffill().values
    age = pos - last_pos
    return pd.Series((~np.isnan(age)) & (age <= max_stale), index=bindex)

m_vix  = fresh_mask(vix,     bidx, MAX_STALE_BDAYS)
m_hy   = fresh_mask(hy,      bidx, MAX_STALE_BDAYS)
m_yc   = fresh_mask(yc,      bidx, MAX_STALE_BDAYS)
m_fund = fresh_mask(fund_raw,bidx, MAX_STALE_BDAYS)
m_dd   = fresh_mask(spx,     bidx, MAX_STALE_BDAYS)

# assemble
df_all = pd.DataFrame({"VIX": vix_b, "HY_OAS": hy_b, "T10Y3M": yc_b, "FUND": fund_b, "SPX": spx_b}).dropna(how="all")

# --------------- Transforms ---------------
roll_max = df_all["SPX"].cummax()
dd_all = 100.0 * (df_all["SPX"] / roll_max - 1.0)
stress_dd_all = -dd_all.clip(upper=0)
inv_curve_all = -df_all["T10Y3M"]

def rolling_percentile(s: pd.Series, window_days: int) -> pd.Series:
    w = max(60, int(window_days * 252 / 365))
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
scores_all.columns = ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]

# lookback subset
start_lb = today - pd.DateOffset(years=years)
scores = scores_all.loc[start_lb:].copy()
panel  = df_all.loc[start_lb:].copy()

# optional smoothing
if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# dynamic weights with stale masks
weights = np.array([w_vix, w_hy, w_curve, w_fund, w_dd], dtype=float)
weights = weights if weights.sum() > 0 else np.ones(5)
weights = weights / weights.sum()

masks = pd.concat([m_vix, m_hy, m_yc, m_fund, m_dd], axis=1)
masks.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]
masks = masks.loc[scores.index].astype(float)

X = scores[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].values
W = weights.reshape(1, -1)
M = masks.values
active_w = (W * M).sum(axis=1)
comp = np.where(active_w > 0, 100.0 * (X * W * M).sum(axis=1) / active_w, np.nan)
comp_s = pd.Series(comp, index=scores.index, name="Composite")
coverage = pd.Series(active_w / W.sum(), index=scores.index, name="ActiveWeightShare")

# --------------- Metrics ---------------
def tag_for_level(x: float) -> str:
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

latest_idx = comp_s.dropna().index[-1]
latest = {
    "date": latest_idx.date(),
    "level": comp_s.loc[latest_idx],
    "regime": tag_for_level(comp_s.loc[latest_idx]),
    "coverage": coverage.loc[latest_idx],
    "vix": panel["VIX"].loc[latest_idx],
    "hy": panel["HY_OAS"].loc[latest_idx],
    "yc": panel["T10Y3M"].loc[latest_idx],
    "fund": panel["FUND"].loc[latest_idx],
    "dd": -100.0 * (panel["SPX"].loc[latest_idx] / panel["SPX"][:latest_idx].cummax().iloc[-1] - 1.0),
}

def fmt2(x): return "N/A" if pd.isna(x) else f"{x:.2f}"
def fmt0(x): return "N/A" if pd.isna(x) else f"{x:.0f}"
def pct(x): return "N/A" if pd.isna(x) else f"{100*x:.0f}%"

# top tiles
t1, t2, t3 = st.columns(3)
t1.metric("Stress Composite", fmt0(latest["level"]))
t2.metric("Regime", latest["regime"])
t3.metric("Active weight in use", pct(latest["coverage"]))

t4, t5, t6, t7, t8 = st.columns(5)
t4.metric("VIX", fmt2(latest["vix"]))
t5.metric("HY OAS (pp)", fmt2(latest["hy"]))
t6.metric("T10Y3M (pp)", fmt2(latest["yc"]))
t7.metric("Funding (pp)", fmt2(latest["fund"]))
t8.metric("SPX Drawdown (%)", fmt2(latest["dd"]))

st.info(
    f"As of {latest['date']}  |  Weights: VIX {weights[0]:.2f}, HY {weights[1]:.2f}, "
    f"Curve {weights[2]:.2f}, Funding {weights[3]:.2f}, Drawdown {weights[4]:.2f}"
)

# --------------- Chart helpers ---------------
def pad_range(lo: float, hi: float, pad: float = 0.06):
    if pd.isna(lo) or pd.isna(hi):
        return None
    d = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1.0)
    return lo - d * pad, hi + d * pad

# --------------- Charts ---------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    specs=[[{}], [{"secondary_y": True}]],
    subplot_titles=("Market Stress Composite (0 to 100)", "SPX and Drawdown (dual axis)")
)

# Row 1: composite
fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

# Row 2: SPX left, Drawdown right
spx_rebased = panel["SPX"] / panel["SPX"].iloc[0] * 100
dd_series = -100.0 * (panel["SPX"] / panel["SPX"].cummax() - 1.0)

llo, lhi = pad_range(spx_rebased.min(), spx_rebased.max())
rlo, rhi = pad_range(max(0.0, dd_series.min()), dd_series.max())

fig.add_trace(go.Scatter(x=panel.index, y=spx_rebased, name="SPX (rebased=100)",
                         line=dict(color="#7f7f7f")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=panel.index, y=dd_series, name="Drawdown (%)",
                         line=dict(color="#ff7f0e")), row=2, col=1, secondary_y=True)

fig.update_yaxes(title="Index", range=[llo, lhi], row=2, col=1, secondary_y=False)
fig.update_yaxes(title="Drawdown %", range=[rlo, rhi], row=2, col=1, secondary_y=True)

fig.update_layout(
    template="plotly_white",
    height=720,
    legend=dict(orientation="h", x=0, y=1.14, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60)
)
fig.update_xaxes(tickformat="%b-%y", title="Date", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# --------------- Download ---------------
with st.expander("Download Data"):
    out = pd.concat(
        [
            panel[["VIX","HY_OAS","T10Y3M","FUND","SPX"]],
            100.0 * scores.rename(columns={"VIX_p":"VIX_pct","HY_p":"HY_pct",
                                           "CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct",
                                           "DD_p":"DD_pct"}),
            masks.rename(columns={"VIX_m":"VIX_fresh","HY_m":"HY_fresh",
                                  "Curve_m":"Curve_fresh","Fund_m":"Fund_fresh",
                                  "DD_m":"DD_fresh"}),
            coverage,
            comp_s.rename("Composite"),
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

# --------------- Methodology ---------------
with st.expander("Methodology"):
    st.markdown(f"""
Composite is a weighted average of component percentile scores on a {perc_years}-year rolling window.
Curve is inverted. Funding is 3m AA CP minus 3m T-bill. Drawdown is percent off high.
If a component is older than {MAX_STALE_BDAYS} business day it is zero-weighted that day.
Smoothing applies a {smooth}-day MA to percentile series.
Regimes: Low ≤ {REGIME_LO}, High ≥ {REGIME_HI}.
""")
st.caption("© 2025 AD Fund Management LP")
