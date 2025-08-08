############################################################
# Market Stress Composite - daily-safe, stale-aware
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
    "vix": "VIXCLS",          # CBOE VIX (close)                [daily]
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M Treasury spread    [daily]
    "hy_oas": "BAMLH0A0HYM2", # ICE BofA US HY OAS              [daily but lags]
    "cp_3m": "DCPF3M",        # 3M AA Financial CP rate         [daily]
    "tbill_3m": "DTB3",       # 3M T-bill secondary market      [daily]
    "spx": "SP500",           # S&P 500 index                   [daily]
}
DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI = 70
REGIME_LO = 30
MAX_STALE_BDAYS = 1   # components older than this are zero-weighted for that date

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
A single **0 to 100 stress score** from VIX, HY OAS, inverted 10y minus 3m,
a funding spread (3m AA CP minus 3m T-bill), and SPX drawdown.
Score is daily. If an input is stale, its weight is set to zero for that day
and other weights are renormalized.
""")
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"],
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
start_all = today - pd.DateOffset(years=max(perc_years + 2, years + 2))

vix   = fred_series(FRED["vix"], start_all, today)
hy    = fred_series(FRED["hy_oas"], start_all, today)
yc    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m  = fred_series(FRED["cp_3m"], start_all, today)
tb3m  = fred_series(FRED["tbill_3m"], start_all, today)
spx   = fred_series(FRED["spx"], start_all, today)

fund_raw = (cp3m - tb3m)

def last_valid_date(s: pd.Series) -> pd.Timestamp:
    try:
        return s.dropna().index.max()
    except Exception:
        return pd.NaT

lasts = [last_valid_date(x) for x in [vix, hy, yc, fund_raw, spx]]
if any(pd.isna(d) for d in lasts):
    st.error("A required series is empty. Check FRED connectivity or mnemonics.")
    st.stop()

# business-day grid through today
bidx = pd.bdate_range(min(x.dropna().index.min() for x in [vix, hy, yc, fund_raw, spx]), today)

def to_bidx(s: pd.Series) -> pd.Series:
    return s.reindex(bidx).ffill()

vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix, hy, yc, fund_raw, spx])

# helper: fresh mask by business-day age relative to last real print
def fresh_mask(original: pd.Series, bindex: pd.DatetimeIndex, max_stale: int) -> pd.Series:
    obs = original.reindex(bindex)               # NaN on days without a print
    pos = np.arange(len(bindex))
    has = obs.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=bindex).ffill().values
    age = pos - last_pos                         # business-day distance
    mask = ~np.isnan(age) & (age <= max_stale)
    return pd.Series(mask, index=bindex)

m_vix  = fresh_mask(vix,  bidx, MAX_STALE_BDAYS)
m_hy   = fresh_mask(hy,   bidx, MAX_STALE_BDAYS)   # often lags
m_yc   = fresh_mask(yc,   bidx, MAX_STALE_BDAYS)
m_fund = fresh_mask(fund_raw, bidx, MAX_STALE_BDAYS)
m_dd   = fresh_mask(spx,  bidx, MAX_STALE_BDAYS)   # drawdown needs SPX

# assemble full panel
df_all = pd.DataFrame({
    "VIX": vix_b, "HY_OAS": hy_b, "T10Y3M": yc_b, "FUND": fund_b, "SPX": spx_b
}).dropna(how="all")

# --------------- Transformations ---------------
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
scores_all.columns = ["VIX_p", "HY_p", "CurveInv_p", "Fund_p", "DD_p"]

# subset to lookback window
start_lb = today - pd.DateOffset(years=years)
scores = scores_all.loc[start_lb:].copy()
panel  = df_all.loc[start_lb:].copy()

# optional smoothing on percentiles
if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# dynamic weights with stale-aware masks
weights = np.array([w_vix, w_hy, w_curve, w_fund, w_dd], dtype=float)
weights = weights if weights.sum() > 0 else np.ones(5)
weights = weights / weights.sum()

masks = pd.concat([m_vix, m_hy, m_yc, m_fund, m_dd], axis=1)
masks.columns = ["VIX_m", "HY_m", "Curve_m", "Fund_m", "DD_m"]
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
    "vix_age": int((~m_vix.loc[[latest_idx]]).astype(int).iloc[0]),   # 0 fresh, 1 stale beyond 1 bday
    "hy_age":  int((~m_hy.loc[[latest_idx]]).astype(int).iloc[0]),
    "yc_age":  int((~m_yc.loc[[latest_idx]]).astype(int).iloc[0]),
    "fund_age":int((~m_fund.loc[[latest_idx]]).astype(int).iloc[0]),
    "dd_age":  int((~m_dd.loc[[latest_idx]]).astype(int).iloc[0]),
}

def fmt2(x): return "N/A" if pd.isna(x) else f"{x:.2f}"
def fmt0(x): return "N/A" if pd.isna(x) else f"{x:.0f}"
def pct(x): return "N/A" if pd.isna(x) else f"{100*x:.0f}%"

c1, c2, c3 = st.columns(3)
c1.metric("Stress Composite", fmt0(latest["level"]))
c2.metric("Regime", latest["regime"])
c3.metric("Active weight in use", pct(latest["coverage"]))
st.caption("Age flags: 0 means fresh today or yesterday. 1 means stale beyond limit. "
           f"VIX {latest['vix_age']}  HY {latest['hy_age']}  Curve {latest['yc_age']}  "
           f"Funding {latest['fund_age']}  Drawdown {latest['dd_age']}")

# --------------- Chart ---------------
fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                    subplot_titles=("Market Stress Composite (0 to 100)",))
fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)
fig.update_layout(template="plotly_white", height=420,
                  legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
                  margin=dict(l=60, r=40, t=60, b=60))
fig.update_xaxes(tickformat="%b-%y", title="Date")
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
Composite = weighted average of component percentile scores, scaled 0 to 100.
Percentiles use a rolling window of {perc_years} years over the full history.
Curve is inverted. Funding is 3m AA CP minus 3m T-bill. Drawdown is percent off high.
If a component print is older than {MAX_STALE_BDAYS} business day, its weight is set to zero for that date and remaining weights are renormalized.
Smoothing applies a {smooth}-day moving average to percentile series.
Regimes: Low <= {REGIME_LO}, High >= {REGIME_HI}.
""")
st.caption("© 2025 AD Fund Management LP")
