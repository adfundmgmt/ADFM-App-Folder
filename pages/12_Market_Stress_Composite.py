############################################################
# Market Stress Composite - trading-day aligned, hardened
# AD Fund Management LP
# - Canonical trading-day index shared by all plots
# - Combined figure: SPX+Drawdown (top), Composite (bottom)
# - Close-only: no intraday nowcast, end-of-day data only
# - 8-factor composite with mask-based renorm of weights
############################################################

# Py3.12 compat shim for pandas_datareader
import sys, types
import packaging.version
try:
    import distutils.version
except Exception:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ui_helpers import render_page_header

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
render_page_header(
    TITLE,
    subtitle="Blended 8-factor stress gauge with aligned trading days.",
    description="The composite now shares the same header treatment as the rest of the dashboards for quicker orientation.",
)

# ---------------- Data maps -----------------
FRED = {
    "vix": "VIXCLS",           # VIX close (daily)
    "yc_10y_3m": "T10Y3M",     # 10Y minus 3M (daily)
    "hy_oas": "BAMLH0A0HYM2",  # ICE HY OAS (daily, lag)
    "cp_3m": "DCPF3M",         # 3M AA CP
    "tbill_3m": "DTB3",        # 3M T-bill
    "spx": "SP500",            # S&P 500 Index
}

# ---------------- Parameters ----------------
DEFAULT_LOOKBACK = "3y"
PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1
REGIME_HI, REGIME_LO = 70, 30

# Allow macro data to be stale for a few sessions before dropping
MAX_STALE_BDAYS = 3

# Weights for 8-factor composite
W_VIX      = 0.20  # implied vol
W_HY_OAS   = 0.15  # HY OAS (FRED)
W_HY_LQD   = 0.15  # HYG/LQD spread
W_CURVE    = 0.10  # 10Y-3M inversion
W_FUND     = 0.10  # CP - TBill funding stress
W_DD       = 0.15  # SPX drawdown
W_RV       = 0.10  # SPX 21d realized vol
W_BREADTH  = 0.05  # SPY/RSP breadth

WEIGHTS_VEC = np.array(
    [W_VIX, W_HY_OAS, W_HY_LQD, W_CURVE, W_FUND, W_DD, W_RV, W_BREADTH],
    dtype=float
)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

weights_text = (
    f"VIX {W_VIX:.2f}, HY_OAS {W_HY_OAS:.2f}, HY_LQD {W_HY_LQD:.2f}, "
    f"Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}, "
    f"RV21 {W_RV:.2f}, Breadth {W_BREADTH:.2f}"
)

# ---------------- Sidebar -------------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"],
                            index=["1y", "2y", "3y", "5y", "10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])

    st.subheader("Weights (fixed)")
    st.caption(weights_text)

    st.markdown("---")
    st.subheader("About this tool")
    st.markdown(
        "- Purpose: a single risk dial blending volatility, credit, curve inversion, funding, equity drawdown, "
        "realized volatility, ETF credit spread, and breadth into a 0-100 score.\n"
        "- Data: FRED for macro series, Yahoo Finance for ETF and equity breadth series.\n"
        "- Method: rolling percentiles over a 3-year window, fixed weights, mask-based renormalization.\n"
        "- Alignment: all factors evaluated on SPX trading days only, close-only (end-of-day data).\n"
        "- When a factor is stale beyond the allowed lag, it is dropped and weights re-normalize across active factors.\n"
        f"- Current weights: {weights_text}"
    )

# ---------------- Time helpers --------------
NY_TZ = "America/New_York"

def today_naive() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz=NY_TZ).date())

def to_naive_date_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
    return obj

def as_naive_date(ts_like) -> pd.Timestamp:
    t = pd.to_datetime(ts_like)
    return pd.Timestamp(t.date())

# ---------------- Loaders -------------------
@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return to_naive_date_index(s.ffill().dropna())
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def yf_history_daily(tickers, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )
        if isinstance(data, pd.Series):
            data = data.to_frame()
        # Prefer Adj Close if present, then Close
        if isinstance(data.columns, pd.MultiIndex):
            if ("Adj Close" in data.columns.get_level_values(0)):
                px = data["Adj Close"].copy()
            elif ("Close" in data.columns.get_level_values(0)):
                px = data["Close"].copy()
            else:
                first_field = data.columns.levels[0][0]
                px = data[first_field].copy()
        else:
            px = data.copy()

        if isinstance(px, pd.Series):
            px = px.to_frame()

        px = px.ffill().dropna(how="all")
        return to_naive_date_index(px)
    except Exception:
        return pd.DataFrame()

# Rolling percentile on trading days, NaN-safe
def rolling_percentile_trading(values: pd.Series, idx: pd.DatetimeIndex, window_trading_days: int) -> pd.Series:
    s = to_naive_date_index(values).reindex(idx)
    def _last_percentile(x):
        arr = pd.Series(x).dropna()
        if arr.empty:
            return np.nan
        return arr.rank(pct=True).iloc[-1]
    out = s.rolling(
        window_trading_days,
        min_periods=max(40, window_trading_days // 6)
    ).apply(_last_percentile, raw=False)
    return to_naive_date_index(out)

# ---------------- Load history --------------
today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

# FRED series
vix_f   = fred_series(FRED["vix"],       start_all, today)
hy_f    = fred_series(FRED["hy_oas"],    start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"],     start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"],  start_all, today)
spx_f   = fred_series(FRED["spx"],       start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f)  # higher = tighter

if all(s.dropna().empty for s in [vix_f, hy_f, yc_f, fund_f, spx_f]):
    st.error("FRED series unavailable. Check network.")
    st.stop()

# ETF and breadth series (HYG, LQD, SPY, RSP)
px_eq_credit = yf_history_daily(["HYG", "LQD", "SPY", "RSP"], start_all, today)
hy_lqd_s = pd.Series(dtype=float)
breadth_s = pd.Series(dtype=float)

if not px_eq_credit.empty:
    cols = list(px_eq_credit.columns)
    colset = set(cols)
    if "HYG" in colset and "LQD" in colset:
        hy_lqd_s = -np.log(px_eq_credit["HYG"] / px_eq_credit["LQD"])
    if "SPY" in colset and "RSP" in colset:
        breadth_s = px_eq_credit["SPY"] / px_eq_credit["RSP"]

# ---------------- Build BD index and align --------------
bidx_start_candidates = [s.index.min() for s in [vix_f, hy_f, yc_f, fund_f, spx_f] if not s.dropna().empty]
if bidx_start_candidates:
    bidx_start = min(bidx_start_candidates)
else:
    bidx_start = start_all

bidx = pd.bdate_range(as_naive_date(bidx_start), today)

def to_bidx(s):
    s = to_naive_date_index(s)
    return s.reindex(bidx).ffill()

panel_dict = {
    "VIX":    to_bidx(vix_f),
    "HY_OAS": to_bidx(hy_f),
    "T10Y3M": to_bidx(yc_f),
    "FUND":   to_bidx(fund_f),
    "SPX":    to_bidx(spx_f),
}

if not hy_lqd_s.dropna().empty:
    panel_dict["HY_LQD"] = to_bidx(hy_lqd_s)

if not breadth_s.dropna().empty:
    panel_dict["SPY_RSP"] = to_bidx(breadth_s)

panel = pd.DataFrame(panel_dict).dropna(how="all")
panel = to_naive_date_index(panel)

# ---------------- Freshness masks ----------- 
def fresh_mask(original: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    original = to_naive_date_index(original).reindex(idx)
    pos = np.arange(len(idx))
    has = original.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=idx).ffill().values
    age = pos - last_pos
    return pd.Series((~np.isnan(age)) & (age <= MAX_STALE_BDAYS), index=idx)

m_vix    = fresh_mask(vix_f,    panel.index)
m_hy     = fresh_mask(hy_f,     panel.index)
m_yc     = fresh_mask(yc_f,     panel.index)
m_fund   = fresh_mask(fund_f,   panel.index)
m_dd     = fresh_mask(spx_f,    panel.index)
m_hy_lqd = fresh_mask(hy_lqd_s, panel.index) if not hy_lqd_s.dropna().empty else pd.Series(False, index=panel.index)
m_breadth = fresh_mask(breadth_s, panel.index) if not breadth_s.dropna().empty else pd.Series(False, index=panel.index)

# ---------------- Trading-day index ---------
trade_idx = panel.index[panel["SPX"].notna()].unique()
panel_tr = panel.reindex(trade_idx).ffill()

# ---------------- Derived series and percentiles ---------
window_td = int(PCTL_WINDOW_YEARS * 252)

# Inverted curve and drawdown-based stress
inv_curve = -panel_tr["T10Y3M"]
roll_max = panel_tr["SPX"].dropna().cummax()
dd_all = 100.0 * (panel_tr["SPX"] / roll_max - 1.0)
stress_dd = -dd_all.clip(upper=0)

# Realized volatility (21d annualized)
ret_spx = panel_tr["SPX"].pct_change()
rv_21 = np.sqrt(252.0) * ret_spx.rolling(21, min_periods=10).std()
panel_tr["RV21"] = rv_21

# Percentiles for original 5
pct_vix    = rolling_percentile_trading(panel_tr["VIX"],     trade_idx, window_td)
pct_hy     = rolling_percentile_trading(panel_tr["HY_OAS"],  trade_idx, window_td)
pct_curve  = rolling_percentile_trading(inv_curve,           trade_idx, window_td)
pct_fund   = rolling_percentile_trading(panel_tr["FUND"],    trade_idx, window_td)
pct_dd     = rolling_percentile_trading(stress_dd,           trade_idx, window_td)

# Percentiles for new signals
if "HY_LQD" in panel_tr.columns:
    pct_hy_lqd = rolling_percentile_trading(panel_tr["HY_LQD"], trade_idx, window_td)
else:
    pct_hy_lqd = pd.Series(index=trade_idx, dtype=float)

pct_rv = rolling_percentile_trading(panel_tr["RV21"], trade_idx, window_td)

if "SPY_RSP" in panel_tr.columns:
    pct_breadth = rolling_percentile_trading(panel_tr["SPY_RSP"], trade_idx, window_td)
else:
    pct_breadth = pd.Series(index=trade_idx, dtype=float)

# Masks on trading index
m_hy_tr       = m_hy.reindex(trade_idx).fillna(False)
m_vix_tr      = m_vix.reindex(trade_idx).fillna(False)
m_yc_tr       = m_yc.reindex(trade_idx).fillna(False)
m_fund_tr     = m_fund.reindex(trade_idx).fillna(False)
m_dd_tr       = m_dd.reindex(trade_idx).fillna(False)
m_hy_lqd_tr   = m_hy_lqd.reindex(trade_idx).fillna(False)
m_breadth_tr  = m_breadth.reindex(trade_idx).fillna(False)
m_rv_tr       = panel_tr["RV21"].reindex(trade_idx).notna()

# ---------------- Scores and composite base ----------------
scores_all = pd.concat(
    [
        pct_vix,
        pct_hy,
        pct_hy_lqd,
        pct_curve,
        pct_fund,
        pct_dd,
        pct_rv,
        pct_breadth,
    ],
    axis=1
)
scores_all.columns = [
    "VIX_p",
    "HY_OAS_p",
    "HY_LQD_p",
    "CurveInv_p",
    "Fund_p",
    "DD_p",
    "RV21_p",
    "Breadth_p",
]
scores_all = to_naive_date_index(scores_all)

# ---------------- Composite -----------------
start_lb = as_naive_date(today - pd.DateOffset(years=years))
scores = scores_all.loc[start_lb:].copy()
if DEFAULT_SMOOTH > 1:
    scores = to_naive_date_index(scores.rolling(DEFAULT_SMOOTH, min_periods=1).mean())

def _align(m):
    return to_naive_date_index(m).reindex(scores.index).astype(float).fillna(0.0)

# Rebuild masks on scores index
m_vix_s      = m_vix_tr.reindex(scores.index).fillna(False)
m_hy_oas_s   = m_hy_tr.reindex(scores.index).fillna(False)
m_hy_lqd_s   = m_hy_lqd_tr.reindex(scores.index).fillna(False)
m_yc_s       = m_yc_tr.reindex(scores.index).fillna(False)
m_fund_s     = m_fund_tr.reindex(scores.index).fillna(False)
m_dd_s       = m_dd_tr.reindex(scores.index).fillna(False)
m_rv_s       = m_rv_tr.reindex(scores.index).fillna(False)
m_breadth_s  = m_breadth_tr.reindex(scores.index).fillna(False)

masks = pd.concat(
    [
        _align(m_vix_s),
        _align(m_hy_oas_s),
        _align(m_hy_lqd_s),
        _align(m_yc_s),
        _align(m_fund_s),
        _align(m_dd_s),
        _align(m_rv_s),
        _align(m_breadth_s),
    ],
    axis=1
)
masks.columns = [
    "VIX_m",
    "HY_OAS_m",
    "HY_LQD_m",
    "Curve_m",
    "Fund_m",
    "DD_m",
    "RV21_m",
    "Breadth_m",
]

W = WEIGHTS_VEC.reshape(1, -1)
X = scores[
    [
        "VIX_p",
        "HY_OAS_p",
        "HY_LQD_p",
        "CurveInv_p",
        "Fund_p",
        "DD_p",
        "RV21_p",
        "Breadth_p",
    ]
].values
M = masks.values

X_masked = np.nan_to_num(X) * M
weight_mask = (W * M)
active_w = weight_mask.sum(axis=1)
num = (X_masked * W).sum(axis=1)
comp = np.where(active_w > 0, 100.0 * num / active_w, np.nan)
comp_s = pd.Series(comp, index=scores.index, name="Composite").dropna()

if comp_s.empty:
    st.error("Composite has no valid points for the selected window.")
    st.stop()

# ---------------- Helpers -------------------
def regime_label(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= REGIME_HI:
        return "High stress"
    if x <= REGIME_LO:
        return "Low stress"
    return "Neutral"

def mean_last(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.tail(n).mean())

def regime_bucket(x: float) -> str:
    if pd.isna(x):
        return "na"
    if x >= REGIME_HI:
        return "high"
    if x <= REGIME_LO:
        return "low"
    return "neutral"

def dd_bucket(dd: float) -> str:
    # dd is positive drawdown: 0 = at peak, higher = deeper drawdown
    if pd.isna(dd):
        return "na"
    if dd < 5:
        return "shallow"
    if dd < 15:
        return "medium"
    return "deep"

def generate_commentary(
    as_of_date: pd.Timestamp,
    daily_val: float,
    weekly_val: float,
    monthly_val: float,
    spx_level: float,
    dd_val: float
) -> str:
    d_reg = regime_bucket(daily_val)
    w_reg = regime_bucket(weekly_val)
    m_reg = regime_bucket(monthly_val)
    dd_b  = dd_bucket(dd_val)

    date_str = as_of_date.date().isoformat()
    dd_pct = f"{dd_val:.1f}" if not pd.isna(dd_val) else "N/A"
    spx_str = f"{spx_level:,.1f}" if not pd.isna(spx_level) else "N/A"

    daily_map = {
        "low": (
            "The composite is signaling a low-stress tape on a daily basis, "
            "which usually lines up with constructive liquidity and tame risk premia."
        ),
        "neutral": (
            "The daily composite sits in a neutral zone, pointing to a market that is balancing "
            "local shocks against still-functioning liquidity and credit conditions."
        ),
        "high": (
            "The daily composite is in high-stress territory, which tells you that volatility, "
            "credit and curve dynamics are collectively skewed toward defense."
        ),
        "na": (
            "The daily composite cannot be interpreted cleanly today because too many inputs are missing, "
            "so the risk read has to lean more on trend and drawdown context."
        ),
    }

    weekly_map = {
        "low": (
            "On a weekly basis the regime remains calm, suggesting that the short-term shocks we see in the tape "
            "have not yet hardened into a persistent stress regime."
        ),
        "neutral": (
            "The weekly profile is neutral, which fits with a market that grinds between fear and relief "
            "without a clear directional stress trend."
        ),
        "high": (
            "The weekly composite is elevated, which means stress has been persistent rather than a single-day flare-up, "
            "often a sign to respect the trend in risk reduction rather than fade it aggressively."
        ),
        "na": (
            "The weekly regime flag is unclear, so you should lean more on the daily and monthly structure "
            "to anchor your risk-taking."
        ),
    }

    monthly_map = {
        "low": (
            "At the monthly horizon the regime is still low, telling you that the bigger backdrop remains benign "
            "even if the nearer term feels noisy."
        ),
        "neutral": (
            "At the monthly horizon the signal is neutral, which usually corresponds to a mid-cycle environment where "
            "positioning and narratives swing around a relatively stable macro core."
        ),
        "high": (
            "At the monthly horizon the composite is high, which is the kind of backdrop where deleveraging waves, "
            "risk parity de-risking and tighter financial conditions can feed on each other."
        ),
        "na": (
            "The monthly regime cannot be pinned down cleanly, so longer-term conclusions need to be made with some humility "
            "around the state of the macro regime."
        ),
    }

    dd_map = {
        "shallow": (
            f"SPX is sitting around {spx_str} with only about {dd_pct}% drawdown from the local peak, "
            "which says price has not yet delivered the kind of cleansing air-pocket that usually resets risk appetite."
        ),
        "medium": (
            f"SPX trades near {spx_str} with a mid-teens drawdown of roughly {dd_pct}%, "
            "a zone where forced selling begins to slow and new money debates whether this is a range or the start of a larger break."
        ),
        "deep": (
            f"SPX is down near {spx_str} with a deep drawdown of roughly {dd_pct}%, "
            "the kind of damage that tends to flush weak hands, create forced rebalancing and set the stage for reflexive turns "
            "if policy or data surprise positively."
        ),
        "na": (
            "The drawdown profile is not well defined here, so the composite should be read more as a cross-asset risk barometer "
            "than a precise timing tool."
        ),
    }

    daily_sentence = daily_map.get(d_reg, daily_map["na"])
    weekly_sentence = weekly_map.get(w_reg, weekly_map["na"])
    monthly_sentence = monthly_map.get(m_reg, monthly_map["na"])
    dd_sentence = dd_map.get(dd_b, dd_map["na"])

    commentary = (
        f"As of {date_str}, {daily_sentence} "
        f"{weekly_sentence} "
        f"{monthly_sentence} "
        f"{dd_sentence}"
    )
    return commentary

latest_idx = comp_s.index[-1]
latest_val = float(comp_s.iloc[-1])
weekly_val = mean_last(comp_s, 5)
monthly_val = mean_last(comp_s, 21)

# ---------------- Unified plotting index ----------------
plot_idx = comp_s.index.copy()                 # canonical x-axis for both panes
panel_plot = panel_tr.reindex(plot_idx).ffill()

spx = panel_plot["SPX"].dropna()
base = spx.iloc[0] if len(spx) else np.nan
spx_rebased = (panel_plot["SPX"] / base) * 100.0
dd_series = -100.0 * (panel_plot["SPX"] / panel_plot["SPX"].cummax() - 1.0)

y_spx = spx_rebased.reindex(plot_idx).values
y_dd  = dd_series.reindex(plot_idx).values
y_comp = comp_s.reindex(plot_idx).values

latest_spx_level = float(panel_plot["SPX"].reindex(plot_idx).iloc[-1]) if len(plot_idx) > 0 else np.nan
latest_dd_val = float(dd_series.reindex(plot_idx).iloc[-1]) if len(plot_idx) > 0 else np.nan

# ---------------- Header commentary --------------------
commentary_text = generate_commentary(
    as_of_date=latest_idx,
    daily_val=latest_val,
    weekly_val=weekly_val,
    monthly_val=monthly_val,
    spx_level=latest_spx_level,
    dd_val=latest_dd_val
)
st.info(commentary_text)

c1, c2, c3 = st.columns(3)
c1.metric("Daily regime", regime_label(latest_val), f"{latest_val:.0f}")
c2.metric("Weekly regime", regime_label(weekly_val), f"{weekly_val:.0f}" if not pd.isna(weekly_val) else "N/A")
c3.metric("Monthly regime", regime_label(monthly_val), f"{monthly_val:.0f}" if not pd.isna(monthly_val) else "N/A")

# ---------------- Combined figure ----------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.65, 0.35], specs=[[{"secondary_y": True}], [{}]],
    subplot_titles=("SPX and Drawdown (daily)", "Market Stress Composite")
)

# Row 1: SPX + Drawdown
fig.add_trace(
    go.Scatter(x=plot_idx, y=y_spx, name="SPX (rebased=100)", line=dict(width=2)),
    row=1, col=1, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=plot_idx, y=y_dd, name="Drawdown (%)", line=dict(width=1.5)),
    row=1, col=1, secondary_y=True
)

# Row 2: Composite with regime bands
fig.add_trace(
    go.Scatter(x=plot_idx, y=y_comp, name="Composite", line=dict(width=2, color="#111111")),
    row=2, col=1
)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=2, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=2, col=1)

# Axes and layout
dd_max = np.nanmax(y_dd) if np.isfinite(np.nanmax(y_dd)) else 5.0
fig.update_yaxes(title_text="Rebased", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Drawdown %", row=1, col=1, secondary_y=True, range=[0, max(5, dd_max * 1.1)])
fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 100])
fig.update_xaxes(tickformat="%b-%d-%y", title_text="Date", row=2, col=1)

fig.update_layout(
    template="plotly_white", height=760,
    legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    latest = plot_idx[-1]
    # Get helper series with safe defaults
    hy_lqd_plot = panel_plot.get("HY_LQD", pd.Series(index=plot_idx))
    spy_rsp_plot = panel_plot.get("SPY_RSP", pd.Series(index=plot_idx))
    rv_plot = panel_plot.get("RV21", pd.Series(index=plot_idx))

    scores_all_plot = scores_all.reindex(plot_idx)

    last_masks = masks.reindex(plot_idx).iloc[-1]

    diag = pd.DataFrame({
        "raw_VIX": [panel_plot.loc[latest, "VIX"]],
        "raw_HY_OAS": [panel_plot.get("HY_OAS", pd.Series(index=plot_idx)).reindex(plot_idx).loc[latest]],
        "raw_HY_LQD(-log(HYG/LQD))": [hy_lqd_plot.loc[latest]],
        "raw_curve_inv(-T10Y3M)": [(-panel_plot["T10Y3M"]).loc[latest]],
        "raw_fund(CP-TBill)": [panel_plot.loc[latest, "FUND"]],
        "raw_dd_stress": [(-100.0 * (panel_plot["SPX"] / panel_plot["SPX"].cummax() - 1.0)).loc[latest]],
        "raw_RV21": [rv_plot.loc[latest]],
        "raw_SPY_RSP(SPY/RSP)": [spy_rsp_plot.loc[latest]],
        "pct_VIX": [scores_all_plot.loc[latest, "VIX_p"]],
        "pct_HY_OAS": [scores_all_plot.loc[latest, "HY_OAS_p"]],
        "pct_HY_LQD": [scores_all_plot.loc[latest, "HY_LQD_p"]],
        "pct_CurveInv": [scores_all_plot.loc[latest, "CurveInv_p"]],
        "pct_Fund": [scores_all_plot.loc[latest, "Fund_p"]],
        "pct_DD": [scores_all_plot.loc[latest, "DD_p"]],
        "pct_RV21": [scores_all_plot.loc[latest, "RV21_p"]],
        "pct_Breadth": [scores_all_plot.loc[latest, "Breadth_p"]],
        "mask_VIX": [bool(last_masks["VIX_m"])],
        "mask_HY_OAS": [bool(last_masks["HY_OAS_m"])],
        "mask_HY_LQD": [bool(last_masks["HY_LQD_m"])],
        "mask_Curve": [bool(last_masks["Curve_m"])],
        "mask_Fund": [bool(last_masks["Fund_m"])],
        "mask_DD": [bool(last_masks["DD_m"])],
        "mask_RV21": [bool(last_masks["RV21_m"])],
        "mask_Breadth": [bool(last_masks["Breadth_m"])],
        "active_weight_sum": [float((WEIGHTS_VEC * last_masks.values).sum())],
        "Composite": [latest_val],
    })
    st.dataframe(diag.T, use_container_width=True)

# ---------------- Download ------------------
with st.expander("Download Data"):
    export_idx = plot_idx
    export_scores = (100.0 * scores.reindex(export_idx).rename(columns={
        "VIX_p": "VIX_pct",
        "HY_OAS_p": "HY_OAS_pct",
        "HY_LQD_p": "HY_LQD_pct",
        "CurveInv_p": "CurveInv_pct",
        "Fund_p": "Fund_pct",
        "DD_p": "DD_pct",
        "RV21_p": "RV21_pct",
        "Breadth_p": "Breadth_pct",
    }))

    cols = ["VIX", "HY_OAS", "T10Y3M", "FUND", "SPX"]
    if "HY_LQD" in panel_tr.columns:
        cols.append("HY_LQD")
    if "SPY_RSP" in panel_tr.columns:
        cols.append("SPY_RSP")
    if "RV21" in panel_tr.columns:
        cols.append("RV21")

    out = pd.concat(
        [panel_tr.reindex(export_idx)[cols], export_scores, comp_s.reindex(export_idx).rename("Composite")],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
