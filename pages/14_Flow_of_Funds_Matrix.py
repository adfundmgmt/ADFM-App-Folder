# 14_Cross_Asset_Compass.py
# Self-contained, yfinance-only, decision-first dashboard.
# Actionables: regime read, OW/UW tilts from trend + vol, pairs from correlation extremes,
# hedge suggestions from VIX term structure and IV-RV gap.

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ────────────────────────── Page & Style ──────────────────────────
st.set_page_config(page_title="Cross-Asset Compass", layout="wide")

# subtle CSS for cleanliness
st.markdown("""
<style>
.reportview-container { background: #ffffff; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
div[data-testid="stMetricValue"] { font-size: 24px; }
thead tr th { font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Cross-Asset Compass")

st.caption("Data source: Yahoo Finance via yfinance. Everything updates from market data only. No external APIs required.")

# ────────────────────────── Universe & Controls ──────────────────────────
DEFAULT = [
    # Equities
    "SPY","QQQ","IWM","EFA","EEM",
    # Duration & Credit
    "TLT","IEF","LQD","HYG",
    # Sectors / real assets
    "XLE","XLF","XLI","XLK","XLU","VNQ",
    # Commodities & FX proxies
    "GLD","SLV","USO","UUP",
    # Crypto
    "BTC-USD",
]
PAIR_PANEL = [
    ("^GSPC","^TNX","Equity vs 10Y"),      # rates beta
    ("^GSPC","HYG","Equity vs HY"),        # credit beta
    ("^GSPC","CL=F","Equity vs Oil"),      # growth vs oil
    ("^STOXX50E","EURUSD=X","Europe vs EURUSD"),
    ("^N225","USDJPY=X","Japan vs USDJPY"),
    ("IWM","LQD","Small Caps vs IG"),
]

with st.sidebar:
    st.header("Controls")
    user_list = st.text_input("Tickers (comma separated)", ",".join(DEFAULT)).replace(" ","").split(",")
    tickers = [t for t in user_list if t]  # sanitize
    start_date = st.date_input("Start date", pd.to_datetime("2015-01-01"))
    win_corr = st.number_input("Correlation window (days)", 10, 63, 21, step=1)
    win_rv = st.number_input("Realized vol window (days)", 10, 63, 21, step=1)
    st.markdown("---")
    st.subheader("Action thresholds")
    pos_cut = st.slider("Strong positive corr", 0.60, 0.95, 0.80, 0.05)
    neg_cut = st.slider("Strong negative corr", -0.95, -0.60, -0.80, 0.05)
    z_hot = st.slider("Hot vol Z", 1.0, 3.0, 1.5, 0.1)
    z_cold = st.slider("Cold vol Z", -3.0, -1.0, -1.5, 0.1)
    st.markdown("---")
    st.caption("Tip: keep ~20 tickers. App is resilient to missing series.")

# ────────────────────────── Data Pull ──────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_prices(tickers, start, end, interval="1d"):
    data = yf.download(
        tickers=list(dict.fromkeys(tickers)), start=start, end=end,
        interval=interval, auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()
    # normalize to wide Close
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            try:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t]
                col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
                frames.append(df[col].rename(t).to_frame())
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        res = pd.concat(frames, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])
        res = data[[col]]
        if len(tickers) == 1:
            res = res.rename(columns={col: tickers[0]})
    if res.index.tz is not None:
        res.index = res.index.tz_localize(None)
    return res.sort_index()

today = datetime.now().date()
start = str(start_date)
end = str(today + timedelta(days=1))

core_list = list(dict.fromkeys(
    tickers + [p for ab in PAIR_PANEL for p in (ab[0], ab[1])] + ["^VIX","^VIX3M","^GSPC","^TNX","CL=F","HYG"]
))
PX = fetch_prices(core_list, start, end, "1d")
if PX.empty:
    st.error("No data downloaded. Check tickers or connectivity.")
    st.stop()

RET = PX.pct_change().replace([np.inf, -np.inf], np.nan)
LOG = np.log(PX)

# helpers
def realized_vol(returns, window=21, ann=252):
    return returns.rolling(window).std() * math.sqrt(ann)

def zscore(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return s
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def pct_rank_last(s):
    s = pd.Series(s).dropna()
    return float(s.rank(pct=True).iloc[-1]*100.0) if not s.empty else np.nan

# ────────────────────────── Regime & Actionables ──────────────────────────
# 1) Equity vs Rates correlation
def roll_corr(ret_df, a, b, w):
    if a not in ret_df.columns or b not in ret_df.columns:
        return pd.Series(dtype=float)
    df = ret_df[[a,b]].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df[a].rolling(w).corr(df[b])

rho_er = roll_corr(RET, "^GSPC", "^TNX", win_corr)
rho_eh = roll_corr(RET, "^GSPC", "HYG", win_corr)
rho_eo = roll_corr(RET, "^GSPC", "CL=F", win_corr)

# 2) Vol diagnostics
RV_SPX = realized_vol(RET["^GSPC"], window=win_rv) * 100 if "^GSPC" in RET.columns else pd.Series(dtype=float)
VIX = PX.get("^VIX", pd.Series(dtype=float))
TS = (PX.get("^VIX3M", pd.Series(dtype=float)) / VIX - 1.0) if not VIX.empty else pd.Series(dtype=float)
IV_RV = (VIX - RV_SPX) if (not VIX.empty and not RV_SPX.empty) else pd.Series(dtype=float)

# Current reads
rho_er_now = float(rho_er.dropna().iloc[-1]) if not rho_er.dropna().empty else np.nan
rho_eh_now = float(rho_eh.dropna().iloc[-1]) if not rho_eh.dropna().empty else np.nan
ts_now = float(TS.dropna().iloc[-1]) if not TS.dropna().empty else np.nan
ivr_now = float(IV_RV.dropna().iloc[-1]) if not IV_RV.dropna().empty else np.nan
rv_now = float(RV_SPX.dropna().iloc[-1]) if not RV_SPX.dropna().empty else np.nan
vix_now = float(VIX.dropna().iloc[-1]) if not VIX.dropna().empty else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("SPX vs 10Y, 21D corr", "NA" if np.isnan(rho_er_now) else f"{rho_er_now:.1%}",
          delta=None if np.isnan(rho_er_now) else f"{pct_rank_last(rho_er):.0f}th pct")
k2.metric("VIX3M/VIX − 1", "NA" if np.isnan(ts_now) else f"{ts_now:+.3f}")
k3.metric("VIX − SPX RV", "NA" if np.isnan(ivr_now) else f"{ivr_now:+.1f} pts")
k4.metric("SPX 1M Realized", "NA" if np.isnan(rv_now) else f"{rv_now:.1f}%")

# Actionable narrative
bullets = []

if np.isfinite(rho_er_now):
    if rho_er_now <= -0.4:
        bullets.append("Rates rally supportive for equities; fade panic spikes in yields rather than chase.")
    elif rho_er_now >= 0.4:
        bullets.append("Rates selloff pressuring equities; respect duration shocks and keep index hedges ready.")

if np.isfinite(rho_eh_now) and rho_eh_now >= 0.5:
    bullets.append("Equity-credit coupling tight; equity drawdowns likely travel with HY widening. HYG hedges are efficient.")

if np.isfinite(ts_now):
    if ts_now < 0:
        bullets.append("VIX backwardation; demand for protection elevated. Avoid short vol without offsets.")
    elif ts_now > 0.10:
        bullets.append("Healthy VIX contango; overwriting and time spreads attractive.")

if np.isfinite(ivr_now):
    if ivr_now >= 5:
        bullets.append("Implied rich to realized; consider call overwrites, calendars, or short premium around events.")
    elif ivr_now <= -2:
        bullets.append("Implied cheap vs realized; own convexity via cheap puts or collars on high beta.")

# Trend-based OW/UW: price vs 50/200 DMA plus 1M and 3M returns
DMAs = PX.assign(
    **{f"MA{w}": PX.rolling(w).mean() for w in [50, 200]}
)
valid_cols = [c for c in tickers if c in PX.columns]
RET_1M = PX[valid_cols].pct_change(21)
RET_3M = PX[valid_cols].pct_change(63)

def trend_state(sym):
    if sym not in PX.columns:
        return None
    p = PX[sym].dropna()
    if p.empty or len(p) < 200:
        return "NA"
    ma50 = DMAs[f"MA50"][sym].iloc[-1]
    ma200 = DMAs[f"MA200"][sym].iloc[-1]
    r1 = RET_1M[sym].iloc[-1] if not RET_1M[sym].dropna().empty else np.nan
    r3 = RET_3M[sym].iloc[-1] if not RET_3M[sym].dropna().empty else np.nan
    up = p.iloc[-1] > ma50 > ma200 and (r1 > 0) and (r3 > 0)
    down = p.iloc[-1] < ma50 < ma200 and (r1 < 0) and (r3 < 0)
    if up:
        return "Uptrend"
    if down:
        return "Downtrend"
    return "Mixed"

trend_table = pd.DataFrame({
    "Asset": valid_cols,
    "State": [trend_state(s) for s in valid_cols],
    "1M": [RET_1M[s].dropna().iloc[-1] if s in RET_1M.columns and not RET_1M[s].dropna().empty else np.nan for s in valid_cols],
    "3M": [RET_3M[s].dropna().iloc[-1] if s in RET_3M.columns and not RET_3M[s].dropna().empty else np.nan for s in valid_cols],
})
# realized vol z-scores
RV = realized_vol(RET[valid_cols], window=win_rv) * 100
rv_latest = RV.tail(1).T.rename(columns={RV.index[-1]: "RV%"}).round(1) if not RV.empty else pd.DataFrame()

if not rv_latest.empty:
    rv_z = RV.apply(zscore).tail(1).T.rename(columns={RV.index[-1]: "RV_z"})
    trend_table = trend_table.set_index("Asset").join(rv_latest).join(rv_z)
    trend_table = trend_table.reset_index()

# Compose OW/UW recommendations
ow = trend_table[(trend_table["State"] == "Uptrend") & (trend_table["RV_z"] <= z_hot)].copy() if "RV_z" in trend_table.columns else trend_table[trend_table["State"]=="Uptrend"].copy()
uw = trend_table[(trend_table["State"] == "Downtrend") & (trend_table["RV_z"] <= z_hot)].copy() if "RV_z" in trend_table.columns else trend_table[trend_table["State"]=="Downtrend"].copy()

# Narrative from OW/UW
if not ow.empty:
    bullets.append("Overweights: " + ", ".join(ow.sort_values(["3M","1M"], ascending=False)["Asset"].head(5).tolist()))
if not uw.empty:
    bullets.append("Underweights: " + ", ".join(uw.sort_values(["3M","1M"], ascending=True)["Asset"].head(5).tolist()))

# Print decision box
st.subheader("Decision box")
if bullets:
    st.markdown("\n".join(f"- {b}" for b in bullets))
else:
    st.info("No strong signals at current thresholds. Use panels below.")

# ────────────────────────── Trend & Vol Panel ──────────────────────────
st.subheader("Cross-asset trend and vol snapshot")

def tbl_format(df):
    df2 = df.copy()
    for c in ["1M","3M"]:
        if c in df2.columns:
            df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    if "RV%" in df2.columns:
        df2["RV%"] = df2["RV%"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "RV_z" in df2.columns:
        df2["RV_z"] = df2["RV_z"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
    return df2

if not trend_table.empty:
    show = trend_table[["Asset","State","1M","3M"] + ([ "RV%","RV_z"] if "RV_z" in trend_table.columns else [])]
    show = show.sort_values(["State","3M"], ascending=[True, False])
    st.dataframe(tbl_format(show), use_container_width=True, height=min(500, 36 + 28*len(show)))

# ────────────────────────── Correlation Regime ──────────────────────────
st.subheader("Correlation regime with context")

fig = make_subplots(rows=2, cols=3, shared_xaxes=False, shared_yaxes=False,
                    vertical_spacing=0.18, horizontal_spacing=0.08,
                    specs=[[{"type":"scatter"},{"type":"scatter"},{"type":"scatter"}],
                           [{"type":"scatter"},{"type":"scatter"},{"type":"scatter"}]])

pairs = PAIR_PANEL
series = [roll_corr(RET, a, b, win_corr) for a,b,_ in pairs]

def add_corr(ax_row, ax_col, s, title):
    s = s.dropna()
    if s.empty:
        return
    mean = s.rolling(126).mean()
    std = s.rolling(126).std()
    r = ax_row; c = ax_col
    fig.add_trace(go.Scatter(x=s.index, y=mean, name="6M mean", line=dict(width=1.3)), row=r, col=c)
    fig.add_trace(go.Scatter(x=s.index, y=s, name="21D corr", line=dict(width=1.0)), row=r, col=c)
    fig.add_trace(go.Scatter(x=s.index, y=(mean+std), name="band+", line=dict(width=0.5, dash="dot")), row=r, col=c)
    fig.add_trace(go.Scatter(x=s.index, y=(mean-std), name="band−", line=dict(width=0.5, dash="dot")), row=r, col=c)
    fig.add_hline(y=0, line=dict(color="#bbbbbb", width=1, dash="dot"), row=r, col=c)
    fig.update_yaxes(range=[-1, 1], tickformat=".0%", row=r, col=c, title=title)

for i, ((a,b,t), s) in enumerate(zip(pairs, series)):
    add_corr(1 if i < 3 else 2, (i % 3)+1, s, t)

fig.update_layout(height=560, margin=dict(l=40, r=10, t=30, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ────────────────────────── Pair ideas from extremes ──────────────────────────
st.subheader("Pairs from correlation extremes")

def matrix_extremes(ret_df, tickers, window=21, pos_cut=0.8, neg_cut=-0.8):
    if len(ret_df) < window:
        return [], []
    corr = ret_df.tail(window).corr().replace([-np.inf, np.inf], np.nan)
    order = [t for t in tickers if t in corr.columns]
    pos, neg = [], []
    for i in range(1, len(order)):
        for j in range(0, i):
            a, b = order[i], order[j]
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            if val >= pos_cut:
                pos.append((a, b, float(val)))
            if val <= neg_cut:
                neg.append((a, b, float(val)))
    pos = sorted(pos, key=lambda x: -x[2])[:6]
    neg = sorted(neg, key=lambda x: x[2])[:6]
    return pos, neg

pos_ext, neg_ext = matrix_extremes(RET, valid_cols, win_corr, pos_cut, neg_cut)

def pairs_table(tag, rows):
    if not rows:
        st.info(f"No {tag} pairs beyond threshold.")
        return
    df = pd.DataFrame(rows, columns=["A","B","ρ(1M)"])
    df["ρ(1M)"] = df["ρ(1M)"].map(lambda x: f"{x:.2f}")
    st.dataframe(df, use_container_width=True, height=min(60 + 32*len(df), 360))

cL, cR = st.columns(2)
with cL:
    st.markdown("**Strong positives** — favor relative value over outright hedges")
    pairs_table("strong positive", pos_ext)
with cR:
    st.markdown("**Strong negatives** — clean hedge and dispersion candidates")
    pairs_table("strong negative", neg_ext)

# ────────────────────────── Vol overlays ──────────────────────────
st.subheader("Vol overlays")

vol_fig = make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.08)
# SPX: RV vs VIX
if not RV_SPX.dropna().empty and not VIX.dropna().empty:
    vol_fig.add_trace(go.Scatter(x=RV_SPX.index, y=RV_SPX, name="SPX 1M RV", line=dict(width=1.4)), row=1, col=1)
    vol_fig.add_trace(go.Scatter(x=VIX.index, y=VIX, name="VIX", line=dict(width=1.1)), row=1, col=1)
    vol_fig.update_yaxes(title="%", row=1, col=1)

# Oil: realized vs OVX if present
OVX = PX.get("^OVX", pd.Series(dtype=float))
if "CL=F" in RET.columns:
    RV_OIL = realized_vol(RET["CL=F"], window=win_rv) * 100
    vol_fig.add_trace(go.Scatter(x=RV_OIL.index, y=RV_OIL, name="WTI 1M RV", line=dict(width=1.4)), row=1, col=2)
    if not OVX.empty:
        vol_fig.add_trace(go.Scatter(x=OVX.index, y=OVX, name="OVX", line=dict(width=1.1)), row=1, col=2)

# FX: pick first FX proxy present
fx_choice = None
for fx in ["USDJPY=X","EURUSD=X","GBPUSD=X"]:
    if fx in RET.columns:
        fx_choice = fx
        break
if fx_choice:
    RV_FX = realized_vol(RET[fx_choice], window=win_rv) * 100
    vol_fig.add_trace(go.Scatter(x=RV_FX.index, y=RV_FX, name=f"{fx_choice} 1M RV", line=dict(width=1.4)), row=1, col=3)

vol_fig.update_layout(height=360, margin=dict(l=40, r=10, t=10, b=10), legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
st.plotly_chart(vol_fig, use_container_width=True)

# ────────────────────────── Weekly summary at bottom ──────────────────────────
st.subheader("This week vs last: correlations and vol")

# weekly corr changes for key pairs
def weekly_last_val(s):
    # week ending Friday
    return s.resample("W-FRI").last()

rows = []
for a,b,title in PAIR_PANEL:
    rc = roll_corr(RET, a, b, win_corr)
    if rc.empty:
        continue
    w = weekly_last_val(rc).dropna()
    if w.empty:
        continue
    now = w.iloc[-1]
    prev = w.iloc[-2] if len(w) >= 2 else np.nan
    d = now - prev if np.isfinite(prev) else np.nan
    pct = pct_rank_last(rc)
    rows.append([title, a, b, now, d, pct])

wk_corr = pd.DataFrame(rows, columns=["Pair","A","B","ρ_now","Δρ_w/w","Percentile"]).sort_values("Δρ_w/w", ascending=True)
if not wk_corr.empty:
    wk_corr_fmt = wk_corr.copy()
    wk_corr_fmt["ρ_now"] = wk_corr_fmt["ρ_now"].map(lambda x: f"{x:.2%}")
    wk_corr_fmt["Δρ_w/w"] = wk_corr_fmt["Δρ_w/w"].map(lambda x: f"{x:+.2%}")
    wk_corr_fmt["Percentile"] = wk_corr_fmt["Percentile"].map(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
    st.dataframe(wk_corr_fmt, use_container_width=True, height=min(60+32*len(wk_corr_fmt), 360))
else:
    st.info("Weekly correlation deltas unavailable with current window.")

# weekly vol changes across a small panel
series_map = {}
if not VIX.dropna().empty:    series_map["VIX"] = VIX
if "^OVX" in PX.columns:      series_map["OVX"] = PX["^OVX"]
if not RV_SPX.dropna().empty: series_map["SPX 1M RV"] = RV_SPX
if "HYG" in RET.columns:      series_map["HY 1M RV"] = realized_vol(RET["HYG"], window=win_rv)*100
if "CL=F" in RET.columns:     series_map["WTI 1M RV"] = realized_vol(RET["CL=F"], window=win_rv)*100
if fx_choice:                 series_map[f"{fx_choice} 1M RV"] = realized_vol(RET[fx_choice], window=win_rv)*100

vol_rows = []
for name, s in series_map.items():
    w = weekly_last_val(s.dropna())
    if w.empty:
        continue
    now = w.iloc[-1]
    prev = w.iloc[-2] if len(w) >= 2 else np.nan
    dz = now - prev if np.isfinite(prev) else np.nan
    vol_rows.append([name, now, dz, pct_rank_last(s)])

vol_tbl = pd.DataFrame(vol_rows, columns=["Series","Level","Δ w/w","Percentile"])
if not vol_tbl.empty:
    vol_tbl["Level"] = vol_tbl["Level"].map(lambda x: f"{x:.1f}")
    vol_tbl["Δ w/w"] = vol_tbl["Δ w/w"].map(lambda x: f"{x:+.1f}")
    vol_tbl["Percentile"] = vol_tbl["Percentile"].map(lambda x: f"{x:.0f}%")
    st.dataframe(vol_tbl.sort_values("Series"), use_container_width=True, height=min(60+32*len(vol_tbl), 360))

# ────────────────────────── Downloads ──────────────────────────
with st.expander("Download tables"):
    if not trend_table.empty:
        st.download_button("Trend snapshot CSV", trend_table.to_csv(index=False).encode(), file_name="trend_snapshot.csv")
    if not wk_corr.empty:
        st.download_button("Weekly correlation deltas CSV", wk_corr.to_csv(index=False).encode(), file_name="weekly_corr_deltas.csv")
    if not vol_tbl.empty:
        st.download_button("Weekly vol deltas CSV", vol_tbl.to_csv(index=False).encode(), file_name="weekly_vol_deltas.csv")

st.caption("© 2025 AD Fund Management LP")
