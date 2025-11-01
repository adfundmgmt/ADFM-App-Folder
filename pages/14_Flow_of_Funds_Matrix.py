# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix — Actionable, always-populated version
# Outputs: Top OW/UW, Pairs, Triggers + the usual heatmap/compass

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# -------------------- Utilities --------------------
st.set_page_config(page_title="Flow of Funds Matrix", layout="wide")
st.title("Flow of Funds Matrix")

@st.cache_data(ttl=6*3600, show_spinner=False)
def load_prices(tickers, start="2012-01-01"):
    try:
        px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()
    if isinstance(px, pd.Series):
        px = px.to_frame()
    bad = [c for c in px.columns if px[c].dropna().empty]
    if bad:
        st.warning(f"No data for: {', '.join(map(str, bad))}. Excluding.")
        px = px.drop(columns=bad)
    return px.ffill()

@st.cache_data(ttl=24*3600, show_spinner=False)
def fred_series(series_id: str, start="2012-01-01") -> pd.Series:
    # Try pd_datareader first (no key required)
    try:
        s = pdr.DataReader(series_id, "fred", start)
        s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)

def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def align_to_weekly(s: pd.Series, widx: pd.DatetimeIndex) -> pd.Series:
    if s.empty:
        return s.reindex(widx)
    df = pd.DataFrame({"v": s.sort_index()})
    df = df.reindex(df.index.union(widx)).sort_index().ffill()
    return df.loc[widx, "v"]

def robust_scale(s: pd.Series):
    if s.isna().all():
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if np.isfinite(mad) and mad > 1e-12:
        return (s - med) / (1.4826*mad)
    std = s.std(ddof=0)
    return (s - s.mean()) / (std + 1e-12) if np.isfinite(std) and std > 1e-12 else s*0

def build_design(fund_df: pd.DataFrame, flows, horizons, min_rows=12):
    feats = {}
    for c in flows:
        if c not in fund_df.columns:
            continue
        for h in horizons:
            d = fund_df[c] - fund_df[c].shift(h)
            feats[f"{c}_d{h}w"] = robust_scale(d)
    if not feats:
        return pd.DataFrame(index=fund_df.index), []
    X = pd.DataFrame(feats, index=fund_df.index)
    valid = [c for c in X.columns if X[c].notna().sum() >= min_rows]
    return X[valid], valid

def ols_annualized(y: pd.Series, X: pd.DataFrame, scale=52.0):
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return None
    Y = df.iloc[:, 0].values.reshape(-1, 1)
    X0 = df.iloc[:, 1:]
    Xmat = np.column_stack([np.ones(len(df)), X0.values])
    beta_w = np.linalg.lstsq(Xmat, Y, rcond=None)[0].flatten()
    yhat = Xmat @ beta_w
    resid = Y.flatten() - yhat
    dof = max(1, (len(df) - Xmat.shape[1]))
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.pinv(Xmat.T @ Xmat)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    t = beta_w / se
    names = ["Intercept"] + list(X0.columns)
    b_weekly = pd.Series(beta_w, index=names).drop("Intercept")
    tser = pd.Series(t, index=names).drop("Intercept")
    denom = ((Y - Y.mean()) ** 2).sum() + 1e-12
    r2 = float(1.0 - (resid @ resid) / denom)
    return {"weekly": b_weekly, "annual": b_weekly*scale, "t": tser, "r2": r2, "rows": len(df)}

def choose_dtick_and_format(x_index: pd.DatetimeIndex):
    if x_index.empty: return "M12", "%Y"
    span_days = (x_index.max() - x_index.min()).days
    if span_days <= 370: return "M1", "%b %y"
    if span_days <= 3*365: return "M3", "%b %y"
    if span_days <= 7*365: return "M6", "%Y"
    return "M12", "%Y"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    show_annual = st.toggle("Show betas annualized (×52)", value=True)
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
    st.markdown("---")
    st.subheader("FRED IDs")
    st.caption("Defaults cover Fed balance sheet, RRP, TGA. Add reserve series if you like.")
    extra = st.text_area("Extra FRED (name=ID per line)", value="", height=80)
    st.markdown("---")
    st.subheader("Conviction weights")
    t_weight = st.slider("Weight of |t|-mean", 0.0, 1.0, 0.5, 0.05)
    stab_weight = st.slider("Weight of Stability", 0.0, 1.0, 0.3, 0.05)
    r2_weight = st.slider("Weight of R²", 0.0, 1.0, 0.2, 0.05)

st.caption("Identify where liquidity and relative performance co-move, then act: tilts, pairs, triggers. Always populated, no silent filters.")

# -------------------- Data --------------------
fred_ids = {"WALCL":"WALCL", "RRPONTSYD":"RRPONTSYD", "WTREGEN":"WTREGEN", "WDTGAL":"WDTGAL"}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=",1)]
            fred_ids[k] = v

with st.spinner("Loading data"):
    px_d = load_prices(tickers, start=str(start_date))
    if px_d.empty:
        st.error("No price data. Check tickers.")
        st.stop()
    widx = weekly_index(start_date)
    px = px_d.resample("W-FRI").last().reindex(widx).ffill()

    fred_raw = {}
    for name, sid in fred_ids.items():
        s = fred_series(sid, start=str(start_date))
        if not s.empty:
            fred_raw[name] = align_to_weekly(s, widx)
    fred_df = pd.DataFrame(fred_raw, index=widx) if fred_raw else pd.DataFrame(index=widx)

if px.empty:
    st.error("No price data.")
    st.stop()

fund = pd.DataFrame(index=widx)
have = set(fred_df.columns)
tga = fred_df["WTREGEN"] if "WTREGEN" in have else (fred_df["WDTGAL"] if "WDTGAL" in have else None)

if "WALCL" in have and "RRPONTSYD" in have:
    fund["Fed_NetLiq"] = fred_df["WALCL"] - fred_df["RRPONTSYD"] - (tga if tga is not None else 0)
    if tga is not None:
        fund["TGA_Drain"] = -tga
elif "WALCL" in have:
    fund["Fed_NetLiq_proxy"] = fred_df["WALCL"]

fx_cols = [c for c in fred_df.columns if c.upper().startswith(("CHN","JPN")) or c.endswith("_RES")]
if fx_cols:
    fund["FX_Reserves"] = fred_df[fx_cols].sum(axis=1)

flow_priority = [
    ["Fed_NetLiq","TGA_Drain","FX_Reserves"],
    ["Fed_NetLiq","TGA_Drain"],
    ["Fed_NetLiq"],
    ["Fed_NetLiq_proxy"],
]
flow_list = next((ok for cand in flow_priority if (ok := [c for c in cand if c in fund.columns])), [])
if not flow_list:
    st.error("No usable flows. Need WALCL and RRPONTSYD, or at least WALCL.")
    st.stop()

X, valid = build_design(fund, flow_list, horizons, min_rows=12)
if X.empty:
    st.error("No usable flow shocks built. Try a later start date.")
    st.stop()

rets = np.log(px).diff()
eff_window = min(int(reg_window), len(widx))
if eff_window < 26:
    st.warning(f"Short sample ({eff_window} weeks). Consider a longer window.")

# -------------------- Regressions --------------------
betas_a, betas_w, tstats, r2s, rows_used = {}, {}, {}, {}, {}
min_rows = max(20, 5*max(1, len(valid)))
for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna().iloc[-eff_window:]
    if df.shape[0] < min_rows:
        continue
    res = ols_annualized(df.iloc[:, 0], df.iloc[:, 1:], scale=52.0)
    if res is None:
        continue
    betas_a[asset] = res["annual"]
    betas_w[asset] = res["weekly"]
    tstats[asset] = res["t"]
    r2s[asset] = res["r2"]
    rows_used[asset] = res["rows"]

if not betas_a:
    st.error("No assets had sufficient overlap. Try fewer tickers or a later start.")
    st.stop()

B_ann = pd.DataFrame(betas_a).T
B_wk = pd.DataFrame(betas_w).T
T = pd.DataFrame(tstats).T
R2 = pd.Series(r2s, name="R2")
rowsS = pd.Series(rows_used, name="Rows")

Bdisp = B_ann if show_annual else B_wk
unit_label = "Annualized β" if show_annual else "Weekly β"

liq_cols = [c for c in Bdisp.columns if any(k in c for k in ["Fed_NetLiq","TGA_Drain","FX_Reserves","Fed_NetLiq_proxy"])]
Bv = Bdisp[liq_cols].copy()
liq_beta = Bv.sum(axis=1).rename("LiquidityBeta")

# -------------------- Confidence & Tilt --------------------
def half_window_liq_beta(asset):
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 2*min_rows:
        return np.nan
    mid = len(df)//2
    def part_beta(dfp):
        r = ols_annualized(dfp.iloc[:,0], dfp.iloc[:,1:], scale=(52.0 if show_annual else 1.0))
        if r is None: return np.nan
        b = r["annual"] if show_annual else r["weekly"]
        return b.reindex(liq_cols).sum()
    b1, b2 = part_beta(df.iloc[:mid]), part_beta(df.iloc[mid:])
    if not np.isfinite(b1) or not np.isfinite(b2): return np.nan
    denom = abs(b1) + abs(b2) + 1e-6
    return float(np.clip(1.0 - abs(b1 - b2)/denom, 0.0, 1.0))

stability = pd.Series({a: half_window_liq_beta(a) for a in Bv.index}, name="Stability")
abs_t_mean = T.reindex(columns=liq_cols).abs().mean(axis=1).rename("|t|_mean").fillna(0.0)
t_conf = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
r2_conf = (R2.fillna(0.0).clip(0.0, 0.15) / 0.15).rename("R2_conf")
stab_conf = stability.fillna(0.0).clip(0.0, 1.0)

# normalize weights
w_sum = max(1e-6, t_weight + stab_weight + r2_weight)
confidence = ((t_weight/w_sum)*t_conf + (stab_weight/w_sum)*stab_conf + (r2_weight/w_sum)*r2_conf).rename("Confidence")
tilt_score = (liq_beta.fillna(0.0) * confidence).rename("TiltScore")

playbook = pd.concat([liq_beta, confidence, tilt_score, abs_t_mean, stab_conf.rename("Stability"), r2_conf, R2, rowsS], axis=1)
playbook = playbook.sort_values("TiltScore", ascending=False)

# -------------------- Liquidity Pulse & triggers --------------------
pos_cols = [c for c in X.columns if any(k in c for k in ["Fed_NetLiq","FX_Reserves","TGA_Drain","Fed_NetLiq_proxy"])]
pulse_series = X[pos_cols].sum(axis=1).dropna()
def zscore(s, w=26):
    r = s.rolling(w, min_periods=max(8, w//3))
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)
pulse_z = zscore(pulse_series)
pulse_slope = pulse_series.diff(4)
state_z = float(pulse_z.iloc[-1]) if not pulse_z.empty else np.nan
state_slope = float(pulse_slope.iloc[-1]) if not pulse_slope.empty else np.nan

# -------------------- Header metrics --------------------
st.subheader("Actionable Playbook")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidity Pulse z", f"{state_z:+.2f}")
c2.metric("Pulse 4w slope", f"{state_slope:+.2f}")
c3.metric("Avg Confidence", f"{confidence.mean():.2f}")
c4.metric("Coverage", f"{len(playbook)} assets")

st.caption("TiltScore = LiquidityBeta × Confidence. Confidence blends |t|-mean, Stability, capped R² with your weights.")

# -------------------- Top lists & pairs (always populated) --------------------
top_n = min(5, len(playbook))
ow_top = playbook.head(top_n)
uw_top = playbook.tail(top_n).iloc[::-1]
def fmt_tbl(df):
    cols = ["LiquidityBeta","Confidence","TiltScore","|t|_mean","Stability","R2_conf","R2","Rows"]
    view = df.reindex(columns=cols)
    try:
        return view.style.format({
            "LiquidityBeta":"{:+.2f}", "Confidence":"{:.2f}", "TiltScore":"{:+.2f}",
            "|t|_mean":"{:.2f}", "Stability":"{:.0%}", "R2_conf":"{:.0%}", "R2":"{:.1%}"
        }).background_gradient(subset=["TiltScore"], cmap="RdYlGn")
    except Exception:
        return view

st.markdown("**Top Overweights**")
st.dataframe(fmt_tbl(ow_top), use_container_width=True)
st.markdown("**Top Underweights**")
st.dataframe(fmt_tbl(uw_top), use_container_width=True)

def make_pairs(long_df, short_df, n=3):
    pairs = []
    for la, ls in zip(long_df.index[:n], short_df.index[:n]):
        pairs.append([la, ls, float(long_df.loc[la,"TiltScore"] - short_df.loc[ls,"TiltScore"])])
    return pd.DataFrame(pairs, columns=["Long","Short","NetScore"])
pairs = make_pairs(ow_top, uw_top, n=min(3, top_n))
st.markdown("**Pairs for expression**")
try:
    st.dataframe(pairs.style.format({"NetScore":"{:+.2f}"}), use_container_width=True)
except Exception:
    st.dataframe(pairs, use_container_width=True)

# -------------------- Triggers --------------------
st.subheader("Triggers and invalidation rules")
def pulse_regime(z, slope):
    if not np.isfinite(z) or not np.isfinite(slope): return "N/A"
    if z >= 1.0 and slope > 0: return "Liquidity improving"
    if z <= -1.0 and slope < 0: return "Liquidity deteriorating"
    return "Mixed/neutral"
regime = pulse_regime(state_z, state_slope)
st.markdown(
    f"- **Regime:** **{regime}**. Risk-on when Pulse z > +1 and slope > 0. Risk-off when z < −1 and slope < 0.\n"
    "- **Act:** Overweight the OW list, underweight the UW list, or express via pairs above.\n"
    "- **Invalidate:** Flip or reduce if TiltScore changes sign, |t|-mean falls, or Stability drops sharply."
)

# -------------------- Heatmap (context) --------------------
st.subheader(f"Liquidity beta heatmap ({unit_label})")
def relabel(col: str) -> str:
    return (col.replace("Fed_NetLiq","Fed NL").replace("TGA_Drain","TGA drain")
               .replace("FX_Reserves","FX reserves").replace("_d"," Δ"))
label_map = {c: relabel(c) for c in Bv.columns}
Bv_disp = Bv.rename(columns=label_map)
Tmatch = T.reindex(index=Bv.index, columns=Bv.columns).rename(columns=label_map)

def star_marker(x):
    if pd.isna(x): return ""
    ax = abs(x)
    return "★★★" if ax >= 3.0 else ("★★" if ax >= 2.0 else ("★" if ax >= 1.7 else ""))

star_text = Tmatch.applymap(star_marker).fillna("")
custom = np.dstack([Bv_disp.values, Tmatch.values])
colors_pos_green = [[0.0, "#d6604d"], [0.5, "#f7f7f7"], [1.0, "#1a9850"]]
hm = go.Figure(data=go.Heatmap(
    z=Bv_disp.values, x=list(Bv_disp.columns), y=list(Bv_disp.index),
    colorscale=colors_pos_green, zmid=0, colorbar=dict(title=unit_label),
    text=star_text.values, texttemplate="%{text}",
    customdata=custom,
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>"+unit_label+": %{z:.3f}<br>t-stat: %{customdata[1]:.2f}<extra></extra>"
))
hm.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# -------------------- Pulse chart --------------------
st.subheader("Liquidity Pulse (sum of standardized flow shocks)")
pulse_plot = pulse_series.loc[str(start_date):]
dtick, tfmt = choose_dtick_and_format(pulse_plot.index)
fig_pulse = go.Figure()
fig_pulse.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
fig_pulse.add_trace(go.Scatter(x=pulse_plot.index, y=pulse_plot, mode="lines", name="Pulse"))
if not pulse_plot.empty:
    fig_pulse.add_trace(go.Scatter(x=[pulse_plot.index[-1]], y=[pulse_plot.iloc[-1]],
                                   mode="markers+text", text=[f"{pulse_plot.iloc[-1]:+.2f}"],
                                   textposition="top center", showlegend=False))
fig_pulse.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
fig_pulse.update_yaxes(title_text="Standardized shocks")
fig_pulse.update_xaxes(dtick=dtick, tickformat=tfmt, hoverformat="%b %d, %Y")
st.plotly_chart(fig_pulse, use_container_width=True)

# -------------------- Downloads --------------------
with st.expander("Download results"):
    st.download_button("Betas (display units) CSV", Bdisp.to_csv().encode(), file_name="flow_betas.csv")
    st.download_button("Playbook CSV", playbook.to_csv().encode(), file_name="flow_playbook.csv")

st.caption("© 2025 AD Fund Management LP")
