# 15_Flow_of_Funds_Matrix.py
# Flow of Funds Matrix — Actionable Playbook edition
# - Adds conviction-weighted TiltScore, Top OW/UW lists, hedge pairs
# - Regime triggers from Liquidity Pulse (z and slope)
# - Keeps your heatmap, compass, quadrant map, diagnostics

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# ---------------- FRED (optional via fredapi) ----------------
FRED_AVAILABLE, FRED = False, None
try:
    from fredapi import Fred
    fred_key = None
    try:
        fred_key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        fred_key = None
    fred_key = fred_key or os.getenv("FRED_API_KEY")
    FRED = Fred(api_key=fred_key) if fred_key else Fred()
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

if not FRED_AVAILABLE:
    st.info("FRED API key not set, using public FRED access. Higher risk of rate limits.")

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, ttl=6*3600)
def load_prices(tickers, start="2012-01-01") -> pd.DataFrame:
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

@st.cache_data(show_spinner=False, ttl=24*3600)
def fred_series(series_id: str, start="2012-01-01") -> pd.Series:
    s = pd.Series(dtype=float)
    if FRED_AVAILABLE:
        try:
            s = FRED.get_series(series_id)
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    if s.empty:
        try:
            df = pdr.DataReader(series_id, "fred", start)
            s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        except Exception:
            s = pd.Series(dtype=float)
    return s.dropna()

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
        return (s - med) / max(1.4826 * mad, 1e-12)
    std = s.std(ddof=0)
    if np.isfinite(std) and std > 1e-12:
        return (s - s.mean()) / (std + 1e-12)
    mu = s.mean()
    return (s - mu) if np.isfinite(mu) else s

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
    resid = (Y.flatten() - yhat)
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
    b_annual = b_weekly * scale
    return {"weekly": b_weekly, "annual": b_annual, "t": tser, "r2": r2, "rows": len(df)}

def choose_dtick_and_format(x_index: pd.DatetimeIndex):
    if x_index.empty:
        return "M12", "%Y"
    span_days = (x_index.max() - x_index.min()).days
    if span_days <= 370:
        return "M1", "%b %y"
    elif span_days <= 3*365:
        return "M3", "%b %y"
    elif span_days <= 7*365:
        return "M6", "%Y"
    else:
        return "M12", "%Y"

# ---------------- UI ----------------
st.set_page_config(page_title="Flow of Funds Matrix", layout="wide")
st.title("Flow of Funds Matrix")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Purpose**  
Translate liquidity betas into **positioning tilts and hedges** with clear triggers.

**How it works**  
1) Build **4w and 8w flow shocks** from FRED, robust z-scores.  
2) Regress **weekly log returns** on those shocks.  
3) Compute **LiquidityBeta** and a **conviction score** from |t|, Stability, and R².  
4) Output **TiltScore**, top OW/UW lists, **pairs**, and **triggers**.

**Reading guide**  
- Green β = outperformance on liquidity up.  
- TiltScore = LiquidityBeta × Confidence.  
- Triggers use Liquidity Pulse z and slope plus score flips.
        """
    )
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8], default=[4, 8])
    show_annual = st.toggle("Show betas annualized (×52)", value=True)
    default_assets = ["SPY","QQQ","TLT","HYG","XLE","XLF","GLD","BTC-USD"]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")
    st.markdown("---")
    st.subheader("FRED IDs")
    st.caption("Preloaded: WALCL, RRPONTSYD, WTREGEN (chg), WDTGAL (level). Add CHN or JPN reserve IDs if you have them.")
    extra = st.text_area("Extra FRED (name=ID per line)", height=100)
    st.markdown("---")
    st.subheader("Action thresholds")
    t_min = st.slider("|t|-stat min for conviction", 0.0, 3.0, 1.7, 0.1)
    stab_min = st.slider("Stability min", 0.0, 1.0, 0.50, 0.05)
    r2_cap = st.slider("R² cap for confidence", 0.05, 0.50, 0.15, 0.01)
    strong_q = st.slider("Strong tilt at top abs-quantile", 0.50, 0.95, 0.80, 0.01)
    medium_q = st.slider("Medium tilt at mid abs-quantile", 0.30, 0.80, 0.60, 0.01)

st.caption("Identify where incremental liquidity co-moves with relative performance, then turn it into tilts, hedges, and triggers.")

# Base FRED ids
fred_ids = {"WALCL":"WALCL", "RRPONTSYD":"RRPONTSYD", "WTREGEN":"WTREGEN", "WDTGAL":"WDTGAL"}
if extra.strip():
    for line in extra.splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=",1)]
            fred_ids[k] = v

# ---------------- Data ----------------
with st.spinner("Loading data"):
    px_d = load_prices(tickers, start=str(start_date))
    if px_d.empty:
        st.error("No price data for provided tickers.")
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

# Build flows (TGA_Drain sign-corrected so positive means more liquidity)
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
    st.error("No usable flows. Need WALCL and RRPONTSYD or at least WALCL proxy.")
    st.stop()

# Design matrix with fallbacks
X, valid = build_design(fund, flow_list, horizons, min_rows=12)
if X.empty or len(valid) == 0:
    X, valid = build_design(fund, flow_list, [4], min_rows=10)
if (X.empty or len(valid) == 0) and ("Fed_NetLiq_proxy" in flow_list):
    d = fund["Fed_NetLiq_proxy"] - fund["Fed_NetLiq_proxy"].shift(4)
    X = pd.DataFrame({"Fed_NetLiq_proxy_d4w": robust_scale(d)}, index=fund.index)
    valid = ["Fed_NetLiq_proxy_d4w"]
if X.empty or len(valid) == 0:
    st.error("No usable factors after fallbacks. Try start on or after 2013 and use only Fed_NetLiq.")
    st.stop()

# Weekly log returns and window
rets = np.log(px).diff()
eff_window = min(int(reg_window), len(widx))
if eff_window < 20:
    st.warning(f"Thin sample ({eff_window} weeks). Using all available rows. Interpret with caution.")

# ---------------- Per-asset regressions ----------------
betas_a, betas_w, tstats, r2s, rows_used = {}, {}, {}, {}, {}
min_rows = max(20, 5*max(1, len(valid)))

for asset in rets.columns:
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        continue
    df = df.iloc[-eff_window:]
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
    st.error("No asset met minimal overlap. Try start on or after 2013 and 4w horizon.")
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

# ---------------- Stability score (half-window consistency) ----------------
def half_window_liq_beta(asset):
    y = rets[asset]
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty or len(df) < 2*min_rows:
        return np.nan
    mid = len(df)//2
    def part_beta(dfp):
        res = ols_annualized(dfp.iloc[:, 0], dfp.iloc[:, 1:], scale=(52.0 if show_annual else 1.0))
        if res is None:
            return np.nan
        b = res["annual"] if show_annual else res["weekly"]
        return b.reindex(liq_cols).sum()
    b1 = part_beta(df.iloc[:mid])
    b2 = part_beta(df.iloc[mid:])
    if not np.isfinite(b1) or not np.isfinite(b2):
        return np.nan
    if max(abs(b1), abs(b2)) < 1e-3:
        return np.nan
    denom = abs(b1) + abs(b2) + 1e-6
    return float(np.clip(1.0 - abs(b1 - b2)/denom, 0.0, 1.0))

stability = pd.Series({a: half_window_liq_beta(a) for a in Bv.index}, name="Stability")

# ---------------- Liquidity Pulse ----------------
pos_cols = [c for c in X.columns if any(k in c for k in ["Fed_NetLiq","FX_Reserves","TGA_Drain","Fed_NetLiq_proxy"])]
pulse_series = X[pos_cols].sum(axis=1).dropna()
pulse_plot = pulse_series.loc[str(start_date):]

def zscore(s, w=26):
    r = s.rolling(w, min_periods=max(8, w//3))
    return (s - r.mean()) / (r.std(ddof=0) + 1e-12)

pulse_z = zscore(pulse_series)
pulse_slope = pulse_series.diff(4)  # 4-week slope
pulse_state = dict(
    z=float(pulse_z.iloc[-1]) if not pulse_z.empty else np.nan,
    slope=float(pulse_slope.iloc[-1]) if not pulse_slope.empty else np.nan
)

# ---------------- Conviction and TiltScore ----------------
# Confidence components: |t| clipped to [0,3], Stability [0,1], R2 capped at r2_cap and rescaled
abs_t_mean = T[liq_cols].abs().mean(axis=1).rename("|t|_mean")
t_conf = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
stab_conf = stability.fillna(0.0).clip(0.0, 1.0)
r2_conf = (R2.fillna(0.0).clip(0.0, r2_cap) / r2_cap).rename("R2_conf")

confidence = (0.5 * t_conf + 0.3 * stab_conf + 0.2 * r2_conf).rename("Confidence")
tilt_score = (liq_beta * confidence).rename("TiltScore")

# Filters for conviction
meets = (abs_t_mean >= t_min) & (stab_conf >= stab_min)
# Quantile thresholds for sizing
abs_scores = tilt_score.abs().dropna()
q_strong = abs_scores.quantile(strong_q) if not abs_scores.empty else np.nan
q_medium = abs_scores.quantile(medium_q) if not abs_scores.empty else np.nan

def bucket(score):
    if pd.isna(score):
        return "Neutral"
    a = abs(score)
    if a >= q_strong and a > 0:
        return "Strong OW" if score > 0 else "Strong UW"
    if a >= q_medium and a > 0:
        return "Medium OW" if score > 0 else "Medium UW"
    if a > 0:
        return "Small OW" if score > 0 else "Small UW"
    return "Neutral"

# ---------------- Playbook table ----------------
playbook = pd.concat([
    liq_beta, confidence, tilt_score, abs_t_mean, stab_conf.rename("Stability"), r2_conf, R2, rowsS
], axis=1)
playbook["Tilt"] = [bucket(v) if meets.get(a, False) else "Neutral" for a, v in tilt_score.items()]
playbook = playbook.sort_values("TiltScore", ascending=False)

# ---------------- Top lists and pairs ----------------
ow = playbook[(playbook["Tilt"].str.contains("OW")) & meets].copy()
uw = playbook[(playbook["Tilt"].str.contains("UW")) & meets].copy()

top_n = 5
ow_top = ow.head(top_n)
uw_top = uw.head(top_n)

def make_pairs(long_df, short_df, n=3):
    if long_df.empty or short_df.empty:
        return pd.DataFrame(columns=["Long","Short","NetScore"])
    pairs = []
    for la, ls in zip(long_df.index[:n], short_df.index[:n]):
        pairs.append([la, ls, float(long_df.loc[la,"TiltScore"] - short_df.loc[ls,"TiltScore"])])
    return pd.DataFrame(pairs, columns=["Long","Short","NetScore"])

pairs = make_pairs(ow_top, uw_top, n=min(3, len(ow_top), len(uw_top)))

# ---------------- Header metrics ----------------
st.subheader("Actionable Playbook")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidity Pulse z", f"{pulse_state['z']:+.2f}")
c2.metric("Pulse 4w slope", f"{pulse_state['slope']:+.2f}")
c3.metric("Avg Confidence", f"{confidence.mean():.2f}")
c4.metric("Coverage", f"{len(playbook)} assets")

st.caption("TiltScore = LiquidityBeta × Confidence. Confidence blends |t|-mean, Stability, and capped R².")

# ---------------- Display playbook summary ----------------
def fmt_tbl(df):
    cols = ["LiquidityBeta","Confidence","TiltScore","|t|_mean","Stability","R2_conf","R2","Rows","Tilt"]
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

st.markdown("**Pairs for expression**")
if pairs.empty:
    st.info("Insufficient conviction for pairs. Raise horizons or lower thresholds if desired.")
else:
    try:
        st.dataframe(pairs.style.format({"NetScore":"{:+.2f}"}), use_container_width=True)
    except Exception:
        st.dataframe(pairs, use_container_width=True)

# ---------------- Triggers ----------------
st.subheader("Triggers and invalidation rules")
def pulse_regime(z, slope):
    if not np.isfinite(z) or not np.isfinite(slope):
        return "N/A"
    if z >= 1.0 and slope > 0:
        return "Liquidity improving"
    if z <= -1.0 and slope < 0:
        return "Liquidity deteriorating"
    return "Mixed/neutral"

regime = pulse_regime(pulse_state["z"], pulse_state["slope"])
st.markdown(
    f"""
- **Regime:** **{regime}**.  
- **Flip risk ON** when Pulse z > +1 and 4w slope > 0, keep only **OW** tilts with Confidence above thresholds.  
- **Flip risk OFF** when Pulse z < −1 and slope < 0, keep only **UW** tilts with Confidence above thresholds.  
- **Invalidate an asset tilt** if its TiltScore changes sign or |t|-mean falls below {t_min:.1f}, or if Stability drops under {stab_min:.2f}.  
"""
)

# ---------------- Liquidity Compass ----------------
tilt = np.select(
    [liq_beta > 0.15, liq_beta > 0.05, liq_beta < -0.15, liq_beta < -0.05],
    ["Overweight risk (strong)", "Overweight risk (modest)", "Defensive or trim (strong)", "Defensive or trim (modest)"],
    default="Neutral"
)
compass = pd.DataFrame({"LiquidityBeta": liq_beta, "R2": R2.reindex(liq_beta.index), "Stability": stability, "Tilt": tilt})
compass = compass.sort_values("LiquidityBeta", ascending=False)
try:
    compass_fmt = compass.style.format({"LiquidityBeta":"{:+.2f}", "R2":"{:.2%}", "Stability":"{:.0%}"}) \
        .background_gradient(subset=["LiquidityBeta"], cmap="RdYlGn",
                             vmin=-compass["LiquidityBeta"].abs().max(), vmax=compass["LiquidityBeta"].abs().max())
    st.subheader("Liquidity Compass, where to tilt gross exposure")
    st.dataframe(compass_fmt, use_container_width=True)
except Exception:
    st.subheader("Liquidity Compass, where to tilt gross exposure")
    st.dataframe(compass, use_container_width=True)

# ---------------- Heatmap (positive = GREEN) ----------------
st.subheader(f"Liquidity beta heatmap ({unit_label})")

def relabel(col: str) -> str:
    out = (col.replace("Fed_NetLiq", "Fed NL")
               .replace("TGA_Drain", "TGA drain")
               .replace("FX_Reserves","FX reserves")
               .replace("_d"," Δ"))
    return out

label_map = {c: relabel(c) for c in Bv.columns}
Bv_disp = Bv.rename(columns=label_map)
Tmatch = T.reindex(index=Bv.index, columns=Bv.columns).rename(columns=label_map)

def star_marker(x):
    if pd.isna(x):
        return ""
    ax = abs(x)
    return "★★★" if ax >= 3.0 else ("★★" if ax >= 2.0 else ("★" if ax >= 1.7 else ""))

star_text = Tmatch.applymap(star_marker).fillna("")
custom = np.dstack([Bv_disp.values, Tmatch.values])

colors_pos_green = [
    [0.0, "#d6604d"],
    [0.5, "#f7f7f7"],
    [1.0, "#1a9850"],
]

hm = go.Figure(data=go.Heatmap(
    z=Bv_disp.values,
    x=list(Bv_disp.columns),
    y=list(Bv_disp.index),
    colorscale=colors_pos_green,
    zmid=0,
    colorbar=dict(title=unit_label),
    text=star_text.values,
    texttemplate="%{text}",
    customdata=custom,
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>"
                  + unit_label + ": %{z:.3f}"
                  + "<br>t-stat: %{customdata[1]:.2f}<extra></extra>"
))
hm.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)
st.caption("Interpretation: +0.12 beta means a +1σ liquidity shock historically coincided with about +12 percent annualized return. TGA is sign corrected.")

# ---------------- Quadrant Map ----------------
st.subheader("Allocator map, Fed NL vs TGA drain (size is R², color is LiquidityBeta)")
xcol = next((c for c in Bv.columns if c.startswith("Fed_NetLiq")), None)
ycol = next((c for c in Bv.columns if c.startswith("TGA_Drain")), None)
if xcol is not None and ycol is not None:
    xs = Bv[xcol]
    ys = Bv[ycol]
    rsq = R2.reindex(Bv.index).fillna(0).clip(0, 1)
    size = 10 + 60*rsq.clip(0, 0.2)
    scatter = go.Figure()
    scatter.add_hline(y=0, line=dict(color="#bbbbbb", dash="dot"))
    scatter.add_vline(x=0, line=dict(color="#bbbbbb", dash="dot"))
    scatter.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=Bv.index, textposition="top center",
        marker=dict(size=size,
                    color=liq_beta, colorscale=colors_pos_green,
                    cmin=-abs(liq_beta).max(), cmax=abs(liq_beta).max(), showscale=True),
        hovertemplate="<b>%{text}</b><br>"
                      + relabel(xcol) + ": %{x:.3f}<br>"
                      + relabel(ycol) + ": %{y:.3f}<br>"
                      + "R²: %{customdata:.3f}<extra></extra>",
        customdata=rsq
    ))
    scatter.update_xaxes(title=f"{relabel(xcol)} ({unit_label})")
    scatter.update_yaxes(title=f"{relabel(ycol)} ({unit_label})")
    scatter.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[
                              dict(x=0.98, y=0.98, xref="paper", yref="paper", text="risk-on quadrant", showarrow=False, font=dict(color="#1a9850")),
                              dict(x=0.02, y=0.02, xref="paper", yref="paper", text="defensive quadrant", showarrow=False, font=dict(color="#d6604d")),
                          ])
    st.plotly_chart(scatter, use_container_width=True)
else:
    st.info("Need both Fed_NetLiq_* and TGA_Drain_* factors for the quadrant map.")

# ---------------- Liquidity Pulse chart ----------------
st.subheader("Liquidity Pulse, net momentum (standardized)")
dtick, tfmt = choose_dtick_and_format(pulse_plot.index)

fig_pulse = go.Figure()
fig_pulse.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
fig_pulse.add_trace(go.Scatter(x=pulse_plot.index, y=pulse_plot, mode="lines", name="Pulse"))
if not pulse_plot.empty:
    fig_pulse.add_trace(go.Scatter(
        x=[pulse_plot.index[-1]], y=[pulse_plot.iloc[-1]],
        mode="markers+text", text=[f"{pulse_plot.iloc[-1]:+.2f}"], textposition="top center",
        showlegend=False
    ))
fig_pulse.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
fig_pulse.update_yaxes(title_text="Sum of standardized flow shocks")
fig_pulse.update_xaxes(dtick=dtick, tickformat=tfmt, hoverformat="%b %d, %Y")
st.plotly_chart(fig_pulse, use_container_width=True)

# ---------------- Downloads ----------------
with st.expander("Download results"):
    st.download_button("Betas (display units) CSV", Bdisp.to_csv().encode(), file_name="flow_betas.csv")
    st.download_button("Liquidity Compass CSV", pd.concat([playbook, rowsS], axis=1).to_csv().encode(), file_name="flow_liquidity_playbook.csv")

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    st.write("Factors used:", list(X.columns))
    st.write("Conviction components are |t|-mean, Stability, capped R².")
    diag = []
    if 'fred_df' in locals() and not fred_df.empty:
        for c in fred_df.columns:
            s = fred_df[c].dropna()
            if not s.empty:
                diag.append([c, str(s.index.min().date()), str(s.index.max().date()), int(s.notna().sum())])
    if diag:
        st.dataframe(pd.DataFrame(diag, columns=["Series","First","Last","Obs"]), use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
