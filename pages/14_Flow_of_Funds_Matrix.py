import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# -------------------- App --------------------
st.set_page_config(page_title="Flow of Funds Matrix", layout="wide")
st.title("Flow of Funds Matrix")

# -------------------- Utils --------------------
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

def weekly_index(start, end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(pd.to_datetime(start), end, freq="W-FRI")

def robust_scale(s: pd.Series):
    if s.isna().all():
        return s
    med = s.median()
    mad = (s - med).abs().median()
    if np.isfinite(mad) and mad > 1e-12:
        return (s - med) / (1.4826 * mad)
    std = s.std(ddof=0)
    return (s - s.mean()) / (std + 1e-12) if np.isfinite(std) and std > 1e-12 else s * 0

def build_design_from_proxies(px_w: pd.DataFrame, horizons, min_rows=12):
    """
    Build standardized delta features from market proxies only.
    """
    # Define proxy columns we will compute from prices
    need = ["TLT","SHY","HYG","IEF","UUP","USO","GLD","RSP","SPY","BTC-USD","ETH-USD"]
    have = [c for c in need if c in px_w.columns]
    if len(have) == 0:
        return pd.DataFrame(index=px_w.index), []

    # Log prices for additive returns
    L = np.log(px_w[have]).copy()

    # Base single-asset drivers
    drivers = {}
    def add_deltas(base: pd.Series, name_prefix: str):
        for h in horizons:
            d = base - base.shift(h)
            drivers[f"{name_prefix}_d{h}w"] = robust_scale(d)

    # Duration, front-end, credit, FX, commodities, breadth, crypto
    if "TLT" in have:      add_deltas(L["TLT"], "TLT")
    if "SHY" in have:      add_deltas(L["SHY"], "SHY")
    if set(["TLT","SHY"]).issuperset({"TLT","SHY"}) and all(x in have for x in ["TLT","SHY"]):
        add_deltas(L["TLT"] - L["SHY"], "Curve_TLT_minus_SHY")
    if all(x in have for x in ["HYG","IEF"]):
        add_deltas(L["HYG"] - L["IEF"], "Credit_HYG_minus_IEF")
    if "UUP" in have:
        add_deltas(-L["UUP"], "USD_inv")  # weaker USD = looser global conditions
    if "USO" in have:      add_deltas(L["USO"], "Oil_USO")
    if "GLD" in have:      add_deltas(L["GLD"], "Gold_GLD")
    if all(x in have for x in ["RSP","SPY"]):
        add_deltas(L["RSP"] - L["SPY"], "Breadth_RSP_minus_SPY")
    if "BTC-USD" in have:  add_deltas(L["BTC-USD"], "Crypto_BTC")
    if "ETH-USD" in have:  add_deltas(L["ETH-USD"], "Crypto_ETH")

    if not drivers:
        return pd.DataFrame(index=px_w.index), []

    X = pd.DataFrame(drivers, index=px_w.index)
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
    st.header("About This Tool")
    st.markdown(
        """
        Maps how cross asset flows show up in weekly return betas for your chosen assets.

        Philosophy: treat duration, curve, credit, USD, commodities, breadth and crypto as a common proxy
        factor set, and regress each asset on those shocks to recover a stable LiquidityBeta plus a
        confidence score.

        Design
        
        • Weekly closes only from Yahoo Finance  
        • Proxies built from log price differences over several horizons (4w, 8w, 12w)  
        • Regressions on weekly log returns over a rolling window  
        • LiquidityBeta is the sum of betas to the key proxy blocks  
        • Confidence blends average |t|, stability of LiquidityBeta across time, and a capped R²

        Outputs
        
        • TiltScore = LiquidityBeta × Confidence, used to rank overweights and underweights  
        • Heatmap shows the sign and size of betas to each proxy family  
        • Liquidity Pulse tracks recent standardized shocks to the proxy set

        Use cases
        
        • See which assets lean into easing versus tightening liquidity regimes  
        • Build long short expressions that align with the current flow of funds regime
        """
    )

    st.markdown("---")
    st.header("Settings")
    start_date = st.date_input("Start date", pd.to_datetime("2012-01-01"))
    reg_window = st.number_input("Regression window (weeks)", 52, 520, 156, step=13)
    horizons = st.multiselect("Delta horizons (weeks)", [4, 8, 12], default=[4, 8])
    show_annual = st.toggle("Show betas annualized (×52)", value=True)

    # Target assets to rank
    default_assets = [
        "SPY","QQQ","IWM","EFA","EEM","SMH","XLE","XLF","XLU","XLB","XLI",
        "TLT","IEF","HYG","GLD","SLV","USO","UUP","URA","BTC-USD","ETH-USD"
    ]
    tickers = st.text_input("Assets (comma separated)", ",".join(default_assets)).replace(" ","").split(",")

    st.markdown("---")
    st.subheader("Conviction weights")
    t_weight = st.slider("Weight of |t|-mean", 0.0, 1.0, 0.5, 0.05)
    stab_weight = st.slider("Weight of Stability", 0.0, 1.0, 0.3, 0.05)
    r2_weight = st.slider("Weight of R²", 0.0, 1.0, 0.2, 0.05)

st.caption("Market-proxy design: duration, curve, credit, USD, commodities, breadth, and crypto. No FRED or keys.")

# -------------------- Data --------------------
with st.spinner("Loading market data"):
    px_d = load_prices(tickers + ["TLT","SHY","HYG","IEF","UUP","USO","GLD","RSP","SPY","BTC-USD","ETH-USD"], start=str(start_date))
    if px_d.empty:
        st.error("No price data. Check tickers.")
        st.stop()
    widx = weekly_index(start_date)
    px = px_d.resample("W-FRI").last().reindex(widx).ffill()

if px.empty:
    st.error("No price data.")
    st.stop()

# -------------------- Design matrix from proxies --------------------
X, valid = build_design_from_proxies(px, horizons, min_rows=12)
if X.empty:
    st.error("Could not build proxy flows. Try a later start date or add more proxies.")
    st.stop()

rets = np.log(px[tickers]).diff()
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
    res = ols_annualized(df.iloc[:,0], df.iloc[:,1:], scale=52.0)
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
B_wk  = pd.DataFrame(betas_w).T
Tstat = pd.DataFrame(tstats).T
R2    = pd.Series(r2s, name="R2")
rowsS = pd.Series(rows_used, name="Rows")

Bdisp = B_ann if show_annual else B_wk
unit_label = "Annualized β" if show_annual else "Weekly β"

# Aggregate a single LiquidityBeta from key proxy groups
key_cols = [c for c in Bdisp.columns if any(k in c for k in [
    "TLT_", "SHY_", "Curve_TLT_minus_SHY_", "Credit_HYG_minus_IEF_",
    "USD_inv_", "Oil_USO_", "Gold_GLD_", "Breadth_RSP_minus_SPY_", "Crypto_"
])]
Bv = Bdisp[key_cols].copy()
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
        return b.reindex(key_cols).sum()
    b1, b2 = part_beta(df.iloc[:mid]), part_beta(df.iloc[mid:])
    if not np.isfinite(b1) or not np.isfinite(b2): return np.nan
    denom = abs(b1) + abs(b2) + 1e-6
    return float(np.clip(1.0 - abs(b1 - b2)/denom, 0.0, 1.0))

stability = pd.Series({a: half_window_liq_beta(a) for a in Bv.index}, name="Stability")
abs_t_mean = Tstat.reindex(columns=key_cols).abs().mean(axis=1).rename("|t|_mean").fillna(0.0)
t_conf = np.clip(abs_t_mean, 0.0, 3.0) / 3.0
r2_conf = (R2.fillna(0.0).clip(0.0, 0.15) / 0.15).rename("R2_conf")
stab_conf = stability.fillna(0.0).clip(0.0, 1.0)

w_sum = max(1e-6, t_weight + stab_weight + r2_weight)
confidence = ((t_weight/w_sum)*t_conf + (stab_weight/w_sum)*stab_conf + (r2_weight/w_sum)*r2_conf).rename("Confidence")
tilt_score = (liq_beta.fillna(0.0) * confidence).rename("TiltScore")

playbook = pd.concat([liq_beta, confidence, tilt_score, abs_t_mean, stab_conf.rename("Stability"), r2_conf, R2, rowsS], axis=1)
playbook = playbook.sort_values("TiltScore", ascending=False)

# -------------------- Pulse & triggers from proxy shocks --------------------
pos_cols = [c for c in X.columns if any(k in c for k in [
    "TLT_", "Curve_TLT_minus_SHY_", "Credit_HYG_minus_IEF_", "USD_inv_", "Oil_USO_", "Gold_GLD_", "Breadth_RSP_minus_SPY_", "Crypto_"
])]
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

# -------------------- Top lists & pairs --------------------
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

# -------------------- Heatmap --------------------
st.subheader(f"Liquidity beta heatmap ({unit_label})")

def relabel(col: str) -> str:
    return (col.replace("Curve_TLT_minus_SHY","Curve")
               .replace("Credit_HYG_minus_IEF","Credit")
               .replace("USD_inv","USD inv")
               .replace("Oil_USO","Oil")
               .replace("Gold_GLD","Gold")
               .replace("Breadth_RSP_minus_SPY","Breadth")
               .replace("Crypto_BTC","Crypto BTC")
               .replace("Crypto_ETH","Crypto ETH")
               .replace("_d"," Δ"))

label_map = {c: relabel(c) for c in Bv.columns}
Bv_disp = Bv.rename(columns=label_map)
Tmatch = Tstat.reindex(index=Bv.index, columns=Bv.columns).rename(columns=label_map)

hm = go.Figure(data=go.Heatmap(
    z=Bv_disp.values,
    x=list(Bv_disp.columns),
    y=list(Bv_disp.index),
    colorscale=[[0.0, "#d6604d"], [0.5, "#f7f7f7"], [1.0, "#1a9850"]],
    zmid=0,
    colorbar=dict(title=unit_label),
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>"+unit_label+": %{z:.3f}<extra></extra>"
))
hm.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(hm, use_container_width=True)

# -------------------- Pulse chart --------------------
st.subheader("Liquidity Pulse (sum of standardized proxy shocks)")
pulse_plot = pulse_series.loc[str(start_date):]
dtick, tfmt = choose_dtick_and_format(pulse_plot.index)
fig_pulse = go.Figure()
fig_pulse.add_hline(y=0, line=dict(color="#cccccc", dash="dot"))
fig_pulse.add_trace(go.Scatter(x=pulse_plot.index, y=pulse_plot, mode="lines", name="Pulse"))
if not pulse_plot.empty:
    fig_pulse.add_trace(go.Scatter(
        x=[pulse_plot.index[-1]], y=[pulse_plot.iloc[-1]],
        mode="markers+text", text=[f"{pulse_plot.iloc[-1]:+.2f}"],
        textposition="top center", showlegend=False
    ))
fig_pulse.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
fig_pulse.update_yaxes(title_text="Standardized shocks")
fig_pulse.update_xaxes(dtick=dtick, tickformat=tfmt, hoverformat="%b %d, %Y")
st.plotly_chart(fig_pulse, use_container_width=True)

# -------------------- Downloads --------------------
with st.expander("Download results"):
    st.download_button("Betas (display units) CSV", Bdisp.to_csv().encode(), file_name="flow_betas.csv")
    st.download_button("Playbook CSV", playbook.to_csv().encode(), file_name="flow_playbook.csv")

st.caption("© 2025 AD Fund Management LP")
