# flow_of_funds_matrix.py
import os
import math
import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page config and styling
# -----------------------------
st.set_page_config(
    page_title="Flow of Funds Matrix",
    page_icon="💧",
    layout="wide"
)

PASTEL_BG = "#f7f8fc"
PASTEL_CARD = "white"
PASTEL_POS = "#9ad1bc"   # gentle green
PASTEL_NEG = "#f5a6a6"   # gentle red
PASTEL_NEU = "#cfd8e3"   # soft blue gray
PASTEL_ACCENT = "#93c5fd"  # light blue

st.markdown(
    f'''
    <style>
    .stApp {{
        background: linear-gradient(180deg, {PASTEL_BG} 0%, #ffffff 100%);
    }}
    .metric-card {{
        background: {PASTEL_CARD};
        border: 1px solid #e6e8f0;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }}
    .matrix-card {{
        background: {PASTEL_CARD};
        border: 1px solid #e6e8f0;
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }}
    .section-title {{
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: .2px;
        color: #0f172a;
        margin-bottom: 8px;
    }}
    .soft {{
        color: #475569;
        font-size: 0.95rem;
    }}
    .takeaways {{
        background: #fbfdff;
        border-left: 4px solid {PASTEL_ACCENT};
        padding: 10px 14px;
        border-radius: 8px;
    }}
    .good {{ color: #166534; font-weight: 600; }}
    .bad {{ color: #7f1d1d; font-weight: 600; }}
    .neutral {{ color: #334155; font-weight: 600; }}
    </style>
    ''',
    unsafe_allow_html=True
)

# -----------------------------
# Parameters
# -----------------------------
TODAY = dt.date.today()
DEFAULT_YEARS = 5
START_DATE = TODAY - dt.timedelta(days=DEFAULT_YEARS * 365)

HORIZONS = {
    "1D": 1,
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252
}

# Asset universe grouped for the Flow of Funds view
UNIVERSE = {
    "Equities": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000",
        "EFA": "Developed ex US",
        "EEM": "Emerging Markets",
        "SMH": "Semiconductors",
        "XLK": "Tech",
        "XLF": "Financials",
        "XLE": "Energy"
    },
    "Rates & Credit": {
        # Rates via FRED (handled separately): 2s, 10s, breakevens, HY OAS, IG OAS
        "IEF": "US 7-10y Treasuries ETF",
        "TLT": "US 20y+ Treasuries ETF",
        "HYG": "US High Yield ETF",
        "LQD": "US Investment Grade ETF"
    },
    "FX": {
        "^DXY": "US Dollar Index",
        "EURUSD=X": "EURUSD",
        "JPY=X": "USDJPY"
    },
    "Commodities": {
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "WTI Oil",
        "UNG": "US Nat Gas",
        "DBC": "Broad Commodities"
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum"
    },
    "Cash": {
        "BIL": "T-Bills 1-3M"
    },
    "Vol": {
        "^VIX": "VIX"
    }
}

# FRED series used for macro box and rate/credit context
FRED_SERIES = {
    "DGS2": "UST 2y Yield (%)",
    "DGS10": "UST 10y Yield (%)",
    "T10YIE": "10y Breakeven (%)",
    "BAMLH0A0HYM2": "HY OAS (bps)",
    "BAMLC0A0CM": "IG OAS (bps)"
}

# -----------------------------
# Data loaders with caching
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo_data(tickers, start):
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by='ticker')
    frames = []
    for t in tickers:
        if t not in df.columns.get_level_values(0):
            continue
        sub = df[t][["Close", "Volume"]].copy()
        sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
        frames.append(sub)
    if frames:
        out = pd.concat(frames, axis=1)
        return out.sort_index()
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_fred_series(series_codes, start):
    out = {}
    for code in series_codes:
        try:
            s = pdr.DataReader(code, "fred", start)
            out[code] = s[code]
        except Exception:
            out[code] = pd.Series(dtype=float, name=code)
    return pd.DataFrame(out).dropna(how="all")

def rolling_zscore(s, window=63):
    return (s - s.rolling(window).mean()) / (s.rolling(window).std(ddof=0) + 1e-9)

def vol_percentile(vol, window=60):
    return vol.rolling(window).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1] if len(a) > 0 else np.nan)

# -----------------------------
# Compute signals
# -----------------------------
def compute_price_panels(yh):
    if yh.empty:
        return pd.DataFrame(), pd.DataFrame()
    close_cols = [c for c in yh.columns if c[1] == "Close"]
    vol_cols = [c for c in yh.columns if c[1] == "Volume"]
    px = yh[close_cols].copy()
    px.columns = [c[0] for c in close_cols]
    vol = yh[vol_cols].copy()
    vol.columns = [c[0] for c in vol_cols]
    return px, vol

def compute_return_matrix(px, horizons):
    return {name: px.pct_change(n) for name, n in horizons.items()}

def compute_volume_metrics(vol):
    dv = vol
    vp = vol_percentile(dv, window=60)
    return dv, vp

def compute_flow_score(px, vol, horizons):
    ret_5d = px.pct_change(horizons["5D"])
    ret_1m = px.pct_change(horizons["1M"])
    ret_3m = px.pct_change(horizons["3M"])
    volp = vol_percentile(vol, window=60)
    z5 = ret_5d.apply(rolling_zscore, window=63)
    z1m = ret_1m.apply(rolling_zscore, window=126)
    z3m = ret_3m.apply(rolling_zscore, window=252)
    zv = volp.apply(rolling_zscore, window=126)
    return 0.45*z5 + 0.35*z1m + 0.15*z3m + 0.05*zv

def tidy_matrix(flow_score, groups_dict):
    latest = flow_score.dropna(how="all").iloc[-1].dropna()
    rows = []
    for group, members in groups_dict.items():
        for tkr, label in members.items():
            if tkr in latest.index:
                rows.append({"Group": group, "Ticker": tkr, "Label": label, "FlowScore": latest[tkr]})
    return pd.DataFrame(rows).sort_values(["Group", "FlowScore"], ascending=[True, False])

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 2, 15, DEFAULT_YEARS, 1)
    START = TODAY - dt.timedelta(days=365*years)

    show_groups = st.multiselect(
        "Asset groups",
        options=list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys())
    )
    horizon_cols = st.multiselect(
        "Horizons for return matrix",
        options=list(HORIZONS.keys()),
        default=["1D","5D","1M","3M","6M","1Y"]
    )

# -----------------------------
# Load data
# -----------------------------
tickers = sorted({t for g in show_groups for t in UNIVERSE[g].keys()})

with st.spinner("Loading market data..."):
    yh = load_yahoo_data(tickers, START)
    fred = load_fred_series(FRED_SERIES.keys(), START)

px, vol = compute_price_panels(yh)

# -----------------------------
# Macro regime header
# -----------------------------
st.markdown("### Macro regime snapshot")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Treasury curve</div>', unsafe_allow_html=True)
    if not fred.empty and set(["DGS2","DGS10"]).issubset(fred.columns):
        d2 = fred["DGS2"].dropna().iloc[-1]
        d10 = fred["DGS10"].dropna().iloc[-1]
        spread = d10 - d2
        st.metric("2s", f"{d2:.2f}%")
        st.metric("10s", f"{d10:.2f}%")
        st.metric("2s10s", f"{spread:.2f} pp")
    else:
        st.write("FRED unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Breakeven inflation</div>', unsafe_allow_html=True)
    if "T10YIE" in fred.columns and not fred["T10YIE"].dropna().empty:
        bre = fred["T10YIE"].dropna().iloc[-1]
        st.metric("10y BE", f"{bre:.2f}%")
    else:
        st.write("FRED unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

with colC:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Credit spreads</div>', unsafe_allow_html=True)
    if "BAMLH0A0HYM2" in fred.columns and not fred["BAMLH0A0HYM2"].dropna().empty:
        hy = fred["BAMLH0A0HYM2"].dropna().iloc[-1]
        st.metric("HY OAS", f"{hy:.0f} bps")
    if "BAMLC0A0CM" in fred.columns and not fred["BAMLC0A0CM"].dropna().empty:
        ig = fred["BAMLC0A0CM"].dropna().iloc[-1]
        st.metric("IG OAS", f"{ig:.0f} bps")
    st.markdown('</div>', unsafe_allow_html=True)

with colD:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Equity trend & vol</div>', unsafe_allow_html=True)
    if "SPY" in px.columns and not px["SPY"].dropna().empty:
        spy = px["SPY"].dropna()
        st.metric("SPY", f"{spy.iloc[-1]:.2f}")
        for w in [21,50,200]:
            st.caption(f"DMA{w}: {spy.rolling(w).mean().iloc[-1]:.2f}")
    if "^VIX" in px.columns and not px["^VIX"].dropna().empty:
        vix = px["^VIX"].dropna().iloc[-1]
        st.metric("VIX", f"{vix:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Flow of funds matrix
# -----------------------------
st.markdown("### Flow of Funds matrix")

ret_mats = compute_return_matrix(px, HORIZONS)
dv, vp = compute_volume_metrics(vol)
flow = compute_flow_score(px, vol, HORIZONS)
matrix_df = tidy_matrix(flow, {g: UNIVERSE[g] for g in show_groups})

if not matrix_df.empty:
    hm = matrix_df.pivot_table(index=["Group","Label"], columns="Ticker", values="FlowScore")
    z = hm.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=hm.columns.tolist(),
        y=[f"{g} • {l}" for g, l in hm.index],
        colorscale=[
            [0.0, PASTEL_NEG],
            [0.5, PASTEL_NEU],
            [1.0, PASTEL_POS],
        ],
        zmid=0,
        colorbar=dict(title="FlowScore", ticksuffix="σ")
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380 + 8*len(hm.index))
    st.markdown('<div class="matrix-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Matrix unavailable. Check data availability for the selected groups.")

# -----------------------------
# Return matrix table
# -----------------------------
st.markdown("### Return matrix by horizon")
def build_return_snapshot(px, horizons, selected_groups):
    rows = []
    tick2label = {t: lbl for g in selected_groups for t, lbl in UNIVERSE[g].items()}
    for t in tick2label.keys():
        if t not in px.columns:
            continue
        for name, n in horizons.items():
            val = px[t].pct_change(n).dropna()
            if val.empty:
                continue
            rows.append({
                "Group": [g for g in selected_groups if t in UNIVERSE[g]][0],
                "Ticker": t,
                "Label": tick2label[t],
                "Horizon": name,
                "Return": val.iloc[-1]*100.0
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot_table(index=["Group","Label","Ticker"], columns="Horizon", values="Return")
    return pivot.sort_values(["Group","Label"], ascending=[True, True])

ret_snapshot = build_return_snapshot(px, {k:v for k,v in HORIZONS.items() if k in horizon_cols}, show_groups)
st.dataframe(ret_snapshot.round(2), use_container_width=True)

# -----------------------------
# Actionable takeaways
# -----------------------------
st.markdown("### Actionable takeaways")
if not matrix_df.empty:
    latest = flow.dropna(how="all").iloc[-1].dropna()
    keep = set(matrix_df["Ticker"])
    latest = latest[latest.index.isin(keep)]
    top = latest.sort_values(ascending=False).head(5)
    bot = latest.sort_values(ascending=True).head(5)

    notes = []
    if set(["DGS2","DGS10"]).issubset(fred.columns):
        d2 = fred["DGS2"].dropna().iloc[-1]
        d10 = fred["DGS10"].dropna().iloc[-1]
        curve = d10 - d2
        curve_1m = (fred["DGS10"] - fred["DGS2"]).dropna().iloc[-21] if len(fred.dropna())>21 else np.nan
        tilt = "steepened" if curve > curve_1m else "flattened" if curve_1m==curve_1m else "moved"
        notes.append(f"Curve {tilt} to {curve:.2f} pp vs one month ago.")
    if "^VIX" in px.columns and not px["^VIX"].dropna().empty:
        vix_now = px["^VIX"].dropna().iloc[-1]
        vix_60 = px["^VIX"].dropna().rolling(60).quantile(0.2).iloc[-1]
        regime = "calmer tape" if vix_now <= vix_60 else "risk premium rebuilding"
        notes.append(f"VIX at {vix_now:.1f} suggests {regime}.")

    def map_name(t):
        for g in UNIVERSE:
            if t in UNIVERSE[g]:
                return UNIVERSE[g][t]
        return t

    top_named = [f"{t} ({map_name(t)})" for t in top.index]
    bot_named = [f"{t} ({map_name(t)})" for t in bot.index]

    st.markdown(
        f'''
        <div class="takeaways">
        <p class="soft">
        Capital is favoring <span class="good">{", ".join(top_named)}</span> on a flow basis, while leaking away from <span class="bad">{", ".join(bot_named)}</span>. 
        {(" ".join(notes))}
        Use this as a map for incremental tilts. Fade extremes where FlowScore is extended two or more standard deviations and confirm with your time frame.
        </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    st.write("No takeaways available due to missing data.")

# -----------------------------
# Asset detail panel
# -----------------------------
st.markdown("### Asset detail")

left, right = st.columns([1,1])
with left:
    all_labels = []
    for g in show_groups:
        all_labels += [f"{t} • {lbl}" for t,lbl in UNIVERSE[g].items() if t in px.columns]
    choice = st.selectbox("Select an asset", sorted(all_labels)) if len(all_labels)>0 else ""
    if "•" in choice:
        tkr = choice.split("•")[0].strip()
    else:
        tkr = choice.strip()

with right:
    if tkr and tkr in px.columns:
        s = px[tkr].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=tkr))
        for w in [21,50,100,200]:
            fig2.add_trace(go.Scatter(x=s.index, y=s.rolling(w).mean(), mode="lines", name=f"DMA{w}", opacity=0.6))
        fig2.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No price history for this selection.")

# -----------------------------
# Footer
# -----------------------------
st.caption(f"Data sources: Yahoo Finance and FRED. Last updated on {TODAY.isoformat()}.")
