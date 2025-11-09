# flow_of_funds_matrix.py
# Flow of Funds Matrix – cross-asset flows with macro regime header and actionable takeaways
# Data sources: Yahoo Finance (prices, volumes) and FRED (yields, breakevens, credit spreads) via pandas_datareader
# FRED access here uses pandas_datareader's public endpoint (no login or API key)

import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page config and styling
# -----------------------------
st.set_page_config(page_title="Flow of Funds Matrix", layout="wide")

PASTEL_BG = "#f7f8fc"
PASTEL_CARD = "white"
PASTEL_POS = "#9ad1bc"   # gentle green
PASTEL_NEG = "#f5a6a6"   # gentle red
PASTEL_NEU = "#cfd8e3"   # soft blue gray
PASTEL_ACCENT = "#93c5fd"  # light blue

st.markdown(
    f"""
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
        min-height: 180px;
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
        font-size: 1.05rem;
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
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Parameters and universe
# -----------------------------
TODAY = dt.date.today()
DEFAULT_YEARS = 5

HORIZONS = {
    "1D": 1,
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

UNIVERSE: Dict[str, Dict[str, str]] = {
    "Equities": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000",
        "EFA": "Developed ex US",
        "EEM": "Emerging Markets",
        "SMH": "Semiconductors",
        "XLK": "Tech",
        "XLF": "Financials",
        "XLE": "Energy",
    },
    "Rates & Credit": {
        "IEF": "US 7-10y Treasuries ETF",
        "TLT": "US 20y+ Treasuries ETF",
        "HYG": "US High Yield ETF",
        "LQD": "US Investment Grade ETF",
    },
    "FX": {
        "^DXY": "US Dollar Index",
        "EURUSD=X": "EURUSD",
        "JPY=X": "USDJPY",
    },
    "Commodities": {
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "WTI Oil",
        "UNG": "US Nat Gas",
        "DBC": "Broad Commodities",
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    },
    "Cash": {
        "BIL": "T-Bills 1-3M",
    },
    "Vol": {
        "^VIX": "VIX",
    },
}

FRED_SERIES = {
    "DGS2": "UST 2y Yield (%)",
    "DGS10": "UST 10y Yield (%)",
    "T10YIE": "10y Breakeven (%)",
    "BAMLH0A0HYM2": "HY OAS (bps)",   # FRED returns percent; convert to bps
    "BAMLC0A0CM": "IG OAS (bps)",     # FRED returns percent; convert to bps
}
FRED_CODES: Tuple[str, ...] = tuple(FRED_SERIES.keys())  # tuple for Streamlit cache hashing

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    years = st.slider("History (years)", 2, 15, DEFAULT_YEARS, 1)
    START = TODAY - dt.timedelta(days=365 * years)

    show_groups = st.multiselect(
        "Asset groups",
        options=list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys()),
    )

    # FlowScore weights are adjustable
    st.markdown("**FlowScore weights**")
    w5d = st.slider("5D weight", 0.0, 1.0, 0.45, 0.05)
    w1m = st.slider("1M weight", 0.0, 1.0, 0.35, 0.05)
    w3m = st.slider("3M weight", 0.0, 1.0, 0.15, 0.05)
    wvol = st.slider("Volume weight", 0.0, 1.0, 0.05, 0.05)
    wsum = max(w5d + w1m + w3m + wvol, 1e-9)
    W = {  # normalized
        "5D": w5d / wsum,
        "1M": w1m / wsum,
        "3M": w3m / wsum,
        "VOL": wvol / wsum,
    }

    horizon_cols = ["1D", "5D", "1M", "3M", "6M", "1Y"]
    disable_fred = st.checkbox("Disable FRED if service is flaky", value=False)

# -----------------------------
# Data loaders with caching
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo_data(tickers_tuple: Tuple[str, ...], start_date: dt.date) -> pd.DataFrame:
    tickers = list(tickers_tuple)
    df = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    frames = []
    for t in tickers:
        if t in df.columns.get_level_values(0):
            sub = df[t][["Close", "Volume"]].copy()
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def load_fred_series(series_codes_tuple: Tuple[str, ...], start_date: dt.date) -> pd.DataFrame:
    out = {}
    for code in series_codes_tuple:
        try:
            s = pdr.DataReader(code, "fred", start_date)
            out[code] = s[code]
        except Exception:
            out[code] = pd.Series(dtype=float, name=code)
    return pd.DataFrame(out).dropna(how="all")

# -----------------------------
# Helpers and signals
# -----------------------------
def compute_price_panels(yh: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if yh.empty:
        return pd.DataFrame(), pd.DataFrame()
    close_cols = [c for c in yh.columns if c[1] == "Close"]
    vol_cols = [c for c in yh.columns if c[1] == "Volume"]
    px = yh[close_cols].copy()
    px.columns = [c[0] for c in close_cols]
    vol = yh[vol_cols].copy()
    vol.columns = [c[0] for c in vol_cols]
    return px, vol

def rolling_zscore(s: pd.Series, window: int = 63) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / (sd.replace(0, np.nan))

def vol_percentile(vol: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return vol.rolling(window).apply(
        lambda a: pd.Series(a).rank(pct=True).iloc[-1] if len(a) > 0 else np.nan,
        raw=False,
    )

def compute_flow_score(
    px: pd.DataFrame, vol: pd.DataFrame, horizons: Dict[str, int], weights: Dict[str, float]
) -> pd.DataFrame:
    """Weighted z across 5D, 1M, 3M returns and 60D volume percentile z.
       Robust to missing components: reweights per-ticker by available pieces."""
    ret_5d = px.pct_change(horizons["5D"])
    ret_1m = px.pct_change(horizons["1M"])
    ret_3m = px.pct_change(horizons["3M"])
    volp = vol_percentile(vol, window=60)

    z5 = ret_5d.apply(rolling_zscore, window=63)
    z1m = ret_1m.apply(rolling_zscore, window=126)
    z3m = ret_3m.apply(rolling_zscore, window=252)
    zv = volp.apply(rolling_zscore, window=126)

    # Align all components
    all_cols = sorted(set(px.columns))
    z5 = z5.reindex(columns=all_cols)
    z1m = z1m.reindex(columns=all_cols)
    z3m = z3m.reindex(columns=all_cols)
    zv = zv.reindex(columns=all_cols)

    numerator = 0
    denominator = 0
    for comp, w in [(z5, weights["5D"]), (z1m, weights["1M"]), (z3m, weights["3M"]), (zv, weights["VOL"])]:
        numerator = numerator + comp.multiply(w)
        denominator = denominator + comp.notna().astype(float).multiply(w)

    flow = numerator / denominator.replace(0, np.nan)
    return flow

def tidy_matrix(flow_score: pd.DataFrame, groups_dict: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if flow_score.dropna(how="all").empty:
        return pd.DataFrame()
    latest = flow_score.dropna(how="all").iloc[-1].dropna()
    rows = []
    for group, members in groups_dict.items():
        for tkr, label in members.items():
            if tkr in latest.index:
                rows.append(
                    {"Group": group, "Ticker": tkr, "Label": label, "FlowScore": latest[tkr]}
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Group", "FlowScore"], ascending=[True, False])

def map_name(t: str) -> str:
    for g in UNIVERSE:
        if t in UNIVERSE[g]:
            return UNIVERSE[g][t]
    return t

# -----------------------------
# Load data
# -----------------------------
tickers = tuple(sorted({t for g in show_groups for t in UNIVERSE[g].keys()}))

with st.spinner("Loading market data..."):
    yh = load_yahoo_data(tickers, START)
    fred = pd.DataFrame()
    if not disable_fred:
        fred = load_fred_series(FRED_CODES, START)

px, vol = compute_price_panels(yh)

# -----------------------------
# Macro regime header (fixed units and clearer layout)
# -----------------------------
st.markdown("### Macro regime snapshot")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Treasury curve</div>', unsafe_allow_html=True)
    if not fred.empty and {"DGS2", "DGS10"}.issubset(fred.columns):
        d2 = float(fred["DGS2"].dropna().iloc[-1])
        d10 = float(fred["DGS10"].dropna().iloc[-1])
        st.metric("2s", f"{d2:.2f}%")
        st.metric("10s", f"{d10:.2f}%")
        st.metric("2s10s", f"{(d10 - d2):.2f} pp")
    else:
        st.caption("FRED unavailable")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Breakeven inflation</div>', unsafe_allow_html=True)
    if "T10YIE" in fred.columns and not fred["T10YIE"].dropna().empty:
        be10 = float(fred["T10YIE"].dropna().iloc[-1])
        st.metric("10y BE", f"{be10:.2f}%")
    else:
        st.caption("FRED unavailable")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Credit spreads</div>', unsafe_allow_html=True)
    # FRED provides these in percent. Convert to bps.
    if "BAMLH0A0HYM2" in fred.columns and not fred["BAMLH0A0HYM2"].dropna().empty:
        hy_bps = float(fred["BAMLH0A0HYM2"].dropna().iloc[-1]) * 100.0
        st.metric("HY OAS", f"{hy_bps:.0f} bps")
    if "BAMLC0A0CM" in fred.columns and not fred["BAMLC0A0CM"].dropna().empty:
        ig_bps = float(fred["BAMLC0A0CM"].dropna().iloc[-1]) * 100.0
        st.metric("IG OAS", f"{ig_bps:.0f} bps")
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Equity trend & vol</div>', unsafe_allow_html=True)
    if "SPY" in px.columns and not px["SPY"].dropna().empty:
        spy = px["SPY"].dropna()
        st.metric("SPY", f"{spy.iloc[-1]:.2f}")
        for w in [21, 50, 200]:
            st.caption(f"DMA{w}: {spy.rolling(w).mean().iloc[-1]:.2f}")
    if "^VIX" in px.columns and not px["^VIX"].dropna().empty:
        st.metric("VIX", f"{px['^VIX'].dropna().iloc[-1]:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Flow of funds matrix (robust, dense)
# -----------------------------
st.markdown("### Flow of Funds matrix")

flow = compute_flow_score(px, vol, HORIZONS, W)
matrix_df = tidy_matrix(flow, {g: UNIVERSE[g] for g in show_groups})

if not matrix_df.empty:
    # Order assets within each group by FlowScore
    hm = matrix_df.pivot_table(index=["Group", "Label"], columns="Ticker", values="FlowScore")
    fig = go.Figure(
        data=go.Heatmap(
            z=hm.values,
            x=hm.columns.tolist(),
            y=[f"{g} • {l}" for g, l in hm.index],
            colorscale=[
                [0.0, PASTEL_NEG],
                [0.5, PASTEL_NEU],
                [1.0, PASTEL_POS],
            ],
            zmid=0,
            colorbar=dict(title="FlowScore", ticksuffix="σ"),
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=min(900, 380 + 10 * len(hm.index)))
    st.markdown('<div class="matrix-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Matrix unavailable. Check data availability for the selected groups.")

# -----------------------------
# Return matrix snapshot (x.xx% formatting and fixed column order)
# -----------------------------
st.markdown("### Return matrix by horizon")

def build_return_snapshot(px: pd.DataFrame, horizons, selected_groups) -> pd.DataFrame:
    rows = []
    tick2label = {t: lbl for g in selected_groups for t, lbl in UNIVERSE[g].items()}
    for t in tick2label:
        if t not in px.columns:
            continue
        for name, n in horizons.items():
            s = px[t].pct_change(n).dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "Group": [g for g in selected_groups if t in UNIVERSE[g]][0],
                    "Label": tick2label[t],
                    "Ticker": t,
                    "Horizon": name,
                    "Return": s.iloc[-1] * 100.0,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["Group", "Label", "Ticker"], columns="Horizon", values="Return")
    # enforce clean column order and percent strings
    pivot = pivot.reindex(columns=horizon_cols)
    pivot = pivot.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}%")
    return pivot

ret_snapshot = build_return_snapshot(px, {k: HORIZONS[k] for k in horizon_cols}, show_groups)
if ret_snapshot.empty:
    st.caption("No returns available for the selected horizons and assets.")
else:
    st.dataframe(ret_snapshot, use_container_width=True)

# -----------------------------
# Actionable takeaways (now broad-based, not only crypto)
# -----------------------------
st.markdown("### Actionable takeaways")
if not matrix_df.empty:
    latest = flow.dropna(how="all").iloc[-1]
    # keep only displayed tickers
    keep = set(matrix_df["Ticker"])
    latest = latest[latest.index.isin(keep)].dropna()

    # rank across all assets; then show diversified top/bottom
    top = latest.sort_values(ascending=False).head(6)
    bot = latest.sort_values(ascending=True).head(6)

    def fmt_names(idx):
        return [f"{t} ({map_name(t)})" for t in idx]

    # Curve and vol notes
    notes = []
    if not fred.empty and {"DGS2", "DGS10"}.issubset(fred.columns):
        curve_now = (fred["DGS10"] - fred["DGS2"]).dropna()
        if not curve_now.empty:
            cur = curve_now.iloc[-1]
            prev = curve_now.shift(21).dropna().iloc[-1] if len(curve_now) > 21 else np.nan
            tilt = "steepened" if prev == prev and cur > prev else "flattened" if prev == prev else "moved"
            notes.append(f"Curve {tilt} to {cur:.2f} pp vs one month ago.")
    if "^VIX" in px.columns and not px["^VIX"].dropna().empty:
        vix_now = px["^VIX"].dropna().iloc[-1]
        vix_floor = px["^VIX"].dropna().rolling(60).quantile(0.2).iloc[-1] if len(px["^VIX"].dropna()) >= 60 else np.nan
        regime = "calmer tape" if vix_floor == vix_floor and vix_now <= vix_floor else "risk premium rebuilding"
        notes.append(f"VIX at {vix_now:.1f} suggests {regime}.")

    extremes_long = [t for t, v in top.items() if v >= 2.0]
    extremes_short = [t for t, v in bot.items() if v <= -2.0]

    st.markdown(
        f"""
        <div class="takeaways">
        <p class="soft">
        Capital is favoring <span class="good">{", ".join(fmt_names(top.index))}</span> and leaking from <span class="bad">{", ".join(fmt_names(bot.index))}</span>.
        {" ".join(notes)}
        Extremes flagged at |FlowScore| ≥ 2σ:
        <span class="good">{", ".join(fmt_names(extremes_long)) if extremes_long else "none"}</span> /
        <span class="bad">{", ".join(fmt_names(extremes_short)) if extremes_short else "none"}</span>.
        Use as a tilt map; invalidate on a close back inside one sigma.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption("No takeaways available due to missing data.")

st.caption(f"Last updated on {TODAY.isoformat()}. Data: Yahoo Finance, FRED via pandas_datareader.")
