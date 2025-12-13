# pages/18_Cross_Asset_Vol_Surface.py
# Cross Asset Volatility Surface Monitor
# Version: Vol-index-only (CBOE indices via Yahoo Finance)
#
# Core idea:
# - Use only volatility indices (^VIX, ^VXN, ^VVIX, ^OVX, ^GVZ, ^RVX, ^VXD, ^TYVIX if available)
# - Build a "surface" as percentile-of-level for rolling means across multiple horizons
# - Add a second "surface" for "vol of vol" as percentile-of-rolling-stdev of daily returns across horizons
# - Commentary: key drivers first, then conclusion
# - Visuals: scorecard, surfaces, normalized time series (rebased to 0), ratios, diagnostics

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# -----------------------------
# Page setup and style
# -----------------------------
st.set_page_config(page_title="Volatility Surface Monitor", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1550px;}
h1, h2, h3 {font-weight: 650; letter-spacing: 0.2px;}
.small-muted {color: rgba(0,0,0,0.55); font-size: 0.92rem;}
.card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.06);
}
.kpi-title {font-size: 0.85rem; color: rgba(0,0,0,0.55); margin-bottom: 2px;}
.kpi-value {font-size: 1.35rem; font-weight: 750; margin: 0;}
.kpi-sub {font-size: 0.92rem; color: rgba(0,0,0,0.65); margin-top: 6px; line-height: 1.25rem;}
hr {border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 0.85rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Cross Asset Volatility Surface Monitor")
st.markdown(
    '<div class="small-muted">Vol indices only. Levels, moves, and where each index sits versus its own history across horizons.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Config
# -----------------------------
START_YEAR_OPTIONS = [2020, 2010, 2000, 1990, 1980]
DEFAULT_START_YEAR = 2020

HORIZONS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

CORE_INDICES: Dict[str, str] = {
    "VIX (S&P 500 vol)": "^VIX",
    "VXN (Nasdaq 100 vol)": "^VXN",
    "VVIX (VIX vol of vol)": "^VVIX",
    "OVX (Crude Oil vol)": "^OVX",
    "GVZ (Gold vol)": "^GVZ",
}

OPTIONAL_INDICES: Dict[str, str] = {
    "RVX (Russell 2000 vol)": "^RVX",
    "VXD (Dow vol)": "^VXD",
    "TYVIX (10Y Treasury vol)": "^TYVIX",
}

CANONICAL_ORDER = list(CORE_INDICES.keys()) + list(OPTIONAL_INDICES.keys())

def _safe_num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return fmt.format(x)

def _safe_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.2f}%"

def _normalize_to_zero(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * np.nan
    base = float(s.iloc[0])
    if base == 0:
        return series * np.nan
    return (series / base - 1.0) * 100.0

def _percentile_of_value(history: pd.Series, value: float) -> Optional[float]:
    h = history.dropna().values
    if len(h) < 60 or value is None or np.isnan(value):
        return None
    return float((h < value).mean() * 100.0)

def _zscore(history: pd.Series, value: float) -> Optional[float]:
    h = history.dropna()
    if len(h) < 60 or value is None or np.isnan(value):
        return None
    mu = float(h.mean())
    sd = float(h.std(ddof=0))
    if sd == 0:
        return None
    return (float(value) - mu) / sd

def _extract_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust close extraction for yfinance outputs.
    Handles:
      - single ticker: columns include 'Close'
      - multi tickers with MultiIndex in either (Field, Ticker) or (Ticker, Field)
    Returns DataFrame with columns as tickers.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns:
            out = df[["Close"]].copy()
            return out
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
            return out
        return pd.DataFrame()

    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # (Field, Ticker)
    if "Close" in set(lvl0):
        close = df["Close"].copy()
        close.columns = [str(c) for c in close.columns]
        return close
    if "Adj Close" in set(lvl0):
        close = df["Adj Close"].copy()
        close.columns = [str(c) for c in close.columns]
        return close

    # (Ticker, Field)
    if "Close" in set(lvl1):
        close = df.xs("Close", axis=1, level=1).copy()
        close.columns = [str(c) for c in close.columns]
        return close
    if "Adj Close" in set(lvl1):
        close = df.xs("Adj Close", axis=1, level=1).copy()
        close.columns = [str(c) for c in close.columns]
        return close

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_vol_indices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    close = _extract_close_matrix(raw)

    # Fix single-column case naming
    if close.shape[1] == 1 and close.columns.tolist() in [["Close"], ["Adj Close"]]:
        if len(tickers) == 1:
            close.columns = [tickers[0]]
    close = close.loc[~close.index.duplicated(keep="last")].sort_index()
    close.columns = [c.upper() for c in close.columns]
    return close

def _points_change(series: pd.Series, k: int) -> Optional[float]:
    s = series.dropna()
    if len(s) < k + 1:
        return None
    return float(s.iloc[-1] - s.iloc[-(k + 1)])

def _pct_change(series: pd.Series, k: int) -> Optional[float]:
    s = series.dropna()
    if len(s) < k + 1:
        return None
    base = float(s.iloc[-(k + 1)])
    if base == 0:
        return None
    return float((s.iloc[-1] / base - 1.0) * 100.0)

def _rolling_mean_surface(series: pd.Series, windows: Dict[str, int]) -> Dict[str, Optional[float]]:
    out = {}
    s = series.dropna()
    for name, w in windows.items():
        if len(s) < w + 10:
            out[name] = None
            continue
        rm = s.rolling(w).mean()
        cur = float(rm.dropna().iloc[-1]) if not rm.dropna().empty else np.nan
        pr = _percentile_of_value(rm.dropna(), cur)
        out[name] = pr
    return out

def _vol_of_vol_surface(series: pd.Series, windows: Dict[str, int]) -> Dict[str, Optional[float]]:
    """
    "Vol of vol" defined as rolling stdev of daily % changes in the vol index.
    Surface cell is percentile of current rolling stdev vs its own history.
    """
    out = {}
    s = series.dropna()
    if len(s) < 40:
        return {k: None for k in windows.keys()}
    rets = s.pct_change() * 100.0
    for name, w in windows.items():
        if len(rets.dropna()) < w + 10:
            out[name] = None
            continue
        rv = rets.rolling(w).std()
        cur = float(rv.dropna().iloc[-1]) if not rv.dropna().empty else np.nan
        pr = _percentile_of_value(rv.dropna(), cur)
        out[name] = pr
    return out

def _build_scorecard(label_to_ticker: Dict[str, str], prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in CANONICAL_ORDER:
        if label not in label_to_ticker:
            continue
        tkr = label_to_ticker[label].upper()
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        if len(s) < 20:
            continue

        level = float(s.iloc[-1])
        pr = _percentile_of_value(s, level)
        zs = _zscore(s, level)

        d1 = _points_change(s, 1)
        w1 = _points_change(s, 5)
        m1 = _points_change(s, 21)
        q1 = _points_change(s, 63)

        d1p = _pct_change(s, 1)
        w1p = _pct_change(s, 5)
        m1p = _pct_change(s, 21)
        q1p = _pct_change(s, 63)

        # Current vol-of-vol snapshot (21d stdev of daily % changes)
        rets = (s.pct_change() * 100.0).dropna()
        vov21 = float(rets.rolling(21).std().dropna().iloc[-1]) if len(rets) >= 40 else np.nan
        vov_pr = None
        if len(rets) >= 100:
            rv = rets.rolling(21).std().dropna()
            if not rv.empty and not np.isnan(vov21):
                vov_pr = _percentile_of_value(rv, vov21)

        rows.append({
            "Index": label,
            "Ticker": tkr,
            "Level": level,
            "Level %ile": pr,
            "Level z": zs,
            "1D chg (pts)": d1,
            "1W chg (pts)": w1,
            "1M chg (pts)": m1,
            "3M chg (pts)": q1,
            "1D chg (%)": d1p,
            "1W chg (%)": w1p,
            "1M chg (%)": m1p,
            "3M chg (%)": q1p,
            "Vol-of-vol 21d": vov21,
            "VoV %ile": vov_pr,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ordered presentation
    df["__order"] = df["Index"].apply(lambda x: CANONICAL_ORDER.index(x) if x in CANONICAL_ORDER else 999)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return df

def _kpi_card(title: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-title">{title}</div>
            <p class="kpi-value">{value}</p>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _commentary_from_scorecard(df: pd.DataFrame) -> str:
    if df.empty:
        return "No indices loaded. Check tickers and Yahoo availability."

    # Drivers: biggest 1D movers (absolute %), highest level percentiles, elevated vol-of-vol percentiles
    tmp = df.copy()

    tmp["abs_1d"] = tmp["1D chg (%)"].abs()
    movers = tmp.dropna(subset=["1D chg (%)"]).sort_values("abs_1d", ascending=False).head(3)

    stress = tmp.dropna(subset=["Level %ile"]).sort_values("Level %ile", ascending=False).head(3)
    vov_stress = tmp.dropna(subset=["VoV %ile"]).sort_values("VoV %ile", ascending=False).head(2)

    def fmt_row(r, mode: str) -> str:
        if mode == "move":
            return f"{r['Index']} {r['Level']:.1f} ({r['1D chg (%)']:+.2f}% 1D, {r['1W chg (%)']:+.2f}% 1W)"
        if mode == "stress":
            return f"{r['Index']} {r['Level']:.1f} at {r['Level %ile']:.0f}%ile (z {r['Level z']:.2f})"
        return f"{r['Index']} VoV {r['Vol-of-vol 21d']:.2f} at {r['VoV %ile']:.0f}%ile"

    driver_bits = []
    if not movers.empty:
        driver_bits.append("Fast tape: " + "; ".join(fmt_row(r, "move") for _, r in movers.iterrows()) + ".")
    if not stress.empty:
        driver_bits.append("Level pressure: " + "; ".join(fmt_row(r, "stress") for _, r in stress.iterrows()) + ".")
    if not vov_stress.empty:
        driver_bits.append("Reflexive risk: " + "; ".join(fmt_row(r, "vov") for _, r in vov_stress.iterrows()) + ".")

    # Conclusion: simple regime classifier
    def get_level(label: str) -> Optional[float]:
        x = tmp.loc[tmp["Index"] == label, "Level %ile"]
        return float(x.iloc[0]) if len(x) else None

    vix_p = get_level("VIX (S&P 500 vol)")
    vxn_p = get_level("VXN (Nasdaq 100 vol)")
    vvix_p = get_level("VVIX (VIX vol of vol)")
    rvx_p = get_level("RVX (Russell 2000 vol)")
    ovx_p = get_level("OVX (Crude Oil vol)")
    gvz_p = get_level("GVZ (Gold vol)")
    ty_p = get_level("TYVIX (10Y Treasury vol)")

    flags = 0
    for p in [vix_p, vxn_p, vvix_p, rvx_p]:
        if p is not None and p >= 75:
            flags += 1
    if ovx_p is not None and ovx_p >= 80:
        flags += 1
    if gvz_p is not None and gvz_p >= 80:
        flags += 1
    if ty_p is not None and ty_p >= 80:
        flags += 1

    if flags >= 4:
        conclusion = "Conclusion: broad vol regime. Equity vol is elevated and confirmed by vol-of-vol and at least one non-equity sleeve, which usually means positioning is fragile and gap risk is priced."
    elif flags == 2 or flags == 3:
        conclusion = "Conclusion: localized stress. The market is paying up for protection in specific areas, but the signal is not yet a full cross-index squeeze."
    else:
        conclusion = "Conclusion: contained regime. Vol is not screaming across the complex, so the base case is compression unless a catalyst pushes VVIX and VIX higher together."

    return " ".join(driver_bits) + " " + conclusion

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About This Tool")
    st.write(
        "This page monitors CBOE volatility indices available on Yahoo Finance. "
        "The surfaces are percentile maps across rolling horizons: how elevated the index level is, and how unstable the index itself is."
    )
    st.divider()

    start_year = st.selectbox("History start year", START_YEAR_OPTIONS, index=START_YEAR_OPTIONS.index(DEFAULT_START_YEAR))
    end_date = st.date_input("End date", value=date.today())

    st.divider()
    include_optional = st.checkbox("Include RVX, VXD, TYVIX if available", value=True)
    heat_cap = st.slider("Surface color cap (percentile)", min_value=70, max_value=100, value=95, step=1)
    dl_chunk = st.slider("Download chunk size", min_value=3, max_value=12, value=7, step=1)

# -----------------------------
# Build ticker set
# -----------------------------
label_to_ticker = dict(CORE_INDICES)
if include_optional:
    label_to_ticker.update(OPTIONAL_INDICES)

tickers = [t.upper() for t in label_to_ticker.values()]

# Dates: yfinance end often exclusive, add 1 day
start_str = f"{start_year}-01-01"
end_str = str(end_date + timedelta(days=1))

# -----------------------------
# Download
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def _download_in_chunks(tickers: List[str], start: str, end: str, chunk_size: int) -> pd.DataFrame:
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    out = []
    progress = st.progress(0, text="Downloading vol indices from Yahoo Finance...")
    for i, ch in enumerate(chunks):
        px = load_vol_indices(ch, start, end)
        if not px.empty:
            out.append(px)
        progress.progress(int((i + 1) / len(chunks) * 100), text="Downloading vol indices from Yahoo Finance...")
    progress.empty()
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, axis=1)
    df = df.groupby(level=0, axis=1).last()
    df = df.sort_index()
    return df

prices = _download_in_chunks(tickers, start_str, end_str, dl_chunk)

if prices.empty:
    st.error("No data returned from Yahoo Finance. Check connectivity and tickers.")
    st.stop()

# Filter to actually-available tickers
available = [t for t in tickers if t.upper() in prices.columns and prices[t.upper()].dropna().shape[0] > 30]
if not available:
    st.error("Tickers downloaded, but none have enough data to display. Open Diagnostics to see what came back.")
    with st.expander("Diagnostics"):
        st.write("Columns received:", list(prices.columns))
        st.write("Non-null counts:", prices.notna().sum().sort_values(ascending=False))
    st.stop()

# Update label map to only those that exist
label_to_ticker_live = {lbl: tkr for lbl, tkr in label_to_ticker.items() if tkr.upper() in prices.columns}

# -----------------------------
# Scorecard + commentary
# -----------------------------
score = _build_scorecard(label_to_ticker_live, prices)
commentary = _commentary_from_scorecard(score)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<b>Key drivers</b>", unsafe_allow_html=True)
st.write(commentary)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# KPI row
# -----------------------------
def _get_row(idx_label: str) -> Optional[pd.Series]:
    if score.empty:
        return None
    x = score.loc[score["Index"] == idx_label]
    if x.empty:
        return None
    return x.iloc[0]

kpi_cols = st.columns(4)

vix_row = _get_row("VIX (S&P 500 vol)")
vxn_row = _get_row("VXN (Nasdaq 100 vol)")
vvix_row = _get_row("VVIX (VIX vol of vol)")
ovx_row = _get_row("OVX (Crude Oil vol)")

with kpi_cols[0]:
    if vix_row is not None:
        _kpi_card("VIX", f"{vix_row['Level']:.1f}", f"{vix_row['1D chg (pts)']:+.2f} pts 1D, {_safe_num(vix_row['Level %ile'], '{:.0f}')}%ile")
    else:
        _kpi_card("VIX", "n/a", "Unavailable")
with kpi_cols[1]:
    if vxn_row is not None:
        _kpi_card("VXN", f"{vxn_row['Level']:.1f}", f"{vxn_row['1D chg (pts)']:+.2f} pts 1D, {_safe_num(vxn_row['Level %ile'], '{:.0f}')}%ile")
    else:
        _kpi_card("VXN", "n/a", "Unavailable")
with kpi_cols[2]:
    if vvix_row is not None:
        _kpi_card("VVIX", f"{vvix_row['Level']:.1f}", f"{vvix_row['1D chg (%)']:+.2f}% 1D, {_safe_num(vvix_row['Level %ile'], '{:.0f}')}%ile")
    else:
        _kpi_card("VVIX", "n/a", "Unavailable")
with kpi_cols[3]:
    if ovx_row is not None:
        _kpi_card("OVX", f"{ovx_row['Level']:.1f}", f"{ovx_row['1D chg (%)']:+.2f}% 1D, {_safe_num(ovx_row['Level %ile'], '{:.0f}')}%ile")
    else:
        _kpi_card("OVX", "n/a", "Unavailable")

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Surface 1: level percentile of rolling means
# -----------------------------
st.subheader("Vol Level Surface (percentile of rolling mean vs history)")

surface_rows = []
for label in CANONICAL_ORDER:
    if label not in label_to_ticker_live:
        continue
    tkr = label_to_ticker_live[label].upper()
    s = prices[tkr].dropna()
    surf = _rolling_mean_surface(s, HORIZONS)
    surface_rows.append({"Index": label, **surf})

surface_df = pd.DataFrame(surface_rows)
if surface_df.empty:
    st.info("Surface unavailable. Not enough data.")
else:
    z = surface_df[list(HORIZONS.keys())].values.astype(float)
    z = np.clip(z, 0, heat_cap)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(HORIZONS.keys()),
            y=surface_df["Index"].tolist(),
            zmin=0,
            zmax=heat_cap,
            hovertemplate="%{y}<br>%{x}<br>%{z:.0f}%ile<extra></extra>",
            colorbar=dict(title="%ile"),
        )
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="small-muted">Interpretation: higher percentiles mean the index is elevated relative to its own history. Rolling means reduce single-day noise.</div>', unsafe_allow_html=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Surface 2: vol of vol percentile across horizons
# -----------------------------
st.subheader("Vol of Vol Surface (percentile of rolling stdev of daily % changes)")

vov_rows = []
for label in CANONICAL_ORDER:
    if label not in label_to_ticker_live:
        continue
    tkr = label_to_ticker_live[label].upper()
    s = prices[tkr].dropna()
    surf = _vol_of_vol_surface(s, HORIZONS)
    vov_rows.append({"Index": label, **surf})

vov_df = pd.DataFrame(vov_rows)
if vov_df.empty:
    st.info("Vol of vol surface unavailable. Not enough data.")
else:
    z = vov_df[list(HORIZONS.keys())].values.astype(float)
    z = np.clip(z, 0, heat_cap)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(HORIZONS.keys()),
            y=vov_df["Index"].tolist(),
            zmin=0,
            zmax=heat_cap,
            hovertemplate="%{y}<br>%{x}<br>%{z:.0f}%ile<extra></extra>",
            colorbar=dict(title="%ile"),
        )
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="small-muted">Interpretation: higher percentiles mean the vol index itself is whipping around more than usual, which tends to coincide with fragile positioning.</div>', unsafe_allow_html=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Time series: rebased to 0
# -----------------------------
st.subheader("Vol Indices Time Series (rebased to 0 at start, %)")

choices = []
for label in CANONICAL_ORDER:
    if label in label_to_ticker_live:
        choices.append(label)

default_sel = [x for x in [
    "VIX (S&P 500 vol)",
    "VXN (Nasdaq 100 vol)",
    "VVIX (VIX vol of vol)",
    "OVX (Crude Oil vol)",
    "GVZ (Gold vol)",
] if x in choices]

selected = st.multiselect("Select indices", options=choices, default=default_sel)

if selected:
    plot_df = pd.DataFrame(index=prices.index)
    for label in selected:
        tkr = label_to_ticker_live[label].upper()
        plot_df[label] = _normalize_to_zero(prices[tkr])

    plot_df = plot_df.dropna(how="all")
    fig = go.Figure()
    for col in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[col],
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"
        ))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Move from start (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one index to plot.")

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Ratios: VXN/VIX and VVIX/VIX
# -----------------------------
st.subheader("Key Ratios (relative stress and convexity)")

def _series(tkr: str) -> Optional[pd.Series]:
    t = tkr.upper()
    if t not in prices.columns:
        return None
    s = prices[t].dropna()
    return s if not s.empty else None

vix = _series("^VIX")
vxn = _series("^VXN")
vvix = _series("^VVIX")

ratio_df = pd.DataFrame(index=prices.index)

if vix is not None and vxn is not None:
    ratio_df["VXN / VIX"] = (vxn / vix).replace([np.inf, -np.inf], np.nan)
if vix is not None and vvix is not None:
    ratio_df["VVIX / VIX"] = (vvix / vix).replace([np.inf, -np.inf], np.nan)

ratio_df = ratio_df.dropna(how="all")
if ratio_df.empty:
    st.info("Ratios unavailable. Requires VIX plus VXN and VVIX.")
else:
    fig = go.Figure()
    for col in ratio_df.columns:
        fig.add_trace(go.Scatter(
            x=ratio_df.index,
            y=ratio_df[col],
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"
        ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Ratio",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Scorecard table + download
# -----------------------------
st.subheader("Vol Index Scorecard")

if score.empty:
    st.info("Scorecard unavailable.")
else:
    show_cols = [
        "Index", "Ticker", "Level", "Level %ile", "Level z",
        "1D chg (pts)", "1W chg (pts)", "1M chg (pts)", "3M chg (pts)",
        "1D chg (%)", "1W chg (%)", "1M chg (%)", "3M chg (%)",
        "Vol-of-vol 21d", "VoV %ile"
    ]
    out = score[show_cols].copy()

    # Formatting for display
    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in ["Level", "Vol-of-vol 21d"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        for c in ["Level %ile", "VoV %ile"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(0)
        for c in ["Level z"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        for c in ["1D chg (pts)", "1W chg (pts)", "1M chg (pts)", "3M chg (pts)"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        for c in ["1D chg (%)", "1W chg (%)", "1M chg (%)", "3M chg (%)"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        return d

    st.dataframe(_fmt(out), use_container_width=True, hide_index=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scorecard as CSV",
        data=csv,
        file_name="vol_index_scorecard.csv",
        mime="text/csv"
    )

# -----------------------------
# Diagnostics
# -----------------------------
with st.expander("Diagnostics"):
    st.write("Date range:", start_str, "to", end_str)
    st.write("Requested tickers:", tickers)
    st.write("Returned columns:", list(prices.columns))
    st.write("Non-null counts:")
    st.write(prices.notna().sum().sort_values(ascending=False))
