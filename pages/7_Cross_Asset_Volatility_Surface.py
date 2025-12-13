# pages/18_Cross_Asset_Vol_Surface.py
# Cross Asset Volatility Surface Monitor
# Equity, commodity, credit, FX regime read with clean visuals and commentary.

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Optional FRED
try:
    from pandas_datareader import data as pdr
    FRED_OK = True
except Exception:
    FRED_OK = False

# -----------------------------
# Page setup and style
# -----------------------------
st.set_page_config(page_title="Cross Asset Vol Surface", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1500px;}
h1, h2, h3 {font-weight: 650; letter-spacing: 0.2px;}
.small-muted {color: rgba(0,0,0,0.55); font-size: 0.9rem;}
.card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
}
.kpi-title {font-size: 0.85rem; color: rgba(0,0,0,0.55); margin-bottom: 2px;}
.kpi-value {font-size: 1.35rem; font-weight: 700; margin: 0;}
.kpi-sub {font-size: 0.9rem; color: rgba(0,0,0,0.65); margin-top: 6px;}
hr {border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 0.8rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Cross Asset Volatility Surface Monitor")
st.markdown(
    '<div class="small-muted">A tape-style regime read: where volatility is rising, where it is compressing, and what is driving cross-asset stress.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Configuration
# -----------------------------
START_YEAR_OPTIONS = [2020, 2010, 2000, 1990, 1980]
DEFAULT_START_YEAR = 2020

VOL_WINDOWS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

@dataclass
class AssetUniverse:
    name: str
    tickers: Dict[str, str]  # label -> ticker

def _safe_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.2f}%"

def _safe_num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return fmt.format(x)

def _annualized_realized_vol(px: pd.Series, window: int) -> pd.Series:
    r = np.log(px).diff()
    return r.rolling(window).std() * np.sqrt(252.0) * 100.0

def _percentile_rank(history: pd.Series, value: float) -> Optional[float]:
    h = history.dropna().values
    if len(h) < 30 or value is None or np.isnan(value):
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

def _normalize_to_zero(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * np.nan
    base = float(s.iloc[0])
    if base == 0:
        return series * np.nan
    return (series / base - 1.0) * 100.0

def _chunked(iterable: List[str], n: int) -> List[List[str]]:
    return [iterable[i:i+n] for i in range(0, len(iterable), n)]

# -----------------------------
# Universes
# -----------------------------
EQUITY = AssetUniverse(
    name="Equity",
    tickers={
        "SPY": "SPY",
        "QQQ": "QQQ",
        "IWM": "IWM",
        "EEM": "EEM",
        "EFA": "EFA",
    }
)

COMMODITY = AssetUniverse(
    name="Commodity",
    tickers={
        "WTI": "CL=F",
        "Gold": "GC=F",
        "Copper": "HG=F",
        "NatGas": "NG=F",
        "DBC": "DBC",
    }
)

CREDIT = AssetUniverse(
    name="Credit",
    tickers={
        "HYG": "HYG",
        "LQD": "LQD",
        "JNK": "JNK",
        "EMB": "EMB",
    }
)

FX = AssetUniverse(
    name="FX",
    tickers={
        "DXY": "DX-Y.NYB",
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
    }
)

IMPLIED_INDICES = {
    "VIX": "^VIX",
    "VXV (3M)": "^VXV",
    "VXMT (6M)": "^VXMT",
    "OVX (Oil Vol)": "^OVX",
    "GVZ (Gold Vol)": "^GVZ",
    "EVZ (EURUSD Vol)": "^EVZ",
}

FRED_SERIES = {
    "HY OAS (bps)": "BAMLH0A0HYM2",
    "IG OAS (bps)": "BAMLC0A0CM",
}

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("About This Tool")
    st.write(
        "Maps realized volatility across equity, commodities, credit, and FX over multiple horizons, "
        "then overlays implied vol indices where available. Commentary summarizes drivers first, then a regime conclusion."
    )
    st.divider()

    start_year = st.selectbox("History start year", START_YEAR_OPTIONS, index=START_YEAR_OPTIONS.index(DEFAULT_START_YEAR))
    end_date = st.date_input("End date", value=date.today())
    vol_cap = st.slider("Heatmap cap (annualized vol, %)", min_value=15, max_value=120, value=70, step=5)
    dl_chunk = st.slider("Download chunk size", min_value=4, max_value=40, value=18, step=2)

    st.divider()
    show_implied = st.checkbox("Show implied vol overlays (VIX, OVX, GVZ, EVZ)", value=True)
    show_credit_spreads = st.checkbox("Try to pull credit spreads from FRED", value=True)

# -----------------------------
# Data fetch (robust yfinance parsing)
# -----------------------------
def _extract_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return:
      - Single ticker: columns like ['Open','High','Low','Close',...]
      - Multi ticker, group_by="column": MultiIndex (Field, Ticker)
      - Multi ticker, group_by="ticker": MultiIndex (Ticker, Field)
    This function returns a Close matrix with columns = tickers (uppercased).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Single ticker typical shape
    if not isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns:
            out = df[["Close"]].copy()
            # column name unknown here, leave as 'Close' and let caller rename if needed
            return out
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
            return out
        return pd.DataFrame()

    # MultiIndex cases
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # Case A: (Field, Ticker)
    if "Close" in set(lvl0):
        close = df["Close"].copy()
        close.columns = [str(c).upper() for c in close.columns]
        return close

    if "Adj Close" in set(lvl0):
        close = df["Adj Close"].copy()
        close.columns = [str(c).upper() for c in close.columns]
        return close

    # Case B: (Ticker, Field)
    if "Close" in set(lvl1):
        close = df.xs("Close", axis=1, level=1).copy()
        close.columns = [str(c).upper() for c in close.columns]
        return close

    if "Adj Close" in set(lvl1):
        close = df.xs("Adj Close", axis=1, level=1).copy()
        close.columns = [str(c).upper() for c in close.columns]
        return close

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_yf_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",   # keep as ticker, we now parse both layouts
        threads=True,
    )
    close = _extract_close_matrix(df)

    # Fix single ticker return where _extract_close_matrix returns a single 'Close' column
    if close.shape[1] == 1 and close.columns.tolist() in [["Close"], ["Adj Close"]]:
        # rename that single column to the ticker we asked for
        if len(tickers) == 1:
            close.columns = [tickers[0].upper()]
        else:
            # fallback: keep as-is
            pass

    return close

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    if not FRED_OK:
        return pd.Series(dtype=float)
    s = pdr.DataReader(series_id, "fred", start, end)
    if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
        return s.iloc[:, 0].rename(series_id)
    if isinstance(s, pd.Series):
        return s.rename(series_id)
    return pd.Series(dtype=float)

def fetch_all_prices(universes: List[AssetUniverse], implied: Dict[str, str], start: str, end: str, chunk_size: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    labels_to_tickers: Dict[str, str] = {}
    for u in universes:
        labels_to_tickers.update(u.tickers)
    if show_implied:
        labels_to_tickers.update(implied)

    tickers = sorted(set([t.upper() for t in labels_to_tickers.values()]))
    if not tickers:
        return pd.DataFrame(), labels_to_tickers

    chunks = _chunked(tickers, chunk_size)
    out = []
    progress = st.progress(0, text="Downloading market data...")
    for i, ch in enumerate(chunks):
        px_chunk = load_yf_prices(ch, start, end)
        if not px_chunk.empty:
            out.append(px_chunk)
        progress.progress(int((i + 1) / len(chunks) * 100), text="Downloading market data...")
    progress.empty()

    if not out:
        return pd.DataFrame(), labels_to_tickers

    prices = pd.concat(out, axis=1)
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()

    # Deduplicate columns if any overlap and keep last non-null
    prices = prices.groupby(level=0, axis=1).last()

    return prices, labels_to_tickers

# Dates
start_str = f"{start_year}-01-01"
# yfinance end is often exclusive, add 1 day buffer
end_str = str(end_date + timedelta(days=1))

universes = [EQUITY, COMMODITY, CREDIT, FX]
prices, labels_to_tickers = fetch_all_prices(universes, IMPLIED_INDICES, start_str, end_str, dl_chunk)

if prices.empty:
    st.error("No price data returned. Check tickers, connectivity, or date range.")
    st.stop()

# -----------------------------
# Build realized vol surface
# -----------------------------
surface_assets: List[Tuple[str, str, str]] = []
for u in universes:
    for label, tkr in u.tickers.items():
        surface_assets.append((u.name, label, tkr.upper()))

vol_rows = []
for group, label, tkr in surface_assets:
    if tkr not in prices.columns:
        continue
    s = prices[tkr].dropna()
    if len(s) < 10:
        continue

    row = {"Group": group, "Asset": label, "Ticker": tkr}
    for h, w in VOL_WINDOWS.items():
        if len(s) >= (w + 2):
            v = _annualized_realized_vol(s, w)
            row[h] = float(v.dropna().iloc[-1]) if not v.dropna().empty else np.nan
        else:
            row[h] = np.nan
    vol_rows.append(row)

vol_df = pd.DataFrame(vol_rows)

if vol_df.empty:
    # Show helpful diagnostics instead of a dead end
    missing = [f"{g}:{a} ({t})" for g, a, t in surface_assets if t not in prices.columns]
    present = [f"{g}:{a} ({t})" for g, a, t in surface_assets if t in prices.columns]
    st.error("Insufficient data to compute realized vol. This usually means the price matrix came back empty or missing key tickers.")
    with st.expander("Diagnostics"):
        st.write("Columns received:", list(prices.columns)[:50])
        st.write("Assets present:", present)
        st.write("Assets missing:", missing)
        st.write("Date range:", start_str, "to", end_str)
        st.write("Non-null counts (top):")
        st.write(prices.notna().sum().sort_values(ascending=False).head(20))
    st.stop()

heat = vol_df[["Group", "Asset"] + list(VOL_WINDOWS.keys())].copy()
heat = heat.sort_values(["Group", "Asset"]).reset_index(drop=True)
heat_vals = heat[list(VOL_WINDOWS.keys())].clip(lower=0, upper=vol_cap)

# -----------------------------
# Implied overlays and FRED
# -----------------------------
def get_implied_last(label: str) -> Optional[float]:
    tkr = IMPLIED_INDICES.get(label)
    if not tkr:
        return None
    t = tkr.upper()
    if t not in prices.columns:
        return None
    s = prices[t].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])

fred_data = {}
if show_credit_spreads and FRED_OK:
    for nm, sid in FRED_SERIES.items():
        try:
            s = load_fred_series(sid, start_str, str(end_date))
            if not s.empty:
                fred_data[nm] = s
        except Exception:
            pass

# -----------------------------
# Commentary engine
# -----------------------------
def last_returns(ticker: str) -> Dict[str, Optional[float]]:
    if ticker not in prices.columns:
        return {"1D": None, "1W": None, "1M": None}
    s = prices[ticker].dropna()
    if len(s) < 30:
        return {"1D": None, "1W": None, "1M": None}
    r1d = (s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0 if len(s) >= 2 else None
    r1w = (s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0 if len(s) >= 6 else None
    r1m = (s.iloc[-1] / s.iloc[-22] - 1.0) * 100.0 if len(s) >= 22 else None
    return {"1D": float(r1d) if r1d is not None else None,
            "1W": float(r1w) if r1w is not None else None,
            "1M": float(r1m) if r1m is not None else None}

def realized_vol_snapshot(ticker: str, window: int = 21) -> Optional[float]:
    if ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    if len(s) < window + 2:
        return None
    v = _annualized_realized_vol(s, window).dropna()
    if v.empty:
        return None
    return float(v.iloc[-1])

def build_commentary() -> str:
    spy = "SPY"
    hyg = "HYG"
    dbc = "DBC"
    dxy = "DX-Y.NYB"

    vix = get_implied_last("VIX") if show_implied else None
    evz = get_implied_last("EVZ (EURUSD Vol)") if show_implied else None
    ovx = get_implied_last("OVX (Oil Vol)") if show_implied else None

    spy_r = last_returns(spy)
    hyg_r = last_returns(hyg)
    dbc_r = last_returns(dbc)
    dxy_r = last_returns(dxy)

    spy_vol = realized_vol_snapshot(spy, 21)
    hyg_vol = realized_vol_snapshot(hyg, 21)
    dbc_vol = realized_vol_snapshot(dbc, 21)
    dxy_vol = realized_vol_snapshot(dxy, 21)

    hy_oas = None
    if "HY OAS (bps)" in fred_data:
        s = fred_data["HY OAS (bps)"].dropna()
        if not s.empty:
            hy_oas = float(s.iloc[-1])

    drivers = []
    if vix is not None:
        drivers.append(f"Equity implied vol is {vix:.1f}, with SPY {_safe_pct(spy_r['1W'])} over 1W and {_safe_pct(spy_r['1M'])} over 1M.")
    else:
        drivers.append(f"SPY is {_safe_pct(spy_r['1W'])} over 1W and {_safe_pct(spy_r['1M'])} over 1M; realized 1M vol is {_safe_num(spy_vol, '{:.1f}') }.")

    if hy_oas is not None:
        drivers.append(f"Credit stress is visible in HY OAS at {hy_oas:.0f} bps, with HYG {_safe_pct(hyg_r['1W'])} over 1W.")
    else:
        drivers.append(f"Credit proxy HYG is {_safe_pct(hyg_r['1W'])} over 1W; realized 1M vol is {_safe_num(hyg_vol, '{:.1f}') }.")

    if ovx is not None:
        drivers.append(f"Oil implied vol (OVX) is {ovx:.1f}; broad commodities (DBC) are {_safe_pct(dbc_r['1M'])} over 1M.")
    else:
        drivers.append(f"DBC is {_safe_pct(dbc_r['1M'])} over 1M; realized 1M vol is {_safe_num(dbc_vol, '{:.1f}') }.")

    if evz is not None:
        drivers.append(f"FX implied vol (EVZ) is {evz:.1f}; DXY is {_safe_pct(dxy_r['1M'])} over 1M.")
    else:
        drivers.append(f"DXY is {_safe_pct(dxy_r['1M'])} over 1M; realized 1M vol is {_safe_num(dxy_vol, '{:.1f}') }.")

    risk_off_flags = 0
    if spy_r["1W"] is not None and spy_r["1W"] < 0:
        risk_off_flags += 1
    if hyg_r["1W"] is not None and hyg_r["1W"] < 0:
        risk_off_flags += 1
    if dxy_r["1M"] is not None and dxy_r["1M"] > 0:
        risk_off_flags += 1
    if vix is not None and vix >= 18:
        risk_off_flags += 1

    if risk_off_flags >= 3:
        conclusion = "Conclusion: the tape is leaning defensive, with stress showing up across at least three sleeves at once."
    elif risk_off_flags == 2:
        conclusion = "Conclusion: mixed regime. Stress is visible, but it is not yet broad enough to call systemic."
    else:
        conclusion = "Conclusion: constructive regime. Vol is contained and cross-asset confirmation of stress is limited."

    return " ".join(drivers) + " " + conclusion

commentary = build_commentary()

# -----------------------------
# KPI cards
# -----------------------------
def kpi_card(title: str, value: str, sub: str) -> None:
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

def implied_kpi(label: str) -> Tuple[str, str]:
    tkr = IMPLIED_INDICES.get(label)
    if not tkr:
        return ("n/a", "Missing mapping")
    col = tkr.upper()
    if col not in prices.columns:
        return ("n/a", "Unavailable on Yahoo")
    s = prices[col].dropna()
    if s.empty:
        return ("n/a", "No data")
    val = float(s.iloc[-1])
    pr = _percentile_rank(s, val)
    zs = _zscore(s, val)
    return (f"{val:.1f}", f"Percentile {_safe_num(pr, '{:.0f}')}, z {_safe_num(zs, '{:.2f}')}")

k1, s1 = implied_kpi("VIX") if show_implied else ("n/a", "Disabled")
k2, s2 = implied_kpi("OVX (Oil Vol)") if show_implied else ("n/a", "Disabled")
k3, s3 = implied_kpi("EVZ (EURUSD Vol)") if show_implied else ("n/a", "Disabled")

credit_value = "n/a"
credit_sub = "Unavailable"
if "HY OAS (bps)" in fred_data:
    s = fred_data["HY OAS (bps)"].dropna()
    if not s.empty:
        v = float(s.iloc[-1])
        pr = _percentile_rank(s, v)
        zs = _zscore(s, v)
        credit_value = f"{v:.0f} bps"
        credit_sub = f"Percentile {_safe_num(pr, '{:.0f}')}, z {_safe_num(zs, '{:.2f}')}"
else:
    hv = realized_vol_snapshot("HYG", 21)
    if hv is not None:
        credit_value = f"{hv:.1f}%"
        credit_sub = "HYG realized vol (1M)"

# -----------------------------
# Layout
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<b>Key drivers</b>", unsafe_allow_html=True)
st.write(commentary)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Equity implied vol", k1, s1)
with c2:
    kpi_card("Commodity implied vol", k2, s2)
with c3:
    kpi_card("FX implied vol", k3, s3)
with c4:
    kpi_card("Credit stress", credit_value, credit_sub)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Heatmap
# -----------------------------
st.subheader("Realized Volatility Surface (annualized, %)")
heatmap = go.Figure(
    data=go.Heatmap(
        z=heat_vals[list(VOL_WINDOWS.keys())].values,
        x=list(VOL_WINDOWS.keys()),
        y=[f"{g}: {a}" for g, a in zip(heat["Group"], heat["Asset"])],
        zmin=0,
        zmax=vol_cap,
        hovertemplate="Horizon %{x}<br>%{y}<br>Vol %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Vol %"),
    )
)
heatmap.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(heatmap, use_container_width=True)
st.markdown('<div class="small-muted">Surface is capped for consistent scaling. Use the sidebar cap slider to adjust.</div>', unsafe_allow_html=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Implied term structure
# -----------------------------
if show_implied:
    st.subheader("Implied Vol Term Structure (equity)")
    term_labels = ["VIX", "VXV (3M)", "VXMT (6M)"]
    term_vals = [get_implied_last(lbl) for lbl in term_labels]

    term_fig = go.Figure()
    term_fig.add_trace(go.Scatter(
        x=term_labels,
        y=[v if v is not None else np.nan for v in term_vals],
        mode="lines+markers",
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>"
    ))
    term_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Index level")
    st.plotly_chart(term_fig, use_container_width=True)

    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Normalized time series (rebased to 0)
# -----------------------------
st.subheader("Cross Asset Moves (rebased to 0 at start, %)")
surface_choice = []
for u in [EQUITY, COMMODITY, CREDIT, FX]:
    for label, tkr in u.tickers.items():
        surface_choice.append((f"{u.name}: {label}", tkr.upper()))

choice_labels = [x[0] for x in surface_choice]
choice_map = {x[0]: x[1] for x in surface_choice}

defaults = []
for want in ["Equity: SPY", "Credit: HYG", "Commodity: DBC", "FX: DXY"]:
    if want in choice_map:
        defaults.append(want)

selected = st.multiselect("Select series", options=choice_labels, default=defaults)

if not selected:
    st.info("Select at least one series to plot.")
else:
    plot_df = pd.DataFrame(index=prices.index)
    for disp in selected:
        tkr = choice_map.get(disp)
        if tkr and tkr in prices.columns:
            plot_df[disp] = _normalize_to_zero(prices[tkr])

    plot_df = plot_df.dropna(how="all")
    ts_fig = go.Figure()
    for col in plot_df.columns:
        ts_fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[col],
            mode="lines",
            name=col,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ))
    ts_fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Move from start (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(ts_fig, use_container_width=True)

# -----------------------------
# Credit spreads (optional)
# -----------------------------
if show_credit_spreads:
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Credit Spreads (FRED)")

    if not FRED_OK:
        st.info("pandas_datareader not available in this environment. Install pandas_datareader or disable this section.")
    elif len(fred_data) == 0:
        st.info("No FRED series loaded (API blocked or series unavailable).")
    else:
        spread_df = pd.DataFrame({k: v for k, v in fred_data.items()}).dropna(how="all")
        fig = go.Figure()
        for col in spread_df.columns:
            fig.add_trace(go.Scatter(
                x=spread_df.index,
                y=spread_df[col],
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra></extra>"
            ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="bps", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Diagnostics
# -----------------------------
with st.expander("Diagnostics"):
    st.write("Date range:", start_str, "to", end_str)
    st.write("Columns received (sample):", list(prices.columns)[:60])
    st.write("Non-null counts (top):")
    st.write(prices.notna().sum().sort_values(ascending=False).head(20))
