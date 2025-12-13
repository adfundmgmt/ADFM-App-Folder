# pages/18_Cross_Asset_Vol_Surface.py
# Cross Asset Volatility Surface Monitor
# Equity, commodity, credit, FX regime read with clean visuals and commentary.
#
# Run: streamlit run app.py
# Put this file in /pages to show as a page in your multipage Streamlit app.

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# Optional FRED, used for HY OAS. If not available or blocked, app falls back gracefully.
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
st.markdown('<div class="small-muted">A tape-style regime read: where volatility is rising, where it is compressing, and what is driving cross-asset stress.</div>', unsafe_allow_html=True)

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
    v = r.rolling(window).std() * np.sqrt(252.0) * 100.0
    return v

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
        return (series * np.nan)
    return (series / base - 1.0) * 100.0  # % move from start

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
    # Equity implied term structure
    "VIX": "^VIX",
    "VXV (3M)": "^VXV",
    "VXMT (6M)": "^VXMT",
    # Commodity implied
    "OVX (Oil Vol)": "^OVX",
    "GVZ (Gold Vol)": "^GVZ",
    # FX implied
    "EVZ (EURUSD Vol)": "^EVZ",
}

FRED_SERIES = {
    # High yield option-adjusted spread
    "HY OAS (bps)": "BAMLH0A0HYM2",
    # Investment grade OAS (common series name, may fail depending on availability)
    "IG OAS (bps)": "BAMLC0A0CM",
}

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("About This Tool")
    st.write(
        "This monitor maps realized volatility across equity, commodities, credit, and FX over multiple horizons, "
        "then overlays implied vol indices where available. The top commentary summarizes what moved, where stress is building, "
        "and what that implies for regime."
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
# Data fetch
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_yf_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    # yfinance end is exclusive on some endpoints, add buffer by passing end as given.
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize to a simple Close matrix: columns are tickers
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Close, else Adj Close, else last level
        for field in ["Close", "Adj Close"]:
            if field in df.columns.get_level_values(0):
                close = df[field].copy()
                close.columns = [c.upper() for c in close.columns]
                return close
        # fallback
        close = df.xs(df.columns.levels[0][-1], axis=1, level=0).copy()
        close.columns = [c.upper() for c in close.columns]
        return close

    # Single ticker case
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": tickers[0].upper()})
    if "Adj Close" in df.columns:
        return df[["Adj Close"]].rename(columns={"Adj Close": tickers[0].upper()})
    return pd.DataFrame()

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

def fetch_all_prices(universes: List[AssetUniverse], implied: Dict[str, str], start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    labels_to_tickers: Dict[str, str] = {}
    for u in universes:
        labels_to_tickers.update(u.tickers)
    if show_implied:
        labels_to_tickers.update(implied)

    tickers = sorted(set(labels_to_tickers.values()))
    if not tickers:
        return pd.DataFrame(), labels_to_tickers

    # Chunk downloads for stability
    chunks = _chunked(tickers, dl_chunk)
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
    return prices, labels_to_tickers

# Dates
start_str = f"{start_year}-01-01"
end_str = str(end_date)

universes = [EQUITY, COMMODITY, CREDIT, FX]
prices, labels_to_tickers = fetch_all_prices(universes, IMPLIED_INDICES, start_str, end_str)

if prices.empty:
    st.error("No price data returned. Check tickers, connectivity, or date range.")
    st.stop()

# Map ticker -> label(s) for display convenience
ticker_to_label = {}
for label, tkr in labels_to_tickers.items():
    ticker_to_label[tkr.upper()] = label

# -----------------------------
# Build realized vol surface
# -----------------------------
# Build a core list of assets for realized vol (exclude implied indices from surface)
surface_assets: List[Tuple[str, str, str]] = []  # (group, label, ticker)
for u in universes:
    for label, tkr in u.tickers.items():
        surface_assets.append((u.name, label, tkr.upper()))

# Compute realized vol across windows
vol_rows = []
latest = prices.iloc[-1]

for group, label, tkr in surface_assets:
    if tkr not in prices.columns:
        continue
    s = prices[tkr].dropna()
    if len(s) < 260:
        continue

    row = {"Group": group, "Asset": label, "Ticker": tkr}
    for h, w in VOL_WINDOWS.items():
        v = _annualized_realized_vol(s, w)
        row[h] = float(v.iloc[-1]) if not v.empty else np.nan
    vol_rows.append(row)

vol_df = pd.DataFrame(vol_rows)
if vol_df.empty:
    st.error("Insufficient data to compute realized vol.")
    st.stop()

# Surface heatmap matrix
heat = vol_df[["Group", "Asset"] + list(VOL_WINDOWS.keys())].copy()
heat = heat.sort_values(["Group", "Asset"]).reset_index(drop=True)

# Cap for consistent scaling
heat_vals = heat[list(VOL_WINDOWS.keys())].clip(lower=0, upper=vol_cap)

# -----------------------------
# Implied overlays (if available)
# -----------------------------
def get_series_by_label(label: str) -> Optional[pd.Series]:
    tkr = IMPLIED_INDICES.get(label)
    if not tkr:
        return None
    t = tkr.upper()
    if t not in prices.columns:
        return None
    return prices[t]

def get_implied_last(label: str) -> Optional[float]:
    s = get_series_by_label(label)
    if s is None:
        return None
    v = float(s.dropna().iloc[-1]) if not s.dropna().empty else None
    return v

# -----------------------------
# Credit spreads from FRED (optional)
# -----------------------------
fred_data = {}
if show_credit_spreads and FRED_OK:
    for nm, sid in FRED_SERIES.items():
        try:
            s = load_fred_series(sid, start_str, end_str)
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

def realized_vol_snapshot(label: str, ticker: str, window: int = 21) -> Optional[float]:
    if ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    if len(s) < window + 5:
        return None
    v = _annualized_realized_vol(s, window)
    if v.dropna().empty:
        return None
    return float(v.dropna().iloc[-1])

def build_commentary() -> str:
    # Core proxies
    spy = "SPY"
    hyg = "HYG"
    dbc = "DBC"
    dxy = "DX-Y.NYB".upper()

    vix = get_implied_last("VIX") if show_implied else None
    ovx = get_implied_last("OVX (Oil Vol)") if show_implied else None
    gvz = get_implied_last("GVZ (Gold Vol)") if show_implied else None
    evz = get_implied_last("EVZ (EURUSD Vol)") if show_implied else None

    spy_r = last_returns(spy)
    hyg_r = last_returns(hyg)
    dbc_r = last_returns(dbc)
    dxy_r = last_returns(dxy)

    # Vol snapshots
    spy_vol = realized_vol_snapshot(spy, spy, 21)
    hyg_vol = realized_vol_snapshot(hyg, hyg, 21)  # uses label == ticker for these
    dbc_vol = realized_vol_snapshot(dbc, dbc, 21)
    dxy_vol = realized_vol_snapshot("DXY", dxy, 21)

    # Credit spread snapshot if available
    hy_oas = None
    if "HY OAS (bps)" in fred_data:
        s = fred_data["HY OAS (bps)"].dropna()
        if not s.empty:
            hy_oas = float(s.iloc[-1])

    # Build driver sentences
    drivers = []
    if vix is not None:
        drivers.append(f"Equity implied vol is {vix:.1f}, with SPY { _safe_pct(spy_r['1W']) } over 1W and { _safe_pct(spy_r['1M']) } over 1M.")
    else:
        drivers.append(f"SPY is { _safe_pct(spy_r['1W']) } over 1W and { _safe_pct(spy_r['1M']) } over 1M; realized 1M vol is { _safe_num(spy_vol, '{:.1f}') }.")

    if hy_oas is not None:
        drivers.append(f"Credit stress is visible in HY OAS at {hy_oas:.0f} bps, with HYG { _safe_pct(hyg_r['1W']) } over 1W.")
    else:
        drivers.append(f"Credit proxy HYG is { _safe_pct(hyg_r['1W']) } over 1W; realized 1M vol is { _safe_num(hyg_vol, '{:.1f}') }.")

    if ovx is not None or gvz is not None:
        parts = []
        if ovx is not None:
            parts.append(f"OVX {ovx:.1f}")
        if gvz is not None:
            parts.append(f"GVZ {gvz:.1f}")
        drivers.append(f"Commodity implied vol: {', '.join(parts)}; DBC is { _safe_pct(dbc_r['1M']) } over 1M.")
    else:
        drivers.append(f"DBC is { _safe_pct(dbc_r['1M']) } over 1M; realized 1M vol is { _safe_num(dbc_vol, '{:.1f}') }.")

    if evz is not None:
        drivers.append(f"FX implied vol (EVZ) is {evz:.1f}; DXY is { _safe_pct(dxy_r['1M']) } over 1M.")
    else:
        drivers.append(f"DXY is { _safe_pct(dxy_r['1M']) } over 1M; realized 1M vol is { _safe_num(dxy_vol, '{:.1f}') }.")

    # Simple regime conclusion: risk-off if SPY down and VIX up, or if HYG weak and DXY strong.
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
        conclusion = "Conclusion: the tape is pricing a defensive regime where equity downside and credit softness coincide with firmer dollar conditions and elevated implied vol."
    elif risk_off_flags == 2:
        conclusion = "Conclusion: mixed regime. Vol is elevated in pockets, but the signal is not yet a full cross-asset stress cascade."
    else:
        conclusion = "Conclusion: constructive regime. Vol is contained across most sleeves, with risk appetite intact unless credit and dollar tighten together."

    paragraph = " ".join(drivers) + " " + conclusion
    return paragraph

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
    sub = f"Percentile { _safe_num(pr, '{:.0f}') }, z { _safe_num(zs, '{:.2f}') }"
    return (f"{val:.1f}", sub)

k1, s1 = implied_kpi("VIX") if show_implied else ("n/a", "Disabled")
k2, s2 = implied_kpi("OVX (Oil Vol)") if show_implied else ("n/a", "Disabled")
k3, s3 = implied_kpi("EVZ (EURUSD Vol)") if show_implied else ("n/a", "Disabled")

# Credit KPI: HY OAS if possible else HYG 1M realized vol
credit_value = "n/a"
credit_sub = "Unavailable"
if "HY OAS (bps)" in fred_data:
    s = fred_data["HY OAS (bps)"].dropna()
    if not s.empty:
        v = float(s.iloc[-1])
        pr = _percentile_rank(s, v)
        zs = _zscore(s, v)
        credit_value = f"{v:.0f} bps"
        credit_sub = f"Percentile { _safe_num(pr, '{:.0f}') }, z { _safe_num(zs, '{:.2f}') }"
else:
    hv = realized_vol_snapshot("HYG", "HYG", 21)
    if hv is not None:
        credit_value = f"{hv:.1f}%"
        credit_sub = "HYG realized vol (1M)"

# -----------------------------
# Layout: commentary and KPIs
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
# Vol Surface heatmap
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
heatmap.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(heatmap, use_container_width=True)

st.markdown('<div class="small-muted">Surface is capped for consistent scaling across sessions. Use the sidebar cap slider to widen or tighten the range.</div>', unsafe_allow_html=True)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Implied term structure (equity)
# -----------------------------
if show_implied:
    st.subheader("Implied Vol Term Structure (where available)")
    term_labels = ["VIX", "VXV (3M)", "VXMT (6M)"]
    term_vals = []
    for lbl in term_labels:
        v = get_implied_last(lbl)
        term_vals.append(v if v is not None else np.nan)

    term_fig = go.Figure()
    term_fig.add_trace(go.Scatter(
        x=term_labels,
        y=term_vals,
        mode="lines+markers",
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>"
    ))
    term_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Implied vol index level",
    )
    st.plotly_chart(term_fig, use_container_width=True)

    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Normalized time series (starts at 0)
# -----------------------------
st.subheader("Cross Asset Moves (rebased to 0 at start, %)")

default_series = [
    ("Equity", "SPY"),
    ("Equity", "QQQ"),
    ("Commodity", "DBC"),
    ("Credit", "HYG"),
    ("FX", "DX-Y.NYB"),
]
all_choices = []
for group, label, tkr in surface_assets:
    all_choices.append((f"{group}: {label}", tkr))
choice_labels = [x[0] for x in all_choices]
choice_map = {x[0]: x[1] for x in all_choices}

preselect = []
for grp, tk in default_series:
    # Find matching label by ticker
    for disp, tkr in all_choices:
        if tkr.upper() == tk.upper():
            preselect.append(disp)

selected = st.multiselect(
    "Select series",
    options=choice_labels,
    default=preselect[:4] if len(preselect) >= 4 else preselect
)

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

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Detail table (clean, consistent scaling)
# -----------------------------
st.subheader("Surface Detail")

detail = heat.copy()
for h in VOL_WINDOWS.keys():
    detail[h] = pd.to_numeric(detail[h], errors="coerce").round(1)

# Add group separators via sorting
detail = detail.sort_values(["Group", "Asset"]).reset_index(drop=True)

# Style: use background gradient but keep simple and readable
def style_surface(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    vcols = list(VOL_WINDOWS.keys())
    sty = df.style.format({c: "{:.1f}" for c in vcols})
    sty = sty.background_gradient(subset=vcols, vmin=0, vmax=vol_cap)
    sty = sty.set_properties(**{"border-color": "rgba(0,0,0,0.08)"})
    return sty

st.dataframe(
    detail[["Group", "Asset"] + list(VOL_WINDOWS.keys())],
    use_container_width=True,
    hide_index=True,
)

st.markdown('<div class="small-muted">Tip: if you want this table to be sortable by stress, add a weighted vol score (for example 40% 1M, 30% 3M, 20% 6M, 10% 1Y) and sort descending.</div>', unsafe_allow_html=True)

# -----------------------------
# Optional: Credit spread chart
# -----------------------------
if show_credit_spreads and len(fred_data) > 0:
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Credit Spreads (FRED)")

    spread_df = pd.DataFrame(index=pd.Index([]))
    for nm, s in fred_data.items():
        spread_df[nm] = s

    spread_df = spread_df.dropna(how="all")
    if not spread_df.empty:
        fig = go.Figure()
        for col in spread_df.columns:
            fig.add_trace(go.Scatter(
                x=spread_df.index,
                y=spread_df[col],
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra></extra>"
            ))
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="bps",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("FRED series could not be loaded for the selected date range.")

# -----------------------------
# Footer diagnostics
# -----------------------------
with st.expander("Diagnostics"):
    st.write(f"History: {start_str} to {end_str}")
    missing = []
    for label, tkr in labels_to_tickers.items():
        if tkr.upper() not in prices.columns:
            missing.append(f"{label} ({tkr})")
    if missing:
        st.warning("Some tickers did not return data:")
        st.write(missing)
    else:
        st.success("All configured tickers returned data.")
