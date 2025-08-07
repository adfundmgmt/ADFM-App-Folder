import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

plt.style.use("default")
st.set_page_config(page_title="ETF Flows (Strict)", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("ETF Flows — Strict Mode")
st.sidebar.markdown(
    """
**What this does:** Net flow ≈ Δ(Shares Outstanding) × last price  
**What it won't do:** No AUM/price proxies, no guesswork.  
If SO is unavailable or ambiguous → the ETF is skipped.
"""
)

# Lookback
today = pd.Timestamp.today().normalize()
lookbacks = {
    "1M": 30, "3M": 90, "6M": 180, "12M": 365,
    "YTD": (today - pd.Timestamp(year=today.year, month=1, day=1)).days
}
lb_key = st.sidebar.radio("Period", list(lookbacks.keys()), index=1)
days = int(lookbacks[lb_key])
start = today - pd.Timedelta(days=days + 10)  # small buffer for alignment

# Universe (keep it simple; edit as you like)
ETF_INFO = {
    "MAGS": "Mag 7",
    "SMH": "Semiconductors",
    "BOTZ": "Robotics/AI",
    "ICLN": "Clean Energy",
    "URNM": "Uranium",
    "ARKK": "Innovation",
    "KWEB": "China Internet",
    "FXI":  "China Large-Cap",
    "EWZ":  "Brazil",
    "EEM":  "Emerging Markets",
    "VWO":  "Emerging Markets (Vanguard)",
    "VGK":  "Europe Large-Cap",
    "FEZ":  "Eurozone",
    "ILF":  "Latin America",
    "ARGT": "Argentina",
    "GLD":  "Gold",
    "SLV":  "Silver",
    "DBC":  "Commodities",
    "HEDJ": "Hedged Europe",
    "USMV": "US Min Vol",
    "COWZ": "US Free Cash Flow",
    "BITO": "BTC Futures",
    "IBIT": "Spot BTC",
    "BIL":  "1–3m T-Bills",
    "TLT":  "20+y Treasuries",
    "SHV":  "0–1y T-Bills",
}
TICKERS = list(ETF_INFO.keys())

# ---------------- Helpers ----------------
@st.cache_data(ttl=900, show_spinner=True)
def dl_prices(tickers, start_dt, end_dt):
    df = yf.download(tickers, start=start_dt, end=end_dt, auto_adjust=False,
                     progress=False, threads=True, group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        closes = {t: df[(t, "Close")].rename(t) for t in tickers if (t, "Close") in df.columns}
        if not closes: return pd.DataFrame()
        return pd.concat(closes.values(), axis=1).sort_index().ffill()
    elif "Close" in df.columns:
        return pd.DataFrame({tickers[0]: df["Close"]}).sort_index().ffill()
    return pd.DataFrame()

def human_money(x, signed=True, decimals=2):
    if x is None or pd.isna(x): return ""
    sgn = "+" if (signed and x > 0) else ("-" if (signed and x < 0) else "")
    ax = abs(float(x))
    if ax >= 1e9:  val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{sgn}${val:,.{decimals}f}{suf}"

def tick_fmt(x, pos):
    ax = abs(float(x))
    sgn = "-" if x < 0 else ""
    if ax >= 1e9:  val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{sgn}${val:,.2f}{suf}"

def normalize_so_units(so: pd.Series):
    """If the median SO is tiny (<1e4), assume it's in millions → multiply by 1e6."""
    so = (so or pd.Series(dtype=float)).dropna().astype(float)
    if so.empty: return so, "no_SO"
    return (so*1_000_000.0, "SO(millions→shares)") if float(so.median()) < 1e4 else (so, "SO(shares)")

def compute_flow_strict(ticker, px: pd.Series):
    """
    Strict ΔSO × last Close.
    Returns dict with keys: flow, method, note.
    Skips ETF if SO is missing/ambiguous or if alignment fails.
    """
    try:
        so = yf.Ticker(ticker).get_shares_full()
    except Exception:
        so = None
    if so is None or (isinstance(so, (pd.Series, pd.DataFrame)) and pd.Series(so).dropna().empty):
        return {"flow": None, "method": "skip", "note": "no_SO"}

    so_series = pd.Series(so).copy()
    so_series.index = pd.to_datetime(so_series.index).tz_localize(None)
    so_series, note = normalize_so_units(so_series)

    px = px.dropna()
    if px.empty:
        return {"flow": None, "method": "skip", "note": "no_price"}

    # Align SO to trading days inside the window
    px_window = px.loc[start:today]
    so_aligned = so_series.reindex(px_window.index, method="ffill").dropna()
    if so_aligned.empty:
        return {"flow": None, "method": "skip", "note": "no_align"}

    so_start, so_end = float(so_aligned.iloc[0]), float(so_aligned.iloc[-1])
    p_end = float(px_window.iloc[-1])
    flow = (so_end - so_start) * p_end

    # Basic sanity: if |ΔSO| > 50% of SO_end for the period, flag as suspect (likely unit jump/rebalance)
    if so_end != 0 and abs(so_end - so_start) / abs(so_end) > 0.5:
        return {"flow": None, "method": "skip", "note": "suspect_SO_jump"}

    return {"flow": flow, "method": f"ΔSO×P_end[{note}]", "note": ""}

# ---------------- Compute ----------------
prices = dl_prices(TICKERS, start, today)
if prices.empty:
    st.error("Price download failed. Try again later.")
    st.stop()

rows = []
skipped = []
for t in TICKERS:
    px = prices[t] if t in prices.columns else pd.Series(dtype=float)
    res = compute_flow_strict(t, px)
    if res["flow"] is None:
        skipped.append((t, ETF_INFO[t], res["note"]))
        continue
    rows.append({"Ticker": t, "Label": f"{ETF_INFO[t]} ({t})", "Flow ($)": res["flow"], "Method": res["method"]})

df = pd.DataFrame(rows).sort_values("Flow ($)", ascending=False).reset_index(drop=True)

# ---------------- UI ----------------
st.title("ETF Flows — Strict ΔSO × Price")
st.caption(f"Period: **{lb_key}** (ending {today.date()}) • Only ETFs with reliable shares-outstanding history are shown.")

if df.empty:
    st.warning("Nothing to show (no ETFs had clean SO data for this lookback).")
else:
    # Chart
    vals = df["Flow ($)"].to_numpy()
    fig, ax = plt.subplots(figsize=(15, max(5, len(df) * 0.45)))
    colors = np.where(vals > 0, "#2ca02c", "#d62728")
    bars = ax.barh(df["Label"], vals, color=colors, alpha=0.9)

    ax.set_xlabel("Estimated Flow ($ with k/M/B)")
    ax.set_title(f"ETF Proxy Flows — {lb_key}")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_fmt))

    x_lo, x_hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
    pad = (x_hi - x_lo) * 0.15 if x_hi != x_lo else 1.0
    ax.set_xlim([x_lo - pad, x_hi + pad])

    for bar, v in zip(bars, df["Flow ($)"]):
        txt = human_money(v, signed=True, decimals=2)
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        off = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x + (off if x >= 0 else -off),
                bar.get_y() + bar.get_height()/2, txt, va="center", ha=ha, fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # Top movers
    st.markdown("### Top Inflows & Outflows (strict)")
    left, right = st.columns(2)
    with left:
        top_in = df.head(5).copy()
        top_in["Flow ($)"] = top_in["Flow ($)"].map(lambda v: human_money(v, True))
        st.table(top_in[["Label", "Flow ($)", "Method"]].set_index("Label"))
    with right:
        top_out = df.sort_values("Flow ($)").head(5).copy()
        top_out["Flow ($)"] = top_out["Flow ($)"].map(lambda v: human_money(v, True))
        st.table(top_out[["Label", "Flow ($)", "Method"]].set_index("Label"))

# Skipped tickers
if skipped:
    st.markdown("### Skipped (no reliable SO)")
    skip_df = pd.DataFrame(skipped, columns=["Ticker", "Label", "Reason"])
    st.dataframe(skip_df, use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
