import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

# ── Page / Sidebar ────────────────────────────────────────────────────────────
st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

st.sidebar.title("ETF Flows")
st.sidebar.markdown(
    """
Flows = **Δ shares outstanding × last close** (best-effort from Yahoo).  
If shares data isn't available, that ETF is shown in **gray**.
"""
)

lookbacks = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookbacks.keys()), index=1)
period_days = int(lookbacks[period_label])

# Curated, liquid core set (20)
ETF_INFO = {
    # Core beta
    "SPY": "S&P 500", "IVV": "S&P 500 (iShares)", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "EFA": "Developed ex-US", "EEM": "Emerging Markets",
    # Sectors (SPDR)
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLI": "Industrials",
    # Credit & Rates
    "HYG": "High Yield", "LQD": "IG Credit", "TLT": "20+Y Treasuries", "IEF": "7–10Y Treasuries", "SHY": "1–3Y Treasuries",
    # Cash / metals / alt beta
    "BIL": "T-Bills 1–3M", "GLD": "Gold", "SLV": "Silver", "USMV": "US Min Vol", "VLUE": "US Value",
}
TICKERS = list(ETF_INFO.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=True)
def get_price_history(ticker: str, days: int) -> pd.DataFrame:
    # unadjusted close to pair with current SO
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{days+10}d")  # small buffer
    return hist.dropna()

def _normalize_so_units(so: pd.Series) -> pd.Series:
    """If the SO series looks like it's in 'millions', scale to shares."""
    s = pd.Series(so).dropna().astype(float)
    if s.empty:
        return s
    return s * 1_000_000.0 if float(s.median()) < 1e4 else s

@st.cache_data(ttl=900, show_spinner=True)
def flow_delta_so_price(ticker: str, days: int):
    """
    Best-effort: Δ(Shares Outstanding) × last Close over lookback window.
    Returns (flow_dollars or None).
    """
    try:
        hist = get_price_history(ticker, days)
        if hist.empty or "Close" not in hist:
            return None
        close = hist["Close"]
        # shares outstanding history
        try:
            so_raw = yf.Ticker(ticker).get_shares_full()
        except Exception:
            so_raw = None
        if so_raw is None:
            return None
        so = _normalize_so_units(so_raw)
        if so.empty:
            return None
        # align SO to trading days
        so.index = pd.to_datetime(so.index)
        so = so.loc[close.index.min(): close.index.max()].reindex(close.index, method="ffill").dropna()
        if len(so) < 2:
            return None
        flow = float(so.iloc[-1] - so.iloc[0]) * float(close.iloc[-1])
        return flow
    except Exception:
        return None

def label_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    s = "-" if x < 0 else "+"
    ax = abs(float(x))
    if   ax >= 1e9:  return f"{s}${ax/1e9:,.0f}B"
    elif ax >= 1e6:  return f"{s}${ax/1e6:,.0f}M"
    elif ax >= 1e3:  return f"{s}${ax/1e3:,.0f}K"
    else:            return f"{s}${ax:,.0f}"

# ── Compute ────────────────────────────────────────────────────────────────────
rows = []
for t in TICKERS:
    f = flow_delta_so_price(t, period_days)
    rows.append({"Ticker": t, "Label": f"{ETF_INFO[t]} ({t})", "Flow ($)": f})

df = pd.DataFrame(rows).sort_values("Flow ($)", ascending=False, na_position="last").reset_index(drop=True)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("ETF Flows Dashboard")
st.caption(f"Flows are proxies (Δ SO × price). Period: **{period_label}**")

if df.empty:
    st.error("No data available.")
else:
    # Chart
    chart_df = df.copy()
    vals = chart_df["Flow ($)"].fillna(0.0).to_numpy()
    max_val = np.nanmax(np.abs(vals)) if len(vals) else 0.0
    pad = max_val * 0.15 if max_val > 0 else 1.0

    fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
    colors = [
        ("#2ca02c" if (v is not None and v > 0) else "#d62728" if (v is not None and v < 0) else "#bdbdbd")
        for v in chart_df["Flow ($)"]
    ]
    bars = ax.barh(chart_df["Label"], chart_df["Flow ($)"].fillna(0.0), color=colors, alpha=0.85)
    ax.set_title(f"ETF Proxy Flows — {period_label}")
    ax.set_xlabel("Estimated Flow ($)")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e9:,.0f}B' if abs(x) >= 1e9 else f'${x/1e6:,.0f}M'
    ))
    lo = -pad if np.nanmin(vals) < 0 else 0
    hi = max_val + pad
    ax.set_xlim([lo, hi])

    for bar, val in zip(bars, chart_df["Flow ($)"]):
        if val is None or pd.isna(val):
            continue
        txt = label_money(val)
        x = bar.get_width()
        ha = "left" if val >= 0 else "right"
        offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height()/2,
                txt, va="center", ha=ha, fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("*Green = inflow, Red = outflow, Gray = missing shares data.*")

    # Top movers
    st.markdown("#### Top Inflows & Outflows")
    top_in = df.head(3)[["Label", "Flow ($)"]].copy()
    top_in["Flow"] = top_in["Flow ($)"].map(label_money)
    top_out = df.sort_values("Flow ($)").head(3)[["Label", "Flow ($)"]].copy()
    top_out["Flow"] = top_out["Flow ($)"].map(label_money)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Inflows**")
        st.table(top_in[["Flow"]].set_index(top_in["Label"]))
    with c2:
        st.write("**Top Outflows**")
        st.table(top_out[["Flow"]].set_index(top_out["Label"]))

    if df["Flow ($)"].isnull().any():
        st.warning("Some ETFs had no usable shares-outstanding history. Gray bars indicate missing flows.")
    else:
        st.success("Flows computed from Δ shares outstanding × price for all ETFs.")

st.caption("© 2025 AD Fund Management LP")
