# ETF Flows — Efficient & Useful (ΔSO × Price, curated liquid ETFs)
# ---------------------------------------------------------------
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
st.set_page_config(page_title="ETF Flows — Core 20", layout="wide")

# ========= Sidebar =========
st.sidebar.title("ETF Flows — Core 20")
st.sidebar.markdown("""
Flows = **Δ Shares Outstanding × Close**.  
Only includes ETFs with reliable SO history; others are listed as *Missing*.
""")

today = pd.Timestamp.today().normalize()
LOOKBACKS = {
    "1M": 30, "3M": 90, "6M": 180, "12M": 365,
    "YTD": (today - pd.Timestamp(year=today.year, month=1, day=1)).days,
}
lb_key = st.sidebar.radio("Lookback", list(LOOKBACKS.keys()), index=1)
days = int(LOOKBACKS[lb_key])
start = today - pd.Timedelta(days=days + 10)  # buffer for alignment

agg = st.sidebar.selectbox("Aggregate", ["Daily", "Weekly"], index=1)
show_missing = st.sidebar.checkbox("Show Missing table", value=True)

# ========= Curated liquid basket (broad + sector + credit/rates + metals) =========
BASKET = {
    # Core beta
    "SPY": "S&P 500", "IVV": "S&P 500 (iShares)", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "EFA": "Developed ex-US", "EEM": "Emerging Mkts",
    # Sectors (SPDR)
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLI": "Industrials",
    # Credit & Rates (iShares)
    "HYG": "High Yield", "LQD": "IG Corp", "TLT": "20+Y Treasuries", "IEF": "7–10Y Treas", "SHY": "1–3Y Treas",
    # Cash-like & metals
    "BIL": "T-Bills 1–3M", "GLD": "Gold", "SLV": "Silver",
    # Value/low vol & alt beta
    "USMV": "US Min Vol", "VLUE": "US Value"
}
TICKERS = list(BASKET.keys())

# ========= Helpers =========
@st.cache_data(ttl=900, show_spinner=True)
def batch_prices(tickers, start_dt, end_dt):
    """Batch download CLOSE (unadjusted) for proper $ flows with SO."""
    df = yf.download(
        tickers, start=start_dt, end=end_dt,
        auto_adjust=False, progress=False, group_by="ticker", threads=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        closes = {t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns}
        if not closes: return pd.DataFrame()
        out = pd.DataFrame(closes).sort_index().ffill()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out
    elif "Close" in df.columns:
        out = pd.DataFrame({tickers[0]: df["Close"]}).sort_index().ffill()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out
    return pd.DataFrame()

def human_money(x, signed=True, decimals=2):
    if x is None or pd.isna(x): return ""
    s = "+" if (signed and x > 0) else ("-" if (signed and x < 0) else "")
    ax = abs(float(x))
    if   ax >= 1e9: val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{s}${val:,.{decimals}f}{suf}"

def tick_money(x, _):
    ax = abs(float(x)); s = "-" if x < 0 else ""
    if   ax >= 1e9: val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{s}${val:,.2f}{suf}"

def normalize_so_units(so: pd.Series):
    """If median SO looks like 'millions' (<1e4), scale to shares."""
    so = (so or pd.Series(dtype=float)).dropna().astype(float)
    if so.empty: return so, "no_SO"
    return (so*1_000_000.0, "SO(millions→shares)") if float(so.median()) < 1e4 else (so, "SO(shares)")

def fetch_so_series(ticker: str) -> pd.Series | None:
    """yfinance SO (can be spotty). Returns Series indexed by date or None."""
    try:
        so = yf.Ticker(ticker).get_shares_full()
    except Exception:
        return None
    if so is None: return None
    s = pd.Series(so).dropna()
    if s.empty: return None
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s, _ = normalize_so_units(s)
    return s

def compute_flows_for_ticker(ticker: str, px: pd.Series, freq: str):
    """
    Returns:
      df_flow: DataFrame with columns ['Flow ($)'] aggregated to freq
      net: period net flow (float)
    """
    so = fetch_so_series(ticker)
    if so is None: return None, None
    # align to trading days in window
    pxw = px.loc[start:today].dropna()
    if pxw.empty: return None, None
    so_al = so.reindex(pxw.index, method="ffill").dropna()
    if so_al.empty: return None, None
    # daily ΔSO × Close
    dso = so_al.diff().fillna(0.0)
    flow_daily = dso * pxw.loc[so_al.index]
    if freq == "W":
        flow = flow_daily.resample("W-FRI").sum()
    else:
        flow = flow_daily
    return flow.to_frame(name="Flow ($)"), float(flow.sum())

# ========= Compute =========
closes = batch_prices(TICKERS, start, today)
if closes.empty:
    st.error("Couldn’t download prices. Try again later.")
    st.stop()

freq = "W" if agg == "Weekly" else "D"

rows = []
per_ticker_flows = {}
missing = []
for t in TICKERS:
    if t not in closes.columns:
        missing.append((t, BASKET[t], "no_price"))
        continue
    df_flow, net = compute_flows_for_ticker(t, closes[t], freq)
    if df_flow is None:
        missing.append((t, BASKET[t], "no_SO"))
        continue
    per_ticker_flows[t] = df_flow
    rows.append({
        "Ticker": t,
        "Label": f"{BASKET[t]} ({t})",
        "Net Flow ($)": net
    })

# If nothing usable, tell user clearly (but this basket should usually work)
if not rows:
    st.error("No ETFs returned usable Shares Outstanding from Yahoo. "
             "Try switching lookback/aggregation or consider adding issuer feeds later.")
    if show_missing and missing:
        st.markdown("### Missing (no reliable SO)")
        st.dataframe(pd.DataFrame(missing, columns=["Ticker","Label","Reason"]), use_container_width=True)
    st.stop()

df = pd.DataFrame(rows).sort_values("Net Flow ($)", ascending=False).reset_index(drop=True)

# ========= UI =========
st.title("ETF Flows — Core 20")
st.caption(f"Period: **{lb_key}** (ending {today.date()}), Aggregation: **{agg}** • Flows = ΔSO × Close")

# Top bar chart
vals = df["Net Flow ($)"].to_numpy()
fig, ax = plt.subplots(figsize=(15, max(6, len(df)*0.42)))
colors = np.where(vals >= 0, "#2ca02c", "#d62728")
bars = ax.barh(df["Label"], vals, color=colors, alpha=0.9)
ax.invert_yaxis()
ax.set_xlabel("Net Flow ($ with k/M/B)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_money))

x_lo, x_hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
pad = (x_hi - x_lo) * 0.15 if x_hi != x_lo else 1.0
ax.set_xlim([x_lo - pad, x_hi + pad])

for bar, v in zip(bars, vals):
    txt = human_money(v, signed=True, decimals=2)
    x = bar.get_width()
    ha = "left" if x >= 0 else "right"
    off = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.text(x + (off if x >= 0 else -off),
            bar.get_y() + bar.get_height()/2, txt,
            va="center", ha=ha, fontsize=10)

plt.tight_layout()
st.pyplot(fig)

# Summary tiles
c1, c2, c3 = st.columns(3)
total_in = df.loc[df["Net Flow ($)"] > 0, "Net Flow ($)"].sum()
total_out = df.loc[df["Net Flow ($)"] < 0, "Net Flow ($)"].sum()
coverage = len(df) / len(TICKERS)
c1.metric("Total Inflows", human_money(total_in, True))
c2.metric("Total Outflows", human_money(total_out, True))
c3.metric("Coverage (SO available)", f"{coverage:.0%}")

# Time-series panel: stackable flows & cumulative
st.markdown("### Flow Time Series")
# align to common index
panel = pd.concat([per_ticker_flows[t]["Flow ($)"].rename(t) for t in per_ticker_flows], axis=1).fillna(0.0)
panel["Total Flow"] = panel.sum(axis=1)
panel["Cumulative"] = panel["Total Flow"].cumsum()

c4, c5 = st.columns(2)
with c4:
    st.line_chart(panel["Total Flow"], height=260, use_container_width=True)
    st.caption("Total flow per period")
with c5:
    st.line_chart(panel["Cumulative"], height=260, use_container_width=True)
    st.caption("Cumulative flow over the lookback")

# Top In/Out tables
st.markdown("### Top Inflows & Outflows")
left, right = st.columns(2)
with left:
    top_in = df.head(5).copy()
    top_in["Net Flow ($)"] = top_in["Net Flow ($)"].map(lambda v: human_money(v, True))
    st.table(top_in.set_index("Label")[["Net Flow ($)"]])
with right:
    top_out = df.sort_values("Net Flow ($)").head(5).copy()
    top_out["Net Flow ($)"] = top_out["Net Flow ($)"].map(lambda v: human_money(v, True))
    st.table(top_out.set_index("Label")[["Net Flow ($)"]])

# Missing table
if show_missing and missing:
    st.markdown("### Missing (no reliable SO from Yahoo)")
    st.dataframe(pd.DataFrame(missing, columns=["Ticker","Label","Reason"]), use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
