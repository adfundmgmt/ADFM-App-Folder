import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Enhanced Breakout Scanner")
st.sidebar.header("About")
st.sidebar.info(
    """
Flags stocks breaking out to 20D/50D/100D/200D highs and shows RSI(7/14/21).
Enter comma‑separated tickers below and explore price & momentum charts.
"""
)

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):", 
    value="AAPL, MSFT, NVDA, TSLA, AMD"
).upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# ─── Fetch Prices ──────────────────────────────────────────────────────────────
period = "6mo"
price_dict = {}
failed = []

for t in tickers:
    hist = yf.Ticker(t).history(period=period)
    if hist.empty or "Close" not in hist:
        failed.append(t)
    else:
        price_dict[t] = hist["Close"].dropna()

if failed:
    st.sidebar.warning(f"Ignored invalid tickers: {', '.join(failed)}")
if not price_dict:
    st.error("No valid price data. Check tickers / internet.")
    st.stop()

price_df = pd.DataFrame(price_dict)

# ─── RSI Helper ────────────────────────────────────────────────────────────────
def compute_rsi(s, window):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs)))

# ─── Build Signal Table ─────────────────────────────────────────────────────────
records = []
for sym, series in price_df.items():
    if len(series) < 200: continue
    latest = series.iat[-1]
    highs = {
        20: series[-20:].max(),
        50: series[-50:].max(),
        100: series[-100:].max(),
        200: series[-200:].max()
    }
    rsis = {w: compute_rsi(series, w).iat[-1] for w in (7, 14, 21)}
    records.append({
        "Ticker": sym,
        "Price": latest,
        **{f"{w}D High": highs[w] for w in highs},
        **{f"Breakout{w}D": latest >= highs[w] for w in highs},
        **{f"RSI({w})": round(rsis[w], 1) for w in rsis},
    })

df = pd.DataFrame(records)
df = df.sort_values(by=["Breakout200D", "Breakout100D", "Breakout50D", "Breakout20D"], ascending=False)

# ─── Style Table ───────────────────────────────────────────────────────────────
def style_bool(v):
    if v: return "✅"
    else: return ""
  
styled = df.style.format({
    "Price": "{:.2f}",
    "20D High": "{:.2f}", "50D High": "{:.2f}", "100D High": "{:.2f}", "200D High": "{:.2f}"
}).applymap(style_bool, subset=["Breakout20D","Breakout50D","Breakout100D","Breakout200D"]) \
.applymap(lambda v: "background: #b3e6ff" if isinstance(v, (float, int)) and v>90 else "", subset=["RSI(7)","RSI(14)","RSI(21)"])

st.subheader("Breakout & RSI Signals")
st.dataframe(styled, use_container_width=True)

# ─── Per‑Ticker Charts ─────────────────────────────────────────────────────────
sel = st.selectbox("Select ticker to chart:", df["Ticker"].tolist())
s = price_df[sel].dropna()

# Price + Rolling Highs
fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax0 = ax[0]
ax0.plot(s.index, s, label="Close", color="black")
for w,color in [(20,"#1f77b4"),(50,"#ff7f0e"),(100,"#2ca02c"),(200,"#d62728")]:
    ax0.plot(s.index, s.rolling(w).max(), label=f"{w}D High", color=color, linewidth=1)
ax0.set_title(f"{sel} Price & Rolling Highs")
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

# Multi‑RSI
for w,color in zip((7,14,21),
