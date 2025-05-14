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
Enter comma‑separated tickers below and explore price + momentum charts.
"""
)

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):",
    "AAPL, MSFT, NVDA, TSLA, AMD"
).upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# ─── Fetch Prices ──────────────────────────────────────────────────────────────
period = "6mo"
price_dict, failed = {}, []
for t in tickers:
    hist = yf.Ticker(t).history(period=period)
    if hist.empty or "Close" not in hist.columns:
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
for sym, s in price_df.items():
    if len(s) < 200:
        continue

    latest = s.iloc[-1]
    high20 = s[-20:].max()
    high50 = s[-50:].max()
    high100 = s[-100:].max()
    high200 = s[-200:].max()
    rsi7 = compute_rsi(s, 7).iloc[-1]
    rsi14 = compute_rsi(s, 14).iloc[-1]
    rsi21 = compute_rsi(s, 21).iloc[-1]

    rec = {
        "Ticker": sym,
        "Price": latest,
        "20D High": high20,
        "50D High": high50,
        "100D High": high100,
        "200D High": high200,
        "Breakout 20D": latest >= high20,
        "Breakout 50D": latest >= high50,
        "Breakout 100D": latest >= high100,
        "Breakout 200D": latest >= high200,
        "RSI (7)": round(rsi7, 1),
        "RSI (14)": round(rsi14, 1),
        "RSI (21)": round(rsi21, 1),
    }
    records.append(rec)

df = pd.DataFrame(records)
df = df.sort_values(
    by=["Breakout 200D", "Breakout 100D", "Breakout 50D", "Breakout 20D"],
    ascending=False
)

# ─── Style Table ───────────────────────────────────────────────────────────────
def mark(v):
    return "✅" if v else ""

styled = (
    df.style
      .format({"Price":"{:.2f}", **{f"{w}D High":"{:.2f}" for w in (20,50,100,200)}})
      .applymap(mark, subset=["Breakout 20D","Breakout 50D","Breakout 100D","Breakout 200D"])
      .background_gradient(subset=["RSI (7)","RSI (14)","RSI (21)"], cmap="coolwarm", vmin=0, vmax=100)
)

st.subheader("Breakout & RSI Signals")
st.dataframe(styled, use_container_width=True)

# ─── Per‑Ticker Charts ─────────────────────────────────────────────────────────
sel = st.selectbox("Select ticker to chart:", df["Ticker"].tolist())
s = price_df[sel]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)

# Price & rolling highs
ax1.plot(s.index, s, color="black", label="Close")
for w, col in zip((20,50,100,200), ("#1f77b4","#ff7f0e","#2ca02c","#d62728")):
    ax1.plot(s.index, s.rolling(w).max(), color=col, lw=1, label=f"{w}D High")
ax1.set_title(f"{sel} Price & Rolling Highs")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Multi‑RSI
for w, col in zip((7,14,21), ("#9467bd","#8c564b","#e377c2")):
    ax2.plot(s.index, compute_rsi(s, w), color=col, label=f"RSI({w})")
ax2.axhline(70, ls="--", color="gray", lw=0.8)
ax2.axhline(30, ls="--", color="gray", lw=0.8)
ax2.set_title(f"{sel} RSI Indicators")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
