import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings

import matplotlib.pyplot as plt
plt.style.use("default")


warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("Breakout Scanner")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Screen for stocks breaking out to 20D, 50D, 100D, or 200D highs and view multi-timeframe RSI.**
        ---
        - Enter comma‑separated tickers (e.g. `NVDA, MSFT, SPY, CL=F, TLT`)
        - Table shows current price, recent highs, breakout flags, and RSI (7, 14, 21)
        - Click any ticker for annotated price and RSI chart
        - Breakouts: ✅ = new high today vs X-day
        - RSI: Red = overbought (>80), Blue = oversold (<20)
        """
    )

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):",
    "NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, AVGO, TSM"
).upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

min_price = st.sidebar.number_input("Min Price (to filter penny stocks)", value=2.0, step=0.5)

# ─── Fetch Prices (1 Year) ──────────────────────────────────────────────────────
price_dict, failed = {}, []
for t in tickers:
    try:
        hist = yf.Ticker(t).history(period="1y")
    except Exception:
        hist = pd.DataFrame()
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
# Filter by min price (last value)
valid_tickers = [t for t in price_df.columns if price_df[t].iloc[-1] >= min_price]
price_df = price_df[valid_tickers]
if price_df.empty:
    st.error(f"No tickers above min price ${min_price:.2f}.")
    st.stop()

# ─── RSI Helper ────────────────────────────────────────────────────────────────
def compute_rsi(s, window):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ─── Build Signal Table ─────────────────────────────────────────────────────────
records = []
for sym, s in price_df.items():
    if len(s) < 200:
        continue
    latest = s.iloc[-1]
    highs = {w: s.rolling(w).max().iloc[-1] for w in (20, 50, 100, 200)}
    rsis  = {w: compute_rsi(s, w).iloc[-1] for w in (7, 14, 21)}

    rec = {
        "Ticker": sym,
        "Price": round(latest, 2),
    }
    for w in highs:
        rec[f"{w}D High"]    = round(highs[w], 2)
        rec[f"Breakout {w}D"] = latest >= highs[w]
    for w in rsis:
        rec[f"RSI ({w})"] = round(rsis[w], 1)
    records.append(rec)

df = pd.DataFrame(records)

if df.empty:
    st.info("No breakouts detected or insufficient data (need ≥200 days & above min price).")
    st.stop()

# ─── Sort by breakout priority ─────────────────────────────────────────────────
break_cols = [c for c in df.columns if c.startswith("Breakout")]
df = df.sort_values(by=break_cols + ["Price"], ascending=False).reset_index(drop=True)

# --- Table Styling Improvements ---
def styled_table(df):
    df_disp = df.copy()
    # Emoji for breakouts (✅ or blank)
    for col in break_cols:
        df_disp[col] = df_disp[col].map(lambda x: "✅" if x else "")
    # Format floats to two decimals, RSIs to one decimal
    float_cols = [c for c in df_disp.columns if "High" in c or c == "Price"]
    rsi_cols = [c for c in df_disp.columns if c.startswith("RSI")]
    for c in float_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:,.2f}")
    for c in rsi_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:.1f}")

    # Color code RSIs: >80 red (overbought), <20 blue (oversold)
    def color_rsi(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 80:
            return "color: #d62728; font-weight:bold;"   # red
        elif v <= 20:
            return "color: #1f77b4; font-weight:bold;"   # blue
        return ""

    styled = (
        df_disp.style
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '13px'), ('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('font-size', '13px'), ('text-align', 'center')]},
        ])
        .applymap(color_rsi, subset=rsi_cols)
        .set_properties(**{'background-color': '#f8fafc'}, subset=pd.IndexSlice[:, ["Price"]])
        .set_properties(**{'border': '1.5px solid #e0e0e0'})
    )
    return styled

# --- Section header (remove redundancy) ---
st.markdown("### Breakout & RSI Signals", help="✅ = closing at a new X-day high. Table is sorted by most recent breakouts.")

st.dataframe(styled_table(df), use_container_width=True)

# --- Download Button remains below table ---
st.download_button(
    "Download as CSV",
    df.to_csv(index=False),
    file_name="breakout_signals.csv"
)

# ─── Per‑Ticker Charts ─────────────────────────────────────────────────────────
sel = st.selectbox("Select ticker to chart:", df["Ticker"])
s = price_df[sel]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(11,7), sharex=True)

# Price & rolling highs
ax1.plot(s.index, s, color="black", label="Close", linewidth=2)
for w, col in zip((20,50,100,200), ("#1f77b4","#ff7f0e","#2ca02c","#d62728")):
    ax1.plot(s.index, s.rolling(w).max(), color=col, lw=1.1, label=f"{w}D High")
ax1.set_title(f"{sel} Price & Rolling Highs", fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylabel("Price")
fig.autofmt_xdate()

# Multi‑RSI
for w, col in zip((7,14,21), ("#9467bd","#8c564b","#e377c2")):
    ax2.plot(s.index, compute_rsi(s, w), color=col, label=f"RSI({w})")
ax2.axhline(80, ls="--", color="gray", lw=0.9, label="RSI 80")
ax2.axhline(20, ls="--", color="gray", lw=0.9, label="RSI 20")
ax2.set_title(f"{sel} RSI Indicators", fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")

fig.tight_layout(h_pad=2)
st.pyplot(fig)

st.caption("© 2025 AD Fund Management LP")
