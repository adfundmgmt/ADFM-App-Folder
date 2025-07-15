import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── Basket definitions ───────────────────────────────────
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

# ─── Streamlit page setup ─────────────────────────────────
st.set_page_config(layout="wide",
                   page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight")

# ─── Sidebar: info + look‑back selector ───────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
Tracks the equal‑weighted ratio of S&P cyclical vs defensive sector ETFs to
gauge risk‑on / risk‑off positioning.

**Method**
- Cyclical basket: XLK XLI XLF XLC XLY  
- Defensive basket: XLP XLE XLV XLRE XLB XLU  
- Plots the cumulative‑return ratio with 50‑ & 200‑day MAs + RSI‑14.
""")

    st.subheader("Time Frame")
    LOOKBACKS = {
        "3 Months": 90, "6 Months": 180, "9 Months": 270,
        "YTD": None, "1 Year": 365,
        "3 Years": 365*3, "5 Years": 365*5, "10 Years": 365*10
    }
    time_choice = st.selectbox("Look‑back:", list(LOOKBACKS.keys()),
                               index=list(LOOKBACKS.keys()).index("5 Years"))

# ─── Helper: cached price fetch ───────────────────────────
@st.cache_data(ttl=3600, show_spinner="Pulling Yahoo prices…")
def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    return yf.download(tickers, start=start, end=end,
                       group_by="ticker", auto_adjust=True,
                       progress=False)

# ─── Date window set‑up ───────────────────────────────────
today = datetime.today()
hist_start = today - timedelta(days=365*10 + 220)  # 10 yrs + MA buffer
display_start = (
    datetime(today.year, 1, 1)
    if time_choice == "YTD"
    else today - timedelta(days=LOOKBACKS[time_choice])
)
start_str, end_str = hist_start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')

# ─── Basket builder ───────────────────────────────────────
def build_basket(etfs):
    raw = fetch_prices(etfs, start_str, end_str)
    missing = [t for t in etfs if t not in raw.columns.get_level_values(0)]
    if missing:
        st.warning(f"Skipped missing tickers: {', '.join(missing)}")
    closes = (raw.xs("Close", level=1, axis=1)
              if isinstance(raw.columns, pd.MultiIndex)
              else raw[["Close"]])
    closes = closes.fillna(method="ffill").dropna()
    basket = (1 + closes.pct_change()).cumprod().mean(axis=1)
    return basket

cyc = build_basket(CYCLICALS)
defn = build_basket(DEFENSIVES)

ratio = (cyc / defn * 100).dropna()
ma50, ma200 = ratio.rolling(50).mean(), ratio.rolling(200).mean()
rsi = lambda s: 100 - 100 / (1 + (s.diff().clip(lower=0).rolling(14).mean() /
                                  -s.diff().clip(upper=0).rolling(14).mean()))
rsi_series = rsi(ratio)

# ─── Slice to look‑back window (MAs stay intact) ──────────
mask = ratio.index >= display_start
ratio_disp, rsi_disp = ratio[mask], rsi_series[mask]

# ─── Dashboard metrics strip ──────────────────────────────
ret_pct = (ratio_disp.iloc[-1] / ratio_disp.iloc[0] - 1) * 100
rsi_now = rsi_disp.iloc[-1]
regime = "Bullish (50>200)" if ma50.iloc[-1] > ma200.iloc[-1] else "Bearish (50<200)"
col1, col2, col3 = st.columns(3)
col1.metric("Return over window", f"{ret_pct:.1f} %")
col2.metric("RSI‑14", f"{rsi_now:.0f}")
col3.metric("Trend Regime", regime)

# ─── Plot: ratio + MAs ────────────────────────────────────
fig = go.Figure()
fig.add_scatter(x=ratio_disp.index, y=ratio_disp, name="Cyc/Def", line=dict(color="#355E3B", width=2))
fig.add_scatter(x=ma50.index, y=ma50, name="50‑DMA", line=dict(color="blue", width=2))
fig.add_scatter(x=ma200.index, y=ma200, name="200‑DMA", line=dict(color="red", width=2))
fig.update_layout(height=550, margin=dict(l=20, r=20, t=20, b=20),
                  font=dict(size=15), yaxis_title="Relative Ratio",
                  plot_bgcolor="white", legend_orientation="h")

# ─── Plot: RSI ────────────────────────────────────────────
fig_rsi = go.Figure()
fig_rsi.add_scatter(x=rsi_disp.index, y=rsi_disp, name="RSI", line=dict(color="black", width=2))
for lvl, clr in [(70, "red"), (30, "green")]:
    fig_rsi.add_hline(y=lvl, line_dash="dot", line_color=clr)
fig_rsi.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30),
                      font=dict(size=14), yaxis=dict(title="RSI", range=[0, 100]),
                      plot_bgcolor="white", showlegend=False)

# ─── Streamlit output ────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig_rsi, use_container_width=True)

# ─── Download button ─────────────────────────────────────
csv = pd.concat([ratio_disp.rename("Ratio"),
                 ma50.rename("MA50"), ma200.rename("MA200"),
                 rsi_disp.rename("RSI")], axis=1).dropna().to_csv().encode()
st.download_button("Download data (CSV)", csv, "cyc_def_ratio.csv")
