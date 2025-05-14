import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime, timedelta

# … [keep your sidebar, data fetch, and indicator code exactly as before] …

# --------------------------------------------------
# TRIM TO DISPLAY WINDOW
# --------------------------------------------------
if period != "max":
    display_days = period_map.get(period, 365)
    cutoff = datetime.today() - timedelta(days=display_days)
    df_plot = df.loc[df.index >= cutoff]
else:
    df_plot = df.copy()

# Prepare OHLC
ohlc = df_plot[["Open","High","Low","Close"]].copy()
ohlc.reset_index(inplace=True)
ohlc["Date"] = mdates.date2num(ohlc["Date"])
ohlc = ohlc[["Date","Open","High","Low","Close"]]

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
grid_opts = {"height_ratios": [5, 1, 1, 1], "hspace": 0.25}
fig, (ax_price, ax_vol, ax_rsi, ax_macd) = plt.subplots(
    4, 1, figsize=(14, 10), sharex=True, gridspec_kw=grid_opts
)

# --------------------------------------------------
# Price & MAs as Candlestick + moving averages
# --------------------------------------------------
ax_price.set_title(f"{ticker} Price & Moving Averages", fontsize=18, weight="bold", pad=20)

candlestick_ohlc(
    ax_price,
    ohlc.values,
    width=0.6,               # adjust to taste
    colorup="green",
    colordown="red",
    alpha=0.8
)

for w, col in zip((20, 50, 100, 200), ("purple", "blue", "orange", "gray")):
    ax_price.plot(
        df_plot.index,
        df_plot[f"MA{w}"],
        label=f"MA{w}",
        color=col,
        linewidth=1
    )

ax_price.legend(loc="upper left", fontsize=10)
ax_price.grid(alpha=0.3)
ax_price.xaxis_date()        # ensure date axis

# --------------------------------------------------
# Volume
# --------------------------------------------------
vol_colors = ["green" if c>=o else "red"
              for c,o in zip(df_plot["Close"],df_plot["Open"])]
ax_vol.bar(df_plot.index, df_plot["Volume"],
           color=vol_colors, alpha=0.3, width=1)
ax_vol.set_ylabel("Volume", color="gray", fontsize=10)
ax_vol.tick_params(axis="y", labelcolor="gray", labelsize=8)
ax_vol.grid(alpha=0.3)

# … [RSI, MACD, x‑axis formatting, stb as before] …

fig.tight_layout()
st.pyplot(fig)
st.caption("© 2025 AD Fund Management LP")
