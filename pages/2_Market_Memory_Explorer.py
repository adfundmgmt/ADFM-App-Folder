import datetime as dt
import time
from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

MIN_DAYS_REQUIRED = 30
MIN_DAYS_CURRENT_YEAR = 5
MIN_DAYS_FOR_CORR = 10
CACHE_TTL_SECONDS = 3600

st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mount/src/adfm-app-folder/logo.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

with st.sidebar:

    st.header("About This Tool")

    st.markdown(
        """
Explore how this year's cumulative return path compares to history.

• Pulls adjusted daily closes from Yahoo Finance  
• Aligns each calendar year by trading day to build YTD paths  
• Selects historical analogs using correlation to the current year so far  
• Displays full-year paths for selected analogs  
"""
    )

    st.markdown("---")

    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    f_jumps = st.checkbox("Exclude analogs with large daily jumps", value=False)

    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300))

    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25)

col1, col2, col3 = st.columns([2, 1, 1])

ticker = col1.text_input("Ticker", "^GSPC").upper()
top_n = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min Correlation", 0.00, 1.00, 0.00, 0.05)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol):

    attempts = 0
    delay = 1

    while attempts < 4:

        try:
            df = yf.download(symbol, period="max", auto_adjust=True, progress=False)
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            break

        attempts += 1
        time.sleep(delay)
        delay *= 2

    if df.empty:
        raise ValueError("Yahoo returned no usable data")

    df = df[["Close"]].dropna()
    df["Year"] = df.index.year

    return df


def cumret(series):

    return series.div(series.iloc[0]).sub(1)


def build_year_matrix(df):

    paths = {}

    this_year = dt.datetime.now().year

    for yr, grp in df.groupby("Year"):

        closes = grp["Close"].dropna()

        if yr == this_year:
            if len(closes) < MIN_DAYS_CURRENT_YEAR:
                continue
        else:
            if len(closes) < MIN_DAYS_REQUIRED:
                continue

        ytd = cumret(closes)

        if len(ytd) < 2:
            continue

        ytd.index = np.arange(1, len(ytd) + 1)

        paths[int(yr)] = ytd

    if not paths:
        raise ValueError("No usable yearly paths")

    ytd_df = pd.concat(paths, axis=1)

    ytd_df.columns = ytd_df.columns.astype(int)

    return ytd_df


def palette(n):

    cmap = plt.cm.tab20(np.linspace(0, 1, max(n, 20)))

    return cmap[:n, :3]


raw = load_history(ticker)

ytd_df = build_year_matrix(raw)

this_year = dt.datetime.now().year

if this_year not in ytd_df.columns:

    st.warning("Current year data unavailable")
    st.stop()

current = ytd_df[this_year].dropna()

n_days = len(current)

if n_days < MIN_DAYS_FOR_CORR:

    st.info("Too few days for reliable correlation")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(current.index, current.values, color="black", lw=3)

    ax.set_title(f"{ticker} {this_year} YTD")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.grid(True, ls=":")

    st.pyplot(fig)

    st.stop()


hist = ytd_df.drop(columns=this_year)

aligned = hist.iloc[:n_days].dropna(axis=1)

corr = aligned.corrwith(current.iloc[:n_days])

corr = corr[corr >= min_corr].dropna()

if corr.empty:

    st.warning("No analog years meet correlation threshold")

    st.stop()


def keep_year(yr):

    ser = ytd_df[yr].dropna()

    if len(ser) < n_days:
        return False

    ret_n = ser.iloc[n_days - 1]

    daily = (1 + ser).div((1 + ser).shift(1)).sub(1)

    max_move = daily.iloc[:n_days].abs().max()

    if f_outliers:
        if not (lo / 100 <= ret_n <= hi / 100):
            return False

    if f_jumps:
        if max_move > max_jump / 100:
            return False

    return True


eligible = {yr: rho for yr, rho in corr.items() if keep_year(yr)}

top = sorted(eligible.items(), key=lambda x: x[1], reverse=True)[:top_n]

colors = palette(len(top))

fig, ax = plt.subplots(figsize=(14, 7))

for i, (yr, rho) in enumerate(top):

    ser = ytd_df[yr].dropna()

    ax.plot(
        ser.index,
        ser.values,
        "--",
        lw=2,
        color=colors[i],
        label=f"{yr} ρ={rho:.2f}"
    )

ax.plot(current.index, current.values, color="black", lw=3, label=f"{this_year}")

ax.axvline(n_days, color="gray", ls=":")

ax.set_title(f"{ticker} Current Year vs Historical Analogs")

ax.set_xlabel("Trading Day")

ax.set_ylabel("Cumulative Return")

ax.axhline(0, color="gray", ls="--")

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":")

ax.legend(frameon=False)

plt.tight_layout()

st.pyplot(fig)

st.caption("© 2026 AD Fund Management LP")
