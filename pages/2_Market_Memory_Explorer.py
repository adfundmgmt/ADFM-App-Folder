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

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
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
        • Displays full-year paths for selected analogs to show how similar setups resolved  
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Filters (optional)")

    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    f_jumps = st.checkbox("Exclude analogs with large daily jumps", value=False)

    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300), 1)

    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25, 1)

    st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

ticker_in = col1.text_input("Ticker", "^SPX").upper()
top_n = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f")

TICKER_ALIASES = {
    "^SPX": "^GSPC"
}

ticker = TICKER_ALIASES.get(ticker_in, ticker_in)

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str) -> pd.DataFrame:

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

    if df.empty or "Close" not in df.columns:
        raise ValueError("Yahoo returned no usable data.")

    df = df[["Close"]].dropna().copy()
    df["Year"] = df.index.year

    return df


def cumret(series: pd.Series) -> pd.Series:
    return series.div(series.iloc[0]).sub(1)


def compute_palette(n):

    if n <= 0:
        return []

    base = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]

    colors = list(base)

    if n <= len(colors):
        return colors[:n]

    extra = plt.cm.hsv(np.linspace(0, 1, n - len(colors)))[:, :3]

    return colors + list(extra)


try:
    raw = load_history(ticker)

except Exception as e:

    st.error(f"Download failed: {e}")
    st.stop()

this_year = dt.datetime.now().year

paths = {}

for yr, grp in raw.groupby("Year"):

    closes = grp["Close"].dropna()

    if yr == this_year:

        if len(closes) < MIN_DAYS_CURRENT_YEAR:
            continue

    else:

        if len(closes) < MIN_DAYS_REQUIRED:
            continue

    ytd = cumret(closes)

    ytd.index = np.arange(1, len(ytd) + 1)

    paths[int(yr)] = ytd


if not paths:

    st.error("No usable yearly data found.")
    st.stop()


ytd_df = pd.DataFrame(paths)

if this_year not in ytd_df.columns:

    st.warning(f"No YTD data for {this_year}.")
    st.stop()


current = ytd_df[this_year].dropna()

n_days = len(current)


if n_days < MIN_DAYS_FOR_CORR:

    st.info(
        f"{this_year} has only {n_days} trading days so far. "
        f"Correlations stabilize after {MIN_DAYS_FOR_CORR} days."
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(current.index, current.values, color="black", lw=3)

    ax.set_title(f"{ticker} – {this_year} YTD")

    ax.axhline(0, color="gray", ls="--")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.grid(True, ls=":")

    st.pyplot(fig)

    st.stop()


# VECTOR CORRELATION ENGINE

hist_matrix = ytd_df.drop(columns=this_year)

aligned = hist_matrix.iloc[:n_days].dropna(axis=1)

if aligned.empty:

    st.warning("No historical years have sufficient overlap.")
    st.stop()

corr_series = aligned.corrwith(current.iloc[:n_days])

corr_series = corr_series[corr_series >= min_corr].dropna()


if corr_series.empty:

    st.warning("No historical years meet the correlation cutoff.")
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


eligible = {yr: corr for yr, corr in corr_series.items() if keep_year(yr)}

if not eligible:

    st.info("All candidates excluded by your filters.")
    st.stop()


top = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


current_ret = float(current.iloc[-1])

finals = [float(ytd_df[yr].dropna().iloc[-1]) for yr, _ in top]

median_final = np.nanmedian(finals)
sigma_final = np.nanstd(finals)

fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"


m1, m2, m3 = st.columns(3)

m1.metric(f"{this_year} YTD", fmt(current_ret))
m2.metric("Median Full-Year Return (Analogs)", fmt(median_final))
m3.metric("Analog Dispersion (σ)", fmt(sigma_final))


palette = compute_palette(len(top))

fig, ax = plt.subplots(figsize=(14, 7))


for idx, (yr, rho) in enumerate(top):

    ser = ytd_df[yr].dropna()

    ax.plot(
        ser.index,
        ser.values,
        "--",
        lw=2.3,
        color=palette[idx],
        label=f"{yr} (ρ={rho:.2f})"
    )


ax.plot(current.index, current.values, color="black", lw=3.2)

ax.axvline(n_days, color="gray", ls=":", lw=1.2)

ax.set_title(f"{ticker} – {this_year} vs Historical Analogs")

ax.set_xlabel("Trading Day of Year")
ax.set_ylabel("Cumulative Return")

ax.axhline(0, color="gray", ls="--")


xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)

ax.set_xlim(1, xmax)


all_y = np.hstack(
    [current.values] + [ytd_df[yr].dropna().values for yr, _ in top]
)

ymin = float(np.min(all_y))
ymax = float(np.max(all_y))

pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02

ax.set_ylim(ymin - pad, ymax + pad)


span = ax.get_ylim()[1] - ax.get_ylim()[0]

raw_step = max(span / 12, 0.0025)

candidates = np.array(
    [0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00]
)

step = float(candidates[np.argmin(np.abs(candidates - raw_step))])

ax.yaxis.set_major_locator(MultipleLocator(step))
ax.yaxis.set_minor_locator(MultipleLocator(step / 2))

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":", lw=0.7)

ax.legend(loc="best", frameon=False, ncol=2)

plt.tight_layout()

st.pyplot(fig)

st.caption("© 2026 AD Fund Management LP")
