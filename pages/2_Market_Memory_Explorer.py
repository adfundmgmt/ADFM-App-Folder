# ──────────────────────────────────────────────────────────────────────────
#  Market Memory Explorer  –  AD Fund Management LP
#  ------------------------------------------------
#  v1.8  ·  dynamic percent tick density using MultipleLocator
# ──────────────────────────────────────────────────────────────────────────
import datetime as dt
import time
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

TRADING_DAYS_FULL_YEAR  = 253
MIN_DAYS_REQUIRED       = 30
CACHE_TTL_SECONDS       = 3600

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore how this year's cumulative return path compares to history.

        • Pulls adjusted daily closes from Yahoo Finance  
        • Aligns each calendar year by trading day to build YTD paths  
        • Computes correlations (ρ) between the current year and all past years  
        • Overlays the highest-correlation analogue paths for visual comparison  

        Use the filters below to exclude extreme years or large jump days, and export CSV snapshots for further work.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Filters (optional)")
    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    f_jumps    = st.checkbox("Exclude analogs with large daily jumps",  value=False)
    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300), 1)
    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25, 1)
    st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])
ticker   = col1.text_input("Ticker", "^GSPC").upper()
top_n    = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min ρ", min_value=0.00, max_value=1.00,
                       value=0.00, step=0.05, format="%.2f")

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Data fetch helper ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str, auto_adjust: bool = True) -> pd.DataFrame:
    attempts, delay = 0, 1
    df = pd.DataFrame()
    while attempts < 4:
        try:
            df = yf.Ticker(symbol).history(period="max", auto_adjust=auto_adjust)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay)
        delay *= 2
    if df.empty:
        raise ValueError("Yahoo returned no data after 4 attempts.")
    if "Close" not in df.columns:
        raise ValueError("'Close' column missing.")
    df = df.loc[df["Close"].notna(), ["Close"]].copy()
    df["Year"] = df.index.year
    return df

def cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1

# ── Build YTD paths ──────────────────────────────────────────────────────
try:
    raw = load_history(ticker, auto_adjust=True)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

paths = {}
for yr, grp in raw.groupby("Year"):
    closes = grp["Close"].dropna()
    if len(closes) < MIN_DAYS_REQUIRED:
        continue
    ytd = cumret(closes)
    ytd.index = np.arange(1, len(closes) + 1)
    paths[yr] = ytd

if not paths:
    st.error("No usable yearly data found.")
    st.stop()

ytd_df    = pd.DataFrame(paths)
this_year = dt.datetime.now().year
if this_year not in ytd_df.columns:
    st.warning(f"No YTD data for {this_year}")
    st.stop()

current = ytd_df[this_year].dropna()
n_days  = len(current)

# ── Correlations ─────────────────────────────────────────────────────────
corrs = {}
for yr, series in ytd_df.items():
    if yr == this_year:
        continue
    clean = series.dropna()
    if len(clean) < n_days:
        continue
    x = current.values
    y = clean.iloc[:n_days].values
    rho = np.corrcoef(x, y)[0, 1]
    if rho >= min_corr:
        corrs[yr] = rho

if not corrs:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# ── Filters then Top-N ───────────────────────────────────────────────────
def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        return False
    ret_n = ser.iloc[n_days - 1]
    daily_ret = (1.0 + ser).pct_change()
    max_d = daily_ret.abs().max()
    if f_outliers and not (lo/100 <= ret_n <= hi/100):
        return False
    if f_jumps and max_d > max_jump/100:
        return False
    return True

eligible = {yr: rho for yr, rho in corrs.items() if keep_year(yr)}
if not eligible:
    st.info("All candidates excluded by your filters.")
    st.stop()

top = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

# ── Metrics ──────────────────────────────────────────────────────────────
current_ret = float(current.iloc[-1])
finals = [float(ytd_df[yr].dropna().iloc[-1]) for yr, _ in top if not ytd_df[yr].dropna().empty]
median_final = float(np.nanmedian(finals)) if finals else np.nan
sigma_final  = float(np.nanstd(finals))    if finals else np.nan
fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD",         fmt(current_ret))
m2.metric("Median Final Return",      fmt(median_final))
m3.metric("Analog Dispersion (σ)",    fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot ─────────────────────────────────────────────────────────────────
palette = plt.cm.get_cmap("tab10" if len(top) <= 10 else "tab20")(np.linspace(0, 1, len(top)))
fig, ax = plt.subplots(figsize=(14, 7))

for idx, (yr, rho) in enumerate(top):
    ser = ytd_df[yr].dropna()
    ax.plot(ser.index, ser.values, "--", lw=2, alpha=0.7,
            color=palette[idx], label=f"{yr} (ρ={rho:.2f})")

ax.plot(current.index, current.values, color="black", lw=3.2, label=f"{this_year} (YTD)")
ax.axvline(current.index[-1], color="gray", ls=":", lw=1.3, alpha=0.7)

ax.set_title(f"{ticker} - {this_year} vs Historical Analogs", fontsize=16, weight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return",   fontsize=13)
ax.axhline(0, color="gray", ls="--", lw=1)

# x-limit from data
xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
ax.set_xlim(1, xmax)

# y-limits from full range of all plotted paths (capture entire move)
all_y = np.hstack([current.values] + [ytd_df[yr].dropna().values for yr, _ in top])
all_y = all_y[np.isfinite(all_y)]
if all_y.size == 0:
    ymin, ymax = -0.02, 0.02
else:
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02
ax.set_ylim(ymin - pad, ymax + pad)

# adaptive percent ticks: target ~12 major ticks, choose a clean step
span = (ax.get_ylim()[1] - ax.get_ylim()[0])
target_ticks = 12
raw_step = max(span / target_ticks, 0.0025)  # min 0.25%
candidates = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
step = float(candidates[np.argmin(np.abs(candidates - raw_step))])
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":", lw=0.7, color="#888")
ax.legend(loc="best", frameon=False, ncol=2, fontsize=11)
plt.tight_layout()
st.pyplot(fig)

# ── Downloads ────────────────────────────────────────────────────────────
st.subheader("Downloads")
paths_trunc = ytd_df.apply(lambda s: s.dropna().iloc[:n_days])
st.download_button(
    "Download YTD Paths (first n days)",
    data=paths_trunc.to_csv(index_label="TradingDay"),
    file_name=f"{ticker}_ytd_paths_first_{n_days}_days.csv",
    mime="text/csv",
)
corr_df = pd.DataFrame(sorted(corrs.items(), key=lambda kv: kv[1], reverse=True), columns=["Year", "Corr"])
top_df  = pd.DataFrame(top, columns=["Year", "Corr"])
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Correlations (all eligible)",
                       data=corr_df.to_csv(index=False),
                       file_name=f"{ticker}_correlations_all.csv",
                       mime="text/csv")
with c2:
    st.download_button("Download Correlations (top shown)",
                       data=top_df.to_csv(index=False),
                       file_name=f"{ticker}_correlations_top.csv",
                       mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
