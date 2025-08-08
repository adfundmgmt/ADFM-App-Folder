# ──────────────────────────────────────────────────────────────────────────
#  Market Memory Explorer  –  AD Fund Management LP
#  ------------------------------------------------
#  v1.5  ·  fixed jump filter, adjusted prices, dynamic y-limits, filter-first,
#           dynamic tick step, added downloads, minor hygiene
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

TRADING_DAYS_FULL_YEAR  = 253        # long-run average
MIN_DAYS_REQUIRED       = 30         # drop stub years
CACHE_TTL_SECONDS       = 3600       # 1-hour data cache

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
Quickly benchmark **this year’s cumulative return** against history.

**What it does**
- Pulls daily closes from Yahoo Finance for any stock, ETF, or index  
- Aligns every calendar year by trading-day #: 1 → n  
- Finds & overlays the *n* most-correlated analogue years (ρ)

**Why it helps**
- Spot repeating return arcs early  
- Gauge where we stand inside bullish or bearish road-maps  
- Stress-test price targets with real precedent

**Extras**
- Optional filters to drop extreme years or > X % one-day jumps  
- One-click download of YTD paths & correlation table
"""
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
    """Yahoo fetch with retries. Returns df[Close, Year] using adjusted prices."""
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

# ── Download & build YTD paths ───────────────────────────────────────────
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
    ytd.index = np.arange(1, len(closes) + 1)  # trading-day index
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

# ── Correlation table (shape-only option shown in comment) ───────────────
corrs = {}
for yr, series in ytd_df.items():
    if yr == this_year:
        continue
    clean = series.dropna()
    if len(clean) < n_days:
        continue
    x = current.values
    y = clean.iloc[:n_days].values
    # shape-only alternative:
    # x, y = x - x.mean(), y - y.mean()
    rho = np.corrcoef(x, y)[0, 1]
    if rho >= min_corr:
        corrs[yr] = rho

if not corrs:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# ── Optional filters, then take Top-N (filter-first ordering) ────────────
def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        return False
    ret_n = ser.iloc[n_days - 1]  # cumulative YTD
    daily_ret = (1.0 + ser).pct_change()  # true daily move computed on price index
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

finals = []
for yr, _ in top:
    ser = ytd_df[yr].dropna()
    if not ser.empty:
        finals.append(float(ser.iloc[-1]))

median_final = float(np.nanmedian(finals)) if finals else np.nan
sigma_final  = float(np.nanstd(finals))    if finals else np.nan
fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD",         fmt(current_ret))
m2.metric("Median Final Return",      fmt(median_final))
m3.metric("Analog Dispersion (σ)",    fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot ─────────────────────────────────────────────────────────────────
palette = plt.cm.get_cmap("tab10" if len(top) <= 10 else "tab20")(
    np.linspace(0, 1, len(top))
)

fig, ax = plt.subplots(figsize=(14, 7))

for idx, (yr, rho) in enumerate(top):
    ser = ytd_df[yr].dropna()
    ax.plot(ser.index, ser.values, "--", lw=2, alpha=0.7,
            color=palette[idx], label=f"{yr} (ρ={rho:.2f})")

ax.plot(current.index, current.values,
        color="black", lw=3.2, label=f"{this_year} (YTD)")
ax.axvline(current.index[-1], color="gray", ls=":", lw=1.3, alpha=0.7)

ax.set_title(f"{ticker} — {this_year} vs Historical Analogs",
             fontsize=16, weight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return",   fontsize=13)
ax.axhline(0, color="gray", ls="--", lw=1)

# x-limit from data rather than hard 253
xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
ax.set_xlim(1, xmax)

# dynamic y-limits based on first n_days of analogs
all_y = np.hstack([current.values] +
                  [ytd_df[yr].dropna().values[:n_days] for yr, _ in top])
if all_y.size:
    rng = float(all_y.max() - all_y.min())
    pad = 0.08 * rng if rng > 0 else 0.02
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

# dynamic tick step
rng_total = ax.get_ylim()[1] - ax.get_ylim()[0]
step = 0.05 if rng_total <= 1.0 else 0.10
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":", lw=0.7, color="#888")
ax.legend(loc="best", frameon=False, ncol=2, fontsize=11)
plt.tight_layout()
st.pyplot(fig)

# ── Downloads ────────────────────────────────────────────────────────────
st.subheader("Downloads")

# YTD paths, truncated to n_days for comparability
paths_trunc = ytd_df.apply(lambda s: s.dropna().iloc[:n_days])
csv_paths = paths_trunc.to_csv(index_label="TradingDay")
st.download_button(
    "Download YTD Paths (first n days)",
    data=csv_paths,
    file_name=f"{ticker}_ytd_paths_first_{n_days}_days.csv",
    mime="text/csv",
)

# Correlation table (pre- and post-filter views)
corr_df = pd.DataFrame(
    sorted(corrs.items(), key=lambda kv: kv[1], reverse=True),
    columns=["Year", "Corr"]
)
csv_corr_all = corr_df.to_csv(index=False)

top_df = pd.DataFrame(top, columns=["Year", "Corr"])
csv_corr_top = top_df.to_csv(index=False)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Correlations (all eligible)",
        data=csv_corr_all,
        file_name=f"{ticker}_correlations_all.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Download Correlations (top shown)",
        data=csv_corr_top,
        file_name=f"{ticker}_correlations_top.csv",
        mime="text/csv",
    )

st.caption("© 2025 AD Fund Management LP")
