# ──────────────────────────────────────────────────────────────────────────
#  Market Memory Explorer  –  AD Fund Management LP
#  ------------------------------------------------
#  v1.3  ·  guards against analog years shorter than current year
# ──────────────────────────────────────────────────────────────────────────
import datetime as dt
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

START_YEAR              = 1980
TRADING_DAYS_FULL_YEAR  = 253        # NYSE / NASDAQ long-run average
MIN_DAYS_REQUIRED       = 30         # discard stub years entirely
CACHE_TTL_SECONDS       = 3600       # 1-hour data cache

# ── Streamlit page config ────────────────────────────────────────────────
st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

# ── Sidebar – controls and filters ───────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.caption(
        "Overlay this year’s cumulative return path (black) with prior years that "
        "exhibit the highest correlation up to today’s trading date."
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
def load_history(symbol: str) -> pd.DataFrame:
    """Robust Yahoo fetch with retries & single-ticker endpoint."""
    attempts, delay = 0, 1
    while attempts < 4:
        df = yf.Ticker(symbol).history(period="max", auto_adjust=False)
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

# ── Download data & build YTD paths ──────────────────────────────────────
try:
    raw = load_history(ticker)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

paths = {}
for yr, grp in raw.groupby("Year"):
    closes = grp["Close"].dropna()
    if len(closes) < MIN_DAYS_REQUIRED:
        continue
    ytd = cumret(closes)
    ytd.index = np.arange(1, len(closes) + 1)        # trading-day index
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

# ── Correlation league table ─────────────────────────────────────────────
corrs = {}
for yr, series in ytd_df.items():
    if yr == this_year:
        continue
    clean = series.dropna()
    if len(clean) < n_days:                       # NEW: skip shorter years early
        continue
    rho = np.corrcoef(current.values, clean.iloc[:n_days].values)[0, 1]
    if rho >= min_corr:
        corrs[yr] = rho

top = sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
if not top:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# ── Apply optional filters ───────────────────────────────────────────────
def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:          # NEW: guard against IndexError
        return False
    ret   = ser.iloc[n_days - 1]   # YTD as-of equal day count
    max_d = ser.pct_change().abs().max()
    if f_outliers and not (lo/100 < ret < hi/100):
        return False
    if f_jumps and max_d > max_jump/100:
        return False
    return True

valid = [(yr, rho) for yr, rho in top if keep_year(yr)]
if not valid:
    st.info("All top matches excluded by your filters.")
    st.stop()

# ── Metrics ──────────────────────────────────────────────────────────────
current_ret = current.iloc[-1]

finals = []
for yr, _ in valid:
    ser = ytd_df[yr].dropna()
    if not ser.empty:
        finals.append(ser.iloc[-1])

if finals:
    median_final = float(np.nanmedian(finals))
    sigma_final  = float(np.nanstd(finals))
else:
    median_final = sigma_final = np.nan

def fmt(x):
    return "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD",         fmt(current_ret))
m2.metric("Median Final Return",      fmt(median_final))
m3.metric("Analog Dispersion (σ)",    fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot ─────────────────────────────────────────────────────────────────
palette = plt.cm.get_cmap("tab10" if len(valid) <= 10 else "tab20")(
    np.linspace(0, 1, len(valid))
)

fig, ax = plt.subplots(figsize=(14, 7))

for idx, (yr, rho) in enumerate(valid):
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
ax.set_xlim(1, TRADING_DAYS_FULL_YEAR)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, ls=":", lw=0.7, color="#888")

# dynamic y-limits with 8 % padding
all_y = np.hstack([current.values] +
                  [ytd_df[yr].dropna().values for yr, _ in valid])
pad   = 0.08 * (all_y.max() - all_y.min())
ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=11)
plt.tight_layout()
st.pyplot(fig)

st.caption("© 2025 AD Fund Management LP")
