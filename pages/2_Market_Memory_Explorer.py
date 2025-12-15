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
CACHE_TTL_SECONDS = 3600

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
ticker = col1.text_input("Ticker", "^GSPC").upper()
top_n = col2.slider("Top Analogs", 1, 10, 5)
min_corr = col3.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f")

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>", unsafe_allow_html=True)

# ── Data fetch helper ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str) -> pd.DataFrame:
    attempts, delay = 0, 1
    df = pd.DataFrame()
    while attempts < 4:
        try:
            df = yf.Ticker(symbol).history(period="max", auto_adjust=True)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay)
        delay *= 2

    if df.empty or "Close" not in df.columns:
        raise ValueError("Yahoo returned no usable data.")

    df = df.loc[df["Close"].notna(), ["Close"]].copy()
    df["Year"] = df.index.year
    return df

def cumret(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] - 1

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def distinct_palette(n: int):
    """
    Greedy max-min selection from a pool of strong, high-contrast qualitative colors.
    Filters out pale/pastel candidates so analog lines stay easy to separate on white.
    """
    if n <= 0:
        return []

    cmap_names = ["tab10", "Dark2", "Set1", "tab20", "tab20b", "tab20c", "Paired", "Accent"]

    candidates = []
    for name in cmap_names:
        cmap = plt.cm.get_cmap(name)
        if hasattr(cmap, "colors"):
            candidates.extend([tuple(c[:3]) for c in cmap.colors])
        else:
            candidates.extend([tuple(cmap(x)[:3]) for x in np.linspace(0, 1, 24)])

    def key_255(rgb):
        return tuple(int(round(x * 255)) for x in rgb)

    # Deduplicate
    uniq = []
    seen = set()
    for c in candidates:
        k = key_255(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)

    # Keep saturated, darker-ish colors (avoid pastels)
    strong = []
    for rgb in uniq:
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        if s >= 0.75 and 0.25 <= v <= 0.85:
            strong.append(rgb)

    pool = strong if len(strong) >= n else uniq

    def dist(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def sat(rgb):
        return colorsys.rgb_to_hsv(*rgb)[1]

    chosen = [max(pool, key=sat)]
    if n == 1:
        return chosen

    remaining = [c for c in pool if c != chosen[0]]
    while len(chosen) < n and remaining:
        best_i, best_min = None, -1.0
        for i, c in enumerate(remaining):
            dmin = min(dist(c, ch) for ch in chosen)
            if dmin > best_min:
                best_min = dmin
                best_i = i
        chosen.append(remaining.pop(best_i))

    # Absolute fallback if pool is exhausted
    if len(chosen) < n:
        extra = list(plt.cm.get_cmap("tab20")(np.linspace(0, 1, n)))  # returns RGBA
        for c in extra:
            rgb = tuple(c[:3])
            if rgb not in chosen:
                chosen.append(rgb)
            if len(chosen) == n:
                break

    return chosen[:n]

# ── Build YTD paths ──────────────────────────────────────────────────────
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
    ytd.index = np.arange(1, len(ytd) + 1)
    paths[yr] = ytd

if not paths:
    st.error("No usable yearly data found.")
    st.stop()

ytd_df = pd.DataFrame(paths)

this_year = dt.datetime.now().year
if this_year not in ytd_df.columns:
    st.warning(f"No YTD data for {this_year}")
    st.stop()

current = ytd_df[this_year].dropna()
n_days = len(current)

# ── Correlations (selection uses ONLY first n_days) ──────────────────────
corrs = {}
for yr, ser in ytd_df.items():
    if yr == this_year:
        continue
    ser = ser.dropna()
    if len(ser) < n_days:
        continue
    rho = safe_corr(current.values, ser.iloc[:n_days].values)
    if np.isfinite(rho) and rho >= min_corr:
        corrs[yr] = rho

if not corrs:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# ── Filters ──────────────────────────────────────────────────────────────
def keep_year(yr: int) -> bool:
    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        return False

    ret_n = ser.iloc[n_days - 1]

    daily_ret = (1 + ser).div((1 + ser).shift(1)).sub(1)
    max_d = daily_ret.iloc[:n_days].abs().max()

    if f_outliers and not (lo / 100 <= ret_n <= hi / 100):
        return False
    if f_jumps and max_d > max_jump / 100:
        return False
    return True

eligible = {yr: rho for yr, rho in corrs.items() if keep_year(yr)}
if not eligible:
    st.info("All candidates excluded by your filters.")
    st.stop()

top = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

# ── Metrics ──────────────────────────────────────────────────────────────
current_ret = float(current.iloc[-1])
finals = [float(ytd_df[yr].dropna().iloc[-1]) for yr, _ in top]
median_final = float(np.nanmedian(finals)) if finals else np.nan
sigma_final = float(np.nanstd(finals)) if finals else np.nan

fmt = lambda x: "N/A" if np.isnan(x) else f"{x:.2%}"

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} YTD", fmt(current_ret))
m2.metric("Median Full-Year Return (Analogs)", fmt(median_final))
m3.metric("Analog Dispersion (σ)", fmt(sigma_final))

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>", unsafe_allow_html=True)

# ── Plot (FULL YEAR FOR HISTORICAL ANALOGS) ──────────────────────────────
palette = distinct_palette(len(top))
fig, ax = plt.subplots(figsize=(14, 7))

for idx, (yr, rho) in enumerate(top):
    ser_full = ytd_df[yr].dropna()
    ax.plot(
        ser_full.index,
        ser_full.values,
        "--",                # dashed analogs (as requested)
        lw=2.3,
        alpha=0.95,
        color=palette[idx],
        label=f"{yr} (ρ={rho:.2f})",
    )

ax.plot(current.index, current.values, color="black", lw=3.2, label=f"{this_year} (YTD)")
ax.axvline(n_days, color="gray", ls=":", lw=1.3, alpha=0.8)

ax.set_title(f"{ticker} – {this_year} vs Historical Analogs", fontsize=16, weight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return", fontsize=13)
ax.axhline(0, color="gray", ls="--", lw=1)

xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
ax.set_xlim(1, xmax)

all_y = np.hstack([current.values] + [ytd_df[yr].dropna().values for yr, _ in top])
ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02
ax.set_ylim(ymin - pad, ymax + pad)

span = ax.get_ylim()[1] - ax.get_ylim()[0]
raw_step = max(span / 12, 0.0025)
candidates = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
step = float(candidates[np.argmin(np.abs(candidates - raw_step))])
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.grid(True, ls=":", lw=0.7, color="#888")
ax.legend(loc="best", frameon=False, ncol=2, fontsize=11)
plt.tight_layout()
st.pyplot(fig)

st.caption("© 2025 AD Fund Management LP")
