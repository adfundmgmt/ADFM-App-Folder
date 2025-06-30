"""
Market Memory Explorer — AD Fund Management LP
----------------------------------------------
Always show analogs; filtering is *optional* and user-controlled.
"""

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.style.use("default")

START_YEAR = 1980
TRADING_DAYS_FULL_YEAR = 253           # US average (NYSE/NASDAQ)

st.set_page_config(page_title="Market Memory Explorer", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

# ---- Sidebar UI -----------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Compare **this year’s YTD performance** for any ticker (index, ETF, or stock)
to prior years with the most similar path.

- **Black = this year**
- **Dashed = top-correlated analogs**
- **Legend shows correlation coefficients (ρ)**
"""
    )
    st.markdown("---")
    st.subheader("Analog Outlier Filters (optional)")
    filter_outliers = st.checkbox("Exclude analogs with extreme returns", value=False)
    filter_jumps    = st.checkbox("Exclude analogs with large daily jumps", value=False)
    if filter_outliers:
        min_return_sidebar, max_return_sidebar = st.slider(
            "Total Return (%) Range",
            min_value=-100, max_value=1000, value=(-95, 300), step=1,
            help="Exclude analogs with YTD returns outside this range."
        )
    if filter_jumps:
        max_jump_sidebar = st.slider(
            "Allowed Max Daily Jump (%)",
            min_value=5, max_value=100, value=25, step=1,
            help="Exclude analogs where any single-day move exceeds this value."
        )
    st.markdown("---")

input_col1, input_col2, input_col3 = st.columns([2, 1, 1])
with input_col1:
    ticker   = st.text_input("Ticker", value="^GSPC",
                             help="Index, ETF, or equity.").upper()
with input_col2:
    top_n    = st.slider("Top Analogs", 1, 10, 5,
                         help="Number of most-correlated years to overlay.")
with input_col3:
    min_corr = st.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f",
                         help="Minimum correlation cutoff.")

st.markdown("<hr style='margin-top:2px; margin-bottom:15px;'>",
            unsafe_allow_html=True)

# ---- Data layer -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=f"{START_YEAR}-01-01",
        progress=False,                # still accepted, but harmless if ignored
        interval="1d",
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("Yahoo returned no data.")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1, drop_level=True)
    df = df[["Close"]].copy()
    df["Year"] = df.index.year
    return df

def cumulative_returns(prices: pd.Series) -> pd.Series:
    return prices / prices.iloc[0] - 1

# pull data
try:
    raw = fetch_price_history(ticker)
except Exception as err:
    st.error(f"Download failed – check ticker. ({err})")
    st.stop()

# build yearly YTD return paths
returns_by_year: dict[int, pd.Series] = {}
for year, grp in raw.groupby("Year"):
    clean = grp["Close"].dropna()                # ← KEY CHANGE
    if len(clean) < 30:                          # keep 30-day sanity filter
        continue
    ytd = cumulative_returns(clean)
    ytd.index = np.arange(1, len(clean) + 1)
    returns_by_year[year] = ytd

ytd_df = pd.DataFrame(returns_by_year)
current_year = dt.datetime.now().year

if current_year not in ytd_df.columns:
    st.warning(f"No valid YTD data for {current_year}")
    st.stop()

current_ytd = ytd_df[current_year].dropna()
n_days      = len(current_ytd)

# --- Correlation & filtering ----------------------------------------------
correlations = {}
for year in ytd_df.columns:
    if year == current_year:
        continue
    past = ytd_df[year].dropna()
    overlap = min(n_days, len(past))
    if overlap < 30:
        continue
    rho = np.corrcoef(current_ytd[:overlap], past[:overlap])[0, 1]
    correlations[year] = rho

filtered_corr = {yr: rho for yr, rho in correlations.items() if rho >= min_corr}
top_matches   = sorted(filtered_corr.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

if not top_matches:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

# optional outlier / jump screens
valid_top_matches, excluded = [], []
for yr, rho in top_matches:
    analog = ytd_df[yr].dropna()
    if len(analog) < n_days:
        continue
    ytd_return = analog.iloc[n_days - 1]
    max_jump   = analog.pct_change().abs().max()

    # apply filters
    if filter_outliers:
        lo, hi = min_return_sidebar / 100, max_return_sidebar / 100
        if not (lo < ytd_return < hi):
            excluded.append((yr, "outlier"))
            continue
    if filter_jumps and max_jump > max_jump_sidebar / 100:
        excluded.append((yr, "jump"))
        continue

    valid_top_matches.append((yr, rho))

if excluded:
    st.info(f"{len(excluded)} analog(s) excluded by your filters.")

if not valid_top_matches:
    st.info("No analogs left after filtering.")
    st.stop()

# --- Metrics --------------------------------------------------------------
current_ytd_return = current_ytd.iloc[-1]
final_analog_returns = [ytd_df[yr].iloc[-1] for yr, _ in valid_top_matches]
median_final = np.median(final_analog_returns)
std_final    = np.std(final_analog_returns)

met1, met2, met3 = st.columns(3)
met1.metric(f"{current_year} YTD Return", f"{current_ytd_return:.2%}")
met2.metric("Median Final Analog Return", f"{median_final:.2%}")
met3.metric("Analog Return Dispersion", f"{std_final:.2%}")

st.markdown("<hr style='margin-top:0; margin-bottom:6px;'>",
            unsafe_allow_html=True)

# ---- Plot ---------------------------------------------------------------
if len(valid_top_matches) <= 10:
    base_cmap = plt.cm.get_cmap("tab10")
else:
    base_cmap = plt.cm.get_cmap("tab20")
palette = base_cmap(np.linspace(0, 1, len(valid_top_matches)))

fig, ax = plt.subplots(figsize=(14, 7))
for idx, (yr, rho) in enumerate(valid_top_matches):
    analog = ytd_df[yr].dropna()
    ax.plot(analog.index, analog.values, "--", linewidth=2,
            color=palette[idx], alpha=0.7, label=f"{yr} (ρ={rho:.2f})")

ax.plot(current_ytd.index, current_ytd.values,
        color="black", linewidth=3.2, label=f"{current_year} (YTD)")
ax.axvline(current_ytd.index[-1], color="gray", linestyle=":", linewidth=1.3, alpha=0.7)

ax.set_title(f"{ticker} — {current_year} vs Historical Analogs", fontsize=16, fontweight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return", fontsize=13)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlim(1, TRADING_DAYS_FULL_YEAR)
ax.grid(True, linestyle=":", linewidth=0.7, color="#888")
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

# dynamic y-limits
all_y = [current_ytd.values] + [ytd_df[yr].dropna().values for yr, _ in valid_top_matches]
min_y, max_y = np.nanmin(np.hstack(all_y)), np.nanmax(np.hstack(all_y))
pad = max(0.02, (max_y - min_y) * 0.08)
ax.set_ylim(min_y - pad, max_y + pad)

ax.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
plt.tight_layout()
st.pyplot(fig)

st.caption("© 2025 AD Fund Management LP")
