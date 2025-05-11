
"""
YTD Analog Year Correlation Explorer
------------------------------------
Compare the current year’s cumulative return path for any ticker with the
most-correlated historical years.

Author: AD Fund Management LP
Last updated: 2025-05-09
"""

###############################################################################
# Imports
###############################################################################
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator

###############################################################################
# Constants & configuration
###############################################################################
START_YEAR = 1980
TRADING_DAYS_FULL_YEAR = 253

st.set_page_config(page_title="YTD Correlation Dashboard", layout="wide")

###############################################################################
# Optional logo (comment out if not needed)
###############################################################################
LOGO_PATH = Path("/mnt/data/99e61e3b-4191-46e7-ae23-a875849b9192.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

###############################################################################
# Title
###############################################################################
st.markdown(
    """
    <h1 style='text-align:center;color:#1f77b4;font-size:42px;'>
        YTD Analog Year Correlation Explorer
    </h1>
    <h4 style='text-align:center;color:gray;'>
        Compare the current year's return path with history
    </h4>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# Sidebar – usage guide
###############################################################################
# ───────────────────────────────────────────────────────── Sidebar ──
with st.sidebar:
    st.title("About This Tool")      # keep the sidebar header

    st.markdown(
        """
This dashboard compares the **current year-to-date (YTD) performance** of a
chosen ticker (e.g., S&P 500 or Nasdaq) with prior years, based on
**correlation of daily return paths**.

---

### 🔧 Steps:

1. **Enter a ticker**  
   Examples:  
   - `^GSPC` (S&P 500)  
   - `^IXIC` (Nasdaq Composite)  
   - `AAPL`, `TSLA`, etc.

2. **Adjust the Top N slider**  
   This selects how many past years to overlay based on similarity to the
   current year.

3. **Interpret the chart**  
   - **Black line** = current year's YTD path  
   - **Dashed lines** = top-correlated historical years  
   - **Legend** shows correlation coefficients (ρ)

---

### 💡 Tip
This tool is useful for:
- Identifying analog years
- Market-narrative framing
- Back-testing

Developed by **AD Fund Management LP**.
        """
    )
# ────────────────────────────────────────────────────────────────────


###############################################################################
# Input controls
###############################################################################
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter ticker symbol", value="^GSPC").upper()

with col2:
    top_n = st.slider("Top N analog years", 1, 10, 5)

# NEW: cutoff slider placed right below the two-column row
min_corr = st.slider(
    "Correlation cutoff (ρ)",
    0.00,   # min
    1.00,   # max
    0.00,   # default (0 = no filter)
    0.05,   # step
    format="%.2f",
)


###############################################################################
# Helper functions
###############################################################################
@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str) -> pd.DataFrame:
    """Download daily close prices from Yahoo Finance."""
    df = yf.download(symbol, start=f"{START_YEAR}-01-01", auto_adjust=False)

    # Some indices return multi-level columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1, drop_level=True)

    df = df[["Close"]].dropna().copy()
    df["Year"] = df.index.year
    return df


def cumulative_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series to cumulative returns starting at 0."""
    return prices / prices.iloc[0] - 1


###############################################################################
# Data retrieval
###############################################################################
try:
    raw = fetch_price_history(ticker)
except Exception as err:
    st.error(f"Download failed – check ticker. ({err})")
    st.stop()

###############################################################################
# Build YTD return matrix
###############################################################################
returns_by_year: dict[int, pd.Series] = {}
for year, grp in raw.groupby("Year"):
    if len(grp) < 30:
        continue
    ytd = cumulative_returns(grp["Close"])
    ytd.index = grp.index.dayofyear
    if ytd.isnull().any() or len(ytd) < 30:
        continue
    returns_by_year[year] = ytd

ytd_df = pd.DataFrame(returns_by_year)
current_year = dt.datetime.now().year

if current_year not in ytd_df.columns:
    st.warning(f"No valid YTD data for {current_year}")
    st.stop()

###############################################################################
# Correlation ranking
###############################################################################
current_ytd = ytd_df[current_year].dropna()
correlations = {}

for year in ytd_df.columns:
    if year == current_year:
        continue
    past_ytd = ytd_df[year].dropna()
    overlap = min(len(current_ytd), len(past_ytd))
    if overlap < 30:
        continue
    rho = np.corrcoef(current_ytd[:overlap], past_ytd[:overlap])[0, 1]
    correlations[year] = rho

# ▶︎ Apply cutoff
filtered_corr = {
    yr: rho for yr, rho in correlations.items()
    if rho >= min_corr        # <-- uses the slider value
}

top_matches = sorted(filtered_corr.items(),
                     key=lambda kv: kv[1],
                     reverse=True)[: top_n]

# Optional guard if nothing survives
if not top_matches:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

###############################################################################
# Display top matches
###############################################################################
st.markdown(
    f"""
    <div style='background:#f0f2f6;padding:20px;
                border-radius:10px;margin-top:20px;'>
        <h3 style='color:#333;'>
            Top {top_n} Most Correlated Years to {current_year}
        </h3>
        <ul>
            {''.join(f'<li><strong>{yr}</strong>: ρ = {rho:.4f}</li>'
                      for yr, rho in top_matches)}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# Plot
###############################################################################
# ------------------------------------------------------------------
# Palette: vivid, high-contrast colours
# ------------------------------------------------------------------
if top_n <= 10:
    base_cmap = plt.cm.get_cmap("tab10")
else:                       # 11-20 analog years
    base_cmap = plt.cm.get_cmap("tab20")

palette = base_cmap(np.linspace(0, 1, top_n))

# Now start the plot
fig, ax = plt.subplots(figsize=(14, 7))

# Current-year trace
ax.plot(range(1, len(current_ytd) + 1),
        current_ytd,
        color="black",
        linewidth=3,
        label=f"{current_year} (YTD)")

# Analog traces
for idx, (yr, rho) in enumerate(top_matches):
    analog = ytd_df[yr].dropna()
    ax.plot(analog.index,
            analog.values,
            linestyle="--",
            linewidth=2,
            color=palette[idx],
            label=f"{yr} (ρ={rho:.2f})")

ax.set_title(f"{ticker} YTD {current_year} vs Historical Analogs",
             fontsize=16,
             fontweight="bold")
ax.set_xlabel("Trading Day of Year")
ax.set_ylabel("Cumulative Return")

ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlim(1, TRADING_DAYS_FULL_YEAR)
ax.grid(True, linestyle=":", linewidth=0.6)

ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.legend(loc="upper left", fontsize=9, frameon=False)
st.pyplot(fig)

###############################################################################
# Footer
###############################################################################
st.markdown(
    """
    <hr style='margin-top:50px;margin-bottom:10px;'>
    <div style='text-align:center;color:gray;font-size:13px;'>
        © 2025 AD Fund Management LP
    </div>
    """,
    unsafe_allow_html=True,
)
