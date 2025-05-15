"""
Market Memory Explorer – AD Fund Management LP
----------------------------------------------
Compare the current year’s cumulative return path for any ticker with the
most-correlated historical years.
"""

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

st.set_page_config(page_title="Market Memory Explorer", layout="wide")

###############################################################################
# Optional logo (comment out if not needed)
###############################################################################
LOGO_PATH = Path("/mnt/data/1c9f0c52-d1ac-41fc-9e8a-d75d172afc55.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

###############################################################################
# Consistent Title & Subheader (use st.title and st.subheader)
###############################################################################
st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

###############################################################################
# Sidebar – usage guide
###############################################################################
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Compare **this year’s YTD performance** for any ticker (index, ETF, or stock) to prior years with the most similar path.

        - **Black = this year**
        - **Dashed = top-correlated analog years**
        - **Legend shows correlation coefficients (ρ)**

        ---
        **Steps:**
        1. Enter a ticker (e.g. `^GSPC`, `^IXIC`, `AAPL`, `TSLA`)
        2. Adjust the number of analog years and minimum correlation.
        3. Explore chart, metrics, and export options.

        _Built by AD Fund Management LP_
        """
    )

###############################################################################
# Input controls
###############################################################################
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Ticker symbol", value="^GSPC").upper()

with col2:
    top_n = st.slider("Top N analog years", 1, 10, 5)

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
    df = yf.download(symbol, start=f"{START_YEAR}-01-01", auto_adjust=False, progress=False)
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
    if rho >= min_corr
}

top_matches = sorted(filtered_corr.items(),
                     key=lambda kv: kv[1],
                     reverse=True)[: top_n]

if not top_matches:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

###############################################################################
# Quick metrics/summary box (new)
###############################################################################
best_analog_year, best_rho = top_matches[0]
current_ytd_return = current_ytd.iloc[-1]
best_analog_ytd_return = ytd_df[best_analog_year].loc[:len(current_ytd)].dropna().iloc[-1]
worst_analog_year, worst_rho = min(top_matches, key=lambda kv: ytd_df[kv[0]].iloc[-1])

c1, c2, c3 = st.columns([2,2,2])
c1.metric(
    label=f"{current_year} YTD Return",
    value=f"{current_ytd_return:.2%}",
    help=f"Cumulative return for {ticker} YTD."
)
c2.metric(
    label=f"Best Analog ({best_analog_year}) YTD",
    value=f"{best_analog_ytd_return:.2%}",
    help=f"Best-matching year: {best_analog_year} (ρ={best_rho:.2f})"
)
c3.metric(
    label=f"Most Negative Analog",
    value=f"{ytd_df[worst_analog_year].iloc[-1]:.2%}",
    delta=f"({worst_analog_year})",
    help=f"Lowest final YTD analog among top {top_n} matches."
)

###############################################################################
# Display top matches – consistent style
###############################################################################
st.markdown(
    f"""
    <div style='background:#f0f2f6;padding:18px 16px 16px 16px;
                border-radius:8px;margin-top:20px;max-width:700px;'>
        <span style='font-size:1.15rem; font-weight:700; color:#1f77b4;'>
            Top {top_n} Most Correlated Years to {current_year}
        </span>
        <ul style='margin-top:6px;'>
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
# Use tab10/tab20 as base palette
if top_n <= 10:
    base_cmap = plt.cm.get_cmap("tab10")
else:
    base_cmap = plt.cm.get_cmap("tab20")
palette = base_cmap(np.linspace(0, 1, top_n))

fig, ax = plt.subplots(figsize=(14, 7))

# Current-year trace
ax.plot(range(1, len(current_ytd) + 1),
        current_ytd,
        color="black",
        linewidth=3.2,
        label=f"{current_year} (YTD)")

# Analog traces, semi-transparent
for idx, (yr, rho) in enumerate(top_matches):
    analog = ytd_df[yr].dropna()
    ax.plot(analog.index,
            analog.values,
            linestyle="--",
            linewidth=2,
            color=palette[idx],
            alpha=0.65 if yr != best_analog_year else 1.0,
            label=f"{yr} (ρ={rho:.2f})" + (" ⭐" if yr == best_analog_year else ""))

ax.set_title(f"{ticker} YTD {current_year} vs Historical Analogs",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Trading Day of Year", fontsize=13)
ax.set_ylabel("Cumulative Return", fontsize=13)

ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlim(1, TRADING_DAYS_FULL_YEAR)
ax.grid(True, linestyle=":", linewidth=0.7, color="#888")

ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

ax.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
plt.tight_layout()
st.pyplot(fig)

###############################################################################
# Data download/export section
###############################################################################
with st.expander("Download Data"):
    st.download_button(
        label="Download Cumulative Returns (CSV)",
        data=ytd_df.to_csv(index=True),
        file_name=f"{ticker}_market_memory_ytd_paths.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Correlation Table (CSV)",
        data=pd.DataFrame(top_matches, columns=["Year", "Correlation"]).to_csv(index=False),
        file_name=f"{ticker}_market_memory_top_analogs.csv",
        mime="text/csv"
    )

###############################################################################
# Footer
###############################################################################
st.caption("© 2025 AD Fund Management LP")

