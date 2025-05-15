"""
Market Memory Explorer — AD Fund Management LP
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
# Optional logo (update path as needed)
###############################################################################
LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

###############################################################################
# Consistent Title & Subheader (use st.title and st.subheader)
###############################################################################
st.title("Market Memory Explorer")
st.subheader("Compare the current year's return path with history")

###############################################################################
# Input controls (one horizontal row, no wasted space)
###############################################################################
input_col1, input_col2, input_col3 = st.columns([2, 1, 1])
with input_col1:
    ticker = st.text_input("Ticker", value="^GSPC", help="Index, ETF, or equity.").upper()
with input_col2:
    top_n = st.slider("Top Analogs", 1, 10, 5, help="Number of most correlated years to overlay.")
with input_col3:
    min_corr = st.slider("Min ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f", help="Minimum correlation cutoff.")

st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)

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
# Data retrieval & YTD matrix
###############################################################################
try:
    raw = fetch_price_history(ticker)
except Exception as err:
    st.error(f"Download failed – check ticker. ({err})")
    st.stop()

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
# Correlation ranking and filter
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

filtered_corr = {
    yr: rho for yr, rho in correlations.items()
    if rho >= min_corr
}
top_matches = sorted(filtered_corr.items(), key=lambda kv: kv[1], reverse=True)[: top_n]

if not top_matches:
    st.warning("No historical years meet the correlation cutoff.")
    st.stop()

###############################################################################
# Quick metrics row — centered
###############################################################################
best_analog_year, best_rho = top_matches[0]
current_ytd_return = current_ytd.iloc[-1]
best_analog_ytd_return = ytd_df[best_analog_year].loc[:len(current_ytd)].dropna().iloc[-1]

# Most negative (worst) analog among top matches
worst_analog_year, _ = min(top_matches, key=lambda kv: ytd_df[kv[0]].iloc[-1])
worst_analog_ytd_return = ytd_df[worst_analog_year].iloc[-1]

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    st.metric(
        f"{current_year} YTD Return",
        f"{current_ytd_return:.2%}",
        help=f"{ticker} YTD cumulative return."
    )
with metrics_col2:
    st.metric(
        f"Best Analog ({best_analog_year})",
        f"{best_analog_ytd_return:.2%}",
        help=f"YTD return for {best_analog_year} (ρ={best_rho:.2f})"
    )
with metrics_col3:
    arrow = "↑" if worst_analog_ytd_return > 0 else "↓"
    color = "green" if worst_analog_ytd_return > 0 else "red"
    st.metric(
        f"Most Negative Analog",
        f"{worst_analog_ytd_return:.2%}",
        delta=f"{arrow} ({worst_analog_year})",
        help=f"Lowest final YTD analog among top {top_n}."
    )

st.markdown("<hr style='margin-top: 0; margin-bottom: 6px;'>", unsafe_allow_html=True)

###############################################################################
# Sidebar: About, Analogs, Download
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
        1. Enter a ticker (e.g. `^GSPC`, `^IXIC`, `AAPL`, `TSLA`)
        2. Adjust analog settings above.
        3. Chart + metrics update in real time.

        _Built by AD Fund Management LP_
        """
    )
    st.markdown("---")
    # Analogs card
    st.markdown(
        f"""
        <div style='background-color:#f8fafb; border-radius:10px; padding:14px 18px 10px 18px; margin-top:4px; margin-bottom:8px; box-shadow:0 1px 3px 0 rgba(0,0,0,0.03);'>
            <span style='font-size:1.1rem; font-weight:600; color:#1761a0;'>
                Top {top_n} Analogs to {current_year}
            </span>
            <ul style='margin-top:6px; margin-bottom:0;'>
                {''.join(f"<li><b style='color:#1f77b4'>{yr}</b>: <span style='color:#555'>ρ = {rho:.4f}</span></li>" for yr, rho in top_matches)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.download_button(
        label="Download Returns CSV",
        data=ytd_df.to_csv(index=True),
        file_name=f"{ticker}_ytd_paths.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Correlation Table",
        data=pd.DataFrame(top_matches, columns=["Year", "Correlation"]).to_csv(index=False),
        file_name=f"{ticker}_top_analogs.csv",
        mime="text/csv"
    )

###############################################################################
# Main Chart
###############################################################################
if top_n <= 10:
    base_cmap = plt.cm.get_cmap("tab10")
else:
    base_cmap = plt.cm.get_cmap("tab20")
palette = base_cmap(np.linspace(0, 1, top_n))

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(
    range(1, len(current_ytd) + 1),
    current_ytd,
    color="black",
    linewidth=3.2,
    label=f"{current_year} (YTD)"
)
for idx, (yr, rho) in enumerate(top_matches):
    analog = ytd_df[yr].dropna()
    ax.plot(
        analog.index,
        analog.values,
        linestyle="--",
        linewidth=2,
        color=palette[idx],
        alpha=0.7 if yr != best_analog_year else 1.0,
        label=f"{yr} (ρ={rho:.2f})" + (" ⭐" if yr == best_analog_year else "")
    )

ax.set_title(f"{ticker} YTD {current_year} vs Historical Analogs", fontsize=16, fontweight="bold")
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
# Footer
###############################################################################
st.caption("© 2025 AD Fund Management LP")
