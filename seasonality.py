############################################################
# Built by AD Fund Management LP.
############################################################

import datetime as dt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import MultipleLocator, PercentFormatter

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

# ── Constants ──────────────────────────────────────────────

FALLBACK_MAP = {
    '^GSPC': 'SP500',
    '^DJI':  'DJIA',
    '^IXIC': 'NASDAQCOM',
}
MONTH_LABELS = [
    'Jan','Feb','Mar','Apr','May','Jun',
    'Jul','Aug','Sep','Oct','Nov','Dec',
]

# ── Streamlit Page Config & Sidebar ───────────────────────

st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Explore the seasonal patterns behind any stock, index, or commodity:\n\n"
        "- **Broad Coverage**: Pulls data from Yahoo Finance, with automatic fallback to FRED for the S&P 500, Dow, and Nasdaq when going back before 1950.\n"
        "- **Clean, Reliable Metrics**: Median monthly returns show typical performance while hit rates reveal consistency—how often each month finishes positive.\n"
        "- **At-a-Glance Insight**: Green bars for positive months, red for negative, and black diamonds to mark the frequency of gains.\n"
        "- **Customizable Scope**: Enter any ticker and set your preferred start year to tailor the historical depth."
    )
    st.markdown("### Tips")
    st.markdown(
        "- Use any Yahoo-compatible symbol (e.g. `^GSPC`, `AAPL`, `GLD`, `CL=F`).\n"
        "- Change the start year to shift the analysis window.\n"
        "- Review the full stats table for monthly breakdowns."
    )
    st.markdown("---")
    st.markdown("Crafted by **AD Fund Management LP**")

# ── Helper Functions ───────────────────────────────────────

def seasonal_stats(prices: pd.Series) -> pd.DataFrame:
    monthly = prices.resample('ME').last().pct_change().dropna()
    monthly.index = monthly.index.to_period('M')
    grouped = monthly.groupby(monthly.index.month)
    median_ret = grouped.median() * 100
    hit_rate  = grouped.apply(lambda x: x.gt(0).mean() * 100)
    counts    = grouped.size()

    idx = pd.Index(range(1,13), name='month')
    stats = pd.DataFrame(index=idx)
    stats['median_ret'] = median_ret
    stats['hit_rate']   = hit_rate
    stats['count']      = counts
    stats['label']      = MONTH_LABELS
    return stats

def plot_seasonality(stats: pd.DataFrame, title: str) -> None:
    plot_df = stats.dropna(subset=['median_ret','hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    median = plot_df['median_ret'].to_numpy(dtype=float)
    hit    = plot_df['hit_rate'].to_numpy(dtype=float)

    # dynamic Y‑limits
    y1_bot = min(0.0, np.nanmin(median) - 1.0)
    y1_top =  np.nanmax(median) + 1.0
    y2_bot = max(0.0, np.nanmin(hit)    - 5.0)
    y2_top = min(100.0, np.nanmax(hit)    + 5.0)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    # bars: green if ≥0 else red
    bar_cols  = ['mediumseagreen' if v>=0 else 'indianred' for v in median]
    edge_cols = ['darkgreen'    if v>=0 else 'darkred'    for v in median]
    ax1.bar(
        labels, median, width=0.8,
        color=bar_cols, edgecolor=edge_cols, linewidth=1.2,
        zorder=2
    )
    ax1.set_ylabel('Median return', weight='bold')
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.set_ylim(y1_bot, y1_top)
    ax1.grid(axis='y', linestyle='--', color='lightgrey', linewidth=0.5, alpha=0.7, zorder=1)

    # diamonds: black
    ax2.scatter(
        labels, hit, marker='D', s=80,
        facecolors='black', edgecolors='black', linewidths=0.8,
        zorder=3
    )
    ax2.set_ylabel('Hit rate of positive returns', weight='bold')
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(y2_bot, y2_top)

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.tight_layout(pad=2)

    # render in a narrower container
    st.pyplot(fig, use_container_width=False)
    st.caption("Created by AD Fund Management LP")

# ── Streamlit Inputs ───────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC")
with col2:
    start_year = st.number_input(
        "Start year", value=1950,
        min_value=1900, max_value=dt.datetime.today().year
    )

start_date = f"{int(start_year)}-01-01"

# ── Data Fetch & Plot ─────────────────────────────────────

warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

try:
    start_dt = pd.to_datetime(start_date)
    sym_up   = symbol.upper()

    if pdr and sym_up in FALLBACK_MAP and start_dt.year < 1950:
        fred_tk = FALLBACK_MAP[sym_up]
        st.info(f"Using FRED fallback: {fred_tk} from {start_date}")
        df_fred = pdr.DataReader(fred_tk, 'fred', start_dt, dt.date.today())
        prices = df_fred[fred_tk].rename('Close')
    else:
        df = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"No data found for '{symbol}'")
            st.stop()
        prices = df['Close']

    stats = seasonal_stats(prices)
    first_year = prices.index[0].year
    plot_seasonality(stats, f"{symbol} seasonality (since {first_year})")

    st.markdown("### Monthly Stats Table")
    df_table = stats[['label','median_ret','hit_rate','count']].copy()
    df_table.columns = ['Month','Median Return (%)','Hit Rate (%)','Years Observed']
    st.dataframe(df_table.set_index('Month').style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error: {e}")
