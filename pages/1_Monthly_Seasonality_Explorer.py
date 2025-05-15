############################################################
# Built by AD Fund Management LP. Enhanced for usability
############################################################

import datetime as dt
import io
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

FALLBACK_MAP = {
    '^GSPC': 'SP500',
    '^DJI':  'DJIA',
    '^IXIC': 'NASDAQCOM',
}
MONTH_LABELS = [
    'Jan','Feb','Mar','Apr','May','Jun',
    'Jul','Aug','Sep','Oct','Nov','Dec',
]

st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

# ── Sidebar: About, Controls, Download ──────────────────

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        "Explore the seasonal patterns behind any stock, index, or commodity:\n\n"
        "- **Broad Coverage**: Pulls data from Yahoo Finance, with FRED fallback for the S&P 500, Dow, and Nasdaq (pre-1950).\n"
        "- **Clean Metrics**: Median monthly returns + hit rates = real pattern discovery.\n"
        "- **At-a-Glance Insight**: Green bars = positive, red = negative, black diamonds = hit rates.\n"
        "- **Customizable Scope**: Choose ticker, start/end year, filter outlier months if desired."
    )
    st.markdown("---")
    st.subheader("Analysis Controls")
    # Outlier toggle for monthly returns
    filter_outliers = st.checkbox("Exclude months with abs(return) > X%", value=False)
    outlier_thresh = st.slider("Outlier threshold (%)", 5, 100, 30, help="Hide months with extreme one-month moves", disabled=not filter_outliers)
    st.markdown("---")
    st.markdown("Crafted by **AD Fund Management LP**")

# ── Helper Functions ─────────────────────────────────────

def seasonal_stats(prices: pd.Series, filter_outliers=False, outlier_thresh=30):
    monthly = prices.resample('ME').last().pct_change().dropna() * 100
    monthly.index = monthly.index.to_period('M')
    # Optionally filter out months with extreme returns (for robust stats)
    if filter_outliers:
        monthly = monthly[monthly.abs() <= outlier_thresh]
    grouped = monthly.groupby(monthly.index.month)
    median_ret = grouped.median()
    hit_rate  = grouped.apply(lambda x: x.gt(0).mean() * 100)
    counts    = grouped.size()

    idx = pd.Index(range(1,13), name='month')
    stats = pd.DataFrame(index=idx)
    stats['median_ret'] = median_ret
    stats['hit_rate']   = hit_rate
    stats['count']      = counts
    stats['label']      = MONTH_LABELS
    return stats

def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(subset=['median_ret','hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    median = plot_df['median_ret'].to_numpy(dtype=float)
    hit    = plot_df['hit_rate'].to_numpy(dtype=float)

    y1_bot = min(0.0, np.nanmin(median) - 1.0)
    y1_top =  np.nanmax(median) + 1.0
    y2_bot = max(0.0, np.nanmin(hit)    - 5.0)
    y2_top = min(100.0, np.nanmax(hit)    + 5.0)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

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
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Inputs ───────────────────────────────────────────────

col1, col2, col3 = st.columns([2,1,1])
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC")
with col2:
    start_year = st.number_input(
        "Start year", value=1950,
        min_value=1900, max_value=dt.datetime.today().year
    )
with col3:
    end_year = st.number_input(
        "End year", value=dt.datetime.today().year,
        min_value=int(start_year), max_value=dt.datetime.today().year
    )

start_date = f"{int(start_year)}-01-01"
end_date = f"{int(end_year)}-12-31"

warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

try:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    sym_up   = symbol.upper()

    if pdr and sym_up in FALLBACK_MAP and start_dt.year < 1950:
        fred_tk = FALLBACK_MAP[sym_up]
        st.info(f"Using FRED fallback: {fred_tk} from {start_date}")
        df_fred = pdr.DataReader(fred_tk, 'fred', start_dt, end_dt)
        prices = df_fred[fred_tk].rename('Close')
    else:
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"No data found for '{symbol}'")
            st.stop()
        prices = df['Close']

    stats = seasonal_stats(prices, filter_outliers=filter_outliers, outlier_thresh=outlier_thresh)
    first_year = prices.index[0].year
    last_year = prices.index[-1].year

    buf = plot_seasonality(stats, f"{symbol} seasonality ({first_year}–{last_year})")
    st.image(buf, caption=f"Monthly seasonality for {symbol} ({first_year}–{last_year})", use_column_width=False)

    # Download buttons
    st.download_button("Download chart as PNG", buf, file_name=f"{symbol}_seasonality_{first_year}_{last_year}.png")
    st.download_button(
        "Download monthly stats (CSV)",
        stats.to_csv(index=True),
        file_name=f"{symbol}_monthly_stats_{first_year}_{last_year}.csv"
    )

    # Stats Table
    st.markdown("### Monthly Stats Table")
    df_table = stats[['label','median_ret','hit_rate','count']].copy()
    df_table.columns = ['Month','Median Return (%)','Hit Rate (%)','Years Observed']
    st.dataframe(df_table.set_index('Month').style.format("{:.2f}"))

    # "Most seasonal" months: positive/negative extremes
    st.markdown("### Most Seasonal Months")
    best = df_table.loc[df_table['Median Return (%)'].idxmax()]
    worst = df_table.loc[df_table['Median Return (%)'].idxmin()]
    st.success(f"**Best month:** {best['Month']} (median return {best['Median Return (%)']:.2f}%)")
    st.error(f"**Worst month:** {worst['Month']} (median return {worst['Median Return (%)']:.2f}%)")

except Exception as e:
    st.error(f"Error: {e}")

# ── Footnotes ─────────────────────────────────────────────

st.caption("© 2025 AD Fund Management LP")
