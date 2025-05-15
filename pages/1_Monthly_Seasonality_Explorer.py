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

# ---- Sidebar (About) ----
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        "Explore the seasonal patterns behind any stock, index, or commodity.\n\n"
        "- **Broad Coverage**: Yahoo & FRED (S&P 500, Dow, Nasdaq pre-1950)\n"
        "- **Clean Metrics**: Median monthly returns & hit rates\n"
        "- **Green bars** = positive, **red** = negative, **black diamonds** = hit rates."
    )
    st.markdown("---")
    st.markdown("Crafted by **AD Fund Management LP**")

def seasonal_stats(prices: pd.Series):
    monthly = prices.resample('ME').last().pct_change().dropna() * 100
    monthly.index = monthly.index.to_period('M')
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
    # Standard aspect ratio, big enough for desktop, still looks good on mobile (Streamlit resizes with use_column_width)
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=100)
    plot_df = stats.dropna(subset=['median_ret','hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    median = plot_df['median_ret'].to_numpy(dtype=float)
    hit    = plot_df['hit_rate'].to_numpy(dtype=float)

    y1_bot = min(0.0, np.nanmin(median) - 1.0)
    y1_top =  np.nanmax(median) + 1.0
    y2_bot = max(0.0, np.nanmin(hit)    - 5.0)
    y2_top = min(100.0, np.nanmax(hit)    + 5.0)

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

    ax2.scatter(
        labels, hit, marker='D', s=90,
        facecolors='black', edgecolors='black', linewidths=0.8,
        zorder=3
    )
    ax2.set_ylabel('Hit rate of positive returns', weight='bold')
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(y2_bot, y2_top)

    fig.suptitle(title, fontsize=17, weight='bold')
    fig.tight_layout(pad=2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---- Main controls ----
col1, col2, col3 = st.columns([2,1,1])
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC")
with col2:
    start_year = st.number_input(
        "Start year", value=2020,
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

    stats = seasonal_stats(prices)
    first_year = prices.index[0].year
    last_year = prices.index[-1].year

    # Best/worst up top
    st.markdown("<br>", unsafe_allow_html=True)
    best = stats.loc[stats['median_ret'].idxmax()]
    worst = stats.loc[stats['median_ret'].idxmin()]
    st.markdown(
        f"""
        <div style='text-align:center'>
            <span style='font-size:1.18em; font-weight:600; color:#218739'>
                ⬆️ Best month: {best['label']} ({best['median_ret']:.2f}%)
            </span>
            &nbsp;&nbsp;&nbsp;
            <span style='font-size:1.18em; font-weight:600; color:#c93535'>
                ⬇️ Worst month: {worst['label']} ({worst['median_ret']:.2f}%)
            </span>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    buf = plot_seasonality(stats, f"{symbol} seasonality ({first_year}–{last_year})")

    st.image(buf, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([1,1])
    with dl_col1:
        st.download_button("Download chart as PNG", buf, file_name=f"{symbol}_seasonality_{first_year}_{last_year}.png")
    with dl_col2:
        st.download_button(
            "Download monthly stats (CSV)",
            stats.to_csv(index=True),
            file_name=f"{symbol}_monthly_stats_{first_year}_{last_year}.csv"
        )

    st.markdown("<hr style='margin-top: 16px; margin-bottom: 8px;'>", unsafe_allow_html=True)

    with st.expander("Show Monthly Stats Table", expanded=False):
        df_table = stats[['label','median_ret','hit_rate','count']].copy()
        df_table.columns = ['Month','Median Return (%)','Hit Rate (%)','Years Observed']
        st.dataframe(df_table.set_index('Month').style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error: {e}")

st.caption("© 2025 AD Fund Management LP")
