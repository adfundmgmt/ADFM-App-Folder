import datetime as dt
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import PercentFormatter, MaxNLocator

plt.style.use("default")
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

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

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Explore the seasonal patterns behind any stock, index, or commodity.**

        ---
        **Broad Coverage:**  
        • Pulls data from Yahoo Finance for all global equities, ETFs, and commodities.  
        • Automatic fallback to FRED for deep history on S&P 500, Dow, and Nasdaq (pre‑1950).

        **Clean, Reliable Metrics:**  
        • Choose median or mean monthly returns  
        • Hit rates reveal consistency—how often each month finishes positive  
        • Volatility overlays show risk context

        **At-a-Glance Insight:**  
        • <span style='color:mediumseagreen'><b>Green bars</b></span> for positive months  
        • <span style='color:indianred'><b>Red bars</b></span> for negative months  
        • <b>Black diamonds</b> to mark the frequency of gains (hit rate)
        • Error bars for monthly volatility

        ---
        **Customizable Scope:**  
        • Enter any ticker and set your preferred start/end year  
        • Download charts or stats instantly

        ---
        Created by **AD Fund Management LP**
        """, unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def fetch_prices(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            return df['Close']
    except Exception:
        pass
    try:
        if pdr and symbol.upper() in FALLBACK_MAP:
            fred_tk = FALLBACK_MAP[symbol.upper()]
            df_fred = pdr.DataReader(fred_tk, 'fred', start, end)
            if not df_fred.empty:
                return df_fred[fred_tk].rename('Close')
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def seasonal_stats(prices: pd.Series):
    monthly_end = prices.resample('M').last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period('M')
    grouped = monthly_ret.groupby(monthly_ret.index.month)

    stats = pd.DataFrame(index=pd.Index(range(1,13), name='month'))
    stats['median_ret'] = grouped.median()
    stats['mean_ret'] = grouped.mean()
    stats['hit_rate'] = grouped.apply(lambda x: (x > 0).mean() * 100)
    # stats['volatility'] = grouped.std()
    stats['min_ret'] = grouped.min()  # ADD min
    stats['max_ret'] = grouped.max()  # ADD max
    stats['years_observed'] = grouped.apply(lambda x: x.index.year.nunique())
    stats['label'] = MONTH_LABELS
    stats = stats.reindex(range(1,13))
    return stats

def plot_seasonality(
    stats: pd.DataFrame, 
    title: str, 
    return_metric: str = "Median"
) -> io.BytesIO:
    col_map = {
        "Median": "median_ret",
        "Mean": "mean_ret"
    }
    ret_col = col_map.get(return_metric, "median_ret")

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=200)
    plot_df = stats.dropna(subset=[ret_col, 'hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    ret = plot_df[ret_col].to_numpy(dtype=float)
    hit = plot_df['hit_rate'].to_numpy(dtype=float)
    #vol = plot_df['volatility'].to_numpy(dtype=float)   # REMOVE
    min_ret = plot_df['min_ret'].to_numpy(dtype=float)
    max_ret = plot_df['max_ret'].to_numpy(dtype=float)

    # Bar colors
    bar_cols = ['mediumseagreen' if v >= 0 else 'indianred' for v in ret]
    edge_cols = ['darkgreen' if v >= 0 else 'darkred' for v in ret]

    # Range for error bars: [distance below, distance above]
    yerr = np.abs(np.vstack([ret - min_ret, max_ret - ret]))

    # --- Left Y: Return axis ---
    # Pad min/max for clean visual
    y_min = np.nanmin(min_ret)
    y_max = np.nanmax(max_ret)
    y_lower = min(0, y_min) - 0.1 * abs(y_min)
    y_upper = max(0, y_max) + 0.1 * abs(y_max)

    ax1.bar(
        labels, ret, width=0.8,
        color=bar_cols, edgecolor=edge_cols, linewidth=1.2,
        yerr=yerr, capsize=6, alpha=0.85, zorder=2,
        error_kw=dict(ecolor='gray', lw=1.6, alpha=0.7)
    )
    ax1.set_ylabel(f'{return_metric} return (%)', weight='bold')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.set_ylim(y_lower, y_upper)
    ax1.grid(axis='y', linestyle='--', color='lightgrey', linewidth=0.6, alpha=0.7, zorder=1)

    # --- Right Y: Hit Rate (always 0% to 100%) ---
    ax2 = ax1.twinx()
    ax2.scatter(
        labels, hit, marker='D', s=90,
        facecolors='black', edgecolors='black', linewidths=0.8,
        zorder=3
    )
    ax2.set_ylabel('Hit rate of positive returns', weight='bold')
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, prune=None))
    ax2.yaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(title, fontsize=17, weight='bold')
    fig.tight_layout(pad=2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---- Main controls ----
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("Ticker symbol", value="SPY")
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

metric = st.radio(
    "Select return metric for chart:",
    ["Median", "Mean"],
    horizontal=True
)

with st.spinner("Fetching and analyzing data..."):
    prices = fetch_prices(symbol, start_date, end_date)

if prices is None or prices.empty or prices.dropna().empty:
    st.error(f"No data found for '{symbol}' in the given date range. Please check ticker or adjust years.")
    st.stop()

if len(prices) < 13:
    st.warning("Less than 12 months of data—seasonality statistics may not be meaningful.")

stats = seasonal_stats(prices)
first_year = prices.index[0].year
last_year = prices.index[-1].year

ret_col = {
    "Median": "median_ret",
    "Mean": "mean_ret"
}[metric]
best = stats.loc[stats[ret_col].idxmax()]
worst = stats.loc[stats[ret_col].idxmin()]

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            ⬆️ Best month: {best['label']} ({best[ret_col]:.2f}% | High: {best['max_ret']:.2f}% | Low: {best['min_ret']:.2f}%)
        </span>
        &nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            ⬇️ Worst month: {worst['label']} ({worst[ret_col]:.2f}% | High: {worst['max_ret']:.2f}% | Low: {worst['min_ret']:.2f}%)
        </span>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

buf = plot_seasonality(stats, f"{symbol} seasonality ({first_year}–{last_year})", return_metric=metric)
st.image(buf, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

dl_col1, dl_col2 = st.columns([1, 1])
with dl_col1:
    st.download_button(
        "Download chart as PNG", buf, 
        file_name=f"{symbol}_seasonality_{first_year}_{last_year}.png"
    )
with dl_col2:
    csv_df = stats[['label', 'median_ret', 'mean_ret', 'hit_rate', 'min_ret', 'max_ret', 'years_observed']].copy()
    csv_df.rename(columns={
        'label': 'Month', 'median_ret': 'Median Return (%)', 'mean_ret': 'Mean Return (%)',
        'hit_rate': 'Hit Rate (%)', 'min_ret': 'Min Return (%)', 'max_ret': 'Max Return (%)', 'years_observed': 'Years Observed'
    }, inplace=True)
    st.download_button(
        "Download monthly stats (CSV)",
        csv_df.to_csv(index=False),
        file_name=f"{symbol}_monthly_stats_{first_year}_{last_year}.csv"
    )

st.markdown(
    """
    <details>
    <summary><b>How to interpret these stats?</b></summary>
    <ul>
    <li><b>Median/Mean Return:</b> Typical percent gain/loss for each month across years</li>
    <li><b>Hit Rate:</b> % of years each month finished positive</li>
    <li><b>High/Low:</b> Maximum and minimum return observed for each month (across all years)</li>
    </ul>
    </details>
    """, unsafe_allow_html=True
)

st.markdown("<hr style='margin-top: 16px; margin-bottom: 8px;'>", unsafe_allow_html=True)
st.caption("© 2025 AD Fund Management LP")
