import datetime as dt
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import MultipleLocator, PercentFormatter

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
        • Choose median, mean, or percentile monthly returns  
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
    # Try Yahoo Finance
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            return df['Close']
    except Exception:
        pass
    # Try FRED if eligible
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
    # Resample by Month End (standard for seasonality), take last price of each month
    monthly_end = prices.resample('M').last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period('M')
    grouped = monthly_ret.groupby(monthly_ret.index.month)

    # Core stats
    stats = pd.DataFrame(index=pd.Index(range(1,13), name='month'))
    stats['median_ret'] = grouped.median()
    stats['mean_ret'] = grouped.mean()
    stats['q25'] = grouped.quantile(0.25)
    stats['q75'] = grouped.quantile(0.75)
    stats['hit_rate'] = grouped.apply(lambda x: (x > 0).mean() * 100)
    stats['volatility'] = grouped.std()
    stats['years_observed'] = grouped.apply(lambda x: x.index.year.nunique())
    stats['label'] = MONTH_LABELS
    stats = stats.reindex(range(1,13))
    return stats

def plot_seasonality(
    stats: pd.DataFrame, 
    title: str, 
    return_metric: str = "Median"
) -> io.BytesIO:
    # Get correct column for plotting
    col_map = {
        "Median": "median_ret",
        "Mean": "mean_ret",
        "25th Percentile": "q25",
        "75th Percentile": "q75"
    }
    ret_col = col_map.get(return_metric, "median_ret")

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=200)
    plot_df = stats.dropna(subset=[ret_col, 'hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    ret = plot_df[ret_col].to_numpy(dtype=float)
    hit = plot_df['hit_rate'].to_numpy(dtype=float)
    vol = plot_df['volatility'].to_numpy(dtype=float)

    # Bar colors
    bar_cols = ['mediumseagreen' if v >= 0 else 'indianred' for v in ret]
    edge_cols = ['darkgreen' if v >= 0 else 'darkred' for v in ret]

    # Bar plot with error bars for volatility
    ax1.bar(
        labels, ret, width=0.8,
        color=bar_cols, edgecolor=edge_cols, linewidth=1.2,
        yerr=vol, capsize=6, alpha=0.85, zorder=2,
        error_kw=dict(ecolor='gray', lw=1.6, alpha=0.7)
    )
    ax1.set_ylabel(f'{return_metric} return (%)', weight='bold')
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_formatter(PercentFormatter())
    y1_bot = min(0.0, np.nanmin(ret) - np.nanmax(vol) - 1.0) if not np.isnan(np.nanmin(ret)) else -1
    y1_top = np.nanmax(ret) + np.nanmax(vol) + 1.0 if not np.isnan(np.nanmax(ret)) else 1
    ax1.set_ylim(y1_bot, y1_top)
    ax1.grid(axis='y', linestyle='--', color='lightgrey', linewidth=0.6, alpha=0.7, zorder=1)

    # Hit rate overlay
    ax2 = ax1.twinx()
    ax2.scatter(
        labels, hit, marker='D', s=90,
        facecolors='black', edgecolors='black', linewidths=0.8,
        zorder=3
    )
    ax2.set_ylabel('Hit rate of positive returns', weight='bold')
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(0, 100)

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
    # Ticker input with autocomplete (experimental)
    try:
        from streamlit_extras.st_autocomplete import st_autocomplete
        symbol = st_autocomplete("Ticker symbol", value="^GSPC")
    except ImportError:
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

metric = st.radio(
    "Select return metric for chart:",
    ["Median", "Mean", "25th Percentile", "75th Percentile"],
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

# Best/worst by return
ret_col = {
    "Median": "median_ret",
    "Mean": "mean_ret",
    "25th Percentile": "q25",
    "75th Percentile": "q75"
}[metric]
best = stats.loc[stats[ret_col].idxmax()]
worst = stats.loc[stats[ret_col].idxmin()]

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            ⬆️ Best month: {best['label']} ({best[ret_col]:.2f}% | σ={best['volatility']:.2f})
        </span>
        &nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            ⬇️ Worst month: {worst['label']} ({worst[ret_col]:.2f}% | σ={worst['volatility']:.2f})
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
    csv_df = stats[['label', 'median_ret', 'mean_ret', 'q25', 'q75', 'hit_rate', 'volatility', 'years_observed']].copy()
    csv_df.rename(columns={
        'label': 'Month', 'median_ret': 'Median Return (%)', 'mean_ret': 'Mean Return (%)',
        'q25': '25th Percentile (%)', 'q75': '75th Percentile (%)',
        'hit_rate': 'Hit Rate (%)', 'volatility': 'Volatility (%)', 'years_observed': 'Years Observed'
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
    <li><b>Median/Mean/Percentile Return:</b> Typical percent gain/loss for each month across years</li>
    <li><b>Hit Rate:</b> % of years each month finished positive</li>
    <li><b>Volatility (σ):</b> Standard deviation of monthly returns, a risk proxy</li>
    </ul>
    </details>
    """, unsafe_allow_html=True
)

st.markdown("<hr style='margin-top: 16px; margin-bottom: 8px;'>", unsafe_allow_html=True)
st.caption("© 2025 AD Fund Management LP")
