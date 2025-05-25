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
        • Median monthly returns show typical performance  
        • Hit rates reveal consistency—how often each month finishes positive

        **At-a-Glance Insight:**  
        • <span style='color:mediumseagreen'><b>Green bars</b></span> for positive months  
        • <span style='color:indianred'><b>Red bars</b></span> for negative months  
        • <b>Black diamonds</b> to mark the frequency of gains (hit rate)

        ---
        **Customizable Scope:**  
        • Enter any ticker and set your preferred start/end year to tailor historical depth  
        • Download charts or stats instantly

        ---
        Created by **AD Fund Management LP**
        """, unsafe_allow_html=True
    )

def seasonal_stats(prices: pd.Series):
    # Resample by Month End (standard for seasonality), take last price of each month
    monthly_end = prices.resample('M').last()
    monthly_ret = monthly_end.pct_change().dropna() * 100

    # Convert index to period month
    monthly_ret.index = monthly_ret.index.to_period('M')

    grouped = monthly_ret.groupby(monthly_ret.index.month)

    median_ret = grouped.median()
    hit_rate = grouped.apply(lambda x: (x > 0).mean() * 100)
    years_observed = grouped.apply(lambda x: x.index.year.nunique())

    # Ensure all months are present, fill missing with NaN (not 0)
    idx = pd.Index(range(1, 13), name='month')
    stats = pd.DataFrame(index=idx)
    stats['median_ret'] = median_ret
    stats['hit_rate'] = hit_rate
    stats['years_observed'] = years_observed
    stats['label'] = MONTH_LABELS
    stats = stats.reindex(idx)

    return stats

def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=100)
    plot_df = stats.dropna(subset=['median_ret', 'hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    median = plot_df['median_ret'].to_numpy(dtype=float)
    hit = plot_df['hit_rate'].to_numpy(dtype=float)

    y1_bot = min(0.0, np.nanmin(median) - 1.0) if not np.isnan(np.nanmin(median)) else -1
    y1_top = np.nanmax(median) + 1.0 if not np.isnan(np.nanmax(median)) else 1
    y2_bot = max(0.0, np.nanmin(hit) - 5.0) if not np.isnan(np.nanmin(hit)) else 0
    y2_top = min(100.0, np.nanmax(hit) + 5.0) if not np.isnan(np.nanmax(hit)) else 100

    ax2 = ax1.twinx()
    bar_cols = ['mediumseagreen' if v >= 0 else 'indianred' for v in median]
    edge_cols = ['darkgreen' if v >= 0 else 'darkred' for v in median]
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

# ---- NEW: Add TSM-style cumulative chart functions below ----

def get_cumulative_returns(prices: pd.Series):
    rets = prices.pct_change().dropna()
    cumrets = (1 + rets).cumprod() - 1
    return cumrets

def get_historical_seasonality(prices: pd.Series, ref_year=None):
    df = prices.pct_change().dropna().to_frame("ret")
    df["year"] = df.index.year
    df["doy"] = df.index.dayofyear
    if ref_year:
        df = df[df["year"] != ref_year]
    # For each year, calculate cumulative returns per calendar day
    all_cumrets = []
    for y in df["year"].unique():
        yr = df[df["year"] == y]
        cr = (1 + yr["ret"]).cumprod().values - 1
        s = pd.Series(cr, index=yr["doy"])
        all_cumrets.append(s)
    cumrets = pd.DataFrame(all_cumrets).T  # Index: doy, Columns: years
    median = cumrets.median(axis=1)
    lower = cumrets.quantile(0.16, axis=1)
    upper = cumrets.quantile(0.84, axis=1)
    return median, lower, upper

def plot_seasonality_vs_year(prices: pd.Series, year: int, symbol: str):
    prices = prices.sort_index()
    prices = prices.asfreq('B').ffill()  # Fill missing business days for proper alignment

    # Drop years with partial data
    min_days = 200
    year_counts = prices.index.year.value_counts()
    good_years = year_counts[year_counts > min_days].index.tolist()
    prices = prices[prices.index.year.isin(good_years)]

    # Historical (all but selected year)
    median, lower, upper = get_historical_seasonality(prices, ref_year=year)
    # Actual YTD
    this_year = prices[prices.index.year == year]
    if len(this_year) == 0:
        return None
    cum_this_year = get_cumulative_returns(this_year)

    # X-axis mapping: day of year to month
    months = pd.to_datetime(this_year.index)
    doy = (this_year.index - this_year.index[0]).days + 1
    xticks = []
    xticklabels = []
    for m in range(1, 13):
        idx = np.where(months.month == m)[0]
        if len(idx) > 0:
            xticks.append(doy[idx[0]])
            xticklabels.append(MONTH_LABELS[m-1])

    fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
    # Shaded ±1 std band
    ax.fill_between(
        median.index, lower.values * 100, upper.values * 100,
        color="lightgrey", alpha=0.5, label="Historical Seasonality ±1σ"
    )
    # Median historical
    ax.plot(
        median.index, median.values * 100,
        color="indianred", linewidth=2, label="Historical Seasonality"
    )
    # Current year
    ax.plot(
        doy, cum_this_year.values * 100,
        color="#3682F4", linewidth=2, label=f"{year} Returns"
    )

    ax.set_title(f"Seasonality and {year} Performance of {symbol.upper()}", fontsize=16, weight="bold")
    ax.set_ylabel("Cumulative Returns (%)")
    ax.set_xlabel("Month")
    ax.set_xlim(1, 366)
    ax.set_ylim(-20, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(linestyle="--", color="grey", alpha=0.3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.legend(loc="upper left")
    ax.margins(x=0)
    plt.tight_layout()
    return fig

# ---- Main controls ----
col1, col2, col3 = st.columns([2, 1, 1])
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

# For the bottom chart: user can select the "actual" performance year (defaults to latest in range)
actual_year = st.number_input(
    "Show actual performance for year (bottom chart)", 
    value=int(end_year),
    min_value=int(start_year), max_value=int(end_year)
)

start_date = f"{int(start_year)}-01-01"
end_date = f"{int(end_year)}-12-31"
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

try:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    sym_up = symbol.upper()

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

    # --- TOP CHART: Monthly Seasonality ---
    stats = seasonal_stats(prices)
    first_year = prices.index[0].year
    last_year = prices.index[-1].year

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

    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        st.download_button("Download chart as PNG", buf, file_name=f"{symbol}_seasonality_{first_year}_{last_year}.png")
    with dl_col2:
        # Add month labels as first column in CSV for clarity
        csv_df = stats[['label', 'median_ret', 'hit_rate', 'years_observed']].copy()
        csv_df.rename(columns={'label': 'Month', 'median_ret': 'Median Return (%)', 'hit_rate': 'Hit Rate (%)', 'years_observed': 'Years Observed'}, inplace=True)
        st.download_button(
            "Download monthly stats (CSV)",
            csv_df.to_csv(index=False),
            file_name=f"{symbol}_monthly_stats_{first_year}_{last_year}.csv"
        )

    st.markdown("<hr style='margin-top: 16px; margin-bottom: 8px;'>", unsafe_allow_html=True)

    # --- BOTTOM CHART: TSM-style seasonality vs actual ---
    st.subheader("Seasonality vs. Actual Cumulative Returns")
    fig2 = plot_seasonality_vs_year(prices, int(actual_year), symbol)
    if fig2 is not None:
        st.pyplot(fig2)
        st.caption("Data: Yahoo! Finance | Plot: AD Fund Management LP")
    else:
        st.warning(f"No data available for {symbol} in {actual_year}.")

except Exception as e:
    st.error(f"Error: {e}")

st.caption("© 2025 AD Fund Management LP")
