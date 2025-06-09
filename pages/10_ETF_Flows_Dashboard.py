import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

# ----- SIDEBAR -----
st.sidebar.title("ETF Flows")
st.sidebar.markdown("""
This dashboard visualizes **proxy flows** for major U.S. ETFs using changes in shares outstanding and price. 
Flows are shown in dollars and are **approximate**—they may not match official fund flows.
- **Positive flow:** More money entering the ETF (creations)
- **Negative flow:** Money leaving the ETF (redemptions)
Data is from Yahoo Finance. 
""")
lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
period_days = lookback_dict[period_label]

# ----- ETF INFO -----
etf_info = {
    "BITO": ("Crypto", "Bitcoin Futures ETF"),
    "IBIT": ("Crypto", "BlackRock Spot Bitcoin ETF"),
    "SHV": ("Short Duration Bonds", "0-1 Yr T-Bills"),
    "BIL": ("Short Duration Bonds", "1-3 Mo T-Bills"),
    "SGOV": ("Treasury Bill ETF", "0-3 Mo T-Bills"),
    "IEF": ("Intermediate Duration Bonds", "7-10 Yr Treasuries"),
    "AGG": ("Intermediate Duration Bonds", "Core US Bonds"),
    "TLT": ("Long Duration Bonds", "20+ Yr Treasuries"),
    "XLK": ("Tech Sector", "Tech Stocks"),
    "XLF": ("Financials Sector", "Financials"),
    "XLY": ("Consumer Discretionary", "Discretionary Stocks"),
    "XLE": ("Energy", "Energy Stocks"),
    "XLP": ("Consumer Staples", "Staples"),
    "XLV": ("Health Care", "Health Care"),
    "XLU": ("Utilities", "Utilities"),
    "XLI": ("Industrials", "Industrials"),
    "XLB": ("Materials", "Materials"),
    "VTV": ("Value Factor", "US Large Value"),
    "VUG": ("Growth Factor", "US Large Growth"),
    "IWM": ("Small Cap", "US Small Caps"),
}
etf_tickers = list(etf_info.keys())

# ----- DATA COLLECTION -----
@st.cache_data(show_spinner=True)
def robust_flow_estimate(ticker, period_days):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{period_days+10}d")
        hist = hist.dropna()
        if hist.empty or len(hist) < 2:
            return 0
        try:
            so = t.get_shares_full()
            if so is not None and not so.empty:
                so = so.dropna()
                so.index = pd.to_datetime(so.index)
                so = so.loc[hist.index[0]:hist.index[-1]]
                if len(so) >= 2:
                    so = so.reindex(hist.index, method='ffill')
                    start, end = so.iloc[0], so.iloc[-1]
                    close = hist['Close']
                    flow = (end - start) * close.iloc[-1]
                    return flow
        except Exception:
            pass
        flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
        return flow
    except Exception:
        return 0

results = []
for ticker in etf_tickers:
    flow = robust_flow_estimate(ticker, period_days)
    cat, desc = etf_info[ticker]
    results.append({
        "Ticker": ticker,
        "Category": cat,
        "Flow ($)": flow,
        "Description": desc
    })
df = pd.DataFrame(results).sort_values("Flow ($)", ascending=False)

# Format Flow in $000,000,000
df['Flow ($)'] = df['Flow ($)'].apply(lambda x: f"${x/1e9:,.2f}B" if abs(x) > 1e9 else f"${x/1e6:,.2f}M")
df_display = df.copy()
df_display = df_display[['Ticker', 'Category', 'Flow ($)', 'Description']]

# ----- MAIN CONTENT -----
st.title("ETF Proxy Flows")
st.caption(f"Proxy flows (not official) for selected U.S. ETFs. Period: **{period_label}**")

st.dataframe(df_display, hide_index=True)

# Plot with label descriptions
def plot_with_labels(data):
    fig, ax = plt.subplots(figsize=(14, 8))
    raw_flows = []
    tiny_descs = []
    for ticker in data['Ticker']:
        # Pull original (unformatted) flow for plot
        idx = df['Ticker'] == ticker
        # Undo formatting to float for bar chart
        val = results[[r['Ticker'] for r in results].index(ticker)]['Flow ($)']
        raw_flows.append(float(val))
        tiny_descs.append(etf_info[ticker][1])
    bars = ax.barh(data['Ticker'], raw_flows, color=['green' if x > 0 else 'red' for x in raw_flows])
    ax.set_xlabel('Estimated Flow ($)')
    ax.set_title(f'ETF Proxy Flows – {period_label}')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.2f}B' if abs(x)>=1e9 else f'${x/1e6:,.1f}M'))

    # Add tiny description labels on bars
    for bar, val, desc in zip(bars, raw_flows, tiny_descs):
        width = bar.get_width()
        ax.text(width + (1e7 if width > 0 else -1e7), bar.get_y() + bar.get_height()/2,
                f"{desc}", va='center', ha='left' if width > 0 else 'right', fontsize=8, color='black')
    plt.tight_layout()
    return fig

st.pyplot(plot_with_labels(df))

