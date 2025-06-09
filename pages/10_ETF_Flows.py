import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ETF Tickers (diverse, liquid, representative)
etf_tickers = [
    "BITO", "IBIT", "SHV", "BIL", "SGOV",  # Crypto, short T-bills
    "IEF", "AGG", "TLT",                   # Bonds: intermediate, aggregate, long
    "XLK", "XLF", "XLY", "XLE", "XLP",     # Sectors: Tech, Financials, Disc, Energy, Staples
    "XLV", "XLU", "XLI", "XLB",            # Sectors: Health, Utilities, Industrials, Materials
    "VTV", "VUG", "IWM"                    # Value, Growth, Small Cap
]

# Timeframes for flows
periods = {
    'Daily': 1,
    '3M': 90,
    '6M': 180,
    '12M': 365,
    '3Y': 1095
}

# Function to get shares outstanding history (Yahoo, best effort)
def get_shares_outstanding(ticker):
    try:
        t = yf.Ticker(ticker)
        so = t.get_shares_full()
        so = so.dropna()
        so.index = pd.to_datetime(so.index)
        return so
    except Exception as e:
        return None

# Function to estimate flows for one ETF
def estimate_flow(ticker, period_days):
    so = get_shares_outstanding(ticker)
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{period_days+2}d")
    hist = hist.dropna()
    # Use shares outstanding if available
    if so is not None and len(so) > 2:
        # Align SO and Close
        df = pd.DataFrame({'Close': hist['Close']})
        df['SO'] = so.reindex(df.index, method='ffill')
        df = df.dropna()
        if len(df) < 2:
            return None
        start, end = df.iloc[0], df.iloc[-1]
        flow = (end['SO'] - start['SO']) * end['Close']
        return flow
    else:
        # Proxy: change in market cap
        if len(hist) < 2:
            return None
        flow = (hist['Close'][-1] - hist['Close'][0]) * hist['Volume'][-1]
        return flow

# Function to build a full flow DataFrame for selected period
def get_flows(period_days):
    results = []
    for ticker in etf_tickers:
        flow = estimate_flow(ticker, period_days)
        results.append({'ETF': ticker, 'Flow ($)': flow if flow is not None else 0})
    df = pd.DataFrame(results).set_index('ETF').sort_values('Flow ($)', ascending=False)
    return df

# Choose timeframe to run (change as needed)
timeframe = '3M'  # Options: 'Daily', '3M', '6M', '12M', '3Y'
period_days = periods[timeframe]

df = get_flows(period_days)

# Plotting
plt.figure(figsize=(14, 7))
bars = plt.barh(df.index, df['Flow ($)'], color=['green' if x > 0 else 'red' for x in df['Flow ($)']])
plt.xlabel('Estimated ETF Flow ($)')
plt.title(f'ETF Flows Over the Last {timeframe}')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Display DataFrame for inspection
print(df)
