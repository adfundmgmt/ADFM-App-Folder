import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ETF metadata: Ticker → (Category, Description)
etf_info = {
    "BITO": ("Crypto", "Bitcoin Futures ETF, offers exposure to Bitcoin price via futures contracts."),
    "IBIT": ("Crypto", "BlackRock's Spot Bitcoin ETF, directly tracks Bitcoin's price."),
    "SHV": ("Short Duration Bonds", "iShares Short Treasury Bond ETF, invests in US T-bills <1 year."),
    "BIL": ("Short Duration Bonds", "SPDR Bloomberg 1-3 Month T-Bill ETF."),
    "SGOV": ("Treasury Bill ETF", "iShares 0-3 Month Treasury Bond ETF."),
    "IEF": ("Intermediate Duration Bonds", "iShares 7-10 Year Treasury Bond ETF."),
    "AGG": ("Intermediate Duration Bonds", "iShares Core U.S. Aggregate Bond ETF."),
    "TLT": ("Long Duration Bonds", "iShares 20+ Year Treasury Bond ETF."),
    "XLK": ("Tech Sector", "Technology Select Sector SPDR Fund."),
    "XLF": ("Financials Sector", "Financial Select Sector SPDR Fund."),
    "XLY": ("Consumer Discretionary", "Consumer Discretionary Select Sector SPDR Fund."),
    "XLE": ("Energy", "Energy Select Sector SPDR Fund."),
    "XLP": ("Consumer Staples", "Consumer Staples Select Sector SPDR Fund."),
    "XLV": ("Health Care", "Health Care Select Sector SPDR Fund."),
    "XLU": ("Utilities", "Utilities Select Sector SPDR Fund."),
    "XLI": ("Industrials", "Industrials Select Sector SPDR Fund."),
    "XLB": ("Materials", "Materials Select Sector SPDR Fund."),
    "VTV": ("Value Factor", "Vanguard Value ETF, US large-cap value stocks."),
    "VUG": ("Growth Factor", "Vanguard Growth ETF, US large-cap growth stocks."),
    "IWM": ("Small Cap", "iShares Russell 2000 ETF, US small cap stocks."),
}

etf_tickers = list(etf_info.keys())
period_days = 45  # Approximate period since 4/8/25

def robust_flow_estimate(ticker, period_days):
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{period_days+10}d")
    hist = hist.dropna()
    if hist.empty or len(hist) < 2:
        print(f"{ticker}: No data found")
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
    except Exception as e:
        pass
    # Fallback: proxy flow via market cap change (not official flow)
    flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
    return flow

# Run calculation and build result dataframe
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
df = pd.DataFrame(results)
df = df.sort_values("Flow ($)", ascending=False)

# Plotting (with labels)
plt.figure(figsize=(14, 8))
colors = ['green' if x > 0 else 'red' for x in df['Flow ($)']]
plt.barh(df['Ticker'], df['Flow ($)'], color=colors)
plt.xlabel('Estimated Flow ($)')
plt.title(f'ETF Flow Proxies (Last {period_days} Days)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Show DataFrame with category & description
pd.set_option('display.max_colwidth', 100)
print(df[['Ticker', 'Category', 'Flow ($)', 'Description']])
