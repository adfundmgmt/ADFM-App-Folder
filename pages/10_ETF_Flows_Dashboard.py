import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("ETF Flows Dashboard")
st.write("Proxy flows for major US ETFs based on share issuance and price moves. Not official fund flow data; interpret as indicative.")

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

period_days = st.slider('Number of days to look back', 30, 90, 45)  # UI control
st.caption(f"Calculating proxy flows for the last {period_days} days.")

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
        # Fallback: price delta x average volume proxy
        flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
        return flow
    except Exception:
        return 0

# Calculate flows and collect ETF metadata
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

if df.empty or df['Flow ($)'].abs().sum() == 0:
    st.warning("No data available for selected period. Try changing the lookback window or check your internet connection.")
else:
    st.dataframe(df, hide_index=True)
    # Bar chart (horizontal)
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df['Ticker'], df['Flow ($)'], color=['green' if x > 0 else 'red' for x in df['Flow ($)']])
    ax.set_xlabel('Estimated Flow ($)')
    ax.set_title(f'ETF Proxy Flows – Last {period_days} Days')
    ax.invert_yaxis()
    st.pyplot(fig)
    st.caption("Descriptions: " + "; ".join([f"{row.Ticker}: {row.Description}" for row in df.itertuples()]))

