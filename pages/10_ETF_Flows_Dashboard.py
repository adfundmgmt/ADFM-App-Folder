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
A dashboard of **thematic, global, and high-conviction ETF flows**—see where money is moving among major macro and innovation trades, not just vanilla sectors.
- **Flows are proxies**, not official fund flows.
- **Themes:** AI, robotics, Mag 7, semis, clean energy, China tech, EM, LatAm, Europe, Bitcoin, T-bills, and more.
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

# ----- CREATIVE ETF SET -----
etf_info = {
    "MAGS": ("Magnificent 7", "Mag 7 stocks basket ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),
    "BOTZ": ("Robotics & AI", "Global robotics and AI leaders"),
    "ICLN": ("Clean Energy", "Global clean energy stocks"),
    "URNM": ("Uranium", "Uranium miners (Sprott)"),
    "ARKK": ("Innovation", "Disruptive growth stocks (ARK)"),
    "KWEB": ("China Internet", "China internet leaders (KraneShares)"),
    "FXI": ("China Large-Cap", "China mega-cap stocks"),
    "EWZ": ("Brazil", "Brazil large-cap equities"),
    "EEM": ("Emerging Markets", "EM equities (MSCI)"),
    "VWO": ("Emerging Markets", "EM equities (Vanguard)"),
    "EUFN": ("Europe Financials", "European banks and insurers"),
    "EZU": ("Europe", "Eurozone large caps"),
    "VGK": ("Europe Large-Cap", "Developed Europe stocks (Vanguard)"),
    "QQQ": ("Nasdaq 100", "U.S. tech/growth (Nasdaq 100)"),
    "SPY": ("S&P 500", "U.S. large cap equities"),
    "BITO": ("Bitcoin Futures", "Bitcoin futures exposure"),
    "IBIT": ("Spot Bitcoin", "BlackRock spot Bitcoin ETF"),
    "BIL": ("Short T-Bills", "1-3 month U.S. Treasury bills"),
    "TLT": ("Long Duration Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("Short Duration Bond", "Short-term Treasury bonds"),
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
st.title("ETF Proxy Flows (Creative Universe)")
st.caption(f"Flows are proxies (not official). Themes: AI, innovation, EM, China, Bitcoin, T-bills. Period: **{period_label}**")
st.dataframe(df_display, hide_index=True)

def plot_with_labels(data):
    fig, ax = plt.subplots(figsize=(14, 8))
    raw_flows = []
    # Use nicknames or truncate description for cleaner chart
    for ticker in data['Ticker']:
        idx = df['Ticker'] == ticker
        val = results[[r['Ticker'] for r in results].index(ticker)]['Flow ($)']
        raw_flows.append(float(val))
    bars = ax.barh(data['Ticker'], raw_flows, color=['green' if x > 0 else 'red' for x in raw_flows])

    ax.set_xlabel('Estimated Flow ($)')
    ax.set_title(f'ETF Proxy Flows – {period_label}')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.2f}B' if abs(x)>=1e9 else f'${x/1e6:,.1f}M'))

    # SHORT description/nickname for each bar
    for bar, ticker in zip(bars, data['Ticker']):
        nickname = etf_info[ticker][0]  # Use the theme/category as a nickname
        # Truncate if too long
        if len(nickname) > 16:
            nickname = nickname[:15] + "…"
        width = bar.get_width()
        ax.text(width + (1e7 if width > 0 else -1e7),
                bar.get_y() + bar.get_height()/2,
                f"{nickname}",
                va='center',
                ha='left' if width > 0 else 'right',
                fontsize=9,
                color='black')
    plt.tight_layout()
    return fig
