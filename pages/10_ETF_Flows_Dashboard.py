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
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=1)
period_days = lookback_dict[period_label]

# ----- ETF UNIVERSE (Creative/Thematic/Regional) -----
etf_info = {
    "QQQ": ("Nasdaq 100", "U.S. tech/growth (Nasdaq 100)"),
    "SPY": ("S&P 500", "U.S. large cap equities"),
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),
    "BOTZ": ("Robotics/AI", "Global robotics and AI leaders"),
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
    "BITO": ("BTC Futures", "Bitcoin futures ETF"),
    "IBIT": ("Spot BTC", "BlackRock spot Bitcoin ETF"),
    "BIL": ("1-3mo T-Bills", "1-3 month U.S. Treasury bills"),
    "TLT": ("20+yr Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("0-1yr T-Bills", "Short-term Treasury bonds"),
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
            return 0.0
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
        # Fallback: price delta x average volume
        flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
        return flow
    except Exception:
        return 0.0

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
df['Flow ($)'] = df['Flow ($)'].apply(lambda x: f"${x/1e9:,.2f}B" if abs(x) > 1e9 else f"${x/1e6:,.2f}M")
df_display = df[['Ticker', 'Category', 'Flow ($)', 'Description']]

# ----- MAIN CONTENT -----
st.title("ETF Proxy Flows (Creative Universe)")
st.caption(f"Flows are proxies (not official). Themes: AI, innovation, EM, China, Bitcoin, T-bills. Period: **{period_label}**")

# ------ CHART ------
def plot_with_labels(data):
    fig, ax = plt.subplots(figsize=(15, 9))
    raw_flows = []
    nicknames = []
    for ticker in data['Ticker']:
        idx = [r['Ticker'] for r in results].index(ticker)
        val = results[idx]['Flow ($)']
        # Undo formatting for plotting
        if "B" in val:
            val_num = float(val.replace("$", "").replace("B", "").replace(",", "")) * 1e9
        elif "M" in val:
            val_num = float(val.replace("$", "").replace("M", "").replace(",", "")) * 1e6
        else:
            val_num = 0.0
        raw_flows.append(val_num)
        # Use the short nickname for the bar
        nickname = etf_info[ticker][0]
        # Truncate if longer than 14 chars
        if len(nickname) > 14:
            nickname = nickname[:13] + "…"
        nicknames.append(nickname)
    bars = ax.barh(data['Ticker'], raw_flows, color=['green' if x > 0 else 'red' for x in raw_flows])
    ax.set_xlabel('Estimated Flow ($)')
    ax.set_title(f'ETF Proxy Flows – {period_label}')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.2f}B' if abs(x)>=1e9 else f'${x/1e6:,.1f}M'))
    # Add tiny nickname labels to bars
    for bar, nickname in zip(bars, nicknames):
        width = bar.get_width()
        ax.text(width + (1e7 if width > 0 else -1e7), bar.get_y() + bar.get_height()/2,
                f"{nickname}",
                va='center', ha='left' if width > 0 else 'right', fontsize=9, color='black')
    plt.tight_layout()
    return fig

st.pyplot(plot_with_labels(df))

# ------ TABLE ------
st.dataframe(df_display, hide_index=True)
