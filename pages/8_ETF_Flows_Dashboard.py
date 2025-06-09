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
A dashboard of **thematic, and global flows** - see where money is moving among major macro and innovation trades.

- **Flows are proxies**, not official fund flows.
- **Themes:** AI, robotics, Mag 7, semis, clean energy, China tech, EM, LatAm, Europe, gold, commodities, min vol, free cash flow, Bitcoin, T-bills, and more.
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

# ----- ETF UNIVERSE -----
etf_info = {
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
    "VGK": ("Europe Large-Cap", "Developed Europe stocks (Vanguard)"),
    "FEZ": ("Eurozone", "Euro STOXX 50 ETF"),
    "ILF": ("Latin America", "Latin America 40 ETF"),
    "ARGT": ("Argentina", "Global X MSCI Argentina ETF"),
    "GLD": ("Gold", "SPDR Gold Trust ETF"),
    "SLV": ("Silver", "iShares Silver Trust ETF"),
    "DBC": ("Commodities", "Invesco DB Commodity Index ETF"),
    "HEDJ": ("Hedged Europe", "WisdomTree Europe Hedged Equity ETF"),
    "USMV": ("US Min Volatility", "iShares MSCI USA Min Volatility ETF"),
    "COWZ": ("US Free Cash Flow", "Pacer US Cash Cows 100 ETF"),
    "BITO": ("BTC Futures", "Bitcoin futures ETF"),
    "IBIT": ("Spot BTC", "BlackRock spot Bitcoin ETF"),
    "BIL": ("1-3mo T-Bills", "1-3 month U.S. Treasury bills"),
    "TLT": ("20+yr Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("0-1yr T-Bills", "Short-term Treasury bonds"),
}
etf_tickers = list(etf_info.keys())

# ----- DATA COLLECTION -----
@st.cache_data(show_spinner=True)
def fetch_etf_metrics(ticker, period_days):
    flow, mcap, flow_pct = 0.0, None, None
    try:
        t = yf.Ticker(ticker)
        # --- FLOW CALC ---
        hist = t.history(period=f"{period_days+10}d")
        hist = hist.dropna()
        if not hist.empty and len(hist) >= 2:
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
                    else:
                        flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
                else:
                    flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
            except Exception:
                flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
        # --- MARKET CAP + % FLOW CALC ---
        shares_out = None
        last_close = None
        try:
            shares_out = t.info.get("sharesOutstanding", None)
            last_close = t.history(period="1d")['Close'][-1]
        except Exception:
            pass
        if shares_out and last_close:
            mcap = shares_out * last_close
            if mcap:
                flow_pct = 100 * flow / mcap
        else:
            mcap, flow_pct = None, None
    except Exception:
        pass
    return flow, mcap, flow_pct

results = []
for ticker in etf_tickers:
    flow, mcap, flow_pct = fetch_etf_metrics(ticker, period_days)
    cat, desc = etf_info[ticker]
    results.append({
        "Ticker": ticker,
        "Category": cat,
        "Flow ($)": flow,
        "Market Cap": mcap,
        "Flow (% MC)": flow_pct,
        "Description": desc
    })

df = pd.DataFrame(results)

# ------ SORT THE DATAFRAME BY RAW FLOW ------
df = df.sort_values("Flow ($)", ascending=False)
df['Flow (Formatted)'] = df['Flow ($)'].apply(lambda x: f"${x/1e9:,.2f}B" if abs(x) > 1e9 else f"${x/1e6:,.2f}M")

# -------- FLOW % OF MC FORMATTING -------
df['Market Cap (Formatted)'] = df['Market Cap'].apply(lambda x: f"${x/1e9:,.2f}B" if x and abs(x) > 1e9 else (f"${x/1e6:,.2f}M" if x else "N/A"))
df['Flow (% MC)'] = df['Flow (% MC)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

# ------ DISPLAY TABLES ------
df_display = df[['Ticker', 'Category', 'Flow (Formatted)', 'Description']]
df_display_pct = df[['Ticker', 'Category', 'Flow (Formatted)', 'Market Cap (Formatted)', 'Flow (% MC)', 'Description']].sort_values("Flow (% MC)", ascending=False, key=lambda x: pd.to_numeric(x.str.replace('%','').replace('N/A','0'), errors='coerce'))

# ----- MAIN CONTENT -----
st.title("ETF Flows Dashboard")
st.caption(f"Flows are proxies (not official). Themes: AI, innovation, EM, China, Bitcoin, commodities, free cash flow, T-bills. Period: **{period_label}**")

# ------ CHART ------
def plot_with_labels(data):
    fig, ax = plt.subplots(figsize=(15, 9))
    nicknames = []
    raw_flows = []
    for _, row in data.iterrows():
        val_num = row['Flow ($)']
        raw_flows.append(val_num)
        nickname = etf_info[row['Ticker']][0]
        # Truncate if longer than 14 chars
        if len(nickname) > 14:
            nickname = nickname[:13] + "…"
        nicknames.append(nickname)
    bars = ax.barh(data['Ticker'], raw_flows, color=['green' if x > 0 else 'red' for x in raw_flows])
    ax.set_xlabel('Estimated Flow ($)')
    ax.set_title(f'ETF Proxy Flows – {period_label}')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.2f}B' if abs(x)>=1e9 else f'${x/1e6:,.1f}M'))
    for bar, nickname in zip(bars, nicknames):
        width = bar.get_width()
        ax.text(width + (1e7 if width > 0 else -1e7), bar.get_y() + bar.get_height()/2,
                f"{nickname}",
                va='center', ha='left' if width > 0 else 'right', fontsize=9, color='black')
    plt.tight_layout()
    return fig

st.pyplot(plot_with_labels(df))

# ------ TABLES ------
st.subheader("Nominal ETF Flows")
st.dataframe(df_display, hide_index=True)
st.subheader("Flows as % of Market Cap")
st.dataframe(df_display_pct, hide_index=True)

st.caption("© 2025 AD Fund Management LP")
