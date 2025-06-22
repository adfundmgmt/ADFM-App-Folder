import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

# Sidebar
st.sidebar.title("Market Dashboard")
st.sidebar.write("""
This dashboard is built for a hedge fund workflow.  
**Modules:**  
- **Index Charts:** Candlestick charts for DOW, NASDAQ, S&P 500 (5-day intraday)
- **Top Gainers/Losers:** S&P 500 movers by % change
- **Sector Heatmap:** S&P 500 by sector, sized by market cap
- **Market News:** Latest headlines for S&P 500
- **Market Movers:** Most active, 52-week gainers/losers
""")

# --- 1. INDEX CHARTS ---

def plot_index_chart(ticker, name):
    df = yf.download(ticker, period='5d', interval='15m', progress=False)
    if df.empty:
        st.warning(f"No data for {name}")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])])
    fig.update_layout(title=f"{name} (Last 5 days, 15m)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.header("Market Indices (Last 5 days, 15-min bars)")
col1, col2, col3 = st.columns(3)
with col1:
    plot_index_chart('^DJI', 'Dow Jones')
with col2:
    plot_index_chart('^IXIC', 'Nasdaq')
with col3:
    plot_index_chart('^GSPC', 'S&P 500')

# --- 2. S&P 500 GAINERS & LOSERS ---

@st.cache_data(ttl=900)
def get_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist(), df

@st.cache_data(ttl=900)
def get_sp500_prices(symbols):
    data = yf.download(" ".join(symbols), period='1d', interval='1m', progress=False, group_by='ticker')
    last_prices = []
    for sym in symbols:
        try:
            pxs = data[sym]['Close']
            last = pxs.dropna().iloc[-1]
            prev = pxs.dropna().iloc[0]
            change = (last - prev) / prev * 100
            volume = data[sym]['Volume'].dropna().iloc[-1]
            last_prices.append({'Symbol': sym, 'Change (%)': round(change, 2), 'Volume': int(volume)})
        except Exception:
            continue
    return pd.DataFrame(last_prices).sort_values('Change (%)', ascending=False)

st.header("S&P 500 Movers")
symbols, sp500 = get_sp500_constituents()
prices_df = get_sp500_prices(symbols[:150])  # Limiting to first 150 for speed; adjust as needed
gainers = prices_df.head(10)
losers = prices_df.tail(10)

col4, col5 = st.columns(2)
with col4:
    st.subheader("Top Gainers")
    st.dataframe(gainers, use_container_width=True)
with col5:
    st.subheader("Top Losers")
    st.dataframe(losers, use_container_width=True)

# --- 2b. Most Active by Volume ---
st.subheader("Most Active S&P 500 Stocks (by Volume)")
most_active = prices_df.sort_values('Volume', ascending=False).head(10)
st.dataframe(most_active[['Symbol', 'Volume', 'Change (%)']], use_container_width=True)

# --- 2c. 52-Week Highs and Lows (Gainers & Losers) ---

@st.cache_data(ttl=3600)
def get_52week_movers(symbols):
    highs = []
    lows = []
    for sym in symbols[:150]:  # Limiting for speed; increase for full S&P
        try:
            info = yf.Ticker(sym).info
            price = info.get('regularMarketPrice', 0)
            high = info.get('fiftyTwoWeekHigh', 0)
            low = info.get('fiftyTwoWeekLow', 0)
            if abs(price - high) < 1e-2 and price != 0:  # At 52w high
                highs.append({'Symbol': sym, 'Price': price, '52w High': high})
            if abs(price - low) < 1e-2 and price != 0:   # At 52w low
                lows.append({'Symbol': sym, 'Price': price, '52w Low': low})
        except:
            continue
    return pd.DataFrame(highs), pd.DataFrame(lows)

highs, lows = get_52week_movers(symbols)

col6, col7 = st.columns(2)
with col6:
    st.subheader("52-Week Gainers (at High)")
    st.dataframe(highs, use_container_width=True)
with col7:
    st.subheader("52-Week Losers (at Low)")
    st.dataframe(lows, use_container_width=True)

# --- 3. SECTOR HEATMAP ---

@st.cache_data(ttl=3600)
def get_sector_heatmap():
    sp500_symbols, sp500_df = get_sp500_constituents()
    mcaps = []
    for sym in sp500_symbols[:150]:
        try:
            info = yf.Ticker(sym).info
            mcaps.append({'Symbol': sym, 'Sector': info.get('sector', 'Unknown'), 'MarketCap': info.get('marketCap', 0)})
        except:
            continue
    mcap_df = pd.DataFrame(mcaps)
    mcap_df = mcap_df[mcap_df['MarketCap'] > 0]
    return mcap_df

st.header("S&P 500 Sector Heatmap")
mcap_df = get_sector_heatmap()
fig = px.treemap(
    mcap_df,
    path=['Sector', 'Symbol'],
    values='MarketCap',
    title="S&P 500 by Sector & Market Cap (sampled)"
)
st.plotly_chart(fig, use_container_width=True)

# --- 4. NEWS HEADLINES ---

@st.cache_data(ttl=600)
def get_market_news():
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US"
    r = requests.get(url)
    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.content)
    headlines = []
    for item in root.findall(".//item"):
        headlines.append({'title': item.find('title').text, 'link': item.find('link').text})
    return headlines[:8]

st.header("Market Headlines")
for item in get_market_news():
    st.markdown(f"- [{item['title']}]({item['link']})")
