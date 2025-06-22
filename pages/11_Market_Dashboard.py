import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(layout="wide")

# ----- SIDEBAR -----
st.sidebar.title("Finviz-Style Market Dashboard")
st.sidebar.write("""
This dashboard is built for a hedge fund workflow.  
**Modules:**  
- **Index Charts:** Candlestick charts for DOW, NASDAQ, S&P 500 (5-day intraday)
- **Top Gainers/Losers:** S&P 500 movers by % change
- **Sector Heatmap:** S&P 500 by sector, sized by market cap
- **Market News:** Latest headlines for S&P 500 (robust parsing)
- **Market Movers:** Most active, 52-week gainers/losers
""")

# ----- INDEX CHARTS -----
def safe_download(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        # Try fallback (for indices, 15m may fail on weekends or off-market)
        fallback_interval = '1d' if interval == '15m' else interval
        df = yf.download(ticker, period=period, interval=fallback_interval, progress=False)
    return df

def plot_index_chart(ticker, name):
    df = safe_download(ticker, period='5d', interval='15m')
    if df is None or df.empty or 'Open' not in df.columns:
        st.warning(f"No recent data for {name}. Market may be closed or API is rate-limiting.")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])])
    fig.update_layout(title=f"{name} (Last 5 days, 15m or fallback)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.header("Market Indices (Last 5 days, 15-min bars)")
col1, col2, col3 = st.columns(3)
with col1:
    plot_index_chart('^DJI', 'Dow Jones')
with col2:
    plot_index_chart('^IXIC', 'Nasdaq')
with col3:
    plot_index_chart('^GSPC', 'S&P 500')

# ----- S&P 500 MOVERS -----
@st.cache_data(ttl=900)
def get_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist(), df

@st.cache_data(ttl=900)
def get_sp500_prices(symbols):
    data = yf.download(" ".join(symbols), period='1d', interval='5m', progress=False, group_by='ticker')
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
    df = pd.DataFrame(last_prices)
    if not df.empty:
        df = df.sort_values('Change (%)', ascending=False)
    return df

st.header("S&P 500 Movers")
symbols, sp500 = get_sp500_constituents()
# Use up to 300 symbols for breadth and always include the major indices if not present.
sample_symbols = symbols[:300]
prices_df = get_sp500_prices(sample_symbols)

if not prices_df.empty:
    gainers = prices_df.head(10)
    losers = prices_df.tail(10)
    most_active = prices_df.sort_values('Volume', ascending=False).head(10)
else:
    gainers = losers = most_active = pd.DataFrame()

col4, col5, col6 = st.columns(3)
with col4:
    st.subheader("Top Gainers")
    if not gainers.empty:
        st.dataframe(gainers, use_container_width=True)
    else:
        st.info("No gainers data available.")

with col5:
    st.subheader("Top Losers")
    if not losers.empty:
        st.dataframe(losers, use_container_width=True)
    else:
        st.info("No losers data available.")

with col6:
    st.subheader("Most Active (Volume)")
    if not most_active.empty:
        st.dataframe(most_active[['Symbol', 'Volume', 'Change (%)']], use_container_width=True)
    else:
        st.info("No active volume data available.")

# ----- 52-WEEK GAINERS/LOSERS -----
@st.cache_data(ttl=3600)
def get_52week_movers(symbols):
    highs = []
    lows = []
    for sym in symbols[:300]:
        try:
            info = yf.Ticker(sym).info
            price = info.get('regularMarketPrice', None)
            high = info.get('fiftyTwoWeekHigh', None)
            low = info.get('fiftyTwoWeekLow', None)
            if price is None or high is None or low is None or price == 0:
                continue
            # Use a relative tolerance to account for small floats
            if abs(price - high) < 0.01 * high:
                highs.append({'Symbol': sym, 'Price': price, '52w High': high})
            if abs(price - low) < 0.01 * low:
                lows.append({'Symbol': sym, 'Price': price, '52w Low': low})
        except Exception:
            continue
    return pd.DataFrame(highs), pd.DataFrame(lows)

highs, lows = get_52week_movers(symbols)

col7, col8 = st.columns(2)
with col7:
    st.subheader("52-Week Gainers (at High)")
    if not highs.empty:
        st.dataframe(highs, use_container_width=True)
    else:
        st.info("No 52-week high data currently available.")

with col8:
    st.subheader("52-Week Losers (at Low)")
    if not lows.empty:
        st.dataframe(lows, use_container_width=True)
    else:
        st.info("No 52-week low data currently available.")

# ----- SECTOR HEATMAP -----
@st.cache_data(ttl=3600)
def get_sector_heatmap():
    sp500_symbols, sp500_df = get_sp500_constituents()
    mcaps = []
    for sym in sp500_symbols[:300]:
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
if not mcap_df.empty:
    fig = px.treemap(
        mcap_df,
        path=['Sector', 'Symbol'],
        values='MarketCap',
        title="S&P 500 by Sector & Market Cap (sampled)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sector/market cap data available at this time.")

# ----- NEWS HEADLINES -----
@st.cache_data(ttl=600)
def get_market_news():
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=8)
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(r.content)
            headlines = []
            for item in root.findall(".//item"):
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                if title and link:
                    headlines.append({'title': title, 'link': link})
            if headlines:
                return headlines[:8]
        except ET.ParseError:
            pass  # Fallback
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.content, 'xml')
        items = soup.find_all('item')
        headlines = []
        for item in items:
            title = item.title.text if item.title else ""
            link = item.link.text if item.link else ""
            if title and link:
                headlines.append({'title': title, 'link': link})
        return headlines[:8]
    except Exception:
        return [{"title": "Market headlines currently unavailable.", "link": "#"}]

st.header("Market Headlines")
news_items = get_market_news()
if news_items and len(news_items) > 0:
    for item in news_items:
        st.markdown(f"- [{item['title']}]({item['link']})")
else:
    st.info("No news headlines available at this time.")
