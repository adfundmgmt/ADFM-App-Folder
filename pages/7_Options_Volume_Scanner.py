import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

# Simple Options Chain Viewer
st.title('Options Chain Viewer')

# Sidebar Inputs
ticker = st.sidebar.text_input('Ticker', 'AAPL').upper()
if not ticker:
    st.sidebar.error('Enter a valid ticker symbol')
    st.stop()

# Fetch available expiries
try:
    expiries = yf.Ticker(ticker).options
    if not expiries:
        raise ValueError
except Exception:
    st.error(f'Could not fetch expiries for {ticker}')
    st.stop()

expiry = st.sidebar.selectbox('Select Expiry', expiries)

# Retrieve option chain
chain = yf.Ticker(ticker).option_chain(expiry)
calls = chain.calls[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()
puts = chain.puts[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()

# Display tables side by side
st.subheader('Calls')
st.dataframe(calls)
st.subheader('Puts')
st.dataframe(puts)

# Combine for charting
calls['type'] = 'Call'
puts['type'] = 'Put'
combined = pd.concat([calls, puts], ignore_index=True)

# Bar chart: volume by strike, color-coded by type
tooltips = ['type', 'strike', 'volume', 'openInterest']
chart = alt.Chart(combined).mark_bar().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('volume:Q', title='Volume'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call', 'Put'], range=['#1a9641', '#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=tooltips
).properties(width=800, height=400)

st.subheader('Volume by Strike')
st.altair_chart(chart, use_container_width=True)

# Footer
st.markdown('''
**How to use:**
- Change the ticker and expiry on the sidebar.
- Scroll tables to explore bid/ask, volume, and open interest.
- Use the bar chart to quickly spot strikes with the highest trading volume.
''')
