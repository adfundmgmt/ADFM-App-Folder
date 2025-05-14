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

# Retrieve option chain data
chain = yf.Ticker(ticker).option_chain(expiry)
calls = chain.calls[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()
puts = chain.puts[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()

# Combine for charting
calls['type'] = 'Call'
puts['type'] = 'Put'
combined = pd.concat([calls, puts], ignore_index=True)

# Bar chart: Volume by Strike at Top
st.subheader('Volume by Strike')
chart = alt.Chart(combined).mark_bar().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('volume:Q', title='Volume'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call', 'Put'], range=['#1a9641', '#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=['type', 'strike', 'volume', 'openInterest']
).properties(width=800, height=400)
st.altair_chart(chart, use_container_width=True)

# Optional Calls/Puts Tables
display_tables = st.sidebar.checkbox('Show Calls & Puts Tables', False)
if display_tables:
    st.markdown('---')
    st.subheader('Calls')
    st.dataframe(calls)
    st.subheader('Puts')
    st.dataframe(puts)

# Footer
st.markdown('''
**How to use:**
- Change the ticker and expiry on the sidebar.
- Toggle the checkbox to view detailed call/put tables.
- Use the bar chart to spot strikes with the highest volume.
''')
