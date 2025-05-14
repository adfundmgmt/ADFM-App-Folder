import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

# Simple Options Chain Viewer with Spread Heatmap & Summary Metrics
st.title('Options Chain Viewer')

# Sidebar Inputs
ticker = st.sidebar.text_input('Ticker', 'AAPL').upper()
if not ticker:
    st.sidebar.error('Enter a valid ticker symbol')
    st.stop()

# Fetch expiries
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

# Prepare call and put DataFrames
calls = chain.calls[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()
puts = chain.puts[['strike', 'bid', 'ask', 'volume', 'openInterest']].copy()
calls['type'] = 'Call'
puts['type'] = 'Put'
combined = pd.concat([calls, puts], ignore_index=True)

# Calculate bid-ask spread percentage
combined['mid'] = (combined['bid'] + combined['ask']) / 2
combined['spread_pct'] = ((combined['ask'] - combined['bid']) / combined['mid']) * 100

# Top-level: Volume by Strike Bar Chart
st.subheader('Volume by Strike')
vol_chart = alt.Chart(combined).mark_bar().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('volume:Q', title='Volume'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=['type', 'strike', 'volume', 'openInterest']
).properties(width=800, height=400)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown('---')
# Bid-Ask Spread Heatmap
st.subheader('Bid-Ask Spread Heatmap (%)')
spread_heatmap = alt.Chart(combined).mark_rect().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('type:N', title='Option Type'),
    color=alt.Color('spread_pct:Q', title='Spread %', scale=alt.Scale(scheme='redyellowgreen')),
    tooltip=['type', 'strike', alt.Tooltip('spread_pct:Q', format='.2f')]
).properties(width=800, height=300)
st.altair_chart(spread_heatmap, use_container_width=True)

st.markdown('---')
# Summary Metrics Panel
st.subheader('Summary Metrics')
total_call_vol = combined.loc[combined['type'] == 'Call', 'volume'].sum()
total_put_vol = combined.loc[combined['type'] == 'Put', 'volume'].sum()
avg_spread = combined['spread_pct'].mean()
col1, col2, col3 = st.columns(3)
col1.metric('Total Call Volume', f'{int(total_call_vol):,}')
col2.metric('Total Put Volume', f'{int(total_put_vol):,}')
col3.metric('Average Spread %', f'{avg_spread:.2f}%')

# Optional detail tables
display = st.sidebar.checkbox('Show Calls & Puts Tables')
if display:
    st.markdown('---')
    st.subheader('Calls Table')
    st.dataframe(calls)
    st.subheader('Puts Table')
    st.dataframe(puts)

# Footer Notes
st.markdown('''
**How to use:**
- Volume chart highlights top traded strikes.
- Spread heatmap shows relative bid-ask costs.
- Summary metrics give a quick overview of chain activity.
- Toggle tables for deeper inspection.
''')
