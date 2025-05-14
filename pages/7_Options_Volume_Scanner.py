import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math

# Simple Options Chain Viewer with Greek Distribution Panel
st.title('Options Chain Viewer')

# Sidebar: About This Tool
st.sidebar.header('About This Tool')
st.sidebar.markdown(
    """
- **Interactive Options Chain**: Fetch real-time call & put chains for any ticker and expiry via yfinance.
- **Volume by Strike**: Bar chart highlighting today’s trading volume across strikes (calls in green, puts in red).
- **Summary Metrics**: Total call/put volume and average Black‑Scholes delta for quick chain health checks.
- **Delta Distribution Panel**: Density plots of option deltas to visualize skew and moneyness across the chain.
- **Optional Tables**: Toggle to inspect detailed bid/ask, volume, and open interest values per strike.
    """
)

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

# Retrieve option chain & underlying spot price
tk = yf.Ticker(ticker)
spot = tk.history(period='1d')['Close'].iloc[-1]
chain = tk.option_chain(expiry)

# Prepare call and put DataFrames
calls = chain.calls[['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
puts  = chain.puts[['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
calls['type'] = 'Call'
puts['type'] = 'Put'
combined = pd.concat([calls, puts], ignore_index=True)

# Time to expiry in years (using local today date)
today = pd.to_datetime('today').normalize()
expiry_date = pd.to_datetime(expiry)
T = max((expiry_date - today).days, 0) / 365

# Black-Scholes Delta
def bs_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'Call':
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    else:
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1

# Compute Delta for each option
RISK_FREE_RATE = 0.03
combined['delta'] = combined.apply(
    lambda row: bs_delta(row['type'], spot, row['strike'], T, RISK_FREE_RATE, row['impliedVolatility']),
    axis=1
)

# Top-level: Volume by Strike
st.subheader('Volume by Strike')
vol_chart = alt.Chart(combined).mark_bar().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('volume:Q', title='Volume'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call', 'Put'], range=['#1a9641', '#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=['type', 'strike', 'volume', 'openInterest', 'delta']
).properties(width=800, height=400)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown('---')
# Summary Metrics Panel
st.subheader('Summary Metrics')
total_call_vol = combined.loc[combined['type'] == 'Call', 'volume'].sum()
total_put_vol  = combined.loc[combined['type'] == 'Put',  'volume'].sum()
avg_delta      = combined['delta'].mean()
col1, col2, col3 = st.columns(3)
col1.metric('Total Call Volume', f'{int(total_call_vol):,}')
col2.metric('Total Put Volume', f'{int(total_put_vol):,}')
col3.metric('Avg Delta', f'{avg_delta:.2f}')

st.markdown('---')
# Greek Distribution Panel
st.subheader('Delta Distribution by Option Type')

delta_hist = alt.Chart(combined).transform_density(
    'delta',
    as_=['delta', 'density'],
    groupby=['type']
).mark_area(opacity=0.5).encode(
    x=alt.X('delta:Q', title='Delta'),
    y=alt.Y('density:Q', title='Density'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call', 'Put'], range=['#1a9641', '#d7191c']), legend=alt.Legend(title='Option Type'))
).properties(width=800, height=300)
st.altair_chart(delta_hist, use_container_width=True)

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
- Volume by strike identifies the busiest strikes.
- Summary metrics provide quick chain IQ and average delta.
- Delta distribution shows skew and moneyness bias.
- Toggle tables for full chain details.
''')
