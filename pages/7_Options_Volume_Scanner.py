import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

# Options Chain Viewer with ATM Filter & IV Skew
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

# Retrieve chain & spot
tk = yf.Ticker(ticker)
spot = tk.history(period='1d')['Close'].iloc[-1]
chain = tk.option_chain(expiry)

# Prepare data
df_calls = chain.calls[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
df_puts  = chain.puts[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
for df, t in [(df_calls,'Call'), (df_puts,'Put')]:
    df['type'] = t
combined = pd.concat([df_calls, df_puts], ignore_index=True)

# ATM filtering
unique_strikes = sorted(combined['strike'].unique())
atm_strike = min(unique_strikes, key=lambda x: abs(x - spot))
n = st.sidebar.slider('Strikes ±N around ATM', 0, len(unique_strikes)//2, 0)
if n > 0:
    idx = unique_strikes.index(atm_strike)
    keep = unique_strikes[max(idx-n,0): min(idx+n+1, len(unique_strikes))]
    combined = combined[combined['strike'].isin(keep)]

# Volume by Strike chart
st.subheader('Volume by Strike')
bar = alt.Chart(combined).mark_bar().encode(
    x=alt.X('strike:O', title='Strike'),
    y=alt.Y('volume:Q', title='Volume'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=['type','strike','volume','openInterest']
).properties(width=800, height=400)
# Add ATM marker
df_atm = pd.DataFrame({'strike':[atm_strike]})
marker = alt.Chart(df_atm).mark_rule(color='black', strokeDash=[4,2]).encode(x='strike:O')
st.altair_chart(bar + marker, use_container_width=True)

# Implied Volatility Skew
st.subheader('Implied Volatility Skew')
iv_df = combined[['strike','impliedVolatility','type']].copy()
skew = alt.Chart(iv_df).mark_line(point=True).encode(
    x=alt.X('strike:Q', title='Strike'),
    y=alt.Y('impliedVolatility:Q', title='Implied Volatility'),
    color=alt.Color('type:N', scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']), legend=alt.Legend(title='Option Type')),
    tooltip=['type','strike','impliedVolatility']
).properties(width=800, height=300)
st.altair_chart(skew, use_container_width=True)

# Optional detail tables
display = st.sidebar.checkbox('Show Calls & Puts Tables')
if display:
    st.markdown('---')
    st.subheader('Calls')
    st.dataframe(df_calls)
    st.subheader('Puts')
    st.dataframe(df_puts)

# Footer notes
st.markdown('''
**Usage:**
- Adjust slider to limit strikes around ATM.
- Bar chart highlights volume with a dashed ATM line.
- IV Skew chart reveals vol curve shape.
- Toggle tables for deeper detail.
''')
