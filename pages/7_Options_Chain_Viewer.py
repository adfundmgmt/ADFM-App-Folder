import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use("default")


st.set_page_config(page_title="Options Chain Viewer", layout="wide")

st.title('Options Chain Viewer')
st.sidebar.header('About This Tool')
st.sidebar.markdown("""
- **Interactive Options Chain**: Fetch real-time call & put chains via yfinance.  
- **Volume by Strike**: Bars showing call volume (green) vs put volume (red).  
- **Summary Metrics**: Total call/put volume and average Black-Scholes delta.  
- **Delta Distribution**: Density plots of option deltas to show skew.  
- **Optional Tables**: Inspect bid/ask, volume, and open interest per strike.
""")

ticker = st.sidebar.text_input('Ticker', 'AAPL').upper()
if not ticker:
    st.sidebar.error('Enter a valid ticker symbol')
    st.stop()

# Only cache the valid expiries (list of strings)
@st.cache_data(ttl=900, show_spinner=True)
def get_valid_expiries(tkr):
    tk = yf.Ticker(tkr)
    raw_expiries = tk.options
    today = datetime.today().date()
    valid_expiries = []
    for exp in raw_expiries:
        exp_date = pd.to_datetime(exp).date()
        if exp_date < today:
            continue
        try:
            chain = tk.option_chain(exp)
            if not chain.calls.empty or not chain.puts.empty:
                valid_expiries.append(exp)
        except Exception:
            continue
    return valid_expiries

with st.spinner("Loading available expiries..."):
    expiries = get_valid_expiries(ticker)

if not expiries:
    st.error(f"No valid options expiries available for {ticker}.")
    st.stop()

expiry = st.sidebar.selectbox('Select Expiry', expiries)

# Fetch the option chain (do NOT cache this!)
try:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    if chain.calls.empty and chain.puts.empty:
        raise ValueError("Empty chain")
except Exception:
    st.warning(
        f"⚠️ Could not load an options chain for {ticker} @ {expiry}. "
        "Please pick a different expiry."
    )
    st.stop()

# Spot price cache is OK, it's a float
@st.cache_data(ttl=600, show_spinner=False)
def get_spot(tkr):
    hist = yf.Ticker(tkr).history(period='1d')
    return hist['Close'].iloc[-1] if not hist.empty else None

spot = get_spot(ticker)
if spot is None:
    st.error("Could not retrieve spot price.")
    st.stop()

calls = chain.calls[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
puts  = chain.puts [['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
calls['type'], puts['type'] = 'Call','Put'
combined = pd.concat([calls, puts], ignore_index=True)

# Time to expiry (ACT/252 convention)
today       = datetime.today().date()
expiry_date = pd.to_datetime(expiry).date()
days_bd     = np.busday_count(today, expiry_date)
T           = max(days_bd, 0) / 252.0
st.sidebar.caption(f"Time to expiry: {days_bd} business days → {T:.3f} years (ACT/252)")

# Warn about zero/missing IV
if (combined['impliedVolatility'] <= 0).any() or combined['impliedVolatility'].isna().any():
    st.warning("Some strikes have zero or missing IV → their delta will be NaN.")

# Vectorized Black-Scholes Delta Calculation
vec_erf = np.vectorize(math.erf)
σ = combined['impliedVolatility'].values
K = combined['strike'].values
S = spot
r = 0.03
T_arr = np.full_like(K, T, dtype=float)
valid = (σ > 0) & (T_arr > 0)
d1 = np.zeros_like(K, dtype=float)
d1[valid] = (
    np.log(S / K[valid]) +
    (r + 0.5 * σ[valid]**2) * T_arr[valid]
) / (σ[valid] * np.sqrt(T_arr[valid]))
call_delta = 0.5 * (1 + vec_erf(d1 / np.sqrt(2)))
combined['delta'] = np.where(combined['type']=='Call', call_delta, call_delta - 1)
combined.loc[~valid, 'delta'] = np.nan

# Summary Metrics
total_call_vol = combined.query("type=='Call'")['volume'].sum()
total_put_vol  = combined.query("type=='Put'")['volume'].sum()
avg_delta      = combined['delta'].mean()

st.subheader('Summary Metrics')
c1, c2, c3 = st.columns(3)
c1.metric('Total Call Volume', f"{int(total_call_vol):,}")
c2.metric('Total Put Volume',  f"{int(total_put_vol):,}")
c3.metric('Avg Delta',          f"{avg_delta:.2f}")

st.markdown('---')

# Volume by Strike Chart (dynamic width)
st.subheader('Volume by Strike')
vol_chart = (
    alt.Chart(combined)
       .mark_bar()
       .encode(
           x=alt.X('strike:Q', title='Strike'),
           y=alt.Y('volume:Q', title='Volume'),
           color=alt.Color(
               'type:N',
               scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']),
               legend=alt.Legend(title='Option Type')
           ),
           tooltip=['type','strike','volume','openInterest','delta']
       )
       .properties(width='container', height=400)
       .interactive()
)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown('---')

# Delta Distribution Panel (dynamic width)
st.subheader('Delta Distribution by Option Type')
delta_hist = (
    alt.Chart(combined)
       .transform_density('delta', as_=['delta','density'], groupby=['type'])
       .mark_area(opacity=0.5)
       .encode(
           x=alt.X('delta:Q', title='Delta'),
           y=alt.Y('density:Q', title='Density'),
           color=alt.Color(
               'type:N',
               scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']),
               legend=alt.Legend(title='Option Type')
           )
       )
       .properties(width='container', height=300)
       .interactive()
)
st.altair_chart(delta_hist, use_container_width=True)

if st.sidebar.checkbox('Show Calls & Puts Tables'):
    st.markdown('---')
    st.subheader('Calls Table')
    st.dataframe(calls)
    st.subheader('Puts Table')
    st.dataframe(puts)

st.caption("© 2025 AD Fund Management LP")
