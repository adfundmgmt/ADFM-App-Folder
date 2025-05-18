import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime

st.set_page_config(page_title="Options Chain Viewer", layout="wide")
st.title('Options Chain Viewer')

# About box at top instead of sidebar for clarity
with st.expander("ℹ️ About This Tool", expanded=False):
    st.markdown("""
    - **Interactive Options Chain**: Fetch real-time call & put chains via yfinance.  
    - **Volume by Strike**: Bars showing call volume (green) vs put volume (red), ATM highlighted.  
    - **Summary Metrics**: Total call/put volume, put/call ratio, and average Black-Scholes delta.  
    - **Delta Distribution**: Density plots of option deltas to show skew.  
    - **Optional Tables & Download**: Inspect bid/ask, volume, open interest per strike; download full option chain.
    """)

# --- Inputs: Ticker, Expiry, Filtering ---
input1, input2 = st.columns([2, 3])
with input1:
    ticker = st.text_input('Ticker', 'AAPL').upper()

with input2:
    hide_zero_vol = st.checkbox("Hide zero-volume options", value=True, help="Show only strikes with nonzero volume")

# --- Expiry selection with stickiness ---
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

# --- Maintain previous expiry selection if possible ---
prev_expiry = st.session_state.get("prev_expiry")
if prev_expiry in expiries:
    default_expiry_idx = expiries.index(prev_expiry)
else:
    default_expiry_idx = 0

expiry = st.selectbox('Select Expiry', expiries, index=default_expiry_idx, key="expiry_select")

# Save the selected expiry for stickiness
st.session_state["prev_expiry"] = expiry

# --- Fetch the option chain ---
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

if hide_zero_vol:
    combined = combined[combined["volume"] > 0]
    calls = calls[calls["volume"] > 0]
    puts = puts[puts["volume"] > 0]

today       = datetime.today().date()
expiry_date = pd.to_datetime(expiry).date()
days_bd     = np.busday_count(today, expiry_date)
T           = max(days_bd, 0) / 252.0
st.caption(f"Time to expiry: {days_bd} business days → {T:.3f} years (ACT/252)")

if (combined['impliedVolatility'] <= 0).any() or combined['impliedVolatility'].isna().any():
    st.warning("Some strikes have zero or missing IV → their delta will be NaN.")

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

atm_idx = (combined['strike'] - spot).abs().idxmin()
atm_strike = combined.loc[atm_idx, 'strike']
st.caption(f"📌 Spot price: **${spot:.2f}** &nbsp; | &nbsp; ATM Strike: **{atm_strike}**")

# --- Layout: Metrics, Charts, Tables as before ---
# ... (your chart/table code unchanged)

