import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime
import matplotlib.pyplot as plt
from time import sleep

plt.style.use("default")

st.set_page_config(page_title="Options Chain Viewer", layout="wide")
st.title('Options Chain Viewer')
st.sidebar.header('About This Tool')
st.sidebar.markdown("""
- **Interactive Options Chain**: Fetch real-time call & put chains via yfinance.  
- **Volume by Strike**: Bars showing call volume (green) vs put volume (red), ATM highlighted.  
- **Summary Metrics**: Total call/put volume, put/call ratio, and average Black-Scholes delta.  
- **Delta Distribution**: Density plots of option deltas to show skew.  
- **Optional Tables & Download**: Inspect bid/ask, volume, open interest per strike; download full option chain.
""")

ticker = st.sidebar.text_input('Ticker', 'AAPL').upper().strip()
if not ticker:
    st.sidebar.error('Enter a valid ticker symbol')
    st.stop()

hide_zero_vol = st.sidebar.checkbox("Hide zero-volume options", value=True)

# --- Robust Expiry Fetch with Retry ---
@st.cache_data(ttl=300, show_spinner=True)
def get_valid_expiries(tkr, max_retries=3):
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(tkr)
            raw_expiries = tk.options
            if raw_expiries is None:
                raw_expiries = []
            today = datetime.today().date()
            valid_expiries = []
            for exp in raw_expiries:
                try:
                    exp_date = pd.to_datetime(exp).date()
                    if exp_date < today:
                        continue
                    chain = tk.option_chain(exp)
                    if not chain.calls.empty or not chain.puts.empty:
                        valid_expiries.append(exp)
                except Exception:
                    continue
            if raw_expiries:
                return raw_expiries, valid_expiries
        except Exception:
            sleep(1 + attempt)
            continue
    return [], []

with st.spinner("Loading available expiries..."):
    raw_expiries, expiries = get_valid_expiries(ticker)

if not raw_expiries:
    st.error(
        f"Yahoo Finance returned no option expiry dates for **{ticker}**.\n\n"
        "This is *almost always* a Yahoo outage, a rate-limit, or a VPN/IP block. "
        "Steps to fix:\n"
        "- Try a common large cap (MSFT, NVDA, SPY).\n"
        "- If those also fail, wait 5-10 minutes and refresh.\n"
        "- If only this ticker fails, it may be delisted or have no options.\n"
        "- If nothing works, try switching networks or a VPN."
    )
    st.stop()

if not expiries:
    st.error(
        f"**No valid options chains available for {ticker}.**\n"
        f"Raw expiries fetched: {', '.join(raw_expiries) if raw_expiries else 'None'}\n\n"
        "Yahoo returned expiry dates, but all chains are empty or could not be loaded. "
        "This is usually a temporary Yahoo Finance API issue or a sign of rate-limiting. "
        "Try again in 15 minutes, switch tickers, or rotate your IP."
    )
    st.stop()

expiry = st.sidebar.selectbox('Select Expiry', expiries)

# -- Defensive option chain fetch --
try:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    if chain.calls.empty and chain.puts.empty:
        raise ValueError("Empty chain")
except Exception as e:
    st.warning(
        f"⚠️ Could not load an options chain for {ticker} @ {expiry}. "
        f"Exception: {str(e)}\n"
        "Try picking a different expiry or refreshing the page."
    )
    st.stop()

# Spot price fetch
@st.cache_data(ttl=600, show_spinner=False)
def get_spot(tkr):
    try:
        hist = yf.Ticker(tkr).history(period='1d')
        return hist['Close'].iloc[-1] if not hist.empty else None
    except Exception:
        return None

spot = get_spot(ticker)
if spot is None or np.isnan(spot):
    st.error("Could not retrieve spot price for this ticker.")
    st.stop()

calls = chain.calls[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
puts  = chain.puts [['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
calls['type'], puts['type'] = 'Call','Put'
combined = pd.concat([calls, puts], ignore_index=True)

if hide_zero_vol:
    combined = combined[combined["volume"] > 0]
    calls = calls[calls["volume"] > 0]
    puts = puts[puts["volume"] > 0]

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

# Display spot + ATM strike
atm_idx = (combined['strike'] - spot).abs().idxmin()
atm_strike = combined.loc[atm_idx, 'strike']
st.caption(f"📌 Spot price: **${spot:.2f}** &nbsp; | &nbsp; ATM Strike: **{atm_strike}**")

# Summary Metrics & Put/Call Ratio
total_call_vol = combined.query("type=='Call'")['volume'].sum()
total_put_vol  = combined.query("type=='Put'")['volume'].sum()
avg_delta      = combined['delta'].mean()
put_call_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan

st.subheader('Summary Metrics')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Call Volume', f"{int(total_call_vol):,}")
c2.metric('Total Put Volume',  f"{int(total_put_vol):,}")
c3.metric('Put/Call Ratio',    f"{put_call_ratio:.2f}")
c4.metric('Avg Delta',         f"{avg_delta:.2f}")

st.markdown('---')

# Volume by Strike Chart with ATM highlight
st.subheader('Volume by Strike')
highlight_color = "#FFD600"
combined['highlight'] = np.where(combined['strike'] == atm_strike, 'ATM', 'Other')
vol_chart = (
    alt.Chart(combined)
       .mark_bar()
       .encode(
           x=alt.X('strike:Q', title='Strike'),
           y=alt.Y('volume:Q', title='Volume'),
           color=alt.condition(
               alt.datum.highlight == 'ATM',
               alt.value(highlight_color),
               alt.Color(
                   'type:N',
                   scale=alt.Scale(domain=['Call','Put'], range=['#1a9641','#d7191c']),
                   legend=alt.Legend(title='Option Type')
               )
           ),
           tooltip=['type','strike','volume','openInterest','delta']
       )
       .properties(width='container', height=400)
       .interactive()
)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown(f"**Yellow bar highlights the ATM strike ({atm_strike}).**")

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

# Download full option chain
with st.expander("Download Option Chain Table"):
    st.download_button(
        "Download as CSV",
        combined.to_csv(index=False),
        file_name=f"{ticker}_{expiry}_option_chain.csv"
    )

st.caption("© 2025 AD Fund Management LP")
