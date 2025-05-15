import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from math import erf
from datetime import datetime

# ── App Title & Sidebar ─────────────────────────────────────────────────────
st.title('Options Chain Viewer')
st.sidebar.header('About This Tool')
st.sidebar.markdown("""
- **Interactive Options Chain**: Fetch real‑time call & put chains for any ticker and expiry via yfinance.  
- **Volume by Strike**: Bar chart highlighting today’s trading volume across strikes (calls in green, puts in red).  
- **Summary Metrics**: Total call/put volume and average Black‑Scholes delta for quick chain health checks.  
- **Delta Distribution Panel**: Density plots of option deltas to visualize skew and moneyness across the chain.  
- **Optional Tables**: Toggle to inspect detailed bid/ask, volume, and open interest values per strike.
""")

# ── Sidebar Inputs ─────────────────────────────────────────────────────────
ticker = st.sidebar.text_input('Ticker', 'AAPL').upper()
if not ticker:
    st.sidebar.error('Enter a valid ticker symbol')
    st.stop()

# ── Choose caching decorator ────────────────────────────────────────────────
cache_data = getattr(st, "cache_data", st.cache)

# ── Fetch & filter valid expiries (only today or future) ───────────────────
@cache_data(ttl=3600, show_spinner=False)
def get_valid_expiries(tkr):
    raw = yf.Ticker(tkr).options
    today = datetime.today().date()
    return [exp for exp in raw if pd.to_datetime(exp).date() >= today]

expiries = get_valid_expiries(ticker)
if not expiries:
    st.error(f"No upcoming expiries available for {ticker}.")
    st.stop()

expiry = st.sidebar.selectbox('Select Expiry', expiries)

# ── Fetch option chain (no caching) ─────────────────────────────────────────
try:
    ticker_obj = yf.Ticker(ticker)
    chain = ticker_obj.option_chain(expiry)
    # guard against empty chain
    if chain.calls.empty and chain.puts.empty:
        raise ValueError("Empty chain")
except Exception:
    st.warning(
        f"⚠️ Could not load an options chain for {ticker} @ {expiry}. "
        "Please pick a different expiry."
    )
    st.stop()

# ── Spot price fetcher ─────────────────────────────────────────────────────
@cache_data(ttl=600, show_spinner=False)
def get_spot(tkr):
    hist = yf.Ticker(tkr).history(period='1d')
    return hist['Close'].iloc[-1] if not hist.empty else None

spot = get_spot(ticker)
if spot is None:
    st.error("Could not retrieve spot price.")
    st.stop()

# ── Prepare call & put DataFrames ──────────────────────────────────────────
calls = chain.calls[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
puts  = chain.puts[['strike','bid','ask','volume','openInterest','impliedVolatility']].copy()
calls['type'], puts['type'] = 'Call','Put'
combined = pd.concat([calls, puts], ignore_index=True)

# ── Time to expiry (ACT/252 convention) ────────────────────────────────────
today       = datetime.today().date()
expiry_date = pd.to_datetime(expiry).date()
days_bd     = np.busday_count(today, expiry_date)
T           = max(days_bd, 0) / 252.0
st.sidebar.caption(f"Time to expiry: {days_bd} business days → {T:.3f} years (ACT/252)")

# ── Warn about zero/missing IV ─────────────────────────────────────────────
if (combined['impliedVolatility'] <= 0).any() or combined['impliedVolatility'].isna().any():
    st.warning("Some strikes have zero or missing IV → their delta will be NaN.")

# ── Vectorized Black‑Scholes Delta Calculation ────────────────────────────
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

call_delta = 0.5 * (1 + erf(d1 / np.sqrt(2)))
combined['delta'] = np.where(combined['type']=='Call', call_delta, call_delta - 1)
combined.loc[~valid, 'delta'] = np.nan

# ── Summary Metrics ─────────────────────────────────────────────────────────
total_call_vol = combined.loc[combined['type']=='Call', 'volume'].sum()
total_put_vol  = combined.loc[combined['type']=='Put',  'volume'].sum()
avg_delta      = combined['delta'].mean()

st.subheader('Summary Metrics')
c1, c2, c3 = st.columns(3)
c1.metric('Total Call Volume', f"{int(total_call_vol):,}")
c2.metric('Total Put Volume',  f"{int(total_put_vol):,}")
c3.metric('Avg Delta',          f"{avg_delta:.2f}")

st.markdown('---')

# ── Volume by Strike Chart ─────────────────────────────────────────────────
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
        .interactive()
        .properties(height=400)
)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown('---')

# ── Delta Distribution Panel ───────────────────────────────────────────────
st.subheader('Delta Distribution by Option Type')
delta_hist = (
    alt.Chart(combined)
        .transform_density(
            'delta', as_=['delta','density'], groupby=['type']
        )
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
        .interactive()
        .properties(height=300)
)
st.altair_chart(delta_hist, use_container_width=True)

# ── Optional Detail Tables ─────────────────────────────────────────────────
if st.sidebar.checkbox('Show Calls & Puts Tables'):
    st.markdown('---')
    st.subheader('Calls Table')
    st.dataframe(calls)
    st.subheader('Puts Table')
    st.dataframe(puts)

st.caption("© 2025 AD Fund Management LP")
