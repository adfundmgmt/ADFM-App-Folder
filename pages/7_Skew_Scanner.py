import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import altair as alt

# Configuration
RISK_FREE_RATE = 0.03  # 3% annual risk-free assumption

# Normal CDF via error function
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# Black-Scholes Delta
def bs_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if option_type == 'call' else norm_cdf(d1) - 1

@st.cache_data
def fetch_metrics(ticker, expiry):
    tk = yf.Ticker(ticker)
    # Underlying spot
    hist = tk.history(period='1d')
    S = hist['Close'].iloc[-1]
    # Option chain
    opt = tk.option_chain(expiry)
    calls = opt.calls.assign(type='call')
    puts = opt.puts.assign(type='put')
    df = pd.concat([calls, puts], ignore_index=True)
    # Mid IV and prices
    df['mid_iv'] = df['impliedVolatility']
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    # Time to expiry (years)
    exp_date = pd.to_datetime(expiry)
    T_val = (exp_date - pd.Timestamp.utcnow()).days / 365
    df['T'] = T_val
    df['S'] = S
    # Calculate delta
    df['delta'] = df.apply(lambda r: bs_delta(r['type'], r['S'], r['strike'], r['T'], RISK_FREE_RATE, r['mid_iv']), axis=1)
    # ATM IV
    atm_iv = df.loc[(df['strike'] - S).abs().idxmin(), 'mid_iv']
    # 25-delta call & put
    call25 = df[df['type'] == 'call'].iloc[(df[df['type']=='call']['delta'] - 0.25).abs().argsort()[0]]
    put25 = df[df['type'] == 'put'].iloc[(df[df['type']=='put']['delta'] + 0.25).abs().argsort()[0]]
    # Metrics
    rr = call25['mid_iv'] - put25['mid_iv']
    fly = 0.5 * (call25['mid_iv'] + put25['mid_iv']) - atm_iv
    return {'ticker': ticker, 'expiry': expiry, 'rr': rr, 'fly': fly}

@st.cache_data
def build_dataframe(tickers, expiries):
    rows = []
    for e in expiries:
        for t in tickers:
            try:
                rows.append(fetch_metrics(t, e))
            except Exception:
                continue
    return pd.DataFrame(rows)

# Streamlit App
st.title('Vol Surface & Skew Scanner')
# Inputs
tickers = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL,MSFT,GOOG').upper().split(',')
available = yf.Ticker(tickers[0]).options if tickers else []
expiries = st.sidebar.multiselect('Expiries', available, default=available[:3])
metric = st.sidebar.selectbox('Metric', ['rr', 'fly'], format_func=lambda x: 'Risk Reversal' if x == 'rr' else 'Butterfly')

# Data
df = build_dataframe(tickers, expiries)

if df.empty:
    st.warning('No valid data—check tickers and expiries.')
else:
    # Heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x='expiry:N',
        y='ticker:N',
        color=alt.Color(f'{metric}:Q', title=metric.upper()),
        tooltip=['ticker', 'expiry', 'rr', 'fly']
    ).properties(width=800, height=300)
    st.altair_chart(chart, use_container_width=True)
