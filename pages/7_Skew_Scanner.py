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
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1) if option_type == 'call' else norm_cdf(d1) - 1

@st.cache_data
def get_expiries_for_ticker(ticker: str) -> list:
    """Return list of available option expiries for a ticker"""
    try:
        return yf.Ticker(ticker).options
    except Exception:
        return []

@st.cache_data
def fetch_metrics(ticker: str, expiry: str) -> dict:
    tk = yf.Ticker(ticker)
    # Underlying spot price
    hist = tk.history(period='1d')
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")
    S = hist['Close'].iloc[-1]
    # Option chain
    chain = tk.option_chain(expiry)
    calls = chain.calls.assign(type='call')
    puts = chain.puts.assign(type='put')
    df = pd.concat([calls, puts], ignore_index=True)
    # Mid prices and IV
    df['mid_iv'] = df['impliedVolatility']
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    # Time to expiry in years
    exp_date = pd.to_datetime(expiry)
    T_val = max((exp_date - pd.Timestamp.utcnow()).days, 0) / 365
    df['T'] = T_val
    df['S'] = S
    # Calculate delta for each row
    df['delta'] = df.apply(lambda r: bs_delta(r['type'], r['S'], r['strike'], r['T'], RISK_FREE_RATE, r['mid_iv']), axis=1)
    # ATM IV
    atm_row = df.loc[(df['strike'] - S).abs().idxmin()]
    atm_iv = atm_row['mid_iv']
    # 25-delta call & put
    call25 = df[df['type'] == 'call'].iloc[(df[df['type'] == 'call']['delta'] - 0.25).abs().argsort()[0]]
    put25 = df[df['type'] == 'put'].iloc[(df[df['type'] == 'put']['delta'] + 0.25).abs().argsort()[0]]
    # Compute metrics
    rr = call25['mid_iv'] - put25['mid_iv']
    fly = 0.5 * (call25['mid_iv'] + put25['mid_iv']) - atm_iv
    return {
        'ticker': ticker,
        'expiry': expiry,
        'rr': rr,
        'fly': fly
    }

@st.cache_data
def build_dataframe(tickers: list, expiries: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for e in expiries:
            try:
                rows.append(fetch_metrics(t, e))
            except Exception:
                continue
    return pd.DataFrame(rows)

# Streamlit App UI
st.title('Vol Surface & Skew Scanner')

# Sidebar inputs
raw = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL, MSFT, GOOG')
tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
# Aggregate expiries across all tickers
all_exps = sorted(set(exp for t in tickers for exp in get_expiries_for_ticker(t)))
if not all_exps:
    st.warning('No option expiries found. Please check your tickers.')
    st.stop()
expiries = st.sidebar.multiselect('Select Expiries', all_exps, default=all_exps[:3])
metric = st.sidebar.selectbox('Metric to Display', ['rr', 'fly'], format_func=lambda x: 'Risk Reversal' if x == 'rr' else 'Butterfly')

# Build data
df = build_dataframe(tickers, expiries)

if df.empty:
    st.error('No valid data—please verify tickers and expiries.')
else:
    # Render heatmap
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('expiry:N', title='Expiry'),
        y=alt.Y('ticker:N', title='Ticker'),
        color=alt.Color(f'{metric}:Q', title=metric.upper()),
        tooltip=['ticker', 'expiry', 'rr', 'fly']
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)
