import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Streamlit: Options Volume Scanner

@st.cache_data
def fetch_volume_metrics(ticker: str, expiry: str) -> dict:
    tk = yf.Ticker(ticker)
    # Fetch option chain
    try:
        chain = tk.option_chain(expiry)
    except Exception as e:
        raise ValueError(f"Error fetching chain for {ticker} {expiry}: {e}")
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    # Compute total volumes
    total_call_vol = calls['volume'].sum()
    total_put_vol = puts['volume'].sum()
    total_vol = total_call_vol + total_put_vol
    # Call/Put volume ratio
    cp_ratio = total_call_vol / total_put_vol if total_put_vol > 0 else np.nan
    return {
        'ticker': ticker,
        'expiry': expiry,
        'call_vol': total_call_vol,
        'put_vol': total_put_vol,
        'total_vol': total_vol,
        'cp_ratio': cp_ratio
    }

@st.cache_data
def get_expiries(ticker: str) -> list:
    try:
        return yf.Ticker(ticker).options
    except Exception:
        return []

@st.cache_data
def build_volume_df(tickers: list, expiries: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for e in expiries:
            try:
                rows.append(fetch_volume_metrics(t, e))
            except Exception:
                continue
    return pd.DataFrame(rows)

# --- Streamlit UI ---
st.title('Options Volume Scanner')

# Sidebar: Inputs
ticker_input = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL,MSFT,GOOG')
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
if not tickers:
    st.sidebar.error('Enter at least one ticker.')
    st.stop()

# Fetch expiries from first ticker
primary = tickers[0]
all_exps = get_expiries(primary)
if not all_exps:
    st.sidebar.error(f'No expiries for {primary}. Check ticker symbol.')
    st.stop()

expiries = st.sidebar.multiselect('Option Expiries', all_exps, default=all_exps[:3])
if not expiries:
    st.sidebar.warning('Select at least one expiry to analyze.')
    st.stop()

metric = st.sidebar.selectbox(
    'Metric',
    ['total_vol', 'call_vol', 'put_vol', 'cp_ratio'],
    format_func=lambda x: dict(total_vol='Total Volume', call_vol='Call Volume', put_vol='Put Volume', cp_ratio='Call/Put Ratio')[x]
)

# Build data frame
df = build_volume_df(tickers, expiries)
if df.empty:
    st.error('No volume data. Verify tickers and expiries.')
    st.stop()

# Heatmap: expiry vs ticker
heatmap = alt.Chart(df).mark_rect().encode(
    x=alt.X('expiry:N', title='Expiry'),
    y=alt.Y('ticker:N', title='Ticker'),
    color=alt.Color(f'{metric}:Q', title=dict(total_vol='Total Volume', call_vol='Call Volume', put_vol='Put Volume', cp_ratio='Call/Put Ratio')[metric]),
    tooltip=['ticker', 'expiry', 'call_vol', 'put_vol', 'total_vol', 'cp_ratio']
).properties(width=700, height=350)
st.altair_chart(heatmap, use_container_width=True)

# Drill-down: show top strikes by volume for a selected pair
st.markdown('---')
if st.button('Show Strike-Level Volume Details'):
    sel = st.selectbox('Choose ticker-expiry', df.apply(lambda r: f"{r['ticker']} | {r['expiry']}", axis=1).tolist())
    t_sel, e_sel = [s.strip() for s in sel.split('|')]
    # Fetch chain again for details
    chain = yf.Ticker(t_sel).option_chain(e_sel)
    combined = pd.concat([chain.calls.assign(type='call'), chain.puts.assign(type='put')])
    combined['volume_abs'] = combined['volume']
    top_strikes = combined.sort_values('volume_abs', ascending=False).head(10)
    bar = alt.Chart(top_strikes).mark_bar().encode(
        x=alt.X('strike:O', title='Strike'),
        y=alt.Y('volume_abs:Q', title='Volume'),
        color=alt.Color('type:N'),
        tooltip=['type', 'strike', 'volume_abs']
    ).properties(width=700, height=300)
    st.altair_chart(bar, use_container_width=True)
    st.table(top_strikes[['type','strike','volume_abs']].reset_index(drop=True))

# Footer notes
st.markdown("""
**Notes:**
- `cp_ratio` = Call Volume / Put Volume
- Data sourced in real-time via yfinance (subject to API limits)
""")
