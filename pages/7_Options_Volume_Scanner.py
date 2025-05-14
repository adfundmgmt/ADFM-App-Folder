import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Streamlit: Options Volume & Activity Alert Scanner

@st.cache_data
def fetch_volume_metrics(ticker: str, expiry: str) -> dict:
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry)
    except Exception as e:
        raise ValueError(f"Error fetching chain for {ticker} {expiry}: {e}")
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    total_call_vol = calls['volume'].sum()
    total_put_vol = puts['volume'].sum()
    total_vol = total_call_vol + total_put_vol
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
st.title('Options Activity & Alert Scanner')

# Sidebar Inputs
ticker_input = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL,MSFT,GOOG')
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
if not tickers:
    st.sidebar.error('Enter at least one ticker.')
    st.stop()

primary = tickers[0]
all_exps = get_expiries(primary)
if not all_exps:
    st.sidebar.error(f'No expiries for {primary}. Check ticker symbol.')
    st.stop()

expiries = st.sidebar.multiselect('Option Expiries', all_exps, default=all_exps[:3])
if not expiries:
    st.sidebar.warning('Select at least one expiry to analyze.')
    st.stop()

# Top-level Heatmap: total volume or C/P ratio
metric = st.sidebar.selectbox(
    'Overview Metric',
    ['total_vol', 'cp_ratio'],
    format_func=lambda x: dict(total_vol='Total Volume', cp_ratio='Call/Put Volume Ratio')[x]
)

df = build_volume_df(tickers, expiries)
if df.empty:
    st.error('No data. Verify tickers and expiries.')
    st.stop()

heatmap = alt.Chart(df).mark_rect().encode(
    x=alt.X('expiry:N', title='Expiry'),
    y=alt.Y('ticker:N', title='Ticker'),
    color=alt.Color(f'{metric}:Q', title=metric),
    tooltip=['ticker', 'expiry', 'call_vol', 'put_vol', 'total_vol', 'cp_ratio']
).properties(width=700, height=350)
st.altair_chart(heatmap, use_container_width=True)

# Drill-Down: Strike-Level Alerts
st.markdown('---')
if st.checkbox('Enable Strike-Level Activity Alerts'):
    sel = st.selectbox('Choose ticker-expiry', df.apply(lambda r: f"{r['ticker']} | {r['expiry']}", axis=1))
    t_sel, e_sel = [s.strip() for s in sel.split('|')]
    chain = yf.Ticker(t_sel).option_chain(e_sel)
    combined = pd.concat([
        chain.calls.assign(type='call'),
        chain.puts.assign(type='put')
    ], ignore_index=True)
    combined['volume'] = combined['volume'].fillna(0)
    combined['openInterest'] = combined['openInterest'].replace(0, np.nan)
    combined['vol_oi_ratio'] = (combined['volume'] / combined['openInterest']).round(3)
    # Identify top activity by vol/OI
    alerts = combined.sort_values('vol_oi_ratio', ascending=False).head(10)
    alerts = alerts[['type', 'strike', 'volume', 'openInterest', 'vol_oi_ratio']]
    st.markdown('**Top Strikes by Volume/Open Interest**')
    st.dataframe(alerts.reset_index(drop=True))
    # Visual highlight with call/put color spectrum
    bars = alt.Chart(alerts).mark_bar().encode(
        x=alt.X('strike:O', title='Strike'),
        y=alt.Y('vol_oi_ratio:Q', title='Volume/OI Ratio'),
        color=alt.Color('type:N', scale=alt.Scale(domain=['put','call'], range=['#d7191c','#1a9641']), legend=alt.Legend(title='Option Type')),
        tooltip=['type', 'strike', 'volume', 'openInterest', 'vol_oi_ratio']
    ).properties(width=700, height=300)
    st.altair_chart(bars, use_container_width=True)

# Footer Notes
st.markdown("""
**Notes:**
- `cp_ratio` = Call Volume / Put Volume
- `vol_oi_ratio` indicates the proportion of open interest traded today (higher = unusual activity).
- Data via yfinance (API subject to quotas).
""")
