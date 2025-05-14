import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# Streamlit: Options Volume & Activity Alert Scanner with 3D Strike-Level Insights

@st.cache_data
def fetch_volume_metrics(ticker: str, expiry: str) -> dict:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
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
    return yf.Ticker(ticker).options or []

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

# UI
st.title('Options Activity & Alert Scanner')

# Sidebar Inputs
ticker_input = st.sidebar.text_input('Tickers (comma-separated)', 'AAPL,MSFT,GOOG')
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
if not tickers:
    st.sidebar.error('Enter at least one ticker.')
    st.stop()
primary = tickers[0]
expiries = st.sidebar.multiselect('Option Expiries', get_expiries(primary), default=get_expiries(primary)[:3])
if not expiries:
    st.sidebar.warning('Select at least one expiry to analyze.')
    st.stop()

# Overview Metric Selector
overview_metric = st.sidebar.selectbox(
    'Overview Metric',
    ['total_vol', 'cp_ratio'],
    format_func=lambda x: {'total_vol':'Total Volume','cp_ratio':'Call/Put Volume Ratio'}[x]
)

df = build_volume_df(tickers, expiries)
if df.empty:
    st.error('No data. Verify tickers and expiries.')
    st.stop()

# Top-level Heatmap
heatmap = alt.Chart(df).mark_rect().encode(
    x=alt.X('expiry:N', title='Expiry'),
    y=alt.Y('ticker:N', title='Ticker'),
    color=alt.Color(f'{overview_metric}:Q', title=overview_metric),
    tooltip=['ticker','expiry','call_vol','put_vol','total_vol','cp_ratio']
).properties(width=700,height=300)
st.altair_chart(heatmap,use_container_width=True)

st.markdown('---')
# Strike-Level Activity Alerts (always available)
sel = st.selectbox('Choose ticker-expiry for Strike-Level Alerts', df.apply(lambda r: f"{r['ticker']} | {r['expiry']}", axis=1).tolist())
sel_t, sel_e = [s.strip() for s in sel.split('|')]
chain = yf.Ticker(sel_t).option_chain(sel_e)
combined = pd.concat([chain.calls.assign(type='call'), chain.puts.assign(type='put')], ignore_index=True)
combined['volume'] = combined['volume'].fillna(0)
combined['openInterest'] = combined['openInterest'].replace(0, np.nan)
combined['vol_oi_ratio'] = (combined['volume'] / combined['openInterest']).round(3)
# Top 10 strikes by vol/OI
alerts = combined.sort_values('vol_oi_ratio', ascending=False).head(10)[['type','strike','volume','openInterest','vol_oi_ratio']]
st.markdown('**Top 10 Strikes by Volume/Open Interest Ratio**')
st.dataframe(alerts.reset_index(drop=True))
# 2D Bar Chart with call/put color
bar_chart = alt.Chart(alerts).mark_bar().encode(
    x='strike:O',
    y='vol_oi_ratio:Q',
    color=alt.Color('type:N', scale=alt.Scale(domain=['put','call'], range=['#d7191c','#1a9641']), legend=alt.Legend(title='Type')),
    tooltip=['type','strike','volume','openInterest','vol_oi_ratio']
).properties(width=700,height=300)
st.altair_chart(bar_chart,use_container_width=True)

# 3D Scatter for deeper context
fig3d = px.scatter_3d(
    alerts,
    x='strike', y='volume', z='vol_oi_ratio',
    color='type',
    title='3D View: Volume vs. Strike vs. Vol/OI Ratio'
)
st.plotly_chart(fig3d)

# Footer
st.markdown("""
**Notes:**
- `cp_ratio`: call_vol / put_vol
- `vol_oi_ratio`: volume / openInterest (high ≈ unusual activity)
- Chart updates per selection; data via yfinance.
""")
