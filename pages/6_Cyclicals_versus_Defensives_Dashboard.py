import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

CYCLICALS  = ["XLK","XLI","XLF","XLC","XLY"]
DEFENSIVES = ["XLP","XLE","XLV","XLRE","XLB","XLU"]

st.set_page_config(layout="wide",
                   page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight")

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("Look‑back")
    spans = {"3 M":90,"6 M":180,"9 M":270,"YTD":None,"1 Y":365,
             "3 Y":365*3,"5 Y":365*5,"10 Y":365*10}
    span_key = st.selectbox("", list(spans.keys()), index=list(spans).index("5 Y"))

# ─── Date windows ─────────────────────────────────────────
today = datetime.today();  hist_start = today - timedelta(days=365*10+220)
disp_start = datetime(today.year,1,1) if span_key=="YTD" \
             else today - timedelta(days=spans[span_key])

# ─── Cached Yahoo pull ────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def yf_prices(tickers,start,end):
    return yf.download(tickers,start, end,
                       group_by="ticker",auto_adjust=True,progress=False)

def basket(etfs):
    raw=yf_prices(etfs,hist_start, today)
    closes = (raw.xs('Close',level=1,axis=1)
              if isinstance(raw.columns,pd.MultiIndex) else raw[['Close']])
    closes = closes.fillna(method='ffill').dropna()
    return (1+closes.pct_change()).cumprod().mean(axis=1)

cyc,defn = basket(CYCLICALS), basket(DEFENSIVES)
ratio = (cyc/defn*100).dropna()

def rsi(series,n=14):
    delta=series.diff(); up=delta.clip(lower=0); dn=-delta.clip(upper=0)
    rs=(up.rolling(n).mean()/dn.rolling(n).mean())
    return 100-100/(1+rs)
rsi_full=rsi(ratio)

ma50,ma200 = ratio.rolling(50).mean(), ratio.rolling(200).mean()

mask = ratio.index>=disp_start
ratio_v, rsi_v = ratio[mask], rsi_full[mask]

# ─── Metric strip ─────────────────────────────────────────
col1,col2,col3 = st.columns([1,1,1.4])
col1.metric("Return",f"{(ratio_v.iat[-1]/ratio_v.iat[0]-1)*100:,.1f}%")
col2.metric("RSI‑14",f"{rsi_v.iat[-1]:.0f}")
trend = "Bullish (50>200)" if ma50.iat[-1]>ma200.iat[-1] else "Bearish (50<200)"
col3.metric("Trend Regime",trend)

# ─── Figure w/ subplots ───────────────────────────────────
fig = make_subplots(rows=2,cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75,0.25],
                    vertical_spacing=0.03)

fig.add_trace(go.Scatter(x=ratio_v.index,y=ratio_v,
                         line=dict(color="#355E3B",width=2),
                         name="Cyc/Def"),row=1,col=1)
fig.add_trace(go.Scatter(x=ma50.index,y=ma50,
                         line=dict(color="blue",width=2),
                         name="50‑DMA"),row=1,col=1)
fig.add_trace(go.Scatter(x=ma200.index,y=ma200,
                         line=dict(color="red",width=2),
                         name="200‑DMA"),row=1,col=1)

fig.add_trace(go.Scatter(x=rsi_v.index,y=rsi_v,
                         line=dict(color="black",width=2),
                         name="RSI‑14",showlegend=False),row=2,col=1)

# Horizontal RSI bands
for y,clr,txt in [(70,"red","Overbought"),(30,"green","Oversold")]:
    fig.add_hline(y=y,row=2,col=1,line_dash="dot", line_color=clr)
    fig.add_annotation(x=rsi_v.index[0],y=y,text=txt,
                       yshift=4 if y==70 else -8,
                       showarrow=False,font=dict(color=clr),row=2,col=1)

fig.update_layout(height=700,margin=dict(l=20,r=20,t=30,b=20),
                  plot_bgcolor="white",
                  legend=dict(orientation="h",y=1.02,x=1,xanchor="right",
                              font=dict(size=13)))

st.plotly_chart(fig,use_container_width=True)

# ─── Download CSV ────────────────────────────────────────
csv = pd.concat({"Ratio":ratio_v,"MA50":ma50,"MA200":ma200,"RSI":rsi_v},
                axis=1).dropna().to_csv().encode()
st.download_button("Download CSV",csv,"cyc_def_ratio.csv")
