import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

CYCLICALS  = ["XLK","XLI","XLF","XLC","XLY"]
DEFENSIVES = ["XLP","XLE","XLV","XLRE","XLB","XLU"]

st.set_page_config(layout="wide", page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight")

# Sidebar: lookback period
with st.sidebar:
    st.header("Look‑back")
    spans = {"3 M":90,"6 M":180,"9 M":270,"YTD":None,"1 Y":365,
             "3 Y":365*3,"5 Y":365*5,"10 Y":365*10}
    span_key = st.selectbox("", list(spans.keys()), index=list(spans).index("5 Y"))

today = datetime.today()
hist_start = today - timedelta(days=365*10+220)
disp_start = datetime(today.year,1,1) if span_key=="YTD" else today - timedelta(days=spans[span_key])

# Cached Yahoo pull
@st.cache_data(ttl=3600, show_spinner="Fetching ETF prices…")
def fetch_etfs(tickers,start,end):
    df = yf.download(tickers, start, end, group_by="ticker", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df.xs('Close',level=1,axis=1)
    else:
        closes = df[["Close"]]
    closes = closes.fillna(method='ffill').dropna()
    return closes

cyc = (1 + fetch_etfs(CYCLICALS, hist_start, today).pct_change()).cumprod().mean(axis=1)
defn = (1 + fetch_etfs(DEFENSIVES, hist_start, today).pct_change()).cumprod().mean(axis=1)
ratio = (cyc/defn*100).dropna()

# SMH/IGV relative performance
smh_igv = fetch_etfs(["SMH","IGV"], hist_start, today)
smh = (1 + smh_igv["SMH"].pct_change()).cumprod()
igv = (1 + smh_igv["IGV"].pct_change()).cumprod()
smh_igv_ratio = (smh / igv).dropna()

def rsi(series,n=14):
    delta=series.diff(); up=delta.clip(lower=0); dn=-delta.clip(upper=0)
    rs=(up.rolling(n).mean()/dn.rolling(n).mean())
    return 100-100/(1+rs)
rsi_main = rsi(ratio)
rsi_smh_igv = rsi(smh_igv_ratio)

ma50, ma200 = ratio.rolling(50).mean(), ratio.rolling(200).mean()
smh_igv_ma50, smh_igv_ma200 = smh_igv_ratio.rolling(50).mean(), smh_igv_ratio.rolling(200).mean()

# Mask for display
mask = ratio.index>=disp_start
ratio_v, rsi_v = ratio[mask], rsi_main[mask]
ma50_v, ma200_v = ma50, ma200

mask_s = smh_igv_ratio.index>=disp_start
smh_igv_v, smh_igv_rsi_v = smh_igv_ratio[mask_s], rsi_smh_igv[mask_s]
smh_igv_ma50_v, smh_igv_ma200_v = smh_igv_ma50, smh_igv_ma200

# --- Metrics panel
col1,col2,col3 = st.columns([1,1,1.4])
col1.metric("Cyc/Def Return",f"{(ratio_v.iat[-1]/ratio_v.iat[0]-1)*100:,.1f}%")
col2.metric("RSI‑14",f"{rsi_v.iat[-1]:.0f}")
trend = "Bullish (50>200)" if ma50.iat[-1]>ma200.iat[-1] else "Bearish (50<200)"
col3.metric("Trend Regime",trend)

# --- Main cyc/def chart and RSI (subplots)
fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                    row_heights=[0.75,0.25],vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=ratio_v.index,y=ratio_v,
                         line=dict(color="#355E3B",width=2),name="Cyc/Def"),row=1,col=1)
fig.add_trace(go.Scatter(x=ma50.index,y=ma50,
                         line=dict(color="blue",width=2),name="50‑DMA"),row=1,col=1)
fig.add_trace(go.Scatter(x=ma200.index,y=ma200,
                         line=dict(color="red",width=2),name="200‑DMA"),row=1,col=1)
fig.add_trace(go.Scatter(x=rsi_v.index,y=rsi_v,
                         line=dict(color="black",width=2),name="RSI‑14",showlegend=False),row=2,col=1)
for y,clr,txt in [(70,"red","Overbought"),(30,"green","Oversold")]:
    fig.add_hline(y=y,row=2,col=1,line_dash="dot", line_color=clr)
    fig.add_annotation(x=rsi_v.index[0],y=y,text=txt,
                       yshift=4 if y==70 else -8,showarrow=False,
                       font=dict(color=clr),row=2,col=1)
fig.update_layout(height=700,margin=dict(l=20,r=20,t=30,b=20),
                  plot_bgcolor="white",
                  legend=dict(orientation="h",y=1.02,x=1,xanchor="right",
                              font=dict(size=13)))
st.plotly_chart(fig,use_container_width=True)

# --- Regime flip table
def get_regime_flips(ratio, ma50, ma200, lookbacks=[21, 63, 126]):
    flips = ((ma50 > ma200) != (ma50.shift(1) > ma200.shift(1))).astype(int)
    flip_dates = ratio.index[flips==1]
    rows = []
    for d in flip_dates[-12:]:  # last 12 flips
        idx = ratio.index.get_loc(d)
        if np.isnan(ma50[d]) or np.isnan(ma200[d]):
            continue
        regime = "Bullish (50>200)" if ma50[d] > ma200[d] else "Bearish (50<200)"
        nexts = []
        for lb in lookbacks:
            if idx+lb < len(ratio):
                ret = (ratio.iloc[idx+lb] / ratio.iloc[idx] - 1) * 100
            else:
                ret = np.nan
            nexts.append(ret)
        rows.append([d.date(), regime] + [f"{x:.1f}%" if pd.notnull(x) else "" for x in nexts])
    cols = ["Date", "Regime", "1M Return", "3M Return", "6M Return"]
    return pd.DataFrame(rows, columns=cols)

regime_table = get_regime_flips(ratio, ma50, ma200)
if not regime_table.empty:
    st.subheader("Historical Regime Flip Table")
    st.dataframe(regime_table, hide_index=True, use_container_width=True)

# --- SMH/IGV Relative Ratio Table
smh_v, igv_v = smh[smh.index>=disp_start], igv[igv.index>=disp_start]
smh_ret = (smh_v.iloc[-1]/smh_v.iloc[0]-1)*100
igv_ret = (igv_v.iloc[-1]/igv_v.iloc[0]-1)*100
smh_igv_rel_ret = (smh_igv_v.iloc[-1]/smh_igv_v.iloc[0]-1)*100
smh_igv_metrics = pd.DataFrame({
    "ETF": ["SMH", "IGV", "SMH/IGV Ratio"],
    "Return over window": [f"{smh_ret:.1f}%", f"{igv_ret:.1f}%", f"{smh_igv_rel_ret:.1f}%"]
})

st.subheader("SMH vs IGV Performance Table")
st.dataframe(smh_igv_metrics, hide_index=True, use_container_width=True)

# --- SMH/IGV Ratio + RSI chart
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.75, 0.25], vertical_spacing=0.03)
fig2.add_trace(go.Scatter(x=smh_igv_v.index, y=smh_igv_v,
                          line=dict(color="#0a5ba1", width=2),
                          name="SMH/IGV Ratio"), row=1, col=1)
fig2.add_trace(go.Scatter(x=smh_igv_ma50.index, y=smh_igv_ma50,
                          line=dict(color="blue", width=2),
                          name="50‑DMA"), row=1, col=1)
fig2.add_trace(go.Scatter(x=smh_igv_ma200.index, y=smh_igv_ma200,
                          line=dict(color="red", width=2),
                          name="200‑DMA"), row=1, col=1)
fig2.add_trace(go.Scatter(x=smh_igv_rsi_v.index, y=smh_igv_rsi_v,
                          line=dict(color="black", width=2),
                          name="RSI‑14", showlegend=False), row=2, col=1)
for y, clr, txt in [(70, "red", "Overbought"), (30, "green", "Oversold")]:
    fig2.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=clr)
    fig2.add_annotation(x=smh_igv_rsi_v.index[0], y=y, text=txt,
                        yshift=4 if y == 70 else -8, showarrow=False,
                        font=dict(color=clr), row=2, col=1)
fig2.update_layout(height=550, margin=dict(l=20, r=20, t=30, b=20),
                   plot_bgcolor="white",
                   legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                               font=dict(size=13)),
                   title="SMH / IGV Relative Strength & RSI")
st.plotly_chart(fig2, use_container_width=True)

# --- Download CSVs
csv_main = pd.concat({"Ratio":ratio_v,"MA50":ma50,"MA200":ma200,"RSI":rsi_v}, axis=1).dropna().to_csv().encode()
csv_smh_igv = pd.concat({"SMH_IGV_Ratio":smh_igv_v,"MA50":smh_igv_ma50,"MA200":smh_igv_ma200,"RSI":smh_igv_rsi_v}, axis=1).dropna().to_csv().encode()
st.download_button("Download Cyc/Def Data (CSV)", csv_main, "cyc_def_ratio.csv")
st.download_button("Download SMH/IGV Ratio Data (CSV)", csv_smh_igv, "smh_igv_ratio.csv")
