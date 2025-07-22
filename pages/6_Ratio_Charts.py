import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# —— Page config ——
st.set_page_config(layout="wide", page_title="Ratio Charts")

# —— Sidebar: About & Inputs ——
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "This dashboard visualizes regime shifts in US equities:\n"
    "- **Cyclical vs Defensive (Eq‑Wt):** Ratio of cumulative returns for cyclical (XLK, XLI, XLF, XLC, XLY) vs defensive (XLP, XLE, XLV, XLRE, XLB, XLU) ETFs, scaled to 100.\n"
    "- **Preset Ratios:** SMH/IGV, QQQ/IWM, HYG/LQD, HYG/IEF.\n"
    "- **Custom Ratio:** Compare any two tickers over your selected look‑back period."
)

st.sidebar.header("Look‑back")
span_map = {"3M":90, "6M":180, "9M":270, "YTD":None, "1Y":365, "3Y":365*3, "5Y":365*5}
span = st.sidebar.selectbox("Period", list(span_map.keys()), index=list(span_map.keys()).index("5Y"))

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
t1 = st.sidebar.text_input("Ticker 1", "NVDA").upper().strip()
t2 = st.sidebar.text_input("Ticker 2", "SMH").upper().strip()

# —— Date range ——
end = datetime.today()
if span == "YTD":
    start = datetime(end.year, 1, 1)
else:
    start = end - timedelta(days=span_map[span])

# —— Data fetch function ——
@st.cache_data(ttl=3600)
def load_close(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.fillna(method='ffill').dropna()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers)
    return data

# —— Compute metrics ——
def compute_cumrets(df):
    return (1 + df.pct_change()).cumprod()

def compute_ratio(s1, s2, scale=1):
    return (s1 / s2) * scale

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100/(1+rs))

# —— Plotting utility ——
def plot_ratio(fig, ratio, ma50, ma200, rsi, title, row_offset=0):
    r = row_offset + 1
    # Ratio + MAs
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name=title), row=r, col=1)
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="50MA"), row=r, col=1)
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="200MA"), row=r, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI"), row=r+1, col=1)
    fig.add_hline(y=70, line_dash="dash", row=r+1, col=1)
    fig.add_hline(y=30, line_dash="dash", row=r+1, col=1)

# —— Static: Cyclical vs Defensive ——
st.header("Cyclical vs Defensive (Eq‑Weighted)")
cyc = ["XLK","XLI","XLF","XLC","XLY"]
defn = ["XLP","XLE","XLV","XLRE","XLB","XLU"]
static_tickers = cyc + defn

closes = load_close(static_tickers, start, end)
cumrets = compute_cumrets(closes)
ratio_cd = compute_ratio(cumrets[cyc].mean(axis=1), cumrets[defn].mean(axis=1), scale=100)
ma50_cd = ratio_cd.rolling(50).mean()
ma200_cd = ratio_cd.rolling(200).mean()
rsi_cd = compute_rsi(ratio_cd)

fig_cd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.7,0.3], vertical_spacing=0.02,
                       subplot_titles=["Cyclical/Defensive", "RSI"])
plot_ratio(fig_cd, ratio_cd, ma50_cd, ma200_cd, rsi_cd, "C/D Ratio", row_offset=0)
fig_cd.update_layout(height=500, showlegend=True)
fig_cd.update_xaxes(range=[start, end])
fig_cd.update_yaxes(title_text="Ratio", row=1, col=1)
fig_cd.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
st.plotly_chart(fig_cd, use_container_width=True)

# —— Preset Ratios ——
st.markdown("---")
st.header("Preset Ratios")
presets = [("SMH","IGV"),("QQQ","IWM"),("HYG","LQD"),("HYG","IEF")]
for a, b in presets:
    st.subheader(f"{a}/{b}")
    df_pair = load_close([a,b], start, end)
    cum = compute_cumrets(df_pair)
    ratio_p = compute_ratio(cum[a], cum[b])
    ma50_p = ratio_p.rolling(50).mean()
    ma200_p = ratio_p.rolling(200).mean()
    rsi_p = compute_rsi(ratio_p)
    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.7,0.3], vertical_spacing=0.02,
                          subplot_titles=[f"{a}/{b}", "RSI"])
    plot_ratio(fig_p, ratio_p, ma50_p, ma200_p, rsi_p, f"{a}/{b}", row_offset=0)
    fig_p.update_layout(height=450, showlegend=False)
    fig_p.update_xaxes(range=[start, end])
    fig_p.update_yaxes(title_text="Ratio", row=1, col=1)
    fig_p.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
    st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("---")

# —— Custom Ratio ——
if t1 and t2:
    st.header(f"Custom: {t1}/{t2}")
    try:
        df_c = load_close([t1,t2], start, end)
        cum_c = compute_cumrets(df_c)
        ratio_c = compute_ratio(cum_c[t1], cum_c[t2])
        ma50_c = ratio_c.rolling(50).mean()
        ma200_c = ratio_c.rolling(200).mean()
        rsi_c = compute_rsi(ratio_c)
        fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.7,0.3], vertical_spacing=0.02,
                              subplot_titles=[f"{t1}/{t2}", "RSI"])
        plot_ratio(fig_c, ratio_c, ma50_c, ma200_c, rsi_c, f"{t1}/{t2}", row_offset=0)
        fig_c.update_layout(height=450, showlegend=False)
        fig_c.update_xaxes(range=[start, end])
        fig_c.update_yaxes(title_text="Ratio", row=1, col=1)
        fig_c.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
        st.plotly_chart(fig_c, use_container_width=True)
    except Exception:
        st.warning(f"Data not available for {t1}/{t2}.")

# —— Footer ——
st.sidebar.markdown("---")
st.sidebar.markdown("Data source: Yahoo Finance")
