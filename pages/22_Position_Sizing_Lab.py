"""A visual, volatility-based position scaling tool."""
from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.market_data import adjusted_ohlcv, configure_yfinance_cache, fetch_daily_ohlcv
from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header, render_sidebar_about
from adfm_core.volatility_sizing import scale_exposure, volatility_history


CSS = """
<style>
.psl-hero{background:radial-gradient(ellipse at 100% 0%,#263d5b 0%,#101c30 55%);color:#fff;padding:30px 34px;margin:14px 0 12px;overflow:hidden;border-radius:14px;position:relative}
.psl-top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:24px}
.psl-symbol{font:700 13px Arial;letter-spacing:.13em;color:#e9f0fa}
.psl-state{font:700 11px Arial;letter-spacing:.09em;color:var(--psl-accent);background:#ffffff0d;border:1px solid #ffffff24;padding:8px 12px;border-radius:30px}
.psl-main{display:grid;grid-template-columns:1.15fr 1fr;gap:45px;align-items:center}
.psl-exposure{display:flex;align-items:center;gap:22px}
.psl-label{font:600 11px Arial;letter-spacing:.09em;text-transform:uppercase;color:#b7c5d9;margin-bottom:8px}
.psl-number{font:500 clamp(35px,4.6vw,66px) Arial;letter-spacing:-.055em;color:#fff;font-variant-numeric:tabular-nums;line-height:1.12;white-space:nowrap}
.psl-target{color:var(--psl-accent)}
.psl-arrow{font:300 30px Arial;color:#8b9bb2}
.psl-change{font:600 14px Arial;color:var(--psl-accent);margin-top:19px}
.psl-rule{font:400 13px Arial;color:#c0cddd;line-height:1.6;margin-top:9px;max-width:490px}
.psl-ruler{position:relative;padding-top:20px}
.psl-ruler svg{width:100%;height:auto;display:block}
.psl-key{display:flex;justify-content:space-between;gap:10px;font:400 11px Arial;color:#c4cfdd;margin-top:4px}
.psl-bottom{border-top:1px solid #ffffff21;margin-top:25px;padding-top:17px;display:flex;flex-wrap:wrap;gap:10px 26px;font:400 12px Arial;color:#b7c5d9}
.psl-bottom b{color:#edf3fa;font-weight:600}
.psl-legend{font:500 12px Arial;color:#526176;margin:19px 0 3px}
.psl-footnote{font:400 12px Arial;color:#5e6b7b;line-height:1.6;margin:5px 0 20px}
@media(max-width:800px){.psl-hero{padding:23px 20px}.psl-main{grid-template-columns:1fr;gap:10px}.psl-exposure{gap:18px}.psl-number{font-size:47px}.psl-top{margin-bottom:22px}.psl-bottom{gap:10px 17px}.psl-state{font-size:9px}.psl-ruler{padding-top:8px}}
</style>
"""


def render_exposure(ticker, side, current, base, ceiling, result, recent, baseline, window):
    delta = result.change * 100
    relative = result.target / current - 1 if current > 0 else None
    at_target = abs(delta) < .05
    state = "AT MODEL SIZE" if at_target else "REDUCE EXPOSURE" if delta < 0 else "INCREASE EXPOSURE"
    accent = "#bcd9ff" if at_target else "#ffc3b4" if delta < 0 else "#a5e3cf"
    change = "Current exposure matches the volatility model" if at_target else f"{abs(delta):.2f} percentage points {'less' if delta < 0 else 'more'} exposure"
    if relative is not None and not at_target:
        change += f" · {abs(relative):.1%} {'reduction' if delta < 0 else 'increase'}"
    maximum = max(ceiling, current, result.target, .01)
    x_current = 12 + 376 * current / maximum
    x_target = 12 + 376 * result.target / maximum
    ticks = "".join(f'<line x1="{12+376*i/4}" y1="58" x2="{12+376*i/4}" y2="65" stroke="#637188"/><text x="{12+376*i/4}" y="86" fill="#b7c5d9" text-anchor="middle" font-size="11">{maximum*i/4:.0%}</text>' for i in range(5))
    limit = "Exposure ceiling is binding." if result.uncapped > ceiling else "Risk held constant relative to your base size."
    st.markdown(
        f'<div class="psl-hero" style="--psl-accent:{accent}">'
        f'<div class="psl-top"><div class="psl-symbol">{escape(ticker)} / {escape(side).upper()}</div><div class="psl-state">{state}</div></div>'
        '<div class="psl-main"><div><div class="psl-exposure">'
        f'<div><div class="psl-label">Current exposure</div><div class="psl-number">{current:.1%}</div></div>'
        '<div class="psl-arrow">→</div>'
        f'<div><div class="psl-label">Vol-adjusted exposure</div><div class="psl-number psl-target">{result.target:.1%}</div></div></div>'
        f'<div class="psl-change">{change}</div><div class="psl-rule">{base:.1%} base size × {baseline:.1%} normal volatility ÷ {recent:.1%} recent volatility. {limit}</div></div>'
        '<div class="psl-ruler"><svg viewBox="0 0 400 100" role="img" aria-label="Current and adjusted exposure on a NAV percentage scale">'
        '<rect x="12" y="37" width="376" height="15" rx="7.5" fill="#ffffff12"/>'
        f'<rect x="12" y="37" width="{max(0,x_target-12):.2f}" height="15" rx="7.5" fill="{accent}" opacity=".85"/>'
        f'<line x1="{x_current:.2f}" y1="20" x2="{x_current:.2f}" y2="56" stroke="#fff" stroke-width="2"/>'
        f'<circle cx="{x_target:.2f}" cy="44.5" r="8.5" fill="{accent}" stroke="#101c30" stroke-width="3"/>{ticks}</svg>'
        '<div class="psl-key"><span>│ Current</span><span>● Vol-adjusted</span><span>Exposure / NAV</span></div></div></div>'
        f'<div class="psl-bottom"><span>Recent {window} sessions <b>{recent:.1%} vol</b></span><span>Historical normal <b>{baseline:.1%} vol</b></span>'
        f'<span>Uncapped scaling <b>{result.multiplier:.2f}× base</b></span><span>Ceiling <b>{ceiling:.1%} NAV</b></span></div></div>',
        unsafe_allow_html=True,
    )


def history_chart(history, base, current, ceiling, window):
    displayed = history.tail(252)
    target = (base * displayed.baseline / displayed.recent).clip(upper=ceiling)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, row_heights=[.56,.44])
    fig.add_trace(go.Scatter(x=displayed.index, y=displayed.recent*100, name=f"{window}-session volatility", line=dict(color=PASTEL["cornflower"],width=2.5), fill="tozeroy", fillcolor="rgba(128,175,218,.12)", hovertemplate="%{y:.1f}% annualized<extra>Recent volatility</extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=displayed.index, y=displayed.baseline*100, name="Historical normal", line=dict(color="#a3adbc",width=1.8,dash="dot"), hovertemplate="%{y:.1f}% annualized<extra>Historical normal</extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=displayed.index,y=target*100,name="Vol-adjusted exposure",line=dict(color=PASTEL["seafoam"],width=2.5),fill="tozeroy",fillcolor="rgba(144,207,187,.16)",hovertemplate="%{y:.2f}% NAV<extra>Model exposure</extra>"),row=2,col=1)
    fig.add_hline(y=current*100,row=2,col=1,line=dict(color="#748398",width=1.2,dash="dash"))
    fig.update_layout(height=400,template="plotly_white",margin=dict(l=12,r=12,t=32,b=14),font=dict(family="Arial",color="#45556b",size=12),paper_bgcolor="#fff",plot_bgcolor="#fff",hovermode="x unified",legend=dict(orientation="h",y=1.1,x=0,font=dict(size=11)))
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor="#edf0f4",zeroline=False,ticksuffix="%",rangemode="tozero")
    fig.update_yaxes(title_text="Annualized vol",row=1,col=1)
    fig.update_yaxes(title_text="Exposure / NAV",row=2,col=1)
    return fig


st.set_page_config(page_title="Position Sizing Lab",layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1350)
render_page_header(PageHeader(title="Position Sizing Lab",description="Scale a position with the volatility regime. Set your base exposure, compare the adjustment, and see the change in risk.",eyebrow="ADFM Risk + Execution"))
st.markdown(CSS,unsafe_allow_html=True)
with st.sidebar:
    render_sidebar_about("22_Position_Sizing_Lab.py")
    window = st.selectbox("Volatility window",[10,20,60],index=1,format_func=lambda x:f"{x} trading sessions",key="psl_window")
    ceiling_pct = st.number_input("Exposure ceiling (% NAV)",min_value=0.0,max_value=500.0,value=30.0,step=5.0,key="psl_ceiling")
    with st.expander("Dollar sizing (optional)"):
        nav = st.number_input("Current NAV (USD)",min_value=0.0,value=0.0,step=100000.0,format="%.0f",key="psl_nav")
        st.caption("Enter NAV to translate exposures into dollar notionals. No portfolio assumptions are stored here.")

c1,c2,c3,c4 = st.columns([1.05,.85,1.3,1.3])
ticker = c1.text_input("Ticker","TLT",key="psl_ticker",help="Equity or ETF ticker. Exposure refers to the entered instrument.").strip().upper()
side = c2.selectbox("Direction",["Long","Short"],key="psl_side")
current_pct = c3.number_input("Current exposure (% NAV)",min_value=0.0,max_value=500.0,value=10.0,step=1.0,key="psl_current")
base_pct = c4.number_input("Base exposure (% NAV)",min_value=0.0,max_value=500.0,value=10.0,step=1.0,key="psl_base",help="Desired size when volatility is normal. All exposures are absolute magnitudes.")
if not ticker:
    st.info("Enter a ticker to see its volatility-adjusted size.")
    st.stop()
with st.spinner(f"Reading {ticker} volatility..."):
    frames,_ = fetch_daily_ohlcv([ticker],period="3y")
frame = frames.get(ticker)
if frame is None or frame.empty or "Close" not in frame:
    st.error(f"Price history is unavailable for {ticker}. Try another equity or ETF ticker.")
    st.stop()
close = pd.to_numeric(adjusted_ohlcv(frame)["Close"],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
close = close.loc[close.gt(0)].sort_index()
close = close.loc[~close.index.duplicated(keep="last")]
history = volatility_history(close,window)
if history.empty or history.index[-1] != close.index[-1]:
    st.warning(f"{ticker} needs at least {252+2*window+1} valid daily closes and nonzero volatility for this window. No sizing estimate is available.")
    st.stop()
as_of = pd.Timestamp(close.index[-1])
age = (pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())-as_of.tz_localize(None).normalize()).days
st.caption(f"{escape(ticker)} · Data through {as_of:%b %d, %Y} · Yahoo Finance adjusted daily closes · Exposure as % of NAV")
if age > 5:
    st.warning(f"Price history is {age} calendar days old. Treat the size below as dated, rather than a current adjustment.")
recent,baseline = history.iloc[-1][["recent","baseline"]]
base,current,ceiling = base_pct/100,current_pct/100,ceiling_pct/100
result = scale_exposure(base,current,recent,baseline,ceiling)
render_exposure(ticker,side,current,base,ceiling,result,recent,baseline,window)
st.markdown('<div class="psl-legend">VOLATILITY & POSITION SIZE · Past year · dashed exposure line = your current size</div>',unsafe_allow_html=True)
st.plotly_chart(history_chart(history,base,current,ceiling,window),width="stretch",theme=None,config={"displayModeBar":False,"scrollZoom":False})
st.caption("Historical exposure uses the same base size and ceiling you entered, with volatility known on each date. It is a sizing illustration, not a backtest.")
daily = recent/np.sqrt(252)
rows = [
    {"Measure":"Exposure / NAV","Current":f"{current:.2%}","Vol-adjusted":f"{result.target:.2%}"},
    {"Measure":"1σ daily NAV move","Current":f"±{current*daily:.2%}","Vol-adjusted":f"±{result.target*daily:.2%}"},
    {"Measure":"2σ adverse day / NAV","Current":f"−{current*daily*2:.2%}","Vol-adjusted":f"−{result.target*daily*2:.2%}"},
]
if nav > 0:
    rows.append({"Measure":"Absolute position notional","Current":f"${current*nav:,.0f}","Vol-adjusted":f"${result.target*nav:,.0f}"})
    rows.append({"Measure":"Notional adjustment","Current":"—","Vol-adjusted":f"{'Reduce' if result.change < 0 else 'Add'} ${abs(result.change)*nav:,.0f}"})
st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
st.markdown('<div class="psl-footnote">Daily moves isolate this position, using recent realized volatility. A 2σ move is a scenario, not a loss limit; gaps and tail events can exceed it. Long and short exposures use the same volatility scaling. Options and futures need instrument-specific risk treatment.</div>',unsafe_allow_html=True)
with st.expander("Portfolio stress testing"):
    if st.checkbox("Load portfolio stress tool",key="psl_stress"):
        from adfm_core.portfolio_stress_page import render_portfolio_stress
        render_portfolio_stress()
render_footer(data_note="Sizing = base exposure × normal volatility ÷ recent volatility, capped at your ceiling. Inputs are illustrative until changed. Source: Yahoo Finance adjusted daily closes.")
