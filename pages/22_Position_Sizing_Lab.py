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
from adfm_core.volatility_sizing import (
    comparison_table,
    downside_statistics,
    invalidation_distance,
    scale_exposure,
    volatility_context,
    volatility_history,
)


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


def render_exposure(ticker, side, current, base, ceiling, result, recent, baseline, window, context):
    delta = result.change * 100
    relative = result.target / current - 1 if current > 0 else None
    at_target = abs(delta) < .05
    state = "AT PERMITTED SIZE" if at_target else "ABOVE PERMITTED SIZE" if delta < 0 else "BELOW PERMITTED SIZE"
    accent = "#bcd9ff" if at_target else "#ffc3b4" if delta < 0 else "#a5e3cf"
    change = "Current exposure matches your sizing limits" if at_target else f"{abs(delta):.2f} percentage points {'less' if delta < 0 else 'more'} exposure"
    if relative is not None and not at_target:
        change += f" · {abs(relative):.1%} {'reduction' if delta < 0 else 'increase'}"
    maximum = max(ceiling, current, result.target, .01)
    x_current = 12 + 376 * current / maximum
    x_target = 12 + 376 * result.target / maximum
    ticks = "".join(f'<line x1="{12+376*i/4}" y1="58" x2="{12+376*i/4}" y2="65" stroke="#637188"/><text x="{12+376*i/4}" y="86" fill="#b7c5d9" text-anchor="middle" font-size="11">{maximum*i/4:.0%}</text>' for i in range(5))
    limit = f"Binding limit: {result.binding}."
    context_html = "".join(f'<span>{n}D vol <b>{context.loc[n, "recent"]:.1%}</b></span>' for n in (10, 20, 60))
    st.markdown(
        f'<div class="psl-hero" style="--psl-accent:{accent}">'
        f'<div class="psl-top"><div class="psl-symbol">{escape(ticker)} / {escape(side).upper()}</div><div class="psl-state">{state}</div></div>'
        '<div class="psl-main"><div><div class="psl-exposure">'
        f'<div><div class="psl-label">Current exposure</div><div class="psl-number">{current:.1%}</div></div>'
        '<div class="psl-arrow">→</div>'
        f'<div><div class="psl-label">Permitted exposure</div><div class="psl-number psl-target">{result.target:.1%}</div></div></div>'
        f'<div class="psl-change">{change}</div><div class="psl-rule">{base:.1%} base size × {baseline:.1%} normal volatility ÷ {recent:.1%} recent volatility. {limit}</div></div>'
        '<div class="psl-ruler"><svg viewBox="0 0 400 100" role="img" aria-label="Current and adjusted exposure on a NAV percentage scale">'
        '<rect x="12" y="37" width="376" height="15" rx="7.5" fill="#ffffff12"/>'
        f'<rect x="12" y="37" width="{max(0,x_target-12):.2f}" height="15" rx="7.5" fill="{accent}" opacity=".85"/>'
        f'<line x1="{x_current:.2f}" y1="20" x2="{x_current:.2f}" y2="56" stroke="#fff" stroke-width="2"/>'
        f'<circle cx="{x_target:.2f}" cy="44.5" r="8.5" fill="{accent}" stroke="#101c30" stroke-width="3"/>{ticks}</svg>'
        '<div class="psl-key"><span>│ Current</span><span>● Permitted</span><span>Exposure / NAV</span></div></div></div>'
        f'<div class="psl-bottom">{context_html}<span>{window}D historical normal <b>{baseline:.1%}</b></span>'
        f'<span>Vol scaling <b>{result.multiplier:.2f}×</b></span><span>Ceiling <b>{ceiling:.1%} NAV</b></span></div></div>',
        unsafe_allow_html=True,
    )


def history_chart(history, base, current, ceiling, window, loss_cap=None):
    displayed = history.tail(252)
    cap = min(ceiling, loss_cap) if loss_cap is not None else ceiling
    target = (base * displayed.baseline / displayed.recent).clip(upper=cap)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, row_heights=[.56,.44])
    fig.add_trace(go.Scatter(x=displayed.index, y=displayed.recent*100, name=f"{window}-session volatility", line=dict(color=PASTEL["cornflower"],width=2.5), fill="tozeroy", fillcolor="rgba(128,175,218,.12)", hovertemplate="%{y:.1f}% annualized<extra>Recent volatility</extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=displayed.index, y=displayed.baseline*100, name="Historical normal", line=dict(color="#a3adbc",width=1.8,dash="dot"), hovertemplate="%{y:.1f}% annualized<extra>Historical normal</extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=displayed.index,y=target*100,name="Permitted exposure",line=dict(color=PASTEL["seafoam"],width=2.5),fill="tozeroy",fillcolor="rgba(144,207,187,.16)",hovertemplate="%{y:.2f}% NAV<extra>Permitted exposure</extra>"),row=2,col=1)
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
    with st.expander("Dollar sizing (optional)", expanded=True):
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
    frames,_ = fetch_daily_ohlcv([ticker],period="10y")
frame = frames.get(ticker)
if frame is None or frame.empty or "Close" not in frame:
    st.error(f"Price history is unavailable for {ticker}. Try another equity or ETF ticker.")
    st.stop()
adjusted = adjusted_ohlcv(frame)
close = pd.to_numeric(adjusted["Close"],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
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
raw_close = pd.to_numeric(frame["Close"],errors="coerce").sort_index()
raw_close = raw_close.loc[~raw_close.index.duplicated(keep="last")]
mark = float(raw_close.loc[close.index[-1]])
with st.sidebar:
    with st.expander("Invalidation loss budget (optional)", expanded=True):
        st.caption(f"Latest {ticker} close: ${mark:,.2f}. Set an invalidation price to cap size from today's mark.")
        invalidation = st.number_input("Invalidation price",min_value=0.0,value=0.0,step=1.0,format="%.2f",key=f"psl_invalidation_{ticker}_{side}",help="Below the latest close for a long; above it for a short. Zero disables this cap.")
        loss_budget_pct = st.number_input("Loss budget (% NAV)",min_value=0.0,max_value=100.0,value=1.0,step=.25,key="psl_loss_budget")
        st.caption("Price gaps, slippage and costs can exceed this budget. An invalidation price is not a guaranteed exit.")
try:
    stop_distance = invalidation_distance(mark,invalidation,side)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
result = scale_exposure(base,current,recent,baseline,ceiling,loss_budget=loss_budget_pct/100,stop_distance=stop_distance)
context = volatility_context(close)
render_exposure(ticker,side,current,base,ceiling,result,recent,baseline,window,context)
selected = context.loc[window]
change_text = f"{'up' if selected['change'] >= 0 else 'down'} {abs(selected['change']):.0%} over 20 sessions" if np.isfinite(selected['change']) else "20-session change unavailable"
st.caption(f"{window}D volatility is higher than {selected['percentile']:.0f}% of its 252 prior observations; {change_text}. Vol reference: {result.uncapped:.2%} NAV before caps."
           + (f" Invalidation cap: {result.loss_cap:.2%} NAV at a {stop_distance:.1%} adverse move." if stop_distance is not None else ""))
st.markdown('<div class="psl-legend">VOLATILITY & POSITION SIZE · Past year · dashed exposure line = your current size</div>',unsafe_allow_html=True)
st.plotly_chart(history_chart(history,base,current,ceiling,window,result.loss_cap),width="stretch",theme=None,config={"displayModeBar":False,"scrollZoom":False})
st.caption("Historical exposure uses your current base size, ceiling and optional invalidation exposure cap throughout, with volatility known on each date. This illustrates sizing, rather than strategy returns.")
daily = recent/np.sqrt(252)
stats = downside_statistics(adjusted,side)
table = comparison_table(current,result,stats,side,daily_sigma=daily,stop_distance=stop_distance,nav=nav)
st.markdown('<div class="psl-legend">SIZE COMPARISON · Scenario rows show this position’s impact on NAV</div>',unsafe_allow_html=True)
numeric_columns = list(table.columns[1:])
styled = table.style.format("{:.2f}%",subset=numeric_columns,na_rep="—")
styled = styled.set_properties(subset=["Permitted"],**{"background-color":"#e7f5ee","font-weight":"600"})
if nav > 0:
    styled = styled.format("${:,.0f}",subset=(table.index[table.Scenario.eq("Position notional (USD)")],numeric_columns[1:]))
if stop_distance is not None:
    scenario_index = table.index[~table.Scenario.isin(["Exposure / NAV","Position notional (USD)"])]
    styled = styled.map(lambda value:"background-color:#fde8e4;color:#904433" if pd.notna(value) and value < -loss_budget_pct-1e-9 else "",subset=(scenario_index,numeric_columns[1:]))
st.dataframe(styled,width="stretch",hide_index=True,height=36+35*len(table))
st.caption(f"Historical adverse moves: {adjusted.index.min():%b %Y}–{as_of:%b %Y}, for the selected {side.lower()} direction. Week = five sessions; tail averages use the worst 5% of observations. — means insufficient data. Rose cells exceed your optional loss budget.")
st.markdown('<div class="psl-footnote">Market move shows the instrument’s return; other scenario columns show position-only NAV P&amp;L at fixed initial notional. These are descriptive shocks, not forecasts or loss limits. The 2σ row uses recent realized volatility. Portfolio offsets, financing, borrow and trading costs are excluded. Options and futures need instrument-specific risk treatment.</div>',unsafe_allow_html=True)
with st.expander("Portfolio stress testing"):
    if st.checkbox("Load portfolio stress tool",key="psl_stress"):
        from adfm_core.portfolio_stress_page import render_portfolio_stress
        render_portfolio_stress()
render_footer(data_note="Permitted exposure = minimum of the volatility reference, exposure ceiling and optional invalidation loss-budget cap. Historical normal = median of 252 rolling-volatility observations preceding the current window. Source: Yahoo Finance adjusted daily OHLCV; invalidation uses the latest unadjusted close.")
