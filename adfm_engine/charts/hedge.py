"""Interactive equivalent of the original static trading-session chart."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
from adfm_engine.analytics.hedge import sessions_for_years,rolling_ma,signal_onset,chart_style_for_years,tick_rule_for_years,tick_label_for_years
from adfm_engine.palette import PASTEL

def chart_data(price,score,meta,t_short,years):
    dfp=pd.DataFrame({'price':price,'score':score}).dropna(subset=['price'])
    if len(dfp)>sessions_for_years(years):dfp=dfp.iloc[-sessions_for_years(years):].copy()
    idx=dfp.index
    dfp['MA50']=meta.get('ma50',rolling_ma(price,50)).reindex(idx)
    dfp['MA200']=meta.get('ma200',rolling_ma(price,200)).reindex(idx)
    dfp['onset']=signal_onset(score.reindex(idx),{k:v.reindex(idx) for k,v in meta.items()},t_short)
    return dfp

def hedge_chart(price,score,meta,t_short,title_prefix,years=1):
    dfp=chart_data(price,score,meta,t_short,years);fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[3.,1.25],vertical_spacing=.06)
    if dfp.empty:return fig
    idx=dfp.index;x=np.arange(len(dfp));dates=idx.strftime('%Y-%m-%d').tolist();style=chart_style_for_years(years)
    for col,label,color,width in [('price','Price','#111111',style['price_lw']),('MA50','MA50',PASTEL['blue'],style['ma_lw']),('MA200','MA200',PASTEL['lavender'],style['ma_lw'])]:
        fig.add_trace(go.Scatter(x=x,y=dfp[col],name=label,line={'color':color,'width':width},customdata=dates,hovertemplate='%{customdata}<br>'+label+': %{y:,.2f}<extra></extra>'),row=1,col=1)
    onset=dfp.onset
    if onset.any():fig.add_trace(go.Scatter(x=x[onset],y=dfp.price[onset],customdata=np.array(dates)[onset],mode='markers',name='Short signal (new)',marker={'symbol':'triangle-down','size':max(6,np.sqrt(style['marker_s'])),'color':PASTEL['rose'],'line':{'color':'white','width':style['marker_lw']}},hovertemplate='%{customdata}<br>New short signal: %{y:,.2f}<extra></extra>'),row=1,col=1)
    y=dfp.score.fillna(0.).to_numpy();scale=[[0,PASTEL['sage']],[.5,PASTEL['amber']],[1,PASTEL['rose']]]
    # Group segments by original discrete score; preserve all vertices and avoid thousands of traces.
    for value in np.unique(y[:-1]):
        positions=np.flatnonzero(y[:-1]==value);sx=[];sy=[]
        for i in positions:sx.extend([int(x[i]),int(x[i+1]),None]);sy.extend([float(y[i]),float(y[i+1]),None])
        fig.add_trace(go.Scatter(x=sx,y=sy,mode='lines',line={'color':sample_colorscale(scale,[max(0,min(100,float(value)))/100])[0],'width':style['score_lw']},showlegend=False,hoverinfo='skip'),row=2,col=1)
    fig.add_trace(go.Scatter(x=x,y=y,customdata=dates,name='Score',showlegend=False,mode='lines',line={'color':'rgba(0,0,0,.10)','width':max(.55,style['score_lw']*.3)},hovertemplate='%{customdata}<br>Score: %{y:.2f}<extra></extra>'),row=2,col=1)
    fig.add_hline(y=t_short,line_color='#111111',opacity=.7,row=2,col=1)
    fig.add_hline(y=max(40,t_short-12),line_color=PASTEL['amber'],opacity=.55,row=2,col=1)
    ticks=pd.date_range(idx.min().normalize(),idx.max().normalize(),freq=tick_rule_for_years(years));tick_pos=[];tick_lbl=[];spacing={1:18,2:26,3:32,5:45,10:70}.get(years,32)
    for d in ticks:
        loc=idx.get_indexer([d],method='nearest')[0]
        if 0<=loc<len(idx) and (not tick_pos or loc-tick_pos[-1]>=spacing):tick_pos.append(int(loc));tick_lbl.append(tick_label_for_years(d,years))
    last=len(idx)-1
    if not tick_pos or last-tick_pos[-1]>=max(10,spacing//2):tick_pos.append(last);tick_lbl.append(idx[-1].strftime('%b %Y') if years<=3 else idx[-1].strftime('%Y'))
    pmin=float(np.nanmin(dfp.price));pmax=float(np.nanmax(dfp.price));pad=(pmax-pmin)*.04 if pmax>pmin else 1.
    fig.update_yaxes(range=[pmin-pad,pmax+pad],tickformat=',.2f',row=1,col=1)
    fig.update_yaxes(range=[0,100],title_text='Score (0–100)',row=2,col=1)
    fig.update_xaxes(range=[-.5,len(idx)-.5],tickmode='array',tickvals=tick_pos,ticktext=tick_lbl)
    fig.update_layout(title=f'{title_prefix} ({years} Year'+('s' if years!=1 else '')+')',height=740,hovermode='x',legend={'orientation':'h','y':1.04},paper_bgcolor='white',plot_bgcolor='white',margin={'l':45,'r':25,'t':90,'b':35})
    return fig
