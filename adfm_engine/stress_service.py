from datetime import date,timedelta
import pandas as pd
from adfm_engine.analytics.stress import *
from adfm_engine.charts.stress import stress_chart
from adfm_engine.data.stress import load_prices
from adfm_engine.serialization import records,figure_json
from adfm_engine.services import DataUnavailable
SPEEDS={'Fast - 3D':3,'Base - 5D':5,'Slow - 10D':10,'21D':21,'63D':63}
def stress(prices,today=None,lookback_years=5,target_mode='Auto',z_window_years=3,smoothing_mode='Slow - 10D'):
    if prices.empty or SPX not in prices or prices[SPX].dropna().empty:raise DataUnavailable('Yahoo Finance did not return enough market data to build the Global Fracture Monitor.')
    today=today or date.today();v=compute_stress(prices,z_window_years,SPEEDS[smoothing_mode],target_mode)
    health=[]
    for ticker in ALL_TICKERS:
        s=v['px'][ticker].dropna() if ticker in v['px'] else pd.Series(dtype=float)
        health.append({'Ticker':ticker,'Obs':len(s),'Last':s.index.max().date().isoformat() if len(s) else ''})
    moves=market_moves(**{key:v[key] for key in ['px','eq_cols','carry_cols','haven_cols','bond_cols','z_window']})
    figure=stress_chart(**{key:v[key] for key in ['target_px','target_label','risk_score','dislocation_score','onset_dates']},lookback_years=lookback_years,today=today)
    return {'warnings':v['warnings'],'asof':v['px'].index[-1].date().isoformat(),'figure':figure_json(figure),'moves':records(moves),'health':health,'regime':v['regime'],'action':v['action'],'risk':fmt_score(v['risk_now']),'dislocation':fmt_score(v['dislocation_now']),'target':v['target_label'],'drawdown':fmt_pct(v['us_dd63_now']),'signal_age':v['signal_age'],'csv':moves.to_csv(index=False)}
def load_stress(lookback_years=5,target_mode='Auto',z_window_years=3,smoothing_mode='Slow - 10D'):
    start=date.today()-timedelta(days=int(max(12,lookback_years+z_window_years+2)*365.25))
    return stress(load_prices(tuple(ALL_TICKERS),start),lookback_years=lookback_years,target_mode=target_mode,z_window_years=z_window_years,smoothing_mode=smoothing_mode)
