from datetime import date,timedelta
import pandas as pd
from adfm_engine.analytics.hedge import *
from adfm_engine.charts.hedge import hedge_chart,chart_data
from adfm_engine.data.hedge import yf_download
from adfm_engine.serialization import records,figure_json
from adfm_engine.services import DataUnavailable

def hedge(close,chart_years=1):
    if close.empty or any(t not in close or close[t].dropna().empty for t in [SPX_TICKER,NDX_TICKER]):raise DataUnavailable('Yahoo feed failed for ^SPX/^NDX. Retry later.')
    v=compute_hedge(close)
    if v['df'].empty:raise DataUnavailable('The index histories have no overlapping sessions.')
    score=v['score_ndx'] if v['target']==NDX_TICKER else v['score_spx'];meta=v['meta_target'];price=v['df'][v['target']]
    table=pd.concat([summarize_eps(SPX_LABEL,v['df'][SPX_TICKER],v['score_spx'],v['meta_spx'],v['t_short']),summarize_eps(NDX_LABEL,v['df'][NDX_TICKER],v['score_ndx'],v['meta_ndx'],v['t_short'])],ignore_index=True)
    stats=pd.DataFrame([{'Index':SPX_LABEL,**v['stats_spx']},{'Index':NDX_LABEL,**v['stats_ndx']}])
    data=chart_data(price,score,meta,v['t_short'],chart_years)
    return {'asof':price.index[-1].date().isoformat(),'target':v['target_label'],'stance':v['stance_target'],'badge':v['badge_target'],'score':int(round(v['score_target'])),'threshold':v['t_short'],'bias':v['t_bias'],
       'indices':[[SPX_LABEL,fmt_num(v['spx_last'],2),fmt_pct(v['dd_spx']),v['stance_spx']],[NDX_LABEL,fmt_num(v['ndx_last'],2),fmt_pct(v['dd_ndx']),v['stance_ndx']]],
       'vix':fmt_num(v['vix_last'],2),'rsi':fmt_num(v['rsi_today'],1),'drawdown63':fmt_pct(v['dd63_today']),'early_stage':v['early_today'],'oversold_block':v['oversold_today'],
       'stats':records(stats),'episodes':records(table),'figure':figure_json(hedge_chart(price,score,meta,v['t_short'],v['target_label'],chart_years)),
       'csv':data.rename_axis('Date').reset_index().to_csv(index=False),'warnings':(['Some risk-layer inputs are missing: '+', '.join(t for t in TICKERS if t not in close or close[t].dropna().empty)] if any(t not in close or close[t].dropna().empty for t in TICKERS) else [])}
def load_hedge(chart_years=1):
    start=date.today()-timedelta(days=int(10*365.25)+180)
    return hedge(extract_close(yf_download(tuple(TICKERS),start),TICKERS),chart_years)
