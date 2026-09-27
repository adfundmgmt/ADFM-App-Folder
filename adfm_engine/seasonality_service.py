"""Native monthly seasonality, with a shared sample for all outputs."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from adfm_engine.data.primary import read_fred
from adfm_engine import seasonality_math as m
from adfm_engine.cache import ttl_cache
from adfm_engine.services import DataUnavailable
from adfm_engine.serialization import records

def _download(symbol, start, end):
    try:
        frame = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False, timeout=18)
        value = frame["Close"]
        if isinstance(value, pd.DataFrame): value = value.iloc[:,0]
        value = pd.to_numeric(value,errors="coerce").dropna()
        value.index = pd.to_datetime(value.index).tz_localize(None)
        return value.sort_index()
    except Exception:
        return pd.Series(dtype=float,index=pd.DatetimeIndex([]))

def _fred(code,start,end):
    try:
        value=read_fred(code,start,end)[code].dropna()
        value.index=pd.to_datetime(value.index).tz_localize(None)
        return value
    except Exception:
        return pd.Series(dtype=float,index=pd.DatetimeIndex([]))

@ttl_cache(seconds=3600,max_entries=48)
def _history(symbol,start,end):
    price=_download(symbol,start,end)
    used=symbol
    if price.empty and symbol=="SPY":
        price=_download("^GSPC",start,end);used="^GSPC"
    if price.empty and symbol in {"^SPX","^GSPC","^DJI","^IXIC"}:
        code={"^SPX":"SP500","^GSPC":"SP500","^DJI":"DJIA","^IXIC":"NASDAQCOM"}[symbol]
        price=_fred(code,start,end);used=code
    if price.empty: raise DataUnavailable(f"No price history found for {symbol}. Try another symbol or sample.")
    extended=str((price.index.min()-pd.DateOffset(years=2)).date())
    months=pd.period_range(price.index.min().to_period("M"),price.index.max().to_period("M"),freq="M")
    regime=pd.DataFrame(index=months)
    recession=_fred("USREC",extended,end).resample("ME").last()
    recession.index=recession.index.to_period("M")
    funds=_fred("FEDFUNDS",extended,end).resample("ME").last()
    funds.index=funds.index.to_period("M")
    regime["is_recession"]=recession.reindex(months).fillna(0).astype(int)
    regime["fedfunds"]=funds.reindex(months)
    delta=regime["fedfunds"].diff(3)
    regime["fed_regime"]=np.where(delta>.05,"Hiking",np.where(delta<-.05,"Cutting","Steady"))
    regime.loc[regime.fedfunds.isna(),"fed_regime"]="Unknown"
    regime["regime_cycle"]=np.where(regime.is_recession==1,"Recession","Expansion")
    market=pd.DataFrame()
    market_start=str((price.index.min()-pd.DateOffset(years=1)).date())
    for key,ticker in [("vix","^VIX"),("tnx","^TNX"),("dxy","DX-Y.NYB")]:
        series=_download(ticker,market_start,end)
        if key=="dxy" and series.empty:
            for fallback in ("DX=F","UUP"):
                series=_download(fallback,market_start,end)
                if not series.empty: break
        if not series.empty: market[key]=series/10 if key=="tnx" else series
    return price,used,m.build_filter_table(price,regime,m.build_monthly_regime_features(market))

def _number(value):
    return None if value is None or not np.isfinite(float(value)) else float(value)

def load_monthly_seasonality(*,symbol="^SPX",lookback="10Y",start_year=None,end_year=None,
                             cycle="All years",complete_only=True,fed="All Fed regimes",
                             vix="All VIX regimes",teny="All 10Y regimes",dxy="All dollar regimes",
                             month=None,year=None,session_hour=None):
    now=datetime.now(ZoneInfo("America/New_York"))
    fetch_year=1900 if lookback=="All" else max(1900,(start_year or now.year-9)-2) if lookback=="Custom" else now.year-22
    start=str((pd.Timestamp(fetch_year,1,1)-pd.DateOffset(days=45)).date())
    end=str((pd.Timestamp(now.date())+pd.Timedelta(days=1)).date())
    price,used,table=_history(symbol,start,end)
    preset={"5Y":"Last 5 years","10Y":"Last 10 years","20Y":"Last 20 years","All":"All history","Custom":"Custom"}[lookback]
    start_y,end_y=m.resolve_year_window(preset,start_year or now.year-9,end_year or now.year,m._latest_complete_year(price))
    filtered=m.apply_filters(table,start_y,end_y,cycle,complete_only,fed,vix,teny,dxy)
    if filtered.empty: raise DataUnavailable("No observations match the selected sample and regime filters.")
    stats=m.seasonal_stats_from_filtered(filtered)
    active_month=month or int(price.index.max().month)
    years=[int(price.index.max().year)]+[int(y) for y in sorted(filtered.year.unique(),reverse=True) if int(y)!=int(price.index.max().year)]
    selected_year=year if year in years else years[0]
    matrix=[]
    for y in years:
        observed=table.loc[table.year==y]
        row={"Year":y}
        for i,label in enumerate(m.MONTH_LABELS,1):
            values=observed.loc[observed.month==i,"total_ret"]
            row[label]=_number(values.iloc[-1]) if len(values) else None
        valid=[row[label] for label in m.MONTH_LABELS if row[label] is not None]
        row["Year return"]=float((np.prod(1+np.array(valid)/100)-1)*100) if valid else None
        matrix.append(row)
    avg={"Year":f"{lookback} AVG" if lookback in {"5Y","10Y","20Y"} else "FILTER AVG"}
    for i,label in enumerate(m.MONTH_LABELS,1):avg[label]=_number(stats.loc[i,"mean_total"])
    valid=[avg[label] for label in m.MONTH_LABELS if avg[label] is not None]
    avg["Year return"]=float((np.prod(1+np.array(valid)/100)-1)*100) if valid else None
    matrix.insert(0,avg)
    summary=m.build_intra_month_summary(price,filtered,active_month,used,selected_year)
    paths,path=m._month_paths_prev_eom_equal_weight_from_filtered(price,filtered,active_month)
    comparison=m._year_month_path(price,selected_year,active_month)
    spread=paths.std(axis=1).reindex(path.index).fillna(0) if not paths.empty else pd.Series(dtype=float)
    x=list(range(len(path)))
    profile={"data":[
        {"type":"bar","x":m.MONTH_LABELS,"y":[_number(stats.loc[i,"mean_total"]) for i in range(1,13)],"name":"Average return","marker":{"color":["#a4c9b7" if i!=active_month else "#7e9ec9" for i in range(1,13)]}},
        {"type":"scatter","mode":"lines+markers","x":m.MONTH_LABELS,"y":[_number(stats.loc[i,"hit_rate"]) for i in range(1,13)],"name":"Hit rate (%)","yaxis":"y2","line":{"color":"#b096bf"}}],
        "layout":{"height":390,"margin":{"l":55,"r":55,"t":25,"b":55},"paper_bgcolor":"#fff","plot_bgcolor":"#fff","yaxis":{"title":"Return (%)"},"yaxis2":{"title":"Hit rate (%)","overlaying":"y","side":"right","range":[0,100]},"legend":{"orientation":"h","y":-0.25}}}
    path_figure={"data":[
        {"type":"scatter","mode":"lines","x":x,"y":(path+spread).tolist(),"line":{"width":0},"showlegend":False},
        {"type":"scatter","mode":"lines","x":x,"y":(path-spread).tolist(),"fill":"tonexty","fillcolor":"rgba(194,177,208,.28)","line":{"width":0},"name":"±1 standard deviation"},
        {"type":"scatter","mode":"lines","x":x,"y":path.tolist(),"name":"Filtered average","line":{"color":"#688f7d","width":3}},
        {"type":"scatter","mode":"lines","x":comparison.index.tolist(),"y":comparison.tolist(),"name":str(selected_year),"line":{"color":"#8a9fc8","width":2}}],
        "layout":{"height":390,"margin":{"l":55,"r":20,"t":25,"b":55},"paper_bgcolor":"#fff","plot_bgcolor":"#fff","xaxis":{"title":"Trading day of month"},"yaxis":{"title":"Return from prior month-end (%)"},"legend":{"orientation":"h","y":-0.25}}}
    audit=filtered.reset_index().rename(columns={"period":"Period","index":"Period","total_ret":"Monthly return (%)","h1_ret":"First half (%)","h2_ret":"Second half (%)"})
    audit["Period"]=audit["Period"].astype(str)
    cols=["Period","year","month","pres_cycle_bucket","regime_cycle","fed_regime","vix_bucket","teny_trend","dxy_trend","is_complete_month","Monthly return (%)","First half (%)","Second half (%)"]
    selected={key:_number(value) if isinstance(value,(float,np.floating)) else value for key,value in summary.items() if key not in {"avg_path","df_paths"}}
    return {"symbol":used,"as_of":str(price.index.max().date()),"start_year":start_y,"end_year":end_y,
            "sample_months":len(filtered),"sample_years":int(filtered.year.nunique()),"thin_sample":len(filtered)<24 or filtered.year.nunique()<5,
            "month":active_month,"year":selected_year,"years":years,"matrix":matrix,
            "profile":records(stats.reset_index()),"profile_figure":profile,"path_figure":path_figure,
            "summary":selected,"audit":records(audit[cols].sort_values(["year","month"])),
            "filter_options":{key:sorted(str(value) for value in table[key].dropna().unique() if str(value)!="Unknown") for key in ["fed_regime","vix_bucket","teny_trend","dxy_trend"]}}
