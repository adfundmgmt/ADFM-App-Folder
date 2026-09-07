"""CFTC scanner and selected-market workflows independent of the web UI."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from adfm_engine.analytics.cftc import COHORTS, DEFAULT_COHORT, REPORT_LABELS, build_scanner, compact_signal_rows, full_scanner_table, rolling_metrics, percentile_rank, positioning_signal, price_proxy, pm_read, fmt_pct, fmt_pp
from adfm_engine.charts.cftc import positioning_chart, cohort_chart
from adfm_engine.data.cftc import load_report, load_history, load_price
from adfm_engine.serialization import records, figure_json
from adfm_engine.services import DataUnavailable

LOOKBACKS = {"1Y":52,"2Y":104,"3Y":156,"5Y":260}
SORTS = ["Most crowded shorts","Most crowded longs","Largest 1W shift","Largest 4W contract change","Largest absolute z-score"]


def scanner_frame(tff, disagg, lookback, tff_cohort, disagg_cohort):
    if lookback not in LOOKBACKS or tff_cohort not in COHORTS["TFF"] or disagg_cohort not in COHORTS["Disaggregated"]:
        raise ValueError("Unsupported CFTC controls.")
    parts = [build_scanner(frame, report, cohort, LOOKBACKS[lookback]) for frame,report,cohort in [(tff,"TFF",tff_cohort),(disagg,"Disaggregated",disagg_cohort)] if not frame.empty]
    scanner = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if scanner.empty:
        raise DataUnavailable("CFTC Public Reporting did not return usable positioning data.")
    scanner["one_week_oi_shift"] = scanner["one_week_change"] / scanner["open_interest"].replace(0,np.nan)
    return scanner


def selection_frame(scanner):
    selection=scanner.sort_values(["asset_class","market"]).reset_index(drop=True)
    selection["key"]=selection["report_type"]+"|"+selection["contract_code"]
    selection["label"]=selection["market"]+" · "+selection["asset_class"]
    duplicated=selection["label"].duplicated(keep=False)
    selection.loc[duplicated,"label"]=selection.loc[duplicated,"label"]+" · "+selection.loc[duplicated,"contract_code"]
    return selection


def selected_row(selection, selected):
    key=selected or ("TFF|209742" if selection["key"].eq("TFF|209742").any() else selection.iloc[0]["key"])
    matches=selection.loc[selection["key"].eq(key)]
    return matches.iloc[0] if not matches.empty else selection.iloc[0]


def cftc(tff,disagg,history_raw,price=None, *,lookback="3Y",tff_cohort=DEFAULT_COHORT["TFF"],disagg_cohort=DEFAULT_COHORT["Disaggregated"],selected=None,assets=None,sort="Most crowded shorts",report_errors=None,history_error="",price_warning=""):
    scanner=scanner_frame(tff,disagg,lookback,tff_cohort,disagg_cohort)
    if sort not in SORTS:raise ValueError("Unsupported scanner rank.")
    selection=selection_frame(scanner);row=selected_row(selection,selected)
    report,code,market=str(row["report_type"]),str(row["contract_code"]),str(row["market"])
    cohort=tff_cohort if report=="TFF" else disagg_cohort
    usable=scanner.dropna(subset=["percentile"])
    shorts=usable.sort_values("percentile").head(5)
    longs=usable.sort_values("percentile",ascending=False).head(5)
    shifts=scanner.dropna(subset=["one_week_oi_shift"]).assign(_abs_shift=lambda f:f["one_week_oi_shift"].abs()).sort_values("_abs_shift",ascending=False).drop(columns="_abs_shift").head(5)
    advanced=scanner.copy()
    if assets:advanced=advanced.loc[advanced["asset_class"].isin(assets)]
    if sort in SORTS[:2]:advanced=advanced.sort_values("percentile",ascending=sort==SORTS[0])
    else:
        col={SORTS[2]:"one_week_oi_shift",SORTS[3]:"four_week_change",SORTS[4]:"zscore"}[sort]
        advanced=advanced.assign(_rank=advanced[col].abs()).sort_values("_rank",ascending=False).drop(columns="_rank")
    display=full_scanner_table(advanced,lookback)
    result=dict(schema_version=1,shorts=records(compact_signal_rows(shorts,lookback)),longs=records(compact_signal_rows(longs,lookback)),shifts=records(compact_signal_rows(shifts,lookback)),
        selection=records(selection[["key","label"]]),selected=str(row["key"]),assets=sorted(scanner["asset_class"].dropna().unique()),
        dates=[f"{report} {frame['report_date'].max().date().isoformat()}" for report,frame in [("TFF",tff),("Disaggregated",disagg)] if not frame.empty],
        warnings=["One CFTC report failed to load, so the dashboard is running on partial coverage."] if report_errors else [],
        scanner=records(display),scanner_csv=display.to_csv(index=False),main=None,cohorts=None,cards=[],narrative=None,history=[],history_csv=None,history_filename=f"adfm_cftc_{report.lower()}_{code}.csv",market=market,report_label=REPORT_LABELS[report])
    if history_raw.empty:
        result["warnings"].append(f"No historical CFTC data returned for {market}."+(f" {history_error}" if history_error else ""))
        return result
    history=rolling_metrics(history_raw,report,cohort,LOOKBACKS[lookback])
    pct_history=history["net_pct_oi"].tail(LOOKBACKS[lookback]).dropna()
    percentile=percentile_rank(pct_history) if len(pct_history)>=26 else np.nan
    latest=history.iloc[-1]
    weekly_shift=float(pct_history.iloc[-1]-pct_history.iloc[-2]) if len(pct_history)>=2 else np.nan
    proxy=price_proxy(code);price_label=proxy[1] if proxy else None
    price=pd.Series(dtype=float) if price is None else price
    result["narrative"]=pm_read(market,percentile,weekly_shift,lookback)
    result["cards"]=[("Signal",positioning_signal(percentile),cohort),("Net / open interest",fmt_pct(float(latest["net_pct_oi"])),"Normalized crowding"),(f"{lookback} percentile",f"{percentile:,.0f}th" if np.isfinite(percentile) else "N/A","Historical rank"),("1W shift",fmt_pp(weekly_shift),"More bullish" if weekly_shift>0 else "More bearish" if weekly_shift<0 else "Unchanged")]
    result["main"]=figure_json(positioning_chart(history,price,market,cohort,price_label))
    result["cohorts"]=figure_json(cohort_chart(history_raw,report))
    if proxy is None:result["warnings"].append("No mapped continuous-futures price proxy is available for this contract yet. Positioning history remains available.")
    elif price_warning:result["warnings"].append(f"Price proxy warning: {price_warning}")
    data=history[["report_date","market_name","contract_code","open_interest","cohort_long","cohort_short","net_contracts","net_pct_oi","rolling_zscore","rolling_percentile"]].tail(520)
    result["history"]=records(data);result["history_csv"]=data.to_csv(index=False)
    return result


def load_cftc(**parameters):
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(load_report,"TFF");b=pool.submit(load_report,"Disaggregated")
        tff,e1=a.result();disagg,e2=b.result()
    scanner=scanner_frame(tff,disagg,parameters.get("lookback","3Y"),parameters.get("tff_cohort",DEFAULT_COHORT["TFF"]),parameters.get("disagg_cohort",DEFAULT_COHORT["Disaggregated"]))
    row=selected_row(selection_frame(scanner),parameters.get("selected"))
    history,error=load_history(str(row["report_type"]),str(row["contract_code"]))
    proxy=price_proxy(str(row["contract_code"]))
    price,warning=load_price(proxy[0]) if proxy and not history.empty else (None,"")
    return cftc(tff,disagg,history,price,report_errors=[e for e in [e1,e2] if e],history_error=error,price_warning=warning,**parameters)
