from datetime import date,timedelta
from typing import List
import pandas as pd
from adfm_engine.analytics import calendar as model
from adfm_engine.analytics.calendar import *
from adfm_engine.data.calendar import _fetch_market,_fetch_macro
from adfm_engine.charts.calendar import _timeline,_heatmap
from adfm_engine.serialization import records,figure_json
from adfm_engine.analytics.calendar import _next_weekday,_previous_weekday,_first_weekday,_nth_weekday,_first_business_day,_nth_business_day,_last_day_of_month,_last_business_day,_add_months,_normalize_event_type,_build_rule_calendar,_parse_custom_events,_close_from_yfinance,_trailing_return,_build_market_table,_market_stress,_action,_score_events,_risk_label,_latest_pair,_fmt_period,_fmt_macro,_macro_prints,_event,_official_dated_calendar,_format_event_date,_format_days

def calendar(market,macro_panel,macro_status,today=None,horizon_days=90,include_macro=True,include_fed=True,hide_low=False,custom_text=""):
    today=today or date.today()
    warnings=[]
    stress_bonus,stress_label=_market_stress(market)
    frames: List[pd.DataFrame] = []
    if include_macro:
        frames.append(_official_dated_calendar(today, horizon_days, include_fed))

    custom = _parse_custom_events(custom_text,warnings)
    if not custom.empty:
        custom["Source"] = "Custom"
        frames.append(custom)

    calendar = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not calendar.empty:
        if "Precision" not in calendar.columns:
            calendar["Precision"] = "Rule"
        if "Source" not in calendar.columns:
            calendar["Source"] = "Calendar rule"
        calendar["Precision"] = calendar["Precision"].fillna("Rule")
        calendar["Source"] = calendar["Source"].fillna("Custom")
        calendar = calendar[(calendar["Date"] >= today) & (calendar["Date"] <= today + timedelta(days=horizon_days))]
        calendar = calendar.drop_duplicates(subset=["Date", "Event", "Type"], keep="last")
        calendar = _score_events(calendar, today, stress_bonus)
        if hide_low and not calendar.empty:
            calendar = calendar[calendar["Risk Score"] >= 65].reset_index(drop=True)

    perf = _build_market_table(market, today)
    if include_macro and (today < date(2026,9,1) or today+timedelta(days=horizon_days)>date(2026,12,31)):
        warnings.append('The original confirmed macro-date catalog covers September–December 2026, plus the January 27, 2027 FOMC meeting. Other scheduled macro dates require an update; calendar rules and custom events remain available.')
    result={'asof':today.isoformat(),'warnings':warnings,'cards':[],'timeline':None,'heatmap':None,'macro':records(_macro_prints(macro_panel)),'decision':[],'details':[],'status':records(macro_status[[c for c in ['key','symbol','provider','data_through','status'] if c in macro_status]]),'csv':''}
    if calendar.empty:return result
    next_event = calendar.iloc[0]
    highest = calendar.sort_values("Risk Score", ascending=False).iloc[0]
    next_week = calendar[calendar["Days"] <= 7]
    cluster_days = int((calendar["Cluster"] > 0).sum())

    result["cards"]=[
        ["Next Catalyst",str(next_event["Event"]),f"{_format_event_date(next_event['Date'])} · {_format_days(int(next_event['Days']))}",TYPE_COLORS.get(str(next_event["Type"]), RISK_COLORS["neutral"])],
        ["Highest Risk",str(highest["Event"]),f"{_format_event_date(highest['Date'])} · {_risk_label(float(highest['Risk Score']))} risk, score {float(highest['Risk Score']):.0f}",RISK_COLORS["high"] if float(highest["Risk Score"]) >= 82 else RISK_COLORS["medium"]],
        ["Next 7 Days",str(len(next_week)),f"{int((next_week['Risk Score'] >= 82).sum())} high-risk event(s)",RISK_COLORS["high"] if int((next_week["Risk Score"] >= 82).sum()) else RISK_COLORS["neutral"]],
        ["Clustered Days",str(cluster_days),"Same-day or nearby catalysts",RISK_COLORS["medium"] if cluster_days else RISK_COLORS["neutral"]],
        ["Vol Backdrop",stress_label,f"+{stress_bonus:.1f} added to event score" if stress_bonus else "No risk-score add-on",RISK_COLORS["high"] if stress_bonus >= 5 else RISK_COLORS["neutral"]]
    ]
    decision = calendar[["Date", "Days", "Event", "Type", "Precision", "Source", "Risk Score", "Exposure", "Action"]].copy()
    decision["Date"] = decision["Date"].map(_format_event_date)
    decision["When"] = decision["Days"].map(lambda x: _format_days(int(x)))
    decision["Status"] = decision["Precision"].replace({"Official": "Confirmed", "Rule": "Rule-based", "Custom": "Custom", "Estimated": "Estimated"})
    decision["Risk"] = decision["Risk Score"].map(lambda x: _risk_label(float(x)))
    decision["Risk Score"] = decision["Risk Score"].map(lambda x: f"{float(x):.0f}")
    decision = decision[["Date", "When", "Event", "Type", "Status", "Source", "Risk", "Risk Score", "Exposure", "Action"]]
    details = calendar.copy()
    details["Date"] = details["Date"].map(_format_event_date)
    details["When"] = details["Days"].map(lambda x: _format_days(int(x)))
    details["Status"] = details["Precision"].replace({"Official": "Confirmed", "Rule": "Rule-based", "Custom": "Custom", "Estimated": "Estimated"})
    result.update(timeline=figure_json(_timeline(calendar,today)),heatmap=figure_json(_heatmap(perf)) if not perf.empty else None,decision=records(decision),details=records(details[['Date','When','Event','Type','Status','Source','Region','Risk Score','Cluster','Why It Matters','Exposure','Action']]),csv=calendar.to_csv(index=False))
    return result

def load_calendar(**kwargs):
    today=date.today()
    market=_fetch_market(min(date(today.year,1,1)-timedelta(days=10),today-timedelta(days=460)).isoformat())
    panel,status=_fetch_macro(date(today.year-3,1,1).isoformat(),today.isoformat())
    return calendar(market,panel,status,today=today,**kwargs)
