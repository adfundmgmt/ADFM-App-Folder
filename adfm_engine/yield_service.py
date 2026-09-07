"""Yield Curve monitor orchestration independent of presentation."""
from datetime import date, timedelta
import pandas as pd
from adfm_engine.analytics.yields import *
from adfm_engine.charts.yields import history_chart, snapshot_chart, pressure_chart, chart_display_mode
from adfm_engine.data.yields import fetch_yahoo_close
from adfm_engine.serialization import records, figure_json
from adfm_engine.services import DataUnavailable
LOOKBACKS = {"6M":190,"1Y":380,"2Y":760,"3Y":1140,"5Y":1900,"10Y":3800}

def yields(yahoo_close, diagnostics=(), *, history="5Y", regime_period="1M", selected_curve="3m10y", curve_compare="1M", market_date=None):
    if history not in LOOKBACKS or regime_period not in PERIODS or selected_curve not in CURVE_OPTIONS or curve_compare not in {"1W","1M","3M","YTD"}:
        raise ValueError("Unsupported yield monitor controls.")
    if yahoo_close.empty:
        raise DataUnavailable("No usable Yahoo Finance yield data loaded.")
    rates = add_derived_yahoo_rates(split_yahoo_yields(yahoo_close))
    if rates.empty or "Y10" not in rates or rates["Y10"].dropna().empty:
        raise DataUnavailable("No usable 10Y Treasury yield loaded from Yahoo Finance. The page needs ^TNX to classify the rates regime.")
    curve_cols = available_curve_columns(rates)
    if not curve_cols:
        raise DataUnavailable("Yahoo loaded the 10Y yield, but not enough curve points to calculate a curve spread.")
    if selected_curve not in curve_cols:
        selected_curve = curve_cols[0]
    warnings=[]
    last_obs=latest_date(rates)
    if last_obs is not None and (pd.Timestamp(market_date or date.today())-last_obs.normalize()).days>4:
        warnings.append(f"Last Yahoo yield observation is {last_obs.date()}. The rates tape may be stale.")
    missing=[t for t in YAHOO_YIELD_TICKERS if t not in yahoo_close]
    if missing:
        warnings.append("Missing Yahoo yield symbols: "+", ".join(missing)+". Outputs are recalculated from available data only.")
    regime, regime_note, regime_color=classify_regime(rates, regime_period, selected_curve)
    cards = [
        ("Regime", regime, regime_note, regime_color),
        (
            "3M Treasury",
            fmt_pct(latest(rates["Y3M"])) if "Y3M" in rates else "N/A",
            f"{regime_period} {fmt_bp(change_bp(rates['Y3M'], regime_period))}"
            if "Y3M" in rates
            else "Unavailable",
            COLORS["slate"],
        ),
        (
            "5Y Treasury",
            fmt_pct(latest(rates["Y5"])) if "Y5" in rates else "N/A",
            f"{regime_period} {fmt_bp(change_bp(rates['Y5'], regime_period))}"
            if "Y5" in rates
            else "Unavailable",
            COLORS["grey"],
        ),
        (
            "10Y Treasury",
            fmt_pct(latest(rates["Y10"])),
            f"{regime_period} {fmt_bp(change_bp(rates['Y10'], regime_period))}",
            COLORS["blue"],
        ),
        (
            "30Y Treasury",
            fmt_pct(latest(rates["Y30"])) if "Y30" in rates else "N/A",
            f"{regime_period} {fmt_bp(change_bp(rates['Y30'], regime_period))}"
            if "Y30" in rates
            else "Unavailable",
            COLORS["purple"],
        ),
        (
            label_for_series(selected_curve),
            fmt_bp(latest(rates[selected_curve]) * 100.0),
            f"{regime_period} {fmt_bp(change_bp(rates[selected_curve], regime_period))}",
            COLORS["amber"],
        ),
    ]



    available_yields=[c for c in ["Y3M","Y5","Y10","Y30"] if c in rates and rates[c].dropna().any()]
    curve_data=rates[available_yields].dropna(how="all")
    matrix=period_matrix(rates,["Y3M","Y5","Y10","Y30",selected_curve])
    table=pd.DataFrame(index=yahoo_close.index)
    for ticker,meta in YAHOO_YIELD_TICKERS.items():
        field,label=str(meta["field"]),str(meta["label"])
        if field in rates:table[f"{label} Yield"]=rates[field]
    for curve in CURVE_OPTIONS:
        if curve in rates:table[f"{label_for_series(curve)} bp"]=rates[curve]*100.0
    return dict(schema_version=1, selected_curve=selected_curve, warnings=warnings, diagnostics=list(diagnostics[-80:]),
        data_through=last_obs.date().isoformat() if last_obs is not None else None,
        available=[t for t in YAHOO_YIELD_TICKERS if t in yahoo_close and yahoo_close[t].dropna().any()],
        cards=cards, narrative=regime_read(regime), config=chart_display_mode(),
        snapshot=figure_json(snapshot_chart(curve_data,curve_compare,available_yields)) if len(available_yields)>=2 and not curve_data.empty else None,
        pressure=figure_json(pressure_chart(matrix)) if not matrix.empty else None,
        history=figure_json(history_chart(rates,selected_curve,available_yields)),
        curve_label=label_for_series(selected_curve), curve_level=fmt_bp(latest(rates[selected_curve])*100.),curve_move=fmt_bp(change_bp(rates[selected_curve],regime_period)),
        rows=records(table.tail(260).rename_axis("Date").reset_index()))

def load_yields(**parameters):
    today=date.today();history=parameters.get("history","5Y")
    close,diagnostics=fetch_yahoo_close(tuple(YAHOO_YIELD_TICKERS),today-timedelta(days=LOOKBACKS[history]+10),today)
    return yields(close,diagnostics,market_date=today,**parameters)
