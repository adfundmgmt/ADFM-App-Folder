"""Options workflows callable by the website, reports, or scheduled jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from adfm_engine.analytics.options import (
    DEFAULT_UNIVERSE, close_series, fmt, latest_value, money, nearest_expiry,
    normalize_ticker, parse_universe,
)
from adfm_engine.analytics.options_positioning import (
    add_cross_sectional_ranks, build_positioning_commentary, option_snapshot,
    ordinal, prepare_chain,
)
from adfm_engine.analytics.relative_volatility import annualized_realized_volatility
from adfm_engine.charts.options import compass_chart, term_structure_chart, iv_surface_chart
from adfm_engine.data.market import fetch_daily_ohlcv
from adfm_engine.data.options import available_expirations, fetch_chain
from adfm_engine.serialization import records, figure_json
from adfm_engine.services import DataUnavailable


def options(raw_prices, price_failures, calendars, chains, *, selected="QQQ",
            universe_text=DEFAULT_UNIVERSE, target_dte=45, term_count=6,
            risk_free_rate=0.04, as_of_date=None):
    """Calculate a snapshot from supplied price/chain observations, without I/O."""
    selected = normalize_ticker(selected)
    universe = parse_universe(universe_text, selected)
    if not selected or len(universe) < 2:
        raise ValueError("Enter a focus ticker and at least one comparison ticker.")
    as_of_date = as_of_date or datetime.now(ZoneInfo("America/New_York")).date()
    price_metrics = {}
    for symbol in universe:
        close = close_series(raw_prices, symbol)
        rvol = annualized_realized_volatility(close, 21)
        price_metrics[symbol] = {
            "spot": latest_value(close),
            "realized_vol_21d": latest_value(rvol) / 100.0,
            "return_5d": float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else np.nan,
        }
    universe_rows, provider_errors = [], []
    for symbol in universe:
        expiry = nearest_expiry(calendars.get(symbol, ()), target_dte, as_of_date)
        if expiry is None:
            provider_errors.append({"Ticker":symbol,"Issue":"No eligible option expiration returned"})
            continue
        calls, puts, underlying, error, source, timestamp = chains[(symbol, expiry)]
        if error or calls.empty or puts.empty:
            provider_errors.append({"Ticker":symbol,"Issue":error or "Empty option chain"})
            continue
        spot = price_metrics[symbol]["spot"]
        if not np.isfinite(spot):
            spot = float(underlying.get("regularMarketPrice", np.nan))
        if not np.isfinite(spot) or spot <= 0:
            provider_errors.append({"Ticker":symbol,"Issue":"No valid underlying price"})
            continue
        snapshot = option_snapshot(calls, puts, spot=spot, expiry=expiry,
                                   as_of=as_of_date, risk_free_rate=float(risk_free_rate))
        # Preserve the original price-metric fields, including missing price
        # history. The valid chain spot is retained for downstream calculations.
        metrics = {**price_metrics[symbol], "spot": spot}
        universe_rows.append({"ticker":symbol,"chain_source":source,
                              "source_timestamp":timestamp,**snapshot,**metrics})
    universe_frame = add_cross_sectional_ranks(pd.DataFrame(universe_rows)) if universe_rows else pd.DataFrame()
    if universe_frame.empty or selected not in set(universe_frame.get("ticker", [])):
        raise DataUnavailable(f"Neither Yahoo nor Cboe returned a usable option chain for {selected}.", diagnostics=provider_errors)
    row = universe_frame.loc[universe_frame["ticker"].eq(selected)].iloc[0]
    expiry = str(row["expiry"])
    calls, puts, _, _, source, timestamp = chains[(selected, expiry)]
    source_detail = f"{source} · snapshot {timestamp} UTC" if timestamp else source
    cards = [
        ("ATM implied vol",fmt(float(row["atm_iv"])*100.0,"%"),f"{int(row['dte'])} DTE expiration"),
        ("21D realized vol",fmt(float(row["realized_vol_21d"])*100.0,"%"),"Annualized close-to-close"),
        ("IV richness rank",ordinal(float(row["iv_richness_percentile"])),"Current selected universe"),
        ("25D put skew",fmt(float(row["put_skew"])*100.0," vol",1),"Put IV minus call IV"),
        ("Put/call volume",fmt(float(row["put_call_volume"]),"x",2),"Aggregate current chain"),
        ("Premium activity",money(float(row["premium_activity"])),"Mid/last × volume × 100"),
    ]
    compass_table = universe_frame[["ticker","expiry","dte","atm_iv","realized_vol_21d",
        "iv_richness","iv_richness_percentile","put_skew","put_skew_percentile",
        "put_call_volume","put_call_oi","return_5d","chain_source","source_timestamp"]].sort_values("put_skew_percentile",ascending=False)
    eligible = [e for e in calendars.get(selected, ()) if 2 <= (pd.Timestamp(e).date()-as_of_date).days <=365][:term_count]
    term_rows, term_chains = [], []
    for term in eligible:
        tc, tp, _, error, _, _ = chains[(selected, term)]
        if error or tc.empty or tp.empty:
            continue
        snapshot = option_snapshot(tc,tp,spot=float(row["spot"]),expiry=term,
                                   as_of=as_of_date,risk_free_rate=float(risk_free_rate))
        term_rows.append(snapshot)
        term_chains.append((snapshot,tc,tp))
    term_frame = pd.DataFrame(term_rows)
    time_years = max(float(row["dte"]),1.0)/365.0
    activity = pd.concat([prepare_chain(frame,kind,spot=float(row["spot"]),
        time_years=time_years,risk_free_rate=float(risk_free_rate))
        for frame,kind in [(calls,"call"),(puts,"put")]],ignore_index=True)
    activity["expiry"] = expiry
    activity["moneyness"] = activity["strike"]/float(row["spot"])
    activity = activity.sort_values("premium_activity",ascending=False).head(30)
    display_activity = activity[["contractSymbol","type","strike","moneyness","lastPrice",
        "bid","ask","mid","impliedVolatility","iv_source","volume","openInterest",
        "premium_activity","lastTradeDate"]]
    diagnostics = provider_errors.copy()
    for failure in price_failures.to_dict("records"):
        diagnostics.append({"Ticker":str(failure.get("Ticker","")),"Issue":str(failure.get("Reason","Price history unavailable"))})
    return dict(schema_version=1,selected=selected,expiry=expiry,dte=int(row["dte"]),
        as_of=as_of_date.isoformat(),chain_count=len(universe_frame),source=source_detail,
        cards=cards,narrative=build_positioning_commentary(row),compass=figure_json(compass_chart(universe_frame,selected)),
        structure=figure_json(term_structure_chart(term_frame)) if not term_frame.empty else None,
        surface=figure_json(iv_surface_chart(term_chains,float(risk_free_rate))) if not term_frame.empty else None,
        compass_rows=records(compass_table),term_rows=records(term_frame),activity_rows=records(display_activity),
        diagnostics=records(pd.DataFrame(diagnostics).drop_duplicates()),
        compass_csv=universe_frame.to_csv(index=False),term_csv=term_frame.to_csv(index=False) if not term_frame.empty else None,
        activity_csv=display_activity.to_csv(index=False))


def load_options(**parameters):
    """Retrieve only the requested expirations, reusing 15-minute chain caches."""
    selected = normalize_ticker(parameters.get("selected","QQQ"))
    universe = parse_universe(parameters.get("universe_text",DEFAULT_UNIVERSE),selected)
    as_of = datetime.now(ZoneInfo("America/New_York")).date()
    raw, failures = fetch_daily_ohlcv(universe,period="1y")
    with ThreadPoolExecutor(max_workers=4) as pool:
        calendars = dict(zip(universe, pool.map(available_expirations, universe)))
    wanted = {(symbol,expiry) for symbol in universe
        if (expiry:=nearest_expiry(calendars[symbol],parameters.get("target_dte",45),as_of))}
    wanted.update((selected,e) for e in [e for e in calendars[selected]
        if 2 <= (pd.Timestamp(e).date()-as_of).days <=365][:parameters.get("term_count",6)])
    keys = sorted(wanted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        chains = dict(zip(keys, pool.map(lambda key:fetch_chain(*key), keys)))
    return options(raw,failures,calendars,chains,as_of_date=as_of,**parameters)
