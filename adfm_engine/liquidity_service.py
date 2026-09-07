"""Liquidity result orchestration for the website, reports and jobs."""
import pandas as pd
from adfm_engine.analytics.liquidity import compute_liquidity, build_primary, scorecard, market_tickers, _color_score
from adfm_engine.analytics.liquidity_definitions import FRED_IDS, FRED_LABELS
from adfm_engine.charts.liquidity import main_chart, financial_conditions_chart, driver_charts
from adfm_engine.data.liquidity import load_fred, load_market, load_fcig
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable

LOOKBACKS = ["6m", "1y", "2y", "3y", "5y", "10y", "max"]


def table_colors(frame, columns):
    backgrounds, foregrounds = [], []
    for _, row in frame.iterrows():
        background, foreground = {}, {}
        for column in columns:
            if column in row:
                css = dict(item.split(":", 1) for item in _color_score(row[column]).strip(";").split(";") if ":" in item)
                if css:
                    background[column], foreground[column] = css["background-color"], css["color"]
        backgrounds.append(background)
        foregrounds.append(foreground)
    return dict(backgrounds=backgrounds, foregrounds=foregrounds)


def liquidity(fred, prices, fcig=None, fred_errors=None, *, lookback="5y", z_window=756, min_periods=252, smoothing=3, show_fcig=True):
    if lookback not in LOOKBACKS or not 252 <= z_window <= 1260 or not 126 <= min_periods <= min(756, z_window) or not 1 <= smoothing <= 21:
        raise ValueError("Minimum observations must not exceed the score lookback; check the selected controls.")
    if fred.empty:
        raise DataUnavailable("Primary-source liquidity data could not be loaded. Failed series will be retried on the next request.")
    if build_primary(fred)[0].empty:
        raise DataUnavailable("The primary liquidity components could not be constructed.")
    errors = fred_errors or {}
    model = compute_liquidity(fred, prices, z_window, min_periods, smoothing, lookback)
    primary_card = scorecard(model["primary"], model["primary_levels"], model["primary_impulses"], model["primary_specs"])
    market_card = scorecard(model["market"], model["market_levels"], model["market_impulses"], model["market_specs"]) if not model["market"].empty else pd.DataFrame()
    driver_figs = driver_charts(model["display_sleeve_impulses"], model["primary_impulses"])
    diagnostics = pd.DataFrame([{"Series": FRED_LABELS.get(key, key), "FRED ID": key, "Status": "Unavailable" if key in errors else "Loaded", "Latest Observation": fred[key].dropna().index.max().date().isoformat() if key in fred and fred[key].notna().any() else "N/A", "Error": errors.get(key, "")} for key in FRED_IDS])
    export = pd.concat({"Liquidity Level": model["liquidity_level"], "Liquidity Impulse": model["liquidity_impulse"], "Weighted Breadth": model["easing_breadth"], "Coverage": model["impulse_coverage"], "Market Confirmation": model["market_confirmation"]}, axis=1).reset_index(names="Date")
    warnings = []
    if errors:
        warnings.append("Unavailable primary series: " + ", ".join(FRED_LABELS.get(key, key) for key in errors) + ". Coverage rules prevent incomplete sleeves from printing as full signals.")
    return dict(schema_version=1, warnings=warnings,
        current=records(pd.DataFrame([{k: model[k] for k in ("current_level", "current_impulse", "level_read", "impulse_read")}]))[0],
        main=figure_json(main_chart(model["display_level"], model["display_impulse"])),
        fcig=figure_json(financial_conditions_chart(fcig, lookback)) if show_fcig and fcig is not None and not fcig.empty else None,
        drivers=[figure_json(fig) for fig in driver_figs if fig is not None],
        primary=records(primary_card), market=records(market_card), diagnostics=records(diagnostics),
        primary_colors=table_colors(primary_card, ["Level Score", "Impulse Score"]), market_colors=table_colors(market_card, ["Impulse Score"]),
        csv=export.to_csv(index=False))


def load_liquidity(**parameters):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        fred_task = pool.submit(load_fred, FRED_IDS)
        market_task = pool.submit(load_market, tuple(market_tickers()), "10y")
        fcig_task = pool.submit(load_fcig) if parameters.get("show_fcig", True) else None
        fred, errors = fred_task.result()
        prices = market_task.result()
        fcig = fcig_task.result()[0] if fcig_task else None
    return liquidity(fred, prices, fcig, errors, **parameters)
