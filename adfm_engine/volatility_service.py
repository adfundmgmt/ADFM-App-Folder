"""Volatility service with original pair, implied, downside, and fixed diagnostics."""
import pandas as pd

from adfm_engine.analytics.relative_volatility import pair_volatility_diagnostics, realized_volatility_ratio, relative_volatility_frame, rolling_zscore_previous
from adfm_engine.charts.relative_volatility import aligned_level_ratio, close_series, display_ticker, latest_value, normalized_chart, overview_chart
from adfm_engine.data.market import fetch_daily_ohlcv, unique_tickers
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable
from adfm_engine.volatility_methodology import methodology


def volatility(raw_frames, *, primary="^NDX", comparison="^GSPC", primary_implied="^VXN", comparison_implied="^VIX", history="5y", rvol_window=21, normalization_window=252, missing=None):
    missing = pd.DataFrame() if missing is None else missing
    pc, cc, pi, ci, soxx, ndx, qew, qqq = [close_series(raw_frames, ticker) for ticker in (primary, comparison, primary_implied, comparison_implied, "SOXX", "^NDX", "QEW", "QQQ")]
    required = [ticker for ticker, close in ((primary, pc), (comparison, cc)) if close.empty]
    if required:
        raise DataUnavailable("No valid daily price history was returned for: " + ", ".join(required))
    warnings = []
    optional = [ticker for ticker, close in ((primary_implied, pi), (comparison_implied, ci), ("SOXX", soxx), ("^NDX", ndx), ("QEW", qew), ("QQQ", qqq)) if ticker and close.empty]
    if optional:
        warnings.append("Optional diagnostics are unavailable for: " + ", ".join(dict.fromkeys(optional)) + ". Core pair analysis is unaffected.")
    minimum = max(10, min(63, normalization_window // 2))
    analysis = relative_volatility_frame(pc, cc, rvol_window=rvol_window, normalization_window=normalization_window, normalization_min_periods=minimum)
    analysis = analysis.join(pair_volatility_diagnostics(pc, cc, short_window=5, long_window=21), how="outer")
    if not pi.empty:
        analysis["primary_implied_level"] = pi
        analysis["primary_implied_zscore"] = rolling_zscore_previous(pi, normalization_window, minimum)
    if not ci.empty:
        analysis["comparison_implied_level"] = ci
        analysis["comparison_implied_zscore"] = rolling_zscore_previous(ci, normalization_window, minimum)
    analysis["implied_ratio"] = aligned_level_ratio(pi, ci)
    analysis["soxx_ndx_rvol_ratio_21d"] = realized_volatility_ratio(soxx, ndx, 21)
    analysis["qew_qqq_rvol_ratio_21d"] = realized_volatility_ratio(qew, qqq, 21)
    usable = analysis.dropna(subset=["primary_rvol", "comparison_rvol", "rvol_ratio"])
    if usable.empty:
        raise DataUnavailable(f"There are not enough overlapping observations to calculate {rvol_window}-session volatility.")
    z_table = pd.DataFrame([
        {"Series": primary, "Current synthetic VIX": latest_value(usable["primary_rvol"]), "Z-score": latest_value(analysis["primary_zscore"])},
        {"Series": comparison, "Current synthetic VIX": latest_value(usable["comparison_rvol"]), "Z-score": latest_value(analysis["comparison_zscore"])},
    ])
    for ticker, prefix in ((primary_implied, "primary"), (comparison_implied, "comparison")):
        if ticker and f"{prefix}_implied_level" in analysis:
            z_table.loc[len(z_table)] = {"Series": ticker, "Current synthetic VIX": latest_value(analysis[f"{prefix}_implied_level"]), "Z-score": latest_value(analysis[f"{prefix}_implied_zscore"])}
    export = analysis.rename(columns={"primary_rvol": f"{primary}_synthetic_vix", "comparison_rvol": f"{comparison}_synthetic_vix", "rvol_ratio": f"{primary}_{comparison}_rvol_ratio", "primary_zscore": f"{primary}_vol_zscore", "comparison_zscore": f"{comparison}_vol_zscore", "primary_implied_level": f"{primary_implied}_level", "primary_implied_zscore": f"{primary_implied}_zscore", "comparison_implied_level": f"{comparison_implied}_level", "comparison_implied_zscore": f"{comparison_implied}_zscore", "implied_ratio": f"{primary_implied}_{comparison_implied}_ratio"})
    export.index.name = "Date"
    labels = [display_ticker(t) for t in (primary, comparison, primary_implied, comparison_implied)]
    return dict(schema_version=1, data_through=usable.index[-1].date().isoformat(), warnings=warnings, overview=figure_json(overview_chart(analysis, labels[0], labels[1], rvol_window, labels[2], labels[3])), normalized=figure_json(normalized_chart(analysis, *labels)), z_table=records(z_table), data=records(export.dropna(how="all").tail(252).sort_index(ascending=False).reset_index()), csv=export.reset_index().to_csv(index=False), filename=f"relative_volatility_{primary}_{comparison}.csv".replace("^", ""), diagnostics=records(missing), methodology=methodology(primary, comparison, primary_implied, comparison_implied, rvol_window, normalization_window))


def load_volatility(**parameters):
    requested = unique_tickers([parameters[k] for k in ("primary", "comparison", "primary_implied", "comparison_implied")] + ["SOXX", "^NDX", "QEW", "QQQ"])
    frames, missing = fetch_daily_ohlcv(requested, parameters["history"])
    return volatility(frames, missing=missing, **parameters)
