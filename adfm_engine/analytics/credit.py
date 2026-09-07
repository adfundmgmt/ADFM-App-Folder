"""Original credit classifications, lookbacks and sovereign transformations."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from adfm_engine.analytics.credit_definitions import *

def clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def latest(series: pd.Series | None) -> float:
    s = clean_series(series)
    return float(s.iloc[-1]) if not s.empty else np.nan


def latest_timestamp(series: pd.Series | None) -> Optional[pd.Timestamp]:
    s = clean_series(series)
    return pd.Timestamp(s.index[-1]) if not s.empty else None


def asof_value(series: pd.Series | None, target: pd.Timestamp) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    eligible = s.loc[s.index <= pd.Timestamp(target)]
    return float(eligible.iloc[-1]) if not eligible.empty else np.nan


def first_on_or_after(series: pd.Series | None, target: pd.Timestamp) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    eligible = s.loc[s.index >= pd.Timestamp(target)]
    return float(eligible.iloc[0]) if not eligible.empty else np.nan


def focus_target(label: str, asof: pd.Timestamp) -> pd.Timestamp:
    if label == "5D":
        return asof - pd.Timedelta(days=8)
    if label == "1M":
        return asof - pd.DateOffset(months=1)
    if label == "3M":
        return asof - pd.DateOffset(months=3)
    if label == "YTD":
        return pd.Timestamp(asof.year, 1, 1)
    if label == "1Y":
        return asof - pd.DateOffset(years=1)
    if label == "3Y":
        return asof - pd.DateOffset(years=3)
    if label == "5Y":
        return asof - pd.DateOffset(years=5)
    return asof - pd.DateOffset(months=1)


def pct_move(series: pd.Series | None, label: str) -> float:
    s = clean_series(series)
    if len(s) < 2:
        return np.nan
    now = float(s.iloc[-1])
    if label == "5D" and len(s) >= 6:
        base = float(s.iloc[-6])
    elif label == "YTD":
        base = first_on_or_after(s, pd.Timestamp(s.index[-1].year, 1, 1))
    else:
        base = asof_value(s, focus_target(label, pd.Timestamp(s.index[-1])))
    if not np.isfinite(base) or base == 0:
        return np.nan
    return (now / base - 1.0) * 100.0


def absolute_move(series: pd.Series | None, label: str, scale: float = 1.0) -> float:
    s = clean_series(series)
    if len(s) < 2:
        return np.nan
    now = float(s.iloc[-1])
    if label == "5D" and len(s) >= 6:
        base = float(s.iloc[-6])
    elif label == "YTD":
        base = first_on_or_after(s, pd.Timestamp(s.index[-1].year, 1, 1))
    else:
        base = asof_value(s, focus_target(label, pd.Timestamp(s.index[-1])))
    if not np.isfinite(base):
        return np.nan
    return (now - base) * scale


def trailing_percentile(series: pd.Series | None, years: int) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    cutoff = pd.Timestamp(s.index[-1]) - pd.DateOffset(years=years)
    window = s.loc[s.index >= cutoff]
    if len(window) < 30:
        return np.nan
    return float((window <= window.iloc[-1]).mean())


def fmt_pct(value: float, digits: int = 2, signed: bool = True) -> str:
    if not np.isfinite(value):
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def fmt_bp(value: float, digits: int = 0) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:+.{digits}f} bp"


def fmt_yield(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}%"


def fmt_percentile(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value * 100:.0f}th pct"


def ratio_frame(market: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "HYG/LQD": ("HYG", "LQD"),
        "JNK/LQD": ("JNK", "LQD"),
        "BKLN/LQD": ("BKLN", "LQD"),
        "SRLN/LQD": ("SRLN", "LQD"),
        "EMB/LQD": ("EMB", "LQD"),
        "KRE/SPY": ("KRE", "SPY"),
        "XLF/SPY": ("XLF", "SPY"),
        "IWM/SPY": ("IWM", "SPY"),
    }
    out = pd.DataFrame(index=market.index)
    for label, (num, den) in definitions.items():
        if num in market.columns and den in market.columns:
            out[label] = market[num] / market[den].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).dropna(how="all").ffill()


def sovereign_move_rows(series_map: Dict[str, pd.Series], horizon: str, source: str) -> pd.DataFrame:
    meta = {str(row["country"]): row for row in SOVEREIGN_UNIVERSE}
    rows: List[dict] = []
    today = pd.Timestamp(date.today())

    for country, series in series_map.items():
        if country not in meta:
            continue
        s = clean_series(series)
        if len(s) < 2:
            continue
        latest_dt = pd.Timestamp(s.index[-1])
        age_days = (today - latest_dt.normalize()).days
        max_age = 7 if source != "OECD/FRED monthly" else 90
        if age_days > max_age:
            continue
        end_yield = float(s.iloc[-1])

        if horizon == "5D":
            if len(s) < 6:
                continue
            start_yield = float(s.iloc[-6])
            start_dt = pd.Timestamp(s.index[-6])
        elif horizon == "YTD":
            start_target = pd.Timestamp(latest_dt.year, 1, 1)
            eligible = s.loc[s.index <= start_target]
            if eligible.empty:
                eligible = s.loc[s.index >= start_target]
            if eligible.empty:
                continue
            start_yield = float(eligible.iloc[-1] if eligible.index[-1] <= start_target else eligible.iloc[0])
            start_dt = pd.Timestamp(eligible.index[-1] if eligible.index[-1] <= start_target else eligible.index[0])
        else:
            target = focus_target(horizon, latest_dt)
            eligible = s.loc[s.index <= target]
            if eligible.empty:
                continue
            start_yield = float(eligible.iloc[-1])
            start_dt = pd.Timestamp(eligible.index[-1])

        move_bp = (end_yield - start_yield) * 100.0
        if not np.isfinite(move_bp):
            continue
        rows.append(
            {
                "Country": country,
                "Label": meta[country]["label"],
                "Group": meta[country]["group"],
                "Move bp": float(move_bp),
                "Start Yield": start_yield,
                "End Yield": end_yield,
                "Start Date": start_dt,
                "End Date": latest_dt,
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def public_snapshot_rows(
    snapshots: pd.DataFrame,
    horizon: str,
    oecd_map: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    meta = {str(row["country"]): row for row in SOVEREIGN_UNIVERSE}
    oecd_map = oecd_map or {}
    rows: List[dict] = []
    today = pd.Timestamp(date.today())

    for _, snap in snapshots.iterrows():
        country = str(snap.get("Country", ""))
        if country not in meta:
            continue
        end_yield = pd.to_numeric(snap.get("End Yield"), errors="coerce")
        end_dt = pd.to_datetime(snap.get("End Date"), errors="coerce")
        if pd.isna(end_yield) or pd.isna(end_dt):
            continue
        end_yield = float(end_yield)
        end_dt = pd.Timestamp(end_dt)
        if (today - end_dt.normalize()).days > 7:
            continue

        source = "Trading Economics public"
        if horizon in {"1M", "1Y"}:
            move_col = f"{horizon} Move bp"
            move_bp = pd.to_numeric(snap.get(move_col), errors="coerce")
            if pd.isna(move_bp):
                continue
            move_bp = float(move_bp)
            start_yield = end_yield - move_bp / 100.0
            start_dt = focus_target(horizon, end_dt)
        elif horizon in {"YTD", "3Y", "5Y"}:
            anchor = clean_series(oecd_map.get(country))
            if anchor.empty:
                continue
            target = focus_target(horizon, end_dt)
            eligible = anchor.loc[anchor.index <= target]
            if eligible.empty:
                continue
            start_yield = float(eligible.iloc[-1])
            start_dt = pd.Timestamp(eligible.index[-1])
            move_bp = (end_yield - start_yield) * 100.0
            source = "Trading Economics current + OECD/FRED anchor"
        else:
            continue

        if not np.isfinite(move_bp):
            continue
        rows.append(
            {
                "Country": country,
                "Label": meta[country]["label"],
                "Group": meta[country]["group"],
                "Move bp": float(move_bp),
                "Start Yield": float(start_yield),
                "End Yield": end_yield,
                "Start Date": pd.Timestamp(start_dt),
                "End Date": end_dt,
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def _adequate_global_coverage(frame: pd.DataFrame) -> bool:
    if frame.empty or len(frame) < 8:
        return False
    dm = int((frame["Group"] == "Developed").sum())
    em = int((frame["Group"] == "Emerging").sum())
    return dm >= 4 and em >= 3


def compute_credit(fred, market, focus_window="1M"):
    proxy = ratio_frame(market) if not market.empty else pd.DataFrame()


    hy_oas = clean_series(fred["hy_oas"]) if "hy_oas" in fred else pd.Series(dtype=float)


    ig_oas = clean_series(fred["ig_oas"]) if "ig_oas" in fred else pd.Series(dtype=float)


    bbb_oas = clean_series(fred["bbb_oas"]) if "bbb_oas" in fred else pd.Series(dtype=float)


    dgs10 = clean_series(fred["dgs10"]) if "dgs10" in fred else pd.Series(dtype=float)


    dgs30 = clean_series(fred["dgs30"]) if "dgs30" in fred else pd.Series(dtype=float)


    spread_percentiles = {
        "HY OAS": trailing_percentile(hy_oas, 5),
        "IG OAS": trailing_percentile(ig_oas, 5),
        "BBB OAS": trailing_percentile(bbb_oas, 5),
    }


    spread_values = [value for value in spread_percentiles.values() if np.isfinite(value)]


    spread_stress = float(np.mean(spread_values)) if spread_values else np.nan


    rate_percentiles = {
        "10Y": trailing_percentile(dgs10, 10),
        "30Y": trailing_percentile(dgs30, 10),
    }


    rate_values = [value for value in rate_percentiles.values() if np.isfinite(value)]


    funding_pressure = float(np.mean(rate_values)) if rate_values else np.nan


    if np.isfinite(spread_stress) and np.isfinite(funding_pressure):
        if spread_stress >= 0.70 and funding_pressure >= 0.70:
            credit_state = "Broad credit stress"
        elif spread_stress <= 0.40 and funding_pressure >= 0.70:
            credit_state = "High-rate / tight-spread"
        elif spread_stress >= 0.70 and funding_pressure < 0.70:
            credit_state = "Credit-specific stress"
        elif spread_stress <= 0.40 and funding_pressure <= 0.40:
            credit_state = "Easy spreads / easy funding"
        else:
            credit_state = "Mixed credit conditions"
    elif np.isfinite(spread_stress):
        credit_state = "Spread read only"
    elif np.isfinite(funding_pressure):
        credit_state = "Funding-cost read only"
    else:
        credit_state = "Insufficient data"


    hyg_lqd_move = pct_move(proxy["HYG/LQD"], focus_window) if "HYG/LQD" in proxy else np.nan


    kre_spy_move = pct_move(proxy["KRE/SPY"], focus_window) if "KRE/SPY" in proxy else np.nan


    vix_level = latest(market["^VIX"]) if "^VIX" in market else np.nan


    hy_oas_level = latest(hy_oas)


    hy_oas_1m_bp = absolute_move(hy_oas, "1M", scale=100.0)


    ig_oas_level = latest(ig_oas)


    ten_y = latest(dgs10)


    ten_y_pct = trailing_percentile(dgs10, 10)


    cards = [('Credit state', credit_state, f'Spread stress {fmt_percentile(spread_stress)} · funding pressure {fmt_percentile(funding_pressure)}'), ('HY OAS', f'{hy_oas_level * 100:.0f} bp' if np.isfinite(hy_oas_level) else 'N/A', f"1M {fmt_bp(hy_oas_1m_bp)} · 5Y {fmt_percentile(spread_percentiles['HY OAS'])}"), ('IG OAS', f'{ig_oas_level * 100:.0f} bp' if np.isfinite(ig_oas_level) else 'N/A', f"5Y {fmt_percentile(spread_percentiles['IG OAS'])}"), ('10Y Treasury', fmt_yield(ten_y), f'10Y history {fmt_percentile(ten_y_pct)}'), ('HY vs IG', fmt_pct(hyg_lqd_move), f'HYG/LQD · {focus_window}'), ('Bank beta', fmt_pct(kre_spy_move), f'KRE/SPY · {focus_window} · VIX {vix_level:.1f}' if np.isfinite(vix_level) else f'KRE/SPY · {focus_window}')]
    if np.isfinite(spread_stress) and np.isfinite(funding_pressure):
        if credit_state == "High-rate / tight-spread":
            active_read = (
                "Outright borrowing costs are historically expensive while credit spreads remain compressed. "
                "That is a very different regime from low credit stress: the market is charging little incremental "
                "default/liquidity premium on top of a high risk-free base rate."
            )
        elif credit_state == "Broad credit stress":
            active_read = (
                "Both the risk-free base rate and credit risk premia are elevated. This is the cleanest broad "
                "tightening signal and the most hostile configuration for levered balance sheets."
            )
        elif credit_state == "Credit-specific stress":
            active_read = (
                "Credit risk premia are elevated even without unusually high sovereign funding costs. "
                "The stress is coming from credit transmission rather than the risk-free curve."
            )
        else:
            active_read = (
                "Funding costs and spread stress are giving a mixed signal. Treat them as separate dimensions "
                "and use bank, loan, and HY relative performance for confirmation."
            )
    else:
        active_read = "Primary spread or rate data are incomplete; use the loaded series without forcing a composite."


    rows: List[dict] = []


    def add_spread_row(label: str, series: pd.Series) -> None:
        if series.empty:
            return
        rows.append(
            {
                "Signal": label,
                "Latest": f"{latest(series) * 100:.0f} bp",
                "5D": fmt_bp(absolute_move(series, "5D", 100.0)),
                "1M": fmt_bp(absolute_move(series, "1M", 100.0)),
                "YTD": fmt_bp(absolute_move(series, "YTD", 100.0)),
                "1Y": fmt_bp(absolute_move(series, "1Y", 100.0)),
                "Context": fmt_percentile(trailing_percentile(series, 5)),
                "Role": "Primary spread",
            }
        )


    def add_yield_row(label: str, series: pd.Series) -> None:
        if series.empty:
            return
        rows.append(
            {
                "Signal": label,
                "Latest": fmt_yield(latest(series)),
                "5D": fmt_bp(absolute_move(series, "5D", 100.0)),
                "1M": fmt_bp(absolute_move(series, "1M", 100.0)),
                "YTD": fmt_bp(absolute_move(series, "YTD", 100.0)),
                "1Y": fmt_bp(absolute_move(series, "1Y", 100.0)),
                "Context": fmt_percentile(trailing_percentile(series, 10)),
                "Role": "Funding cost",
            }
        )


    def add_ratio_row(label: str, series: pd.Series) -> None:
        if series.empty:
            return
        rows.append(
            {
                "Signal": label,
                "Latest": f"{latest(series):.3f}",
                "5D": fmt_pct(pct_move(series, "5D")),
                "1M": fmt_pct(pct_move(series, "1M")),
                "YTD": fmt_pct(pct_move(series, "YTD")),
                "1Y": fmt_pct(pct_move(series, "1Y")),
                "Context": "Higher = stronger banks" if label == "KRE/SPY" else "Higher = stronger",
                "Role": "Market confirmation",
            }
        )


    add_spread_row("US HY OAS", hy_oas)


    add_spread_row("US BBB OAS", bbb_oas)


    add_spread_row("US IG OAS", ig_oas)


    add_yield_row("US 10Y Treasury", dgs10)


    add_yield_row("US 30Y Treasury", dgs30)


    for label in ["HYG/LQD", "BKLN/LQD", "SRLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY"]:
        if label in proxy:
            add_ratio_row(label, proxy[label])


    if "^VIX" in market:
        vix = clean_series(market["^VIX"])
        rows.append(
            {
                "Signal": "VIX",
                "Latest": f"{latest(vix):.1f}",
                "5D": f"{absolute_move(vix, '5D'):+.1f}",
                "1M": f"{absolute_move(vix, '1M'):+.1f}",
                "YTD": f"{absolute_move(vix, 'YTD'):+.1f}",
                "1Y": f"{absolute_move(vix, '1Y'):+.1f}",
                "Context": fmt_percentile(trailing_percentile(vix, 5)),
                "Role": "Volatility",
            }
        )


    return locals()
