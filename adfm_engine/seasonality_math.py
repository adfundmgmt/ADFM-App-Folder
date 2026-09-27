"""Pure month and regime calculations from the Streamlit seasonality explorer."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()

def build_monthly_regime_features(regime_daily: pd.DataFrame) -> pd.DataFrame:
    if regime_daily is None or regime_daily.empty:
        return pd.DataFrame()

    monthly = pd.DataFrame()
    daily = regime_daily.copy().sort_index()

    if "vix" in daily:
        vix_m = daily["vix"].resample("ME").mean()
        monthly["avg_vix"] = vix_m
        monthly["vix_bucket"] = pd.cut(
            monthly["avg_vix"],
            bins=[-np.inf, 15, 20, 25, np.inf],
            labels=["VIX <15", "VIX 15-20", "VIX 20-25", "VIX >25"],
        ).astype(str)

    if "tnx" in daily:
        tnx_m = daily["tnx"].resample("ME").last()
        tnx_delta = tnx_m.diff(3)
        monthly["teny"] = tnx_m
        monthly["teny_3m_chg"] = tnx_delta
        monthly["teny_trend"] = np.where(
            tnx_delta > 0.15,
            "10Y rising",
            np.where(tnx_delta < -0.15, "10Y falling", "10Y flat"),
        )

    if "dxy" in daily:
        dxy_m = daily["dxy"].resample("ME").last()
        dxy_delta = dxy_m.pct_change(3) * 100.0
        monthly["dxy"] = dxy_m
        monthly["dxy_3m_chg_pct"] = dxy_delta
        monthly["dxy_trend"] = np.where(
            dxy_delta > 1.5,
            "Dollar rising",
            np.where(dxy_delta < -1.5, "Dollar falling", "Dollar flat"),
        )

    if monthly.empty:
        return monthly

    monthly.index = monthly.index.to_period("M")
    return monthly

def _presidential_cycle_bucket(year: int) -> str:
    mod = year % 4
    if mod == 0:
        return "Election years"
    if mod == 1:
        return "Post-election years"
    if mod == 2:
        return "Midterm years"
    return "Pre-election years"

def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

    out_rows = []
    months = pd.period_range(
        prices.index.min().to_period("M"),
        prices.index.max().to_period("M"),
        freq="M",
    )

    current_month = _today().to_period("M")

    for m in months:
        m_mask = prices.index.to_period("M") == m
        month_days = prices.loc[m_mask]
        if month_days.shape[0] < 3:
            continue

        prev_month_days = prices.loc[prices.index.to_period("M") == (m - 1)]
        if prev_month_days.empty:
            continue

        prev_eom = float(prev_month_days.iloc[-1])
        last = float(month_days.iloc[-1])

        n = month_days.shape[0]
        mid_idx = (n // 2) - 1
        if mid_idx < 0:
            continue

        mid_close = float(month_days.iloc[mid_idx])
        tot = (last / prev_eom - 1.0) * 100.0
        h1 = (mid_close / prev_eom - 1.0) * 100.0
        h2 = tot - h1

        out_rows.append(
            {
                "period": m,
                "year": m.year,
                "month": m.month,
                "total_ret": tot,
                "h1_ret": h1,
                "h2_ret": h2,
                "month_obs": int(month_days.shape[0]),
                "is_complete_month": bool(m < current_month),
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df

def build_filter_table(
    prices: pd.Series,
    regime_df: pd.DataFrame,
    market_regime_df: pd.DataFrame,
) -> pd.DataFrame:
    halves = _intra_month_halves(prices)
    if halves.empty:
        return halves

    df = halves.copy()
    df["pres_cycle_bucket"] = df["year"].apply(_presidential_cycle_bucket)
    df["is_post_gfc"] = df["year"] >= 2009
    df["is_post_covid"] = df.index >= pd.Period("2020-01", freq="M")

    if regime_df is not None and not regime_df.empty:
        keep_cols = ["is_recession", "regime_cycle", "fed_regime", "fedfunds"]
        df = df.join(regime_df[keep_cols], how="left")
    else:
        df["is_recession"] = 0
        df["regime_cycle"] = "Expansion"
        df["fed_regime"] = "Unknown"
        df["fedfunds"] = np.nan

    if market_regime_df is not None and not market_regime_df.empty:
        df = df.join(market_regime_df, how="left")

    df["regime_cycle"] = df["regime_cycle"].fillna("Unknown")
    df["fed_regime"] = df["fed_regime"].fillna("Unknown")
    df["vix_bucket"] = df.get(
        "vix_bucket", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")
    df["teny_trend"] = df.get(
        "teny_trend", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")
    df["dxy_trend"] = df.get(
        "dxy_trend", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")

    return df

def _latest_complete_year(prices: pd.Series) -> int:
    if prices.empty:
        return _today().year - 1

    latest_period = prices.index.max().to_period("M")
    current_period = _today().to_period("M")
    if latest_period >= current_period:
        return current_period.year - 1
    return latest_period.year

def resolve_year_window(
    preset: str,
    custom_start_year: int,
    custom_end_year: int,
    latest_complete_year: int,
) -> Tuple[int, int]:
    if preset == "Custom":
        return int(custom_start_year), int(custom_end_year)
    if preset == "All history":
        return 1900, int(latest_complete_year)
    if preset == "Last 5 years":
        return int(latest_complete_year - 4), int(latest_complete_year)
    if preset == "Last 10 years":
        return int(latest_complete_year - 9), int(latest_complete_year)
    if preset == "Last 20 years":
        return int(latest_complete_year - 19), int(latest_complete_year)
    if preset == "Post-GFC (2009+)":
        return 2009, int(latest_complete_year)
    if preset == "Post-COVID (2020+)":
        return 2020, int(latest_complete_year)
    return int(custom_start_year), int(custom_end_year)

def apply_filters(
    filter_table: pd.DataFrame,
    start_year: int,
    end_year: int,
    cycle_filter: str,
    complete_months_only: bool,
    fed_filter: str,
    vix_filter: str,
    teny_filter: str,
    dxy_filter: str,
) -> pd.DataFrame:
    df = filter_table.copy()

    df = df[(df["year"] >= int(start_year)) & (df["year"] <= int(end_year))]

    if complete_months_only:
        df = df[df["is_complete_month"]]

    if cycle_filter != "All years":
        df = df[df["pres_cycle_bucket"] == cycle_filter]

    if fed_filter != "All Fed regimes":
        df = df[df["fed_regime"] == fed_filter]

    if vix_filter != "All VIX regimes":
        df = df[df["vix_bucket"] == vix_filter]

    if teny_filter != "All 10Y regimes":
        df = df[df["teny_trend"] == teny_filter]

    if dxy_filter != "All dollar regimes":
        df = df[df["dxy_trend"] == dxy_filter]

    return df.sort_index()

def seasonal_stats_from_filtered(filtered_halves: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["label"] = MONTH_LABELS

    if filtered_halves.empty:
        for col in [
            "hit_rate",
            "min_ret",
            "max_ret",
            "years_observed",
            "mean_h1",
            "mean_h2",
            "mean_total",
            "median_total",
            "p25_total",
            "p75_total",
            "downside_freq",
            "observations",
        ]:
            stats[col] = np.nan
        return stats

    grouped = filtered_halves.groupby("month")

    stats["hit_rate"] = grouped["total_ret"].apply(lambda x: (x > 0).mean() * 100.0)
    stats["min_ret"] = grouped["total_ret"].min()
    stats["max_ret"] = grouped["total_ret"].max()
    stats["years_observed"] = grouped["year"].nunique()
    stats["mean_h1"] = grouped["h1_ret"].mean()
    stats["mean_h2"] = grouped["h2_ret"].mean()
    stats["mean_total"] = grouped["total_ret"].mean()
    stats["median_total"] = grouped["total_ret"].median()
    stats["p25_total"] = grouped["total_ret"].quantile(0.25)
    stats["p75_total"] = grouped["total_ret"].quantile(0.75)
    stats["downside_freq"] = grouped["total_ret"].apply(
        lambda x: (x < -2.0).mean() * 100.0
    )
    stats["observations"] = grouped["total_ret"].count()

    return stats

def _month_paths_prev_eom_equal_weight_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if filtered_halves.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    target_periods = filtered_halves.loc[
        filtered_halves["month"] == month_int
    ].index.tolist()
    if not target_periods:
        return pd.DataFrame(), pd.Series(dtype=float)

    raw_paths = {}

    for p in target_periods:
        y = p.year
        m_num = p.month

        month_days = prices.loc[
            (prices.index.year == y) & (prices.index.month == m_num)
        ]
        if month_days.shape[0] < 3:
            continue

        prev_p = p - 1
        prev_month_days = prices.loc[
            (prices.index.year == prev_p.year) & (prices.index.month == prev_p.month)
        ]
        if prev_month_days.empty:
            continue

        prev_eom = float(prev_month_days.iloc[-1])
        cum_ret = (month_days / prev_eom - 1.0) * 100.0
        cum_ret.index = pd.RangeIndex(start=1, stop=1 + len(cum_ret), step=1)
        cum_ret.loc[0] = 0.0
        cum_ret = cum_ret.sort_index()
        raw_paths[str(p)] = cum_ret

    if not raw_paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_days = max(int(s.index.max()) for s in raw_paths.values())
    full_index = pd.RangeIndex(0, max_days + 1)

    df = pd.DataFrame(index=full_index)
    for k, s in raw_paths.items():
        df[k] = s.reindex(full_index).ffill()

    avg_path = df.mean(axis=1)
    return df, avg_path

def _year_month_path(prices: pd.Series, year: int, month_int: int) -> pd.Series:
    month_prices = prices.loc[
        (prices.index.year == int(year)) & (prices.index.month == int(month_int))
    ]
    previous_period = pd.Period(year=int(year), month=int(month_int), freq="M") - 1
    previous_prices = prices.loc[
        (prices.index.year == previous_period.year)
        & (prices.index.month == previous_period.month)
    ]
    if month_prices.empty or previous_prices.empty:
        return pd.Series(dtype=float)
    path = (month_prices / float(previous_prices.iloc[-1]) - 1.0) * 100.0
    path.index = pd.RangeIndex(start=1, stop=1 + len(path), step=1)
    path.loc[0] = 0.0
    return path.sort_index()

def _avg_calendar_day_for_ordinal_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    ordinal: int,
) -> Optional[int]:
    month_periods = filtered_halves.loc[
        filtered_halves["month"] == month_int
    ].index.tolist()
    if not month_periods or ordinal <= 0:
        return None

    days = []
    for p in month_periods:
        m = prices.loc[(prices.index.year == p.year) & (prices.index.month == p.month)]
        if len(m) >= ordinal:
            days.append(m.index[ordinal - 1].day)

    if not days:
        return None

    return int(round(np.mean(days)))

def build_intra_month_summary(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    symbol_shown: str,
    comparison_year: Optional[int] = None,
) -> Dict[str, Any]:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(
        prices, filtered_halves, month_int
    )

    summary: Dict[str, Any] = {
        "ok": False,
        "symbol": symbol_shown,
        "month_int": month_int,
        "month_label": MONTH_LABELS[month_int - 1],
        "sample_months": 0,
        "sample_years": 0,
        "avg_month_end": np.nan,
        "avg_low_day": None,
        "avg_low_val": np.nan,
        "avg_low_dom": None,
        "avg_high_day": None,
        "avg_high_val": np.nan,
        "avg_high_dom": None,
        "day5_val": np.nan,
        "day10_val": np.nan,
        "day15_val": np.nan,
        "day20_val": np.nan,
        "front_loaded": None,
        "recovery_strength": np.nan,
        "comparison_year": int(comparison_year or _today().year),
        "comparison_year_month_available": False,
        "comparison_year_end": np.nan,
        "comparison_vs_hist_end_gap": np.nan,
        "avg_path": avg_sel,
        "df_paths": df_sel,
    }

    month_filtered = filtered_halves.loc[filtered_halves["month"] == month_int]
    if not month_filtered.empty:
        summary["sample_months"] = int(month_filtered.shape[0])
        summary["sample_years"] = int(month_filtered["year"].nunique())

    if avg_sel.empty or df_sel.empty:
        return summary

    summary["ok"] = True
    summary["avg_month_end"] = float(avg_sel.iloc[-1])

    avg_ex = avg_sel.copy()
    if 0 in avg_ex.index:
        avg_ex = avg_ex.drop(index=0)

    if not avg_ex.empty:
        low_idx = int(avg_ex.idxmin())
        high_idx = int(avg_ex.idxmax())

        summary["avg_low_day"] = low_idx
        summary["avg_low_val"] = float(avg_ex.loc[low_idx])
        summary["avg_low_dom"] = _avg_calendar_day_for_ordinal_from_filtered(
            prices, filtered_halves, month_int, low_idx
        )

        summary["avg_high_day"] = high_idx
        summary["avg_high_val"] = float(avg_ex.loc[high_idx])
        summary["avg_high_dom"] = _avg_calendar_day_for_ordinal_from_filtered(
            prices, filtered_halves, month_int, high_idx
        )

    for k in [5, 10, 15, 20]:
        if k in avg_sel.index:
            summary[f"day{k}_val"] = float(avg_sel.loc[k])

    d10 = summary["day10_val"]
    month_end = summary["avg_month_end"]
    summary["front_loaded"] = bool(
        pd.notna(d10) and pd.notna(month_end) and abs(d10) >= 0.6 * abs(month_end)
    )

    if summary["avg_low_day"] is not None and pd.notna(month_end):
        summary["recovery_strength"] = float(month_end - summary["avg_low_val"])

    comparison_path = _year_month_path(
        prices, int(summary["comparison_year"]), month_int
    )
    if not comparison_path.empty:
        summary["comparison_year_month_available"] = True
        summary["comparison_year_end"] = float(comparison_path.iloc[-1])
        if pd.notna(summary["avg_month_end"]):
            summary["comparison_vs_hist_end_gap"] = float(
                summary["comparison_year_end"] - summary["avg_month_end"]
            )

    return summary
