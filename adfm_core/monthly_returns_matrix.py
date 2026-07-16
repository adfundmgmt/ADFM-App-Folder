"""Monthly return matrix used by the Monthly Seasonality Explorer.

The renderer keeps the year-by-month return tape separate from the filtered
seasonality sample: calendar rows show realized close-to-close returns, while
seasonality averages and the live comparison respect the active sample.
"""

from __future__ import annotations

from html import escape
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
POSITIVE = "56, 102, 73"
NEGATIVE = "145, 88, 63"
NEUTRAL = "148, 163, 184"


def _finite(values: Iterable[object]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    return series[np.isfinite(series)].to_numpy(dtype=float)


def _compound_pct(values: Iterable[object]) -> float:
    clean = _finite(values)
    if clean.size == 0:
        return np.nan
    return float((np.prod(1.0 + clean / 100.0) - 1.0) * 100.0)


def _fmt_pct(value: object, digits: int = 2, plus: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    prefix = "+" if plus else ""
    return f"{float(value):{prefix}.{digits}f}%"


def _tone(value: object, scale: float) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return f"rgba({NEUTRAL}, 0.055)", "muted"

    numeric = float(value)
    intensity = min(abs(numeric) / max(scale, 0.01), 1.0)
    alpha = 0.10 + 0.66 * intensity
    rgb = POSITIVE if numeric > 0 else NEGATIVE if numeric < 0 else NEUTRAL
    text_class = "strong" if intensity >= 0.55 else ""
    return f"rgba({rgb}, {alpha:.3f})", text_class


def _sparkline_svg(monthly_returns: Iterable[object], positive: bool) -> str:
    values = _finite(monthly_returns)
    if values.size == 0:
        return ""

    cumulative = (np.cumprod(1.0 + values / 100.0) - 1.0) * 100.0
    if cumulative.size == 1:
        cumulative = np.array([0.0, float(cumulative[0])])

    width, height, pad = 88.0, 25.0, 2.0
    x = np.linspace(pad, width - pad, cumulative.size)
    lo = float(np.nanmin(cumulative))
    hi = float(np.nanmax(cumulative))
    span = max(hi - lo, 1e-9)
    y = height - pad - ((cumulative - lo) / span) * (height - 2.0 * pad)
    points = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(x, y, strict=True))
    stroke = "#4f765f" if positive else "#a06452"
    return (
        f"<svg class='monthly-spark' viewBox='0 0 {int(width)} {int(height)}' "
        f"preserveAspectRatio='none' aria-hidden='true'>"
        f"<polyline points='{points}' fill='none' stroke='{stroke}' stroke-width='1.55' "
        "stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


def _prepare_month_table(months: pd.DataFrame) -> pd.DataFrame:
    if months is None or months.empty:
        return pd.DataFrame(columns=["year", "month", "total_ret"])

    out = months.copy()
    if not isinstance(out.index, pd.PeriodIndex):
        try:
            out.index = pd.PeriodIndex(out.index, freq="M")
        except Exception:
            return pd.DataFrame(columns=["year", "month", "total_ret"])

    if "year" not in out:
        out["year"] = out.index.year
    if "month" not in out:
        out["month"] = out.index.month
    if "total_ret" not in out:
        out["total_ret"] = np.nan

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out["total_ret"] = pd.to_numeric(out["total_ret"], errors="coerce")
    return out.dropna(subset=["year", "month"]).sort_index()


def _sample_month_stats(stats: pd.DataFrame, month: int) -> tuple[float, float]:
    if stats is None or stats.empty or month not in stats.index:
        return np.nan, np.nan
    row = stats.loc[month]
    return float(row.get("mean_total", np.nan)), float(row.get("hit_rate", np.nan))


def _display_years(
    months: pd.DataFrame, years_to_show: int, as_of: pd.Timestamp
) -> list[int]:
    available = set(
        pd.to_numeric(months.get("year", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    latest_year = int(as_of.year)
    preferred = list(range(latest_year, latest_year - years_to_show, -1))
    selected = [year for year in preferred if year in available or year == latest_year]
    if len(selected) < years_to_show:
        remaining = sorted(
            (year for year in available if year not in selected), reverse=True
        )
        selected.extend(remaining[: years_to_show - len(selected)])
    return selected[:years_to_show]


def _seasonality_summary(stats: pd.DataFrame) -> dict[str, object]:
    if stats is None or stats.empty or "mean_total" not in stats:
        return {}
    valid = stats.dropna(subset=["mean_total"])
    if valid.empty:
        return {}
    best_month = int(valid["mean_total"].idxmax())
    worst_month = int(valid["mean_total"].idxmin())
    best_mean, best_hit = _sample_month_stats(stats, best_month)
    worst_mean, worst_hit = _sample_month_stats(stats, worst_month)
    return {
        "best_month": best_month,
        "best_mean": best_mean,
        "best_hit": best_hit,
        "worst_month": worst_month,
        "worst_mean": worst_mean,
        "worst_hit": worst_hit,
    }


def _build_matrix_html(
    months: pd.DataFrame,
    stats: pd.DataFrame,
    years: list[int],
    current_period: pd.Period,
    selected_month: int | None = None,
    selected_year: int | None = None,
    average_label: str = "FILTER AVG",
) -> tuple[str, np.ndarray]:
    visible = months.loc[months["year"].astype(int).isin(years)].copy()
    visible_values = _finite(visible["total_ret"])
    scale = (
        float(np.nanpercentile(np.abs(visible_values), 82))
        if visible_values.size
        else 5.0
    )
    scale = max(scale, 1.0)

    header = "".join(
        f"<th class='{'is-selected-month' if month == selected_month else ''}'>{label}</th>"
        for month, label in enumerate(MONTH_LABELS, start=1)
    )
    rows: list[str] = []

    avg_cells: list[str] = []
    for month in range(1, 13):
        value, hit_rate = _sample_month_stats(stats, month)
        bg, text_class = _tone(value, scale)
        title = f"{MONTH_LABELS[month - 1]} sample average: {_fmt_pct(value)}; hit rate: {_fmt_pct(hit_rate, 0, plus=False)}"
        selected_class = " is-selected-month" if month == selected_month else ""
        avg_cells.append(
            f"<td class='monthly-cell {text_class}{selected_class}' style='background:{bg}' title='{escape(title)}'>"
            f"{_fmt_pct(value, 2, plus=False)}</td>"
        )
    average_values = [_sample_month_stats(stats, month)[0] for month in range(1, 13)]
    average_year_return = _compound_pct(average_values)
    average_spark = _sparkline_svg(average_values, bool(average_year_return >= 0))
    rows.append(
        "<tr class='monthly-average-row'>"
        f"<th class='monthly-row-label'><span>{escape(average_label)}</span></th>"
        + "".join(avg_cells)
        + f"<td class='monthly-year-cell {'is-positive' if average_year_return >= 0 else 'is-negative'}'>"
        f"{average_spark}<div class='monthly-year-return'>{_fmt_pct(average_year_return, 1)}</div>"
        "<div class='monthly-year-caption'>avg profile</div></td></tr>"
    )

    for year in years:
        year_data = months.loc[months["year"].astype(int).eq(year)].copy()
        by_month = year_data.groupby(year_data["month"].astype(int))["total_ret"].last()
        year_values = [by_month.get(month, np.nan) for month in range(1, 13)]
        year_return = _compound_pct(year_values)
        year_cells: list[str] = []

        for month, value in enumerate(year_values, start=1):
            period = pd.Period(year=year, month=month, freq="M")
            is_now = period == current_period
            bg, text_class = _tone(value, scale)
            badge = (
                "<span class='monthly-now-badge'>NOW</span>"
                if is_now and pd.notna(value)
                else ""
            )
            title = f"{MONTH_LABELS[month - 1]} {year}: {_fmt_pct(value)}"
            now_class = " is-now" if is_now and pd.notna(value) else ""
            selected_class = " is-selected-month" if month == selected_month else ""
            display_value = "" if pd.isna(value) else _fmt_pct(value, 2, plus=False)
            year_cells.append(
                f"<td class='monthly-cell {text_class}{now_class}{selected_class}' style='background:{bg}' title='{escape(title)}'>"
                f"{badge}<span>{display_value}</span></td>"
            )

        spark = _sparkline_svg(
            year_values, bool(pd.notna(year_return) and year_return >= 0)
        )
        year_caption = (
            "YTD"
            if year == current_period.year and current_period.month < 12
            else "calendar return"
        )
        selected_year_class = " class='is-selected-year'" if year == selected_year else ""
        rows.append(
            f"<tr{selected_year_class}>"
            f"<th class='monthly-row-label'>{year}</th>"
            + "".join(year_cells)
            + f"<td class='monthly-year-cell {'is-positive' if year_return >= 0 else 'is-negative'}'>"
            f"{spark}<div class='monthly-year-return'>{_fmt_pct(year_return, 1)}</div>"
            f"<div class='monthly-year-caption'>{year_caption}</div></td></tr>"
        )

    html = (
        "<div class='monthly-grid-wrap'><table class='monthly-grid'>"
        f"<thead><tr><th class='monthly-row-label'></th>{header}<th class='monthly-year-head'>YEAR</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return html, visible_values


def render_monthly_returns_matrix(
    months: pd.DataFrame,
    stats: pd.DataFrame,
    years: list[int],
    current_period: pd.Period,
    selected_month: int,
    selected_year: int,
    average_label: str = "FILTER AVG",
) -> None:
    """Render the compact terminal-style matrix used by the coordinated page."""

    months = _prepare_month_table(months)
    matrix_html, _ = _build_matrix_html(
        months,
        stats,
        years,
        current_period,
        selected_month=selected_month,
        selected_year=selected_year,
        average_label=average_label,
    )
    st.markdown(
        """
        <style>
        .monthly-grid-wrap {overflow-x:auto; border:1px solid rgba(100,116,139,.16); border-radius:7px; margin:.2rem 0 .45rem; scrollbar-width:thin;}
        .monthly-grid {width:100%; min-width:780px; border-collapse:separate; border-spacing:1px; background:rgba(100,116,139,.12); table-layout:fixed;}
        .monthly-grid th,.monthly-grid td {height:44px; padding:3px; text-align:center; vertical-align:middle; position:relative;}
        .monthly-grid thead th {height:31px; font-size:.60rem; letter-spacing:.14em; text-transform:uppercase; font-weight:680; color:#64748b; background:var(--background-color, #fff);}
        .monthly-grid .monthly-row-label {width:72px; text-align:left; padding-left:9px; font-size:.74rem; font-weight:780; letter-spacing:.015em; background:var(--background-color, #fff); color:inherit;}
        .monthly-grid .monthly-row-label span {font-size:.57rem; letter-spacing:.07em; color:#4f765f;}
        .monthly-grid .monthly-cell {font-size:.69rem; font-variant-numeric:tabular-nums; color:inherit; transition:filter .12s ease, transform .12s ease;}
        .monthly-grid .monthly-cell:hover {filter:brightness(1.08); transform:translateY(-1px); z-index:4;}
        .monthly-grid .monthly-cell.strong {color:#f8fafc; font-weight:760; text-shadow:0 1px 1px rgba(15,23,42,.22);}
        .monthly-grid .monthly-cell.muted {color:#94a3b8;}
        .monthly-grid .monthly-cell.is-now {box-shadow:inset 0 0 0 2px rgba(15,23,42,.72); z-index:3;}
        .monthly-now-badge {position:absolute; top:2px; right:3px; font-size:.46rem; line-height:1; letter-spacing:.08em; padding:2px 4px; border-radius:2px; background:rgba(15,23,42,.78); color:#f8fafc;}
        .monthly-year-head {width:94px; background:rgba(79,118,95,.20)!important; color:#355845!important;}
        .monthly-year-cell {width:94px; background:var(--background-color, #fff); padding:2px 6px!important;}
        .monthly-year-cell.is-positive {color:#4f765f;}
        .monthly-year-cell.is-negative {color:#a06452;}
        .monthly-spark {display:block; width:100%; height:20px; margin:0 auto -2px; opacity:.92;}
        .monthly-year-return {font-size:.72rem; line-height:1.05; font-weight:760; font-variant-numeric:tabular-nums;}
        .monthly-year-caption {font-size:.48rem; line-height:1.05; letter-spacing:.08em; text-transform:uppercase; color:#64748b; margin-top:2px;}
        .monthly-average-row td,.monthly-average-row th {border-bottom:1px solid rgba(100,116,139,.38);}
        .monthly-grid .is-selected-month {box-shadow:inset 2px 0 rgba(53,88,69,.88), inset -2px 0 rgba(53,88,69,.88); z-index:2;}
        .monthly-grid thead .is-selected-month {color:#355845; background:rgba(79,118,95,.16); box-shadow:inset 2px 0 rgba(53,88,69,.88), inset -2px 0 rgba(53,88,69,.88), inset 0 2px rgba(53,88,69,.88);}
        .monthly-grid tr.is-selected-year > * {border-top:2px solid rgba(53,88,69,.9); border-bottom:2px solid rgba(53,88,69,.9);}
        .monthly-grid tr.is-selected-year > .monthly-row-label {color:#355845; background:rgba(79,118,95,.10);}
        .monthly-grid tr.is-selected-year > .is-selected-month {outline:2px solid #243c2f; outline-offset:-3px; z-index:3;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(matrix_html, unsafe_allow_html=True)


def build_monthly_returns_frame(
    months: pd.DataFrame,
    stats: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Build the selectable year-by-month matrix used by the page controller."""

    months = _prepare_month_table(months)
    rows: list[list[float]] = []
    labels: list[str] = []

    average_row = [_sample_month_stats(stats, month)[0] for month in range(1, 13)]
    rows.append(average_row + [float(np.nanmean(_finite(average_row)))])
    labels.append("FILTER AVG")

    for year in years:
        year_data = months.loc[months["year"].astype(int).eq(int(year))]
        by_month = year_data.groupby(year_data["month"].astype(int))["total_ret"].last()
        values = [float(by_month.get(month, np.nan)) for month in range(1, 13)]
        rows.append(values + [_compound_pct(values)])
        labels.append(str(int(year)))

    frame = pd.DataFrame(rows, index=labels, columns=[*MONTH_LABELS, "YEAR / YTD"])
    frame.index.name = "Sample / Year"
    return frame


def monthly_returns_snapshot(
    months: pd.DataFrame,
    stats: pd.DataFrame,
    years: list[int],
    current_period: pd.Period,
    current_months: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    """Return the headline metrics shared by the matrix and companion charts."""

    months = _prepare_month_table(months)
    current_source = _prepare_month_table(
        current_months if current_months is not None else months
    )
    visible = months.loc[months["year"].astype(int).isin([int(year) for year in years])]
    visible_values = _finite(visible.get("total_ret", []))
    summary = _seasonality_summary(stats)
    current_row = current_source.loc[current_source.index == current_period]
    current_value = (
        float(current_row["total_ret"].iloc[-1]) if not current_row.empty else np.nan
    )
    current_average, _ = _sample_month_stats(stats, int(current_period.month))

    return {
        "best_month": int(summary["best_month"]) if summary else None,
        "best_mean": float(summary["best_mean"]) if summary else np.nan,
        "best_hit": float(summary["best_hit"]) if summary else np.nan,
        "worst_month": int(summary["worst_month"]) if summary else None,
        "worst_mean": float(summary["worst_mean"]) if summary else np.nan,
        "worst_hit": float(summary["worst_hit"]) if summary else np.nan,
        "hit_rate": float(np.mean(visible_values > 0.0) * 100.0)
        if visible_values.size
        else np.nan,
        "monthly_average": float(np.mean(visible_values))
        if visible_values.size
        else np.nan,
        "current_value": current_value,
        "current_average": current_average,
        "current_gap": current_value - current_average
        if pd.notna(current_value) and pd.notna(current_average)
        else np.nan,
    }


def render_monthly_returns_lens(
    all_months: pd.DataFrame,
    filtered_months: pd.DataFrame,
    stats: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
) -> None:
    """Render a compact year-by-month return lens above the legacy charts."""

    all_months = _prepare_month_table(all_months)
    filtered_months = _prepare_month_table(filtered_months)
    as_of = (
        pd.Timestamp(as_of).tz_localize(None)
        if getattr(pd.Timestamp(as_of), "tzinfo", None)
        else pd.Timestamp(as_of)
    )
    current_period = as_of.to_period("M")

    st.markdown(
        """
        <style>
        .monthly-lens-title {font-size:1.42rem; font-weight:760; letter-spacing:-.025em; margin:.15rem 0 .05rem; color:inherit;}
        .monthly-lens-subtitle {font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:#64748b; margin-bottom:.45rem;}
        .monthly-banner {border:1px solid rgba(100,116,139,.18); border-left:3px solid #4f765f; border-radius:6px; background:rgba(100,116,139,.045); padding:9px 12px; margin:.45rem 0 .55rem; font-size:.89rem; line-height:1.45;}
        .monthly-banner.now {border-left-color:#cbd5e1; background:rgba(100,116,139,.025);}
        .monthly-kicker {font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; font-weight:800; color:#4f765f; margin-right:10px;}
        .monthly-banner.now .monthly-kicker {color:inherit; background:rgba(100,116,139,.15); padding:3px 7px; border-radius:2px;}
        .monthly-muted {color:#64748b;}
        .monthly-negative {color:#a06452; font-weight:760;}
        .monthly-positive {color:#4f765f; font-weight:760;}
        .monthly-kpi-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid rgba(100,116,139,.16); border-radius:7px; overflow:hidden; margin:.55rem 0 .7rem;}
        .monthly-kpi {padding:10px 13px 11px; min-height:62px; border-right:1px solid rgba(100,116,139,.14);}
        .monthly-kpi:last-child {border-right:0;}
        .monthly-kpi-label {font-size:.61rem; letter-spacing:.14em; text-transform:uppercase; color:#64748b; margin-bottom:5px;}
        .monthly-kpi-value {font-size:1.13rem; line-height:1; font-weight:780; color:inherit;}
        .monthly-grid-wrap {overflow-x:auto; border:1px solid rgba(100,116,139,.16); border-radius:7px; margin:.2rem 0 .45rem; scrollbar-width:thin;}
        .monthly-grid {width:100%; min-width:780px; border-collapse:separate; border-spacing:1px; background:rgba(100,116,139,.12); table-layout:fixed;}
        .monthly-grid th,.monthly-grid td {height:44px; padding:3px; text-align:center; vertical-align:middle; position:relative;}
        .monthly-grid thead th {height:31px; font-size:.60rem; letter-spacing:.14em; text-transform:uppercase; font-weight:680; color:#64748b; background:var(--background-color, #fff);}
        .monthly-grid .monthly-row-label {width:72px; text-align:left; padding-left:9px; font-size:.74rem; font-weight:780; letter-spacing:.015em; background:var(--background-color, #fff); color:inherit;}
        .monthly-grid .monthly-row-label span {font-size:.57rem; letter-spacing:.07em; color:#4f765f;}
        .monthly-grid .monthly-cell {font-size:.69rem; font-variant-numeric:tabular-nums; color:inherit; transition:filter .12s ease, transform .12s ease;}
        .monthly-grid .monthly-cell:hover {filter:brightness(1.08); transform:translateY(-1px); z-index:2;}
        .monthly-grid .monthly-cell.strong {color:#f8fafc; font-weight:760; text-shadow:0 1px 1px rgba(15,23,42,.22);}
        .monthly-grid .monthly-cell.muted {color:#94a3b8;}
        .monthly-grid .monthly-cell.is-now {box-shadow:inset 0 0 0 2px rgba(15,23,42,.66); z-index:1;}
        .monthly-now-badge {position:absolute; top:2px; right:3px; font-size:.46rem; line-height:1; letter-spacing:.08em; padding:2px 4px; border-radius:2px; background:rgba(15,23,42,.72); color:#f8fafc;}
        .monthly-year-head {width:94px; background:rgba(79,118,95,.20)!important; color:#355845!important;}
        .monthly-year-cell {width:94px; background:var(--background-color, #fff); padding:2px 6px!important;}
        .monthly-spark {display:block; width:100%; height:20px; margin:0 auto -2px; opacity:.92;}
        .monthly-year-return {font-size:.72rem; line-height:1.05; font-weight:760; font-variant-numeric:tabular-nums;}
        .monthly-year-caption {font-size:.48rem; line-height:1.05; letter-spacing:.08em; text-transform:uppercase; color:#64748b; margin-top:2px;}
        .monthly-average-row td,.monthly-average-row th {border-bottom:1px solid rgba(226,232,240,.75);}
        .monthly-lens-caption {font-size:.73rem; line-height:1.4; color:#64748b; margin-bottom:.9rem;}
        @media (max-width:900px) {.monthly-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.monthly-kpi:nth-child(2){border-right:0}.monthly-kpi:nth-child(-n+2){border-bottom:1px solid rgba(100,116,139,.14);}}
        @media (prefers-color-scheme:dark) {
          .monthly-lens-subtitle,.monthly-muted,.monthly-kpi-label,.monthly-grid thead th,.monthly-year-caption,.monthly-lens-caption{color:#94a3b8;}
          .monthly-grid thead th,.monthly-grid .monthly-row-label,.monthly-year-cell{background:#111827;}
          .monthly-grid-wrap{border-color:rgba(148,163,184,.22)}
          .monthly-year-head{background:rgba(79,118,95,.34)!important;color:#d5e6db!important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_col, window_col = st.columns([4.5, 1.35], vertical_alignment="bottom")
    with title_col:
        st.markdown(
            f"<div class='monthly-lens-title'>Monthly Returns</div>"
            f"<div class='monthly-lens-subtitle'>{escape(symbol)} · close-to-close · hover, scan, compare</div>",
            unsafe_allow_html=True,
        )
    with window_col:
        window = st.segmented_control(
            "Displayed history",
            options=["5Y", "10Y", "20Y"],
            default="10Y",
            label_visibility="collapsed",
            key="monthly_returns_display_window",
        )
    years_to_show = int(str(window or "10Y").replace("Y", ""))

    summary = _seasonality_summary(stats)
    if summary:
        st.markdown(
            "<div class='monthly-banner'>"
            "<span class='monthly-kicker'>Seasonality</span>"
            f"Strongest {MONTH_LABELS[int(summary['best_month']) - 1]}: {_fmt_pct(summary['best_mean'])} avg · "
            f"{_fmt_pct(summary['best_hit'], 0, plus=False)} hit. "
            f"Weakest {MONTH_LABELS[int(summary['worst_month']) - 1]}: {_fmt_pct(summary['worst_mean'])} avg · "
            f"{_fmt_pct(summary['worst_hit'], 0, plus=False)} hit."
            "</div>",
            unsafe_allow_html=True,
        )

    current_row = all_months.loc[all_months.index == current_period]
    current_value = (
        float(current_row["total_ret"].iloc[-1]) if not current_row.empty else np.nan
    )
    current_avg, _ = _sample_month_stats(stats, current_period.month)
    current_gap = (
        current_value - current_avg
        if pd.notna(current_value) and pd.notna(current_avg)
        else np.nan
    )
    if pd.notna(current_value):
        gap_class = "monthly-positive" if current_gap >= 0 else "monthly-negative"
        gap_text = f"{abs(current_gap):.2f}pp {'above' if current_gap >= 0 else 'below'} the sample average"
        st.markdown(
            "<div class='monthly-banner now'>"
            "<span class='monthly-kicker'>Now</span>"
            f"{MONTH_LABELS[current_period.month - 1]} {current_period.year} · <b>{_fmt_pct(current_value)} MTD</b> · "
            f"sample {MONTH_LABELS[current_period.month - 1]} avg {_fmt_pct(current_avg)} · "
            f"<span class='{gap_class}'>{gap_text}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    years = _display_years(all_months, years_to_show, as_of)
    matrix_html, visible_values = _build_matrix_html(
        all_months, stats, years, current_period
    )
    if visible_values.size:
        best_value = float(np.nanmax(visible_values))
        worst_value = float(np.nanmin(visible_values))
        hit_rate = float(np.mean(visible_values > 0.0) * 100.0)
        avg_month = float(np.mean(visible_values))
    else:
        best_value = worst_value = hit_rate = avg_month = np.nan

    st.markdown(
        "<div class='monthly-kpi-grid'>"
        f"<div class='monthly-kpi'><div class='monthly-kpi-label'>Best month</div><div class='monthly-kpi-value monthly-positive'>{_fmt_pct(best_value)}</div></div>"
        f"<div class='monthly-kpi'><div class='monthly-kpi-label'>Worst month</div><div class='monthly-kpi-value monthly-negative'>{_fmt_pct(worst_value)}</div></div>"
        f"<div class='monthly-kpi'><div class='monthly-kpi-label'>Hit rate</div><div class='monthly-kpi-value'>{_fmt_pct(hit_rate, 0, plus=False)}</div></div>"
        f"<div class='monthly-kpi'><div class='monthly-kpi-label'>{years_to_show}Y monthly avg</div><div class='monthly-kpi-value'>{_fmt_pct(avg_month)}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(matrix_html, unsafe_allow_html=True)
    filtered_obs = int(filtered_months.shape[0])
    filtered_years = (
        int(filtered_months["year"].nunique()) if not filtered_months.empty else 0
    )
    st.markdown(
        f"<div class='monthly-lens-caption'>Calendar rows show realized monthly returns and compounded year/YTD paths. "
        f"The FILTER AVG row and the live comparison use the active seasonality filters: {filtered_obs} month observations across "
        f"{filtered_years} distinct years. Partial current-month data is marked NOW.</div>",
        unsafe_allow_html=True,
    )


__all__ = [
    "build_monthly_returns_frame",
    "monthly_returns_snapshot",
    "render_monthly_returns_matrix",
    "render_monthly_returns_lens",
    "_compound_pct",
    "_display_years",
    "_prepare_month_table",
]

