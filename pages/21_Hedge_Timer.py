# Hedge Timer
# High-recall drawdown warning system for ^SPX and ^NDX.

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from adfm_core.hedge_timer_data import load_hedge_inputs
from adfm_core.hedge_timer_model import (
    CALIBRATION_START,
    CONFIRM_COMPONENTS,
    CONFIRM_THRESHOLD,
    FROZEN_WATCH_THRESHOLD,
    HORIZON_DAYS,
    LEAD_LOOKBACK,
    MODEL_FIT_END,
    NDX_LABEL,
    NDX_TICKER,
    SPX_LABEL,
    SPX_TICKER,
    TICKERS,
    WATCH_COMPONENTS,
    compute_scores,
    drawdown,
    episode_audit,
    forward_stats,
    onset,
    state_label,
    warning_summary,
    watch_signal,
)
from adfm_core.market_data import _last_completed_us_session, fill_short_calendar_gaps
from adfm_core.palette import PASTEL
from adfm_core.ui import (
    PageHeader,
    inject_institutional_tool_finish,
    render_footer,
    render_page_header,
    render_sidebar_about,
)

st.set_page_config(page_title="Hedge Timer", layout="wide")
inject_institutional_tool_finish()

st.markdown(
    f"""
    <style>
    .hedge-state {{
        display: inline-block;
        border: 1px solid #111111;
        padding: .22rem .48rem;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .05em;
    }}
    .hedge-stand {{ background: {PASTEL['sage']}; }}
    .hedge-watch {{ background: {PASTEL['amber']}; }}
    .hedge-confirmed {{ background: {PASTEL['coral']}; }}
    .hedge-short {{ background: {PASTEL['rose']}; }}
    .hedge-unavailable {{ background: #eeeeee; color: #555555; }}
    .hedge-index-title {{
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.18rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }}
    .hedge-line {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: .80rem;
        line-height: 1.55;
        margin-top: .45rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams["figure.dpi"] = 180
LOOKBACK_OPTIONS = [1, 2, 3, 5, 10]


with st.sidebar:
    render_sidebar_about("21_Hedge_Timer.py")
    st.markdown(
        "**Scoring model**\n\n"
        "Hedge Watch prioritizes early deterioration in credit, breadth, defensive rotation, "
        "volatility acceleration, drawdown velocity, and daily/weekly momentum. Hedge Confirmed "
        "adds price structure, realized-volatility expansion, and longer-term trend confirmation. "
        "RSI and 63-session drawdown gates apply only to fresh short initiation, not to the warning itself."
    )
    st.divider()
    chart_index = st.radio(
        "Chart index",
        options=[SPX_LABEL, NDX_LABEL],
        index=0,
    )
    chart_years = st.radio(
        "Chart lookback",
        options=LOOKBACK_OPTIONS,
        index=4,
        format_func=lambda value: f"{value} year" if value == 1 else f"{value} years",
    )
    audit_basis = st.radio("Drawdown basis", ["Intraday high / low", "Daily close"], index=0)
    st.divider()
    st.markdown("### Early capture since 2020")
    sanity_box = st.empty()


def _today() -> date:
    return date.today()


def _start_date() -> date:
    return _today() - timedelta(days=int(10 * 365.25) + 180)


def _sessions_for_years(years: int) -> int:
    return int(round(252 * years))


def latest_value(series: pd.Series) -> float:
    return float(series.reindex(df.index).iloc[-1]) if len(df) else float("nan")


def last_bool(series: pd.Series, default: bool = False) -> bool:
    value = series.reindex(df.index).iloc[-1] if len(df) else np.nan
    return bool(value) if pd.notna(value) else default


def fmt_num(value: float, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value * 100:.{decimals}f}%"


def state_css(label: str) -> str:
    return {
        "STAND DOWN": "hedge-stand",
        "HEDGE WATCH": "hedge-watch",
        "HEDGE CONFIRMED": "hedge-confirmed",
        "SHORT ALLOWED": "hedge-short",
    }.get(label, "hedge-unavailable")


def format_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Peak", "3% deadline", "Start", "End", "Trough", "First warning"):
        out[column] = out[column].map(
            lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "No warning"
        )
    for column in ("Depth", "Loss at warning"):
        out[column] = out[column].map(lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "No warning")
    out["Lead sessions"] = out["Lead sessions"].map(
        lambda value: int(value) if pd.notna(value) else "No"
    )
    out["Captured"] = out["Captured"].map(lambda value: "Yes" if bool(value) else "No")
    return out


def plot_index(
    price: pd.Series,
    watch_score: pd.Series,
    confirm_score: pd.Series,
    meta: dict[str, pd.Series],
    watch_threshold: float,
    label: str,
    years: int,
    episode: pd.Series | None = None,
) -> plt.Figure:
    frame = pd.DataFrame({"Price": price}).dropna(subset=["Price"])
    if episode is not None:
        price_index = frame.index
        left = max(0, int(price_index.get_indexer([episode["Peak"]], method="nearest")[0]) - LEAD_LOOKBACK - 5)
        right = min(len(price_index), int(price_index.get_indexer([episode["Trough"]], method="nearest")[0]) + 21)
        frame = frame.iloc[left:right].copy()
    elif len(frame) > _sessions_for_years(years):
        frame = frame.iloc[-_sessions_for_years(years) :].copy()

    fig, ax_price = plt.subplots(figsize=(13.5, 5.4))

    x = np.arange(len(frame))
    idx = frame.index
    ma50 = meta["ma50"].reindex(idx)
    ma200 = meta["ma200"].reindex(idx)

    ax_price.plot(x, frame["Price"].values, linewidth=2.0, color="#111111", label="Price")
    ax_price.plot(x, ma50.values, linewidth=1.3, color=PASTEL["blue"], label="MA50")
    ax_price.plot(x, ma200.values, linewidth=1.25, color=PASTEL["lavender"], label="MA200")

    confirmed_onsets = onset((watch_signal(watch_score, watch_threshold) & (confirm_score >= CONFIRM_THRESHOLD)).fillna(False)).reindex(idx)
    if confirmed_onsets.any():
        ax_price.scatter(
            x[confirmed_onsets.values],
            frame["Price"].values[confirmed_onsets.values],
            marker="o",
            s=42,
            color=PASTEL["rose"],
            edgecolors="#111111",
            linewidths=.45,
            label="Confirmed",
            zorder=6,
        )

    ax_price.grid(True, linewidth=.6, alpha=.14)
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_price.legend(loc="upper left", frameon=False, ncol=4, fontsize=8.5)

    tick_count = 8 if years <= 2 else 10
    tick_positions = np.linspace(0, max(len(frame) - 1, 0), min(tick_count, len(frame)), dtype=int)
    tick_positions = np.unique(tick_positions)
    tick_labels = [
        idx[position].strftime("%b %Y") if years <= 3 else idx[position].strftime("%Y-%m")
        for position in tick_positions
    ]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, fontsize=8)

    period = f"{years} Year" if years == 1 else f"{years} Years"
    fig.suptitle(f"{label} Hedge Timer | {episode['Peak'].date().isoformat() if episode is not None else period}", fontsize=14, fontweight="bold", y=.99)
    fig.subplots_adjust(left=.06, right=.99, bottom=.07, top=.93)
    return fig


render_page_header(
    PageHeader(
        title="Hedge Timer",
        description=(
            "High-recall drawdown warning system for the S&P 500 and Nasdaq-100. "
            "Warnings qualify before the peak or within the first 3% of a decline. "
            "Rules are fitted on SPX since 2020 and applied unchanged to NDX; "
            "confirmation and anti-bottom-short gates determine whether a fresh directional short is allowed."
        ),
        eyebrow="ADFM Risk + Execution",
    )
)

df0, expected_session, input_status = load_hedge_inputs(TICKERS, _start_date(), research_snapshot_path=Path(__file__).resolve().parents[1] / "data/hedge_timer")
if df0.empty or not df0[SPX_TICKER].notna().any() or not df0[NDX_TICKER].notna().any():
    st.error("Yahoo Finance did not return usable S&P 500 and Nasdaq-100 data.")
    st.stop()

base_idx = df0[SPX_TICKER].dropna().index.intersection(df0[NDX_TICKER].dropna().index)
if base_idx.empty:
    st.error("No matching completed S&P 500 and Nasdaq-100 sessions are available.")
    st.stop()
df = fill_short_calendar_gaps(df0.reindex(base_idx), limit=2)
current_inputs_fresh = bool(input_status["Status"].eq("Current").all() and df.index[-1] == expected_session)
common_sessions = df0.dropna(subset=list(TICKERS)).index
previous_session = _last_completed_us_session(pd.Timestamp(expected_session).tz_localize("America/New_York")) if expected_session is not None else None
signal_inputs_usable = bool(
    len(common_sessions) and previous_session is not None
    and common_sessions[-1] >= previous_session
    and not input_status["Status"].isin(["Last-good cache", "Research snapshot"]).any()
)
if signal_inputs_usable:
    df = df.loc[df.index <= common_sessions[-1]]

watch_spx, confirm_spx, meta_spx, conditions_spx = compute_scores(df, SPX_TICKER)
watch_ndx, confirm_ndx, meta_ndx, conditions_ndx = compute_scores(df, NDX_TICKER)
complete_rows = df.reindex(columns=list(TICKERS)).notna().all(axis=1)
watch_spx = watch_spx.where(complete_rows)
confirm_spx = confirm_spx.where(complete_rows)
watch_ndx = watch_ndx.where(complete_rows)
confirm_ndx = confirm_ndx.where(complete_rows)

watch_threshold = FROZEN_WATCH_THRESHOLD
warning_active_spx = watch_signal(watch_spx, watch_threshold)
warning_active_ndx = watch_signal(watch_ndx, watch_threshold)
warning_on_spx = onset(warning_active_spx)
warning_on_ndx = onset(warning_active_ndx)


def audit_prices(ticker: str) -> pd.Series | pd.DataFrame:
    ranges = [f"{ticker} High", f"{ticker} Low"]
    if audit_basis == "Intraday high / low" and all(key in df0 for key in ranges):
        bars = df0.loc[:, [ticker, *ranges]].rename(columns={ticker: "Close", ranges[0]: "High", ranges[1]: "Low"})
        bars = bars.reindex(df.index)
        required = df[ticker].dropna().loc[lambda price: price.index >= pd.Timestamp(CALIBRATION_START) - pd.Timedelta(days=90)].index
        if bars.reindex(required).notna().all().all():
            return bars
    return df[ticker]


px_audit_spx, px_audit_ndx = audit_prices(SPX_TICKER), audit_prices(NDX_TICKER)
summary_spx = warning_summary(px_audit_spx, warning_on_spx, warning_active=warning_active_spx)
summary_ndx = warning_summary(px_audit_ndx, warning_on_ndx, warning_active=warning_active_ndx)
audit_spx = episode_audit(SPX_LABEL, px_audit_spx, warning_on_spx, warning_active=warning_active_spx)
audit_ndx = episode_audit(NDX_LABEL, px_audit_ndx, warning_on_ndx, warning_active=warning_active_ndx)
audit = pd.concat([audit_spx, audit_ndx], ignore_index=True)

warning_time_spx = warning_active_spx.loc[warning_active_spx.index >= CALIBRATION_START].mean()
warning_time_ndx = warning_active_ndx.loc[warning_active_ndx.index >= CALIBRATION_START].mean()
sanity_box.markdown(
    f"**SPX fit:** {summary_spx['captured']}/{summary_spx['episodes']} early captures  \n"
    f"**NDX same rules:** {summary_ndx['captured']}/{summary_ndx['episodes']} early captures  \n"
    f"**Warning time:** SPX {fmt_pct(warning_time_spx, 0)} · NDX {fmt_pct(warning_time_ndx, 0)}  \n"
    f"**False alarms:** SPX {summary_spx['false_warnings']} · NDX {summary_ndx['false_warnings']}  \n"
    f"**Late / repeat alerts:** SPX {summary_spx['late_warnings']} · NDX {summary_ndx['late_warnings']}  \n"
    f"**Pending:** SPX {summary_spx['pending_warnings']} · NDX {summary_ndx['pending_warnings']}  \n"
    f"**Frozen threshold:** {watch_threshold:.0f}/100  \n\n"
    "SPX fitted through " + MODEL_FIT_END + ". NDX did not influence the fit."
)


def render_index_state(
    label: str,
    ticker: str,
    watch: pd.Series,
    confirm: pd.Series,
    meta: dict[str, pd.Series],
) -> None:
    watch_now = latest_value(watch)
    confirm_now = latest_value(confirm)
    early_now = last_bool(meta["early_stage"], True) and current_inputs_fresh
    oversold_now = last_bool(meta["oversold_block"], False)
    state = state_label(watch_now, confirm_now, watch_threshold, early_now, oversold_now)
    if not signal_inputs_usable:
        state = "Unavailable: stale or missing inputs"
        early_now = False
        watch_now = confirm_now = float("nan")
    css = state_css(state)
    price_now = latest_value(df[ticker])
    current_dd = latest_value(drawdown(df[ticker]))
    rsi_now = latest_value(meta["rsi_d"])
    dd63_now = latest_value(meta["dd63"])
    sector_weak = latest_value(meta["sector_breadth_share"]) if signal_inputs_usable else float("nan")
    rv_ratio = latest_value(meta["realized_vol_ratio"])
    watch_display = f"{fmt_num(watch_now, 0)}/100" if np.isfinite(watch_now) else "NA"
    confirm_display = f"{fmt_num(confirm_now, 0)}/100" if np.isfinite(confirm_now) else "NA"

    st.markdown(
        f"""
        <div class="hedge-index-title">{label}</div>
        <span class="hedge-state {css}">{state}</span>
        <div class="hedge-line">
            Price <b>{fmt_num(price_now, 2)}</b> &nbsp;·&nbsp; Current drawdown <b>{fmt_pct(current_dd)}</b><br>
            Hedge Watch <b>{watch_display}</b> &nbsp;·&nbsp;
            Confirmation <b>{confirm_display}</b><br>
            RSI14 <b>{fmt_num(rsi_now, 1)}</b> &nbsp;·&nbsp;
            63-session drawdown <b>{fmt_pct(dd63_now)}</b><br>
            Sectors below MA50 <b>{fmt_pct(sector_weak, 0)}</b> &nbsp;·&nbsp;
            RV10 / RV63 <b>{fmt_num(rv_ratio, 2)}x</b><br>
            Fresh-short gate: <b>{'Open' if early_now and not oversold_now else 'Blocked'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(f"{'Signals as of' if signal_inputs_usable else 'Index observations through'} {df.index[-1].date().isoformat()}. Latest completed US session: {expected_session.date().isoformat() if expected_session is not None else 'unavailable'}.")
if input_status["Status"].eq("Research snapshot").any():
    st.warning("Live inputs are unavailable. Browsing the dated research snapshot; current scores and fresh shorts remain blocked.")
elif not current_inputs_fresh:
    unavailable_inputs = input_status.loc[input_status["Status"] != "Current"]
    details = ", ".join(unavailable_inputs["Input"])
    message = "Showing the last complete signal; fresh shorts remain blocked." if signal_inputs_usable else "Current scores and fresh shorts are blocked."
    st.warning(f"Awaiting the latest close for {details}. {message} See Input dates for source dates.")

col_spx, col_ndx = st.columns(2)
with col_spx:
    render_index_state(SPX_LABEL, SPX_TICKER, watch_spx, confirm_spx, meta_spx)
with col_ndx:
    render_index_state(NDX_LABEL, NDX_TICKER, watch_ndx, confirm_ndx, meta_ndx)

st.caption(
    f"Hedge Watch threshold {watch_threshold:.0f}/100 was fitted on SPX through {MODEL_FIT_END} and frozen for both indices. "
    f"Confirmation threshold is {CONFIRM_THRESHOLD:.0f}/100. "
    "An oversold or late-stage tape can block a fresh short while Hedge Watch remains active."
)

st.divider()
selected_ticker = SPX_TICKER if chart_index == SPX_LABEL else NDX_TICKER
selected_watch = watch_spx if selected_ticker == SPX_TICKER else watch_ndx
selected_confirm = confirm_spx if selected_ticker == SPX_TICKER else confirm_ndx
selected_meta = meta_spx if selected_ticker == SPX_TICKER else meta_ndx
selected_audit = audit_spx if selected_ticker == SPX_TICKER else audit_ndx
selected_prices = px_audit_spx if selected_ticker == SPX_TICKER else px_audit_ndx
if audit_basis == "Intraday high / low" and isinstance(selected_prices, pd.Series):
    st.warning("Intraday range history is missing or incomplete for this index. Showing a daily-close audit; intraday coverage cannot be certified.")
historical_signal_gaps = int((~complete_rows.loc[complete_rows.index >= CALIBRATION_START]).sum())
if historical_signal_gaps:
    st.warning(f"Historical signal inputs are incomplete on {historical_signal_gaps} sessions. Full early-warning recall cannot be certified.")
choice = st.selectbox("Browse drawdown", options=list(range(len(selected_audit) + 1)), key="hedge_episode",
                      format_func=lambda value: "All history" if value == 0 else
                      f"{selected_audit.iloc[value - 1]['Peak'].date()} → {selected_audit.iloc[value - 1]['Trough'].date()} · {abs(selected_audit.iloc[value - 1]['Depth']) * 100:.1f}% decline")
selected_episode = selected_audit.iloc[choice - 1] if choice else None
if selected_episode is not None:
    warning_date = selected_episode["First warning"]
    warning_date_text = warning_date.date().isoformat() if pd.notna(warning_date) else "none"
    st.caption(f"Audit warning: {warning_date_text} · {selected_episode['Timing']} · "
               f"Loss at warning: {fmt_pct(selected_episode['Loss at warning'])} · "
               f"Early capture: {'Yes' if selected_episode['Captured'] else 'No'}")

figure = plot_index(
    df[selected_ticker],
    selected_watch,
    selected_confirm,
    selected_meta,
    watch_threshold,
    chart_index,
    chart_years,
    selected_episode,
)
st.pyplot(figure, width="stretch")
plt.close(figure)

st.divider()
st.subheader("10%+ drawdown audit since 2020")
st.caption(
    f"A warning must start within {LEAD_LOOKBACK} sessions before the peak, already be active at the peak, "
    "or start before the first loss beyond 3%. A rebound does not reopen that deadline. "
    "Distinct local-peak legs rearm after a 10% rebound from the trough. "
    "An ongoing warning is dated at the peak, or the preceding close when that peak session already breached 3%; this is not a new alert. "
    "Before-peak loss is shown as 0%; lead is measured to the first 10% crossing. "
    "False alarms have 60 completed sessions of follow-up and are separate from late or repeat drawdown alerts. "
    "Historical recall is fitted evidence, not a guarantee of future warnings."
)
if audit.empty:
    st.info("No qualifying drawdown episodes are available in the current history.")
else:
    st.dataframe(format_audit(selected_audit), width="stretch", hide_index=True)
    st.download_button("Download drawdown audit", audit.to_csv(index=False),
                       file_name="hedge_timer_drawdown_audit.csv", mime="text/csv")

driver_date = selected_episode["First warning"] if selected_episode is not None else df.index[-1]
with st.expander("Warning signal drivers" if selected_episode is not None else "Current signal drivers"):
    if pd.notna(driver_date):
        st.caption(f"Drivers at close on {driver_date.date()}. Watch weights total 100; price retreat triggers at a 1.5% loss from the 20-session high, a 2% five-session fall, or a 3% ten-session fall.")
    rows = []
    component_map = {item.key: item.label for item in (*WATCH_COMPONENTS, *CONFIRM_COMPONENTS)}
    for key, label in component_map.items():
        rows.append(
            {
                "Signal": label,
                SPX_LABEL: ("Active" if bool(conditions_spx[key].get(driver_date, False)) else "Inactive") if pd.notna(driver_date) and (selected_episode is not None or signal_inputs_usable) else "Unavailable",
                NDX_LABEL: ("Active" if bool(conditions_ndx[key].get(driver_date, False)) else "Inactive") if pd.notna(driver_date) and (selected_episode is not None or signal_inputs_usable) else "Unavailable",
                "Weight": next(item.weight for item in (*WATCH_COMPONENTS, *CONFIRM_COMPONENTS) if item.key == key),
                "Layer": "Watch" if key in {item.key for item in WATCH_COMPONENTS} else "Confirm",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with st.expander("Input dates"):
    st.dataframe(input_status, width="stretch", hide_index=True)
    st.download_button("Download input history", df0.to_csv(index_label="Date"),
                       file_name="hedge_timer_inputs.csv", mime="text/csv")

stats_spx = forward_stats(watch_spx[watch_spx.index >= CALIBRATION_START], df[SPX_TICKER], watch_threshold)
stats_ndx = forward_stats(watch_ndx[watch_ndx.index >= CALIBRATION_START], df[NDX_TICKER], watch_threshold)
st.caption(
    f"Forward check, next {HORIZON_DAYS} sessions: average worst return after a Hedge Watch onset was "
    f"{fmt_pct(stats_spx['avg_worst_warning'])} for {SPX_LABEL} and "
    f"{fmt_pct(stats_ndx['avg_worst_warning'])} for {NDX_LABEL}."
)

render_footer()
