# Hedge Timer
# High-recall drawdown warning system for ^SPX and ^NDX.

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from adfm_core.hedge_timer_data import callout_session_inputs, load_hedge_inputs
from adfm_core.hedge_timer_model import (
    CALIBRATION_START,
    CALLOUT_INPUT_TICKERS,
    CALLOUT_LEAD_LOOKBACK,
    FROZEN_CALLOUT_RULES,
    LEAD_LOOKBACK,
    MODEL_FIT_END,
    NDX_LABEL,
    NDX_TICKER,
    SPX_LABEL,
    SPX_TICKER,
    TICKERS,
    compute_callouts,
    compute_scores,
    drawdown,
    episode_audit,
    warning_summary,
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
        "**Hedge alert rules**\n\n"
        "A fresh price break plus two risk groups, a price/volatility shock, or near-high "
        "volatility divergence. Breadth, volatility, and credit each count once. "
        "A new alert requires three recovery closes and at least ten sessions since the prior alert."
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
        index=1,
        format_func=lambda value: f"{value} year" if value == 1 else f"{value} years",
    )
    audit_basis = st.radio("Drawdown basis", ["Intraday high / low", "Daily close"], index=0)
    st.divider()
    st.markdown("### Hedge alert audit since 2020")
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
        "HEDGE ALERT ACTIVE": "hedge-confirmed",
        "NO ACTIVE HEDGE ALERT": "hedge-stand",
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
    return out.rename(columns={"First warning": "First alert", "Loss at warning": "Loss at alert"})


def plot_index(
    price: pd.Series,
    callouts: pd.Series,
    meta: dict[str, pd.Series],
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

    confirmed_onsets = callouts.reindex(idx).fillna(False).astype(bool)
    if confirmed_onsets.any():
        ax_price.scatter(
            x[confirmed_onsets.values],
            frame["Price"].values[confirmed_onsets.values],
            marker="o",
            s=42,
            color=PASTEL["rose"],
            edgecolors="#111111",
            linewidths=.45,
            label="Hedge alert",
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
            "Timely hedge alerts from price breaks and breadth, volatility, and credit. "
            "SPX-fitted rules apply unchanged to NDX. The audit measures the red dots shown on the chart."
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
signal_input_status = input_status.loc[input_status["Input"].isin(CALLOUT_INPUT_TICKERS)]
current_inputs_fresh = bool(signal_input_status["Status"].eq("Current").all() and df.index[-1] == expected_session)
common_sessions = df0.dropna(subset=list(CALLOUT_INPUT_TICKERS)).index
previous_session = _last_completed_us_session(pd.Timestamp(expected_session).tz_localize("America/New_York")) if expected_session is not None else None
signal_inputs_usable = bool(
    len(common_sessions) and previous_session is not None
    and common_sessions[-1] >= previous_session
    and not signal_input_status["Status"].isin(["Last-good cache", "Research snapshot"]).any()
)
if signal_inputs_usable:
    df = df.loc[df.index <= common_sessions[-1]]

_, _, meta_spx, _ = compute_scores(df, SPX_TICKER)
_, _, meta_ndx, _ = compute_scores(df, NDX_TICKER)
# Preserve unknown sessions as well as raw observations; a missing index close
# must interrupt recovery rather than disappearing from the event calendar.
event_inputs = callout_session_inputs(df0.loc[:df.index[-1]])
signals_spx = compute_callouts(event_inputs, SPX_TICKER)
signals_ndx = compute_callouts(event_inputs, NDX_TICKER)
complete_rows = signals_spx["Inputs valid"] & signals_ndx["Inputs valid"]
latest_model_valid = last_bool(complete_rows)
signal_inputs_usable = signal_inputs_usable and latest_model_valid
current_inputs_fresh = current_inputs_fresh and latest_model_valid
warning_on_spx, warning_on_ndx = signals_spx["Callout"], signals_ndx["Callout"]


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
summary_spx = warning_summary(px_audit_spx, warning_on_spx, lookback=CALLOUT_LEAD_LOOKBACK)
summary_ndx = warning_summary(px_audit_ndx, warning_on_ndx, lookback=CALLOUT_LEAD_LOOKBACK)
audit_spx = episode_audit(SPX_LABEL, px_audit_spx, warning_on_spx, lookback=CALLOUT_LEAD_LOOKBACK)
audit_ndx = episode_audit(NDX_LABEL, px_audit_ndx, warning_on_ndx, lookback=CALLOUT_LEAD_LOOKBACK)
for audit_frame, signals in ((audit_spx, signals_spx), (audit_ndx, signals_ndx)):
    audit_frame["Trigger"] = audit_frame["First warning"].map(signals["Trigger"]).fillna("")
audit = pd.concat([audit_spx, audit_ndx], ignore_index=True)

sanity_box.markdown(
    f"**SPX fit:** {summary_spx['captured']}/{summary_spx['episodes']} early captures  \n"
    f"**NDX same rules:** {summary_ndx['captured']}/{summary_ndx['episodes']} early captures  \n"
    f"**Red dots:** SPX {summary_spx['warnings']} · NDX {summary_ndx['warnings']}  \n"
    f"**False alarms:** SPX {summary_spx['false_warnings']} · NDX {summary_ndx['false_warnings']}  \n"
    f"**Late / repeat alerts:** SPX {summary_spx['late_warnings']} · NDX {summary_ndx['late_warnings']}  \n"
    f"**Pending:** SPX {summary_spx['pending_warnings']} · NDX {summary_ndx['pending_warnings']}  \n"
    "\nSPX fitted through " + MODEL_FIT_END + ". NDX did not influence the fit. Only actual red dots count."
)


def render_index_state(
    label: str,
    ticker: str,
    signals: pd.DataFrame,
    meta: dict[str, pd.Series],
) -> None:
    early_now = last_bool(meta["early_stage"], True) and current_inputs_fresh
    oversold_now = last_bool(meta["oversold_block"], False)
    signal_active = last_bool(signals["Latched"])
    state = "HEDGE ALERT ACTIVE" if signal_active else "NO ACTIVE HEDGE ALERT"
    callout_dates = signals.index[signals["Callout"]]
    last_callout = callout_dates[-1].date().isoformat() if len(callout_dates) else "None"
    if not signal_inputs_usable:
        state = "Unavailable: stale or missing inputs"
        early_now = False
        last_callout = "NA"
    css = state_css(state)
    price_now = latest_value(df[ticker])
    current_dd = latest_value(drawdown(df[ticker]))
    rsi_now = latest_value(meta["rsi_d"])

    st.markdown(
        f"""
        <div class="hedge-index-title">{label}</div>
        <span class="hedge-state {css}">{state}</span>
        <div class="hedge-line">
            Price <b>{fmt_num(price_now, 2)}</b> &nbsp;·&nbsp; Current drawdown <b>{fmt_pct(current_dd)}</b><br>
            Last hedge alert <b>{last_callout}</b> &nbsp;·&nbsp; RSI14 <b>{fmt_num(rsi_now, 1)}</b><br>
            New short entry: <b>{'Eligible' if signal_active and early_now and not oversold_now else 'Blocked'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(f"{'Signals as of' if signal_inputs_usable else 'Index observations through'} {df.index[-1].date().isoformat()}. Latest completed US session: {expected_session.date().isoformat() if expected_session is not None else 'unavailable'}.")
if input_status["Status"].eq("Research snapshot").any():
    st.warning("Live inputs are unavailable. Browsing the dated research snapshot; current alerts and new short entries remain unavailable.")
elif not latest_model_valid:
    latest_observed = pd.to_numeric(event_inputs.reindex(columns=list(CALLOUT_INPUT_TICKERS)).iloc[-1], errors="coerce")
    invalid_inputs = latest_observed.index[~(np.isfinite(latest_observed) & latest_observed.gt(0))]
    details = ", ".join(invalid_inputs) or "insufficient observed history"
    st.warning(f"Invalid latest inputs: {details}. Current alerts and new short entries are unavailable. See Input dates for source dates.")
elif not current_inputs_fresh:
    unavailable_inputs = signal_input_status.loc[signal_input_status["Status"] != "Current"]
    details = ", ".join(unavailable_inputs["Input"])
    message = "Showing the last complete alert status; new short entries are blocked." if signal_inputs_usable else "Current alerts and new short entries are unavailable."
    st.warning(f"Awaiting the latest close for {details}. {message} See Input dates for source dates.")

col_spx, col_ndx = st.columns(2)
with col_spx:
    render_index_state(SPX_LABEL, SPX_TICKER, signals_spx, meta_spx)
with col_ndx:
    render_index_state(NDX_LABEL, NDX_TICKER, signals_ndx, meta_ndx)

st.caption(
    "Red dots mark new hedge alerts. Moving averages are chart context; they do not trigger dots. "
    "An oversold or late-stage tape can block a fresh short independently."
)

st.divider()
selected_ticker = SPX_TICKER if chart_index == SPX_LABEL else NDX_TICKER
selected_signals = signals_spx if selected_ticker == SPX_TICKER else signals_ndx
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
    st.caption(f"Hedge alert: {warning_date_text} · {selected_episode['Timing']} · "
               f"Loss at alert: {fmt_pct(selected_episode['Loss at warning'])} · "
               f"Early capture: {'Yes' if selected_episode['Captured'] else 'No'}")

figure = plot_index(
    df[selected_ticker],
    selected_signals["Callout"],
    selected_meta,
    chart_index,
    chart_years,
    selected_episode,
)
st.pyplot(figure, width="stretch")
plt.close(figure)

st.divider()
st.subheader("10%+ drawdown audit since 2020")
st.caption(
    f"Capture requires an actual red dot within {CALLOUT_LEAD_LOOKBACK} sessions before the peak "
    "or before the first loss beyond 3%. An ongoing signal is not credited as a new dot. "
    "Distinct local-peak legs rearm after a 10% rebound from the trough. "
    "Before-peak loss is shown as 0%; lead is measured to the first 10% crossing. "
    "False alarms have 60 completed sessions of follow-up and are separate from late or repeat drawdown alerts. "
    "Historical recall is fitted evidence, not a guarantee of future warnings."
)
if audit.empty:
    st.info("No qualifying drawdown episodes are available in the current history.")
else:
    st.dataframe(format_audit(selected_audit), width="stretch", hide_index=True)
    st.download_button("Download drawdown audit", audit.rename(columns={"First warning": "First alert", "Loss at warning": "Loss at alert"}).to_csv(index=False),
                       file_name="hedge_timer_drawdown_audit.csv", mime="text/csv")

driver_date = selected_episode["First warning"] if selected_episode is not None else df.index[-1]
with st.expander("Alert drivers" if selected_episode is not None else "Current risk conditions"):
    if pd.notna(driver_date):
        st.caption(f"Drivers at close on {driver_date.date()}. Breadth, volatility, and credit each count once; no weighted score.")
    rows = []
    for key in ("Price break", "Breadth", "Volatility", "Credit proxy", "Shock", "Divergence", "Recovery"):
        rows.append(
            {
                "Signal": key,
                SPX_LABEL: ("Active" if bool(signals_spx[key].get(driver_date, False)) else "Inactive") if pd.notna(driver_date) and bool(signals_spx["Inputs valid"].get(driver_date, False)) and (selected_episode is not None or signal_inputs_usable) else "Unavailable",
                NDX_LABEL: ("Active" if bool(signals_ndx[key].get(driver_date, False)) else "Inactive") if pd.notna(driver_date) and bool(signals_ndx["Inputs valid"].get(driver_date, False)) and (selected_episode is not None or signal_inputs_usable) else "Unavailable",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"Price break: prior three-session close low, {FROZEN_CALLOUT_RULES.price_retreat:.2%}–3% below the 20-session close high, plus two risk groups. "
        "Shock: at least a 1.25% one-day loss with VIX up 15%. Divergence: within 0.5% of the close high, "
        "rising over five sessions, weak breadth, and VIX up 15% over five sessions and above its 20-session average by 5%. "
        "Credit uses HYG/LQD as a proxy; verified full-history spread publication dates were unavailable."
    )

with st.expander("Input dates"):
    st.dataframe(input_status, width="stretch", hide_index=True)
    st.download_button("Download input history", df0.to_csv(index_label="Date"),
                       file_name="hedge_timer_inputs.csv", mime="text/csv")

render_footer()
