# Hedge Timer
# High-recall drawdown warning system for ^SPX and ^NDX.

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from adfm_core.hedge_timer_model import (
    CALIBRATION_START,
    CONFIRM_COMPONENTS,
    CONFIRM_THRESHOLD,
    HORIZON_DAYS,
    LEAD_LOOKBACK,
    NDX_LABEL,
    NDX_TICKER,
    SPX_LABEL,
    SPX_TICKER,
    TICKERS,
    WATCH_COMPONENTS,
    calibrate_watch_threshold,
    compute_scores,
    drawdown,
    episode_audit,
    forward_stats,
    onset,
    state_label,
    warning_summary,
    watch_signal,
)
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
        index=0,
        format_func=lambda value: f"{value} year" if value == 1 else f"{value} years",
    )
    st.divider()
    st.markdown("### Sanity check since 2020")
    sanity_box = st.empty()


def _today() -> date:
    return date.today()


def _start_date() -> date:
    return _today() - timedelta(days=int(10 * 365.25) + 180)


def _sessions_for_years(years: int) -> int:
    return int(round(252 * years))


@st.cache_data(ttl=900, show_spinner=False)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )


def extract_close(df_raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    if isinstance(df_raw.columns, pd.MultiIndex):
        out = {}
        for ticker in tickers:
            if (ticker, "Close") in df_raw.columns:
                out[ticker] = df_raw[(ticker, "Close")]
            elif (ticker, "Adj Close") in df_raw.columns:
                out[ticker] = df_raw[(ticker, "Adj Close")]
        result = pd.DataFrame(out)
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    if "Close" in df_raw.columns:
        result = df_raw[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Adj Close" in df_raw.columns:
        result = df_raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        return pd.DataFrame()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def last_valid(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if len(clean) else float("nan")


def last_bool(series: pd.Series, default: bool = False) -> bool:
    clean = series.dropna()
    return bool(clean.iloc[-1]) if len(clean) else default


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
    }.get(label, "hedge-stand")


def format_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Start", "End", "Trough", "First warning"):
        out[column] = out[column].map(
            lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "No warning"
        )
    out["Depth"] = out["Depth"].map(lambda value: f"{value * 100:.1f}%")
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
) -> plt.Figure:
    frame = pd.DataFrame(
        {
            "Price": price,
            "Watch": watch_score,
            "Confirm": confirm_score,
        }
    ).dropna(subset=["Price"])
    if len(frame) > _sessions_for_years(years):
        frame = frame.iloc[-_sessions_for_years(years) :].copy()

    fig = plt.figure(figsize=(13.5, 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.09)
    ax_price = fig.add_subplot(grid[0])
    ax_score = fig.add_subplot(grid[1], sharex=ax_price)

    x = np.arange(len(frame))
    idx = frame.index
    ma50 = meta["ma50"].reindex(idx)
    ma200 = meta["ma200"].reindex(idx)

    ax_price.plot(x, frame["Price"].values, linewidth=2.0, color="#111111", label="Price")
    ax_price.plot(x, ma50.values, linewidth=1.3, color=PASTEL["blue"], label="MA50")
    ax_price.plot(x, ma200.values, linewidth=1.25, color=PASTEL["lavender"], label="MA200")
    ax_price.grid(True, linewidth=.6, alpha=.14)
    ax_price.spines[["top", "right"]].set_visible(False)
    ax_price.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_price.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)

    watch_active = watch_signal(watch_score.reindex(idx), watch_threshold)
    confirmation = confirm_score.reindex(idx) >= CONFIRM_THRESHOLD
    confirmed_onsets = onset((watch_active & confirmation).fillna(False))

    ax_score.plot(
        x,
        frame["Watch"].values,
        linewidth=1.9,
        color=PASTEL["amber"],
        label="Hedge Score",
    )
    ax_score.axhline(
        watch_threshold,
        linewidth=1.0,
        color="#111111",
        alpha=.72,
        label="Threshold",
    )
    if confirmed_onsets.any():
        ax_score.scatter(
            x[confirmed_onsets.values],
            frame["Watch"].values[confirmed_onsets.values],
            marker="o",
            s=42,
            color=PASTEL["rose"],
            edgecolors="#111111",
            linewidths=.45,
            label="Confirmed",
            zorder=6,
        )

    ax_score.set_ylim(0, 100)
    ax_score.set_ylabel("Score")
    ax_score.grid(True, axis="y", linewidth=.6, alpha=.14)
    ax_score.spines[["top", "right"]].set_visible(False)
    ax_score.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)

    tick_count = 8 if years <= 2 else 10
    tick_positions = np.linspace(0, max(len(frame) - 1, 0), min(tick_count, len(frame)), dtype=int)
    tick_positions = np.unique(tick_positions)
    tick_labels = [
        idx[position].strftime("%b %Y") if years <= 3 else idx[position].strftime("%Y-%m")
        for position in tick_positions
    ]
    ax_score.set_xticks(tick_positions)
    ax_score.set_xticklabels(tick_labels, fontsize=8)

    period = f"{years} Year" if years == 1 else f"{years} Years"
    fig.suptitle(f"{label} Hedge Timer | {period}", fontsize=14, fontweight="bold", y=.99)
    fig.tight_layout(rect=[.01, .01, .99, .96])
    return fig


render_page_header(
    PageHeader(
        title="Hedge Timer",
        description=(
            "High-recall drawdown warning system for the S&P 500 and Nasdaq-100. "
            "Hedge Watch is calibrated first for capture of 10%+ drawdowns since 2020; "
            "confirmation and anti-bottom-short gates determine whether a fresh directional short is allowed."
        ),
        eyebrow="ADFM Risk + Execution",
    )
)

raw = yf_download(list(TICKERS), _start_date())
df0 = extract_close(raw, list(TICKERS))
if df0.empty or SPX_TICKER not in df0.columns or NDX_TICKER not in df0.columns:
    st.error("Yahoo Finance did not return usable S&P 500 and Nasdaq-100 data.")
    st.stop()

base_idx = df0[SPX_TICKER].dropna().index.intersection(df0[NDX_TICKER].dropna().index)
df = df0.reindex(base_idx).ffill()

watch_spx, confirm_spx, meta_spx, conditions_spx = compute_scores(df, SPX_TICKER)
watch_ndx, confirm_ndx, meta_ndx, conditions_ndx = compute_scores(df, NDX_TICKER)

calibration = calibrate_watch_threshold(
    watch_spx,
    df[SPX_TICKER],
    watch_ndx,
    df[NDX_TICKER],
)
watch_threshold = float(calibration["threshold"])

warning_on_spx = onset(watch_signal(watch_spx, watch_threshold))
warning_on_ndx = onset(watch_signal(watch_ndx, watch_threshold))
summary_spx = warning_summary(df[SPX_TICKER], warning_on_spx)
summary_ndx = warning_summary(df[NDX_TICKER], warning_on_ndx)

audit_spx = episode_audit(SPX_LABEL, df[SPX_TICKER], warning_on_spx)
audit_ndx = episode_audit(NDX_LABEL, df[NDX_TICKER], warning_on_ndx)
audit = pd.concat([audit_spx, audit_ndx], ignore_index=True)

captured_total = int(summary_spx["captured"] + summary_ndx["captured"])
episode_total = int(summary_spx["episodes"] + summary_ndx["episodes"])
false_warnings_total = int(summary_spx["false_warnings"] + summary_ndx["false_warnings"])
lead_values = audit.loc[audit["Captured"], "Lead sessions"].dropna() if not audit.empty else pd.Series(dtype=float)
median_lead = float(lead_values.median()) if len(lead_values) else float("nan")
full_recall = calibration["spx_coverage"] >= 1.0 and calibration["ndx_coverage"] >= 1.0

sanity_box.markdown(
    (
        f"**Drawdowns captured:** {captured_total}/{episode_total}  \n"
        f"**Median lead:** {fmt_num(median_lead, 0)} sessions  \n"
        f"**False warnings:** {false_warnings_total}  \n"
        f"**Calibrated watch threshold:** {watch_threshold:.0f}/100  \n"
        f"**Full-recall constraint:** {'Met' if full_recall else 'Best available'}  \n\n"
        f"{SPX_LABEL}: {int(summary_spx['captured'])}/{int(summary_spx['episodes'])} captured  \n"
        f"{NDX_LABEL}: {int(summary_ndx['captured'])}/{int(summary_ndx['episodes'])} captured"
    )
)


def render_index_state(
    label: str,
    ticker: str,
    watch: pd.Series,
    confirm: pd.Series,
    meta: dict[str, pd.Series],
) -> None:
    watch_now = last_valid(watch)
    confirm_now = last_valid(confirm)
    early_now = last_bool(meta["early_stage"], True)
    oversold_now = last_bool(meta["oversold_block"], False)
    state = state_label(watch_now, confirm_now, watch_threshold, early_now, oversold_now)
    css = state_css(state)
    price_now = last_valid(df[ticker])
    current_dd = last_valid(drawdown(df[ticker]))
    rsi_now = last_valid(meta["rsi_d"])
    dd63_now = last_valid(meta["dd63"])
    sector_weak = last_valid(meta["sector_breadth_share"])
    rv_ratio = last_valid(meta["realized_vol_ratio"])

    st.markdown(
        f"""
        <div class="hedge-index-title">{label}</div>
        <span class="hedge-state {css}">{state}</span>
        <div class="hedge-line">
            Price <b>{fmt_num(price_now, 2)}</b> &nbsp;·&nbsp; Current drawdown <b>{fmt_pct(current_dd)}</b><br>
            Hedge Watch <b>{fmt_num(watch_now, 0)}/100</b> &nbsp;·&nbsp;
            Confirmation <b>{fmt_num(confirm_now, 0)}/100</b><br>
            RSI14 <b>{fmt_num(rsi_now, 1)}</b> &nbsp;·&nbsp;
            63-session drawdown <b>{fmt_pct(dd63_now)}</b><br>
            Sectors below MA50 <b>{fmt_pct(sector_weak, 0)}</b> &nbsp;·&nbsp;
            RV10 / RV63 <b>{fmt_num(rv_ratio, 2)}x</b><br>
            Fresh-short gate: <b>{'Open' if early_now and not oversold_now else 'Blocked'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


col_spx, col_ndx = st.columns(2)
with col_spx:
    render_index_state(SPX_LABEL, SPX_TICKER, watch_spx, confirm_spx, meta_spx)
with col_ndx:
    render_index_state(NDX_LABEL, NDX_TICKER, watch_ndx, confirm_ndx, meta_ndx)

st.caption(
    f"Hedge Watch threshold {watch_threshold:.0f}/100 is calibrated across both indices from "
    f"{CALIBRATION_START}. Confirmation threshold is {CONFIRM_THRESHOLD:.0f}/100. "
    "An oversold or late-stage tape can block a fresh short while Hedge Watch remains active."
)

st.divider()
selected_ticker = SPX_TICKER if chart_index == SPX_LABEL else NDX_TICKER
selected_watch = watch_spx if selected_ticker == SPX_TICKER else watch_ndx
selected_confirm = confirm_spx if selected_ticker == SPX_TICKER else confirm_ndx
selected_meta = meta_spx if selected_ticker == SPX_TICKER else meta_ndx

figure = plot_index(
    df[selected_ticker],
    selected_watch,
    selected_confirm,
    selected_meta,
    watch_threshold,
    chart_index,
    chart_years,
)
st.pyplot(figure, use_container_width=True)
plt.close(figure)

st.divider()
st.subheader("10%+ drawdown audit since 2020")
st.caption(
    f"Captured means a new Hedge Watch onset occurred within the prior {LEAD_LOOKBACK} trading sessions. "
    "The audit uses the warning layer before RSI and late-drawdown short-entry gates."
)
if audit.empty:
    st.info("No qualifying drawdown episodes are available in the current history.")
else:
    st.dataframe(format_audit(audit), use_container_width=True, hide_index=True)

with st.expander("Current signal drivers"):
    rows = []
    component_map = {item.key: item.label for item in (*WATCH_COMPONENTS, *CONFIRM_COMPONENTS)}
    for key, label in component_map.items():
        rows.append(
            {
                "Signal": label,
                SPX_LABEL: "Active" if last_bool(conditions_spx[key]) else "Inactive",
                NDX_LABEL: "Active" if last_bool(conditions_ndx[key]) else "Inactive",
                "Layer": "Watch" if key in {item.key for item in WATCH_COMPONENTS} else "Confirm",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

stats_spx = forward_stats(watch_spx[watch_spx.index >= CALIBRATION_START], df[SPX_TICKER], watch_threshold)
stats_ndx = forward_stats(watch_ndx[watch_ndx.index >= CALIBRATION_START], df[NDX_TICKER], watch_threshold)
st.caption(
    f"Forward check, next {HORIZON_DAYS} sessions: average worst return after a Hedge Watch onset was "
    f"{fmt_pct(stats_spx['avg_worst_warning'])} for {SPX_LABEL} and "
    f"{fmt_pct(stats_ndx['avg_worst_warning'])} for {NDX_LABEL}."
)

render_footer()
