"""Interactive historical position-sizing simulator."""
from __future__ import annotations

from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.market_data import (
    adjusted_ohlcv,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
    unique_tickers,
)
from adfm_core.palette import PASTEL
from adfm_core.position_sizing import (
    HORIZON_TRADING_DAYS,
    annualized_volatility,
    calculate_sizing,
    daily_gap_proxy,
    earnings_reaction_frame,
    expected_shortfall,
    first_touch_statistics,
    historical_tail_move,
    historical_windows,
    maximum_drawdown_from_prices,
    rolling_annualized_volatility,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_selection_note,
    render_status_line,
)

TITLE = "Position Sizing Lab"
NY_TZ = ZoneInfo("America/New_York")
BENCHMARKS = ("SPY", "QQQ", "TLT", "UUP", "USO", "^VIX")
BLUE, RED, GREEN, ORANGE = (PASTEL[key] for key in ("blue", "rose", "sage", "coral"))
GRID = "rgba(148,163,184,.30)"
PFX = "psl_"
BLOCK_SIZE = 20


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "N/A"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value * 100:,.{digits}f}%"


def money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def close_series(frames: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    frame = frames.get(ticker)
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=ticker)
    series = pd.to_numeric(adjusted_ohlcv(frame).get("Close"), errors="coerce").dropna()
    series.name = ticker
    return series


def adjusted_frame(frames: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    frame = frames.get(ticker)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return adjusted_ohlcv(frame).dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def earnings_dates(symbol: str) -> tuple[pd.Timestamp, ...]:
    try:
        data = yf.Ticker(symbol).get_earnings_dates(limit=48)
        if data is None or data.empty:
            return ()
        index = pd.DatetimeIndex(data.index)
        if index.tz is not None:
            index = index.tz_convert("America/New_York").tz_localize(None)
        return tuple(pd.Timestamp(value) for value in index)
    except Exception:
        return ()


def directional_daily_returns(close: pd.Series, direction: str) -> pd.Series:
    daily = pd.to_numeric(close, errors="coerce").pct_change(fill_method=None)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    daily = daily.loc[daily.gt(-1.0)]
    if direction == "Short":
        daily = -daily
    daily.name = "directional_return"
    return daily


def chart_style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=56, r=24, t=34, b=44),
        font=dict(family="Arial, sans-serif", color="#404040"),
        legend=dict(orientation="h", y=1.03, x=0),
        hovermode="x unified",
    )
    return fig


def bankroll_chart(balances: list[float]) -> go.Figure:
    values = np.asarray(balances, dtype=float)
    sessions = np.arange(values.size)
    fig = go.Figure(
        go.Scatter(
            x=sessions,
            y=values,
            mode="lines",
            name="Balance",
            line=dict(color="#2E7D32", width=3.0),
            fill="tozeroy",
            fillcolor="rgba(112,173,71,.22)",
            hovertemplate="Session %{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=float(values[0]), line=dict(color="#7F8C8D", width=1.2, dash="dash"))
    fig.update_xaxes(title="Trading sessions simulated")
    fig.update_yaxes(title="Balance", tickprefix="$", tickformat=",.3s")
    return chart_style(fig, 350)


def volatility_chart(nav_returns: list[float]) -> go.Figure:
    rolling = (
        pd.Series(nav_returns, dtype=float)
        .rolling(20, min_periods=5)
        .std(ddof=1)
        .mul(np.sqrt(252) * 100)
    )
    fig = go.Figure(
        go.Scatter(
            x=np.arange(1, len(rolling) + 1),
            y=rolling,
            mode="lines",
            name="20-session annualized σ",
            line=dict(color="#C65911", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(237,125,49,.18)",
            hovertemplate="Session %{x}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="Trading sessions simulated")
    fig.update_yaxes(title="Annualized portfolio σ", ticksuffix="%")
    return chart_style(fig, 240)


def distribution_chart(paths: pd.DataFrame, target: float, stop: float) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=paths["return"] * 100,
            nbinsx=45,
            marker_color=BLUE,
            opacity=.86,
        )
    )
    if target > 0:
        fig.add_vline(
            x=target * 100,
            line=dict(color=GREEN, width=2, dash="dash"),
            annotation_text="Target",
        )
    fig.add_vline(
        x=-stop * 100,
        line=dict(color=RED, width=2, dash="dash"),
        annotation_text="Invalidation",
    )
    fig.update_xaxes(title="Directional holding-period return", ticksuffix="%")
    fig.update_yaxes(title="Observations")
    return chart_style(fig, 420)


def empirical_log_optimal(values: pd.Series, ceiling: float) -> float:
    returns = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(returns) < 20 or ceiling <= 0:
        return np.nan
    candidates = np.linspace(0, ceiling, 501)
    scores = np.full(len(candidates), -np.inf)
    for index, fraction in enumerate(candidates):
        gross = 1 + fraction * returns
        if np.all(gross > 0):
            scores[index] = np.mean(np.log(gross))
    return float(candidates[int(np.argmax(scores))]) if np.isfinite(scores).any() else np.nan


def max_balance_drawdown(balances: list[float]) -> float:
    values = np.asarray(balances, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.min(values / np.maximum.accumulate(values) - 1))


def outcome_grid(values: list[float]) -> str:
    recent = values[-100:]
    cells = ["<span class='sim-dot empty'></span>" for _ in range(100 - len(recent))]
    for value in recent:
        state, mark = ("win", "+") if value > 0 else ("loss", "−") if value < 0 else ("flat", "·")
        label = escape(f"{value * 100:+.2f}% observed daily return")
        cells.append(f"<span class='sim-dot {state}' title='{label}'>{mark}</span>")
    return "<div class='sim-grid'>" + "".join(cells) + "</div>"


def reset_simulation(balance: float, signature: str, seed: int) -> None:
    st.session_state[PFX + "signature"] = signature
    st.session_state[PFX + "running"] = False
    st.session_state[PFX + "balances"] = [float(balance)]
    st.session_state[PFX + "returns"] = []
    st.session_state[PFX + "nav_returns"] = []
    st.session_state[PFX + "dates"] = []
    st.session_state[PFX + "seed"] = int(seed)


def sampled_session_index(
    size: int,
    session_no: int,
    mode: str,
    seed: int,
    target_sessions: int,
) -> int:
    if size <= 1:
        return 0
    if mode == "Chronological regime replay":
        maximum_start = max(1, size - target_sessions + 1)
        start = seed % maximum_start
        return min(start + session_no, size - 1)

    block_no, offset = divmod(session_no, BLOCK_SIZE)
    maximum_start = max(1, size - BLOCK_SIZE + 1)
    rng = np.random.default_rng(seed + block_no * 7919)
    if mode == "Recent-regime weighted blocks":
        weights = np.linspace(1.0, 4.0, maximum_start)
        start = int(rng.choice(maximum_start, p=weights / weights.sum()))
    else:
        start = int(rng.integers(0, maximum_start))
    return min(start + offset, size - 1)


def add_session(
    daily_returns: pd.Series,
    fraction: float,
    mode: str,
    seed: int,
    target_sessions: int,
) -> None:
    balances = st.session_state[PFX + "balances"]
    sampled = st.session_state[PFX + "returns"]
    if not balances or balances[-1] <= 0 or daily_returns.empty:
        st.session_state[PFX + "running"] = False
        return
    index = sampled_session_index(
        len(daily_returns),
        len(sampled),
        mode,
        seed,
        target_sessions,
    )
    observed_return = float(daily_returns.iloc[index])
    nav_return = fraction * observed_return
    balances.append(max(0.0, balances[-1] * (1 + nav_return)))
    sampled.append(observed_return)
    st.session_state[PFX + "nav_returns"].append(nav_return)
    st.session_state[PFX + "dates"].append(pd.Timestamp(daily_returns.index[index]))
    if balances[-1] <= 0:
        st.session_state[PFX + "running"] = False


def sensitivity_table(
    selected_returns: pd.Series,
    frames: dict[str, pd.DataFrame],
    direction: str,
) -> pd.DataFrame:
    labels = {
        "SPY": "US equities",
        "QQQ": "Growth / duration",
        "TLT": "Long duration",
        "UUP": "US dollar",
        "USO": "Oil",
        "^VIX": "Equity volatility",
    }
    sign = 1 if direction == "Long" else -1
    rows = []
    for symbol, label in labels.items():
        benchmark = close_series(frames, symbol).pct_change(fill_method=None)
        aligned = pd.concat(
            [selected_returns.rename("asset"), benchmark.rename("benchmark")],
            axis=1,
        ).dropna().tail(756)
        if len(aligned) < 40:
            continue
        variance = aligned["benchmark"].var(ddof=1)
        beta = aligned["asset"].cov(aligned["benchmark"]) / variance if variance > 0 else np.nan
        rows.append(
            (
                label,
                symbol,
                sign * aligned["asset"].corr(aligned["benchmark"]),
                sign * beta,
                len(aligned),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["Exposure", "Proxy", "Position correlation", "Position beta", "Observations"],
    )


st.set_page_config(page_title=TITLE, layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1560)
st.markdown(
    """
    <style>
    .sim-head{border-top:3px solid #000;border-bottom:1px solid #000;margin:1.2rem 0 .75rem;padding:.7rem 0}
    .sim-kicker{font:800 .68rem Arial;letter-spacing:.13em;text-transform:uppercase}
    .sim-title{font:700 1.5rem Georgia;margin-top:.22rem}
    .sim-grid{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:7px;margin:.45rem 0 1rem}
    .sim-dot{display:flex;align-items:center;justify-content:center;aspect-ratio:1;border:2px solid #7f7f7f;border-radius:50%;background:#fff;color:#262626;font:900 .72rem Arial}
    .sim-dot.win{background:#C6E0B4;border-color:#548235;color:#1F4E21}
    .sim-dot.loss{background:#F4B183;border-color:#C00000;color:#9C0006}
    .sim-dot.flat{background:#D9E1F2;border-color:#4472C4;color:#203864}
    .sim-dot.empty{background:#F2F2F2;border-color:#A6A6A6;color:transparent}
    .sim-balance{display:flex;justify-content:space-between;border-top:2px solid #000;border-bottom:2px solid #000;margin:.55rem 0 .85rem;padding:.72rem 0}
    .sim-balance span:first-child{font:800 .7rem Arial;letter-spacing:.1em;text-transform:uppercase;color:#262626}
    .sim-balance span:last-child{font:700 1.35rem Georgia}
    main:has(.sim-head) [data-testid="stBaseButton-primary"]{border:1px solid #7F7F7F!important;background:#E7E6E6!important;color:#000!important;box-shadow:none!important}
    main:has(.sim-head) [data-testid="stBaseButton-primary"]:hover{background:#D9D9D9!important;color:#000!important}
    main:has(.sim-head) [data-testid="stBaseButton-primary"]:disabled{border-color:#A6A6A6!important;background:#E7E6E6!important;color:#595959!important;opacity:1!important}
    @media(max-width:760px){.sim-grid{gap:5px}.sim-dot{font-size:.64rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Position setup")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    direction = st.selectbox("Direction", ("Long", "Short"))
    conviction = st.slider(
        "Conviction level",
        1,
        5,
        3,
        help="1=5%, 2=10%, 3=15%, 4=20%, 5=25% maximum gross exposure.",
    )
    horizon_label = st.selectbox(
        "Intended holding period",
        tuple(HORIZON_TRADING_DAYS),
        index=1,
    )
    portfolio_nav = st.number_input(
        "Portfolio NAV",
        min_value=1_000.0,
        value=5_000_000.0,
        step=100_000.0,
        format="%.0f",
    )
    max_loss_pct = st.number_input(
        "Maximum NAV loss at invalidation",
        .10,
        10.0,
        1.25,
        .05,
        format="%.2f",
    )
    hold_earnings = st.checkbox(
        "Assume position is held through earnings",
        value=horizon_label != "1 month",
    )
    with st.expander("Liquidity assumptions"):
        participation = st.slider(
            "Maximum share of median daily dollar volume",
            1,
            25,
            10,
            format="%d%%",
        )
        liquidation_days = st.slider("Target liquidation window", 1, 10, 3, format="%d days")
    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        "Conviction sets the maximum exposure. The simulator advances through observed daily returns for one selected holding-period path. Historical volatility, invalidation risk, event risk, tails, and liquidity can only reduce the recommended size."
    )

render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Set conviction, define the trade, and run a live path simulation using observed daily returns from the selected ticker's history."
        ),
        eyebrow="ADFM Risk + Decision Support",
    )
)
if not ticker:
    st.error("Enter a ticker.")
    st.stop()

with st.spinner(f"Loading full available history for {ticker}..."):
    frames, missing = fetch_daily_ohlcv(
        unique_tickers([ticker, *BENCHMARKS]),
        period="max",
    )
close = close_series(frames, ticker)
ohlcv = adjusted_frame(frames, ticker)
if len(close) < 63:
    st.error(f"No usable historical series was returned for {ticker}.")
    st.stop()

latest = float(close.iloc[-1])
horizon_days = HORIZON_TRADING_DAYS[horizon_label]
render_status_line(
    ticker=ticker,
    data_through=pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d"),
    history=f"{len(close):,} sessions",
    holding_period=horizon_label,
    source="Yahoo Finance",
)

render_section_header(
    "Trade structure",
    "Target and invalidation distances are applied proportionally to every historical starting date.",
)
c1, c2, c3 = st.columns(3)
entry = c1.number_input(
    "Entry price",
    min_value=.01,
    value=latest,
    step=max(.01, latest * .0025),
    format="%.2f",
    key=f"entry_{ticker}_{direction}",
)
target = c2.number_input(
    "Target price",
    min_value=.01,
    value=latest * (1.15 if direction == "Long" else .85),
    step=max(.01, latest * .0025),
    format="%.2f",
    key=f"target_{ticker}_{direction}",
)
stop = c3.number_input(
    "Invalidation price",
    min_value=.01,
    value=latest * (.92 if direction == "Long" else 1.08),
    step=max(.01, latest * .0025),
    format="%.2f",
    key=f"stop_{ticker}_{direction}",
)
if direction == "Long":
    target_move, stop_distance = target / entry - 1, 1 - stop / entry
else:
    target_move, stop_distance = 1 - target / entry, stop / entry - 1
if stop_distance <= 0:
    st.error("The invalidation is on the wrong side of the entry for the selected direction.")
    st.stop()
if target_move <= 0:
    st.warning("The target is on the wrong side of the entry. Target-hit statistics are disabled until corrected.")

returns = close.pct_change(fill_method=None).dropna()
simulation_returns = directional_daily_returns(close, direction)
rolling_vol = rolling_annualized_volatility(returns, 63)
current_vol = annualized_volatility(returns, 63)
median_vol = float(rolling_vol.dropna().median())
tail_move = historical_tail_move(returns, direction.lower(), 5)
daily_es = expected_shortfall(returns, direction.lower(), .01)
median_dollar_volume = float(
    (
        close.reindex(ohlcv.index)
        * pd.to_numeric(ohlcv["Volume"], errors="coerce")
    ).dropna().tail(252).median()
)

dates = earnings_dates(ticker)
events = earnings_reaction_frame(ohlcv, dates)
if len(events) >= 4:
    event_move = float(events["abs_move"].quantile(.90))
    event_basis = f"90th-percentile earnings reaction across {len(events)} events"
else:
    event_move = daily_gap_proxy(ohlcv, .90)
    event_basis = "90th-percentile overnight gap; earnings history unavailable"
today = pd.Timestamp(datetime.now(NY_TZ).date())
next_earnings = min((date for date in dates if date.normalize() >= today), default=None)

step = {21: 5, 63: 5, 252: 21, 1260: 63}[horizon_days]
paths = historical_windows(close, horizon_days, direction.lower(), step=step)
max_loss = max_loss_pct / 100
sizing = calculate_sizing(
    conviction=conviction,
    max_nav_loss=max_loss,
    stop_distance=stop_distance,
    current_volatility=current_vol,
    historical_median_volatility=median_vol,
    event_move=event_move,
    tail_move=tail_move,
    portfolio_nav=portfolio_nav,
    median_dollar_volume=median_dollar_volume,
    hold_through_earnings=hold_earnings,
    participation_rate=participation / 100,
    liquidation_days=liquidation_days,
)
position_value = portfolio_nav * sizing.suggested_size
loss_at_stop = sizing.suggested_size * stop_distance
target_nav = sizing.suggested_size * target_move if target_move > 0 else np.nan
event_nav = sizing.suggested_size * event_move

verdict = (
    f"{sizing.binding_constraint} reduces the {pct(sizing.conviction_ceiling)} conviction ceiling to {pct(sizing.suggested_size)}."
    if sizing.suggested_size < sizing.conviction_ceiling - 1e-9
    else "The conviction ceiling remains binding; the selected risk constraints do not force a smaller position."
)
render_selection_note("Sizing verdict", verdict)
render_kpi_cards(
    [
        ("Conviction ceiling", pct(sizing.conviction_ceiling), f"Conviction {conviction} of 5"),
        ("Suggested exposure", pct(sizing.suggested_size), sizing.binding_constraint),
        ("Position notional", money(position_value), f"At {money(portfolio_nav)} NAV"),
        ("Loss at invalidation", pct(loss_at_stop, 2), f"Budget {pct(max_loss, 2)}"),
        ("NAV impact at target", pct(target_nav, 2), f"Target move {pct(target_move)}"),
        ("Historical event loss", pct(event_nav, 2), event_basis),
    ]
)

st.markdown(
    "<div class='sim-head'><div class='sim-kicker'>Interactive historical simulation</div>"
    "<div class='sim-title'>Run one holding-period path, session by session, using observed ticker returns.</div></div>",
    unsafe_allow_html=True,
)

if paths.empty or len(simulation_returns) < horizon_days:
    st.warning(
        f"{ticker} does not have enough history for a full {horizon_label} simulation. Choose a shorter holding period or a ticker with longer history."
    )
else:
    horizon_hit_rate = float(paths["return"].gt(0).mean())
    median_horizon_return = float(paths["return"].median())
    daily_hit_rate = float(simulation_returns.gt(0).mean())
    log_optimal = empirical_log_optimal(simulation_returns, sizing.conviction_ceiling)

    a, b, c, d = st.columns([1.1, 1.15, .8, .8])
    starting_balance = a.number_input(
        "Starting balance",
        min_value=1_000.0,
        value=float(portfolio_nav),
        step=100_000.0,
        format="%.0f",
    )
    mode = b.selectbox(
        "Sampling mode",
        (
            "Random historical blocks",
            "Recent-regime weighted blocks",
            "Chronological regime replay",
        ),
        help=(
            "Random modes sample 20-session blocks so serial dependence is partly preserved. Chronological replay follows one continuous historical path."
        ),
    )
    c.number_input(
        "Sessions in run",
        min_value=horizon_days,
        max_value=horizon_days,
        value=horizon_days,
        disabled=True,
    )
    sessions_per_pulse = d.slider("Sessions per pulse", 1, 25, 5)

    ceiling_pct = sizing.conviction_ceiling * 100
    suggested_pct = sizing.suggested_size * 100
    control_signature = (
        f"{ticker}|{direction}|{horizon_label}|{conviction}|{starting_balance:.2f}|"
        f"{mode}|{suggested_pct:.3f}"
    )
    position_key = PFX + "position_pct"
    if st.session_state.get(PFX + "control_signature") != control_signature:
        st.session_state[PFX + "control_signature"] = control_signature
        st.session_state[position_key] = min(
            ceiling_pct,
            max(.5, round(suggested_pct * 2) / 2),
        )
    simulation_position_pct = st.slider(
        "Simulation position size",
        .5,
        max(.5, float(ceiling_pct)),
        step=.5,
        key=position_key,
        help="Gross exposure applied to each observed daily return. It cannot exceed the conviction ceiling.",
    )
    simulation_fraction = simulation_position_pct / 100
    seed = abs(hash((ticker, direction, horizon_label, conviction))) % 1_000_000
    signature = f"{control_signature}|{simulation_position_pct:.2f}|{seed}"
    if st.session_state.get(PFX + "signature") != signature:
        reset_simulation(starting_balance, signature, seed)

    @st.fragment(run_every=.8)
    def live_simulator() -> None:
        if st.session_state[PFX + "running"]:
            remaining = max(
                0,
                horizon_days - len(st.session_state[PFX + "returns"]),
            )
            for _ in range(min(sessions_per_pulse, remaining)):
                add_session(
                    simulation_returns,
                    simulation_fraction,
                    mode,
                    seed,
                    horizon_days,
                )
            if (
                len(st.session_state[PFX + "returns"]) >= horizon_days
                or st.session_state[PFX + "balances"][-1] <= 0
            ):
                st.session_state[PFX + "running"] = False

        balances = st.session_state[PFX + "balances"]
        sampled = st.session_state[PFX + "returns"]
        nav_returns = st.session_state[PFX + "nav_returns"]
        sampled_dates = st.session_state[PFX + "dates"]
        completed = len(sampled)
        current_balance = float(balances[-1])
        cumulative = current_balance / starting_balance - 1
        realized_hit = float(np.mean(np.asarray(sampled) > 0)) if sampled else np.nan

        left, right = st.columns([.82, 1.18], gap="large")
        with left:
            st.markdown(
                f"**{ticker} finished positive in {horizon_hit_rate * 100:.1f}% of rolling {horizon_label} windows.**"
            )
            st.caption(
                "The game advances one observed daily session at a time. It does not stack repeated full holding-period returns."
            )
            st.markdown(outcome_grid(sampled), unsafe_allow_html=True)
            st.markdown(
                f"<div class='sim-balance'><span>Balance</span><span>{escape(money(current_balance))}</span></div>",
                unsafe_allow_html=True,
            )
            buttons = st.columns(4)
            if buttons[0].button(
                "Start simulation",
                type="primary",
                use_container_width=True,
                disabled=(
                    st.session_state[PFX + "running"]
                    or completed >= horizon_days
                    or current_balance <= 0
                ),
            ):
                st.session_state[PFX + "running"] = True
            if buttons[1].button(
                "Run one session",
                use_container_width=True,
                disabled=(
                    st.session_state[PFX + "running"]
                    or completed >= horizon_days
                    or current_balance <= 0
                ),
            ):
                add_session(
                    simulation_returns,
                    simulation_fraction,
                    mode,
                    seed,
                    horizon_days,
                )
            if buttons[2].button(
                "Stop",
                use_container_width=True,
                disabled=not st.session_state[PFX + "running"],
            ):
                st.session_state[PFX + "running"] = False
            if buttons[3].button("Reset", use_container_width=True):
                reset_simulation(starting_balance, signature, seed)

            if simulation_fraction > sizing.suggested_size + .0025:
                st.error(
                    f"Above the risk-adjusted suggestion by {(simulation_fraction - sizing.suggested_size) * 100:.1f} percentage points."
                )
            elif simulation_fraction + .0025 < sizing.suggested_size:
                st.info(
                    f"Below the risk-adjusted suggestion by {(sizing.suggested_size - simulation_fraction) * 100:.1f} percentage points."
                )
            else:
                st.success("Simulation size is aligned with the risk-adjusted suggestion.")
            if sampled_dates:
                st.caption(
                    f"Last draw: {sampled_dates[-1]:%Y-%m-%d} · ticker return {sampled[-1] * 100:+.2f}% · NAV contribution {nav_returns[-1] * 100:+.3f}%"
                )
            else:
                st.caption(
                    "Each step uses an observed daily return. Random modes preserve returns in 20-session blocks."
                )

        with right:
            st.plotly_chart(bankroll_chart(balances), use_container_width=True)
            st.plotly_chart(volatility_chart(nav_returns), use_container_width=True)

        render_kpi_cards(
            [
                ("Sessions completed", f"{completed:,} / {horizon_days:,}", mode),
                ("Cumulative return", pct(cumulative, 1, True), money(current_balance)),
                ("Maximum drawdown", pct(max_balance_drawdown(balances)), "Simulation balance"),
                ("Realized daily hit rate", pct(realized_hit, 0), f"Historical daily rate {pct(daily_hit_rate, 0)}"),
                ("Position size", pct(simulation_fraction), f"Suggested {pct(sizing.suggested_size)}"),
                ("Historical log-optimal", pct(log_optimal), "Daily-return estimate within conviction ceiling"),
            ]
        )

    live_simulator()
    with st.expander("Simulation methodology and historical edge inputs"):
        st.markdown(
            "One simulation now represents one selected holding period. The balance advances through observed daily returns rather than compounding 100 separate one-, three-, or five-year outcomes. Random modes resample 20-session blocks, while chronological replay follows one continuous historical path. This keeps the path economically interpretable and avoids the explosive double-counting visible in the prior version."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ("Historical daily hit rate", pct(daily_hit_rate)),
                    (f"Rolling {horizon_label} hit rate", pct(horizon_hit_rate)),
                    (f"Median {horizon_label} return", pct(median_horizon_return, 1, True)),
                    ("Daily log-growth maximizing size", pct(log_optimal)),
                    ("Sessions in one run", f"{horizon_days:,}"),
                    ("Bootstrap block length", f"{BLOCK_SIZE} sessions"),
                ],
                columns=["Metric", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

caps = pd.DataFrame(
    [
        ("Conviction ceiling", sizing.conviction_ceiling, f"Conviction {conviction} × 5%"),
        ("Volatility adjustment", sizing.volatility_cap, "Current 63D volatility versus full-history median"),
        ("Invalidation loss budget", sizing.invalidation_cap, "NAV loss budget divided by invalidation distance"),
        ("Earnings / event risk", sizing.event_cap, event_basis if hold_earnings else "Disabled"),
        ("Historical tail risk", sizing.tail_cap, "NAV loss budget divided by five-day adverse tail"),
        ("Liquidity", sizing.liquidity_cap, f"{participation}% of median dollar volume across {liquidation_days} days"),
    ],
    columns=["Constraint", "Maximum position", "Method"],
)
caps["Reduction vs ceiling"] = sizing.conviction_ceiling - caps["Maximum position"]
for column in ("Maximum position", "Reduction vs ceiling"):
    caps[column] *= 100

sizing_tab, paths_tab, events_tab, sensitivity_tab = st.tabs(
    ("Sizing attribution", "Historical paths", "Event + tail risk", "Sensitivities")
)
with sizing_tab:
    st.dataframe(
        caps,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Maximum position": st.column_config.NumberColumn(format="%.1f%%"),
            "Reduction vs ceiling": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    metrics = pd.DataFrame(
        [
            ("Latest price", money(latest)),
            ("21D realized volatility", pct(annualized_volatility(returns, 21))),
            ("63D realized volatility", pct(current_vol)),
            ("252D realized volatility", pct(annualized_volatility(returns, 252))),
            ("Historical median 63D volatility", pct(median_vol)),
            ("Maximum historical drawdown", pct(maximum_drawdown_from_prices(close))),
            ("1% daily expected shortfall", pct(daily_es)),
            ("Five-day adverse tail", pct(tail_move)),
            ("Median daily dollar volume", money(median_dollar_volume)),
        ],
        columns=["Metric", "Value"],
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

with paths_tab:
    if paths.empty:
        st.info("Insufficient history for this holding period.")
    else:
        st.plotly_chart(
            distribution_chart(paths, max(target_move, 0), stop_distance),
            use_container_width=True,
        )
        touch = first_touch_statistics(
            ohlcv,
            horizon_days,
            direction.lower(),
            max(target_move, 0),
            stop_distance,
            step=step,
        )
        if touch:
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Target reached first", touch["target_first_rate"] * 100, touch["target_first"]),
                        ("Invalidation reached first", touch["stop_first_rate"] * 100, touch["stop_first"]),
                        ("Both reached same session", touch["same_day_rate"] * 100, touch["same_day"]),
                        ("Neither reached", touch["neither_rate"] * 100, touch["neither"]),
                    ],
                    columns=["Outcome", "Rate", "Observations"],
                ),
                use_container_width=True,
                hide_index=True,
                column_config={"Rate": st.column_config.NumberColumn(format="%.1f%%")},
            )
        dataframe_download(
            "Download historical paths",
            paths,
            f"{ticker.lower()}_{horizon_label.replace(' ', '_')}_paths.csv",
        )

with events_tab:
    render_kpi_cards(
        [
            ("Event move", pct(event_move), event_basis),
            ("Event NAV loss", pct(event_nav, 2), "Suggested exposure × event move"),
            ("Five-day tail", pct(tail_move), "Historical adverse tail"),
            ("Next earnings", next_earnings.strftime("%Y-%m-%d") if next_earnings is not None else "N/A", "Yahoo schedule when available"),
        ]
    )
    if events.empty:
        st.info("No earnings history was returned. Event sizing uses the historical overnight-gap distribution.")
    else:
        display = events.sort_values("date", ascending=False).copy()
        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
        for column in ("gap", "session", "close_to_close", "abs_move"):
            display[column] *= 100
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.1f%%")
                for column in ("gap", "session", "close_to_close", "abs_move")
            },
        )

with sensitivity_tab:
    sensitivity = sensitivity_table(returns, frames, direction)
    if sensitivity.empty:
        st.info("Insufficient overlapping data for sensitivity analysis.")
    else:
        st.dataframe(
            sensitivity,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Position correlation": st.column_config.NumberColumn(format="%.2f"),
                "Position beta": st.column_config.NumberColumn(format="%.2f"),
            },
        )

if not missing.empty:
    with st.expander("Data diagnostics"):
        st.dataframe(missing, use_container_width=True, hide_index=True)

render_footer(
    data_note=(
        "Primary inputs: Yahoo Finance adjusted daily OHLCV, available earnings dates, and liquid market proxies. The live simulation advances through observed daily returns for one selected holding period and is descriptive rather than predictive."
    )
)
