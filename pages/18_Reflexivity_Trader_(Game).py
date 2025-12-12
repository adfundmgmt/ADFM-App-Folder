# app.py
# Reflexivity Trader: a market micro-sim game for Streamlit
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="Reflexivity Trader", layout="wide")
st.title("Reflexivity Trader")
st.caption("A reflexivity-driven trading game: macro regime, narrative, flows, and feedback loops.")

# ---------------------------
# Helpers
# ---------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fmt_pct(x):
    return f"{x*100:.2f}%"

def fmt_money(x):
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.0f}"

def dd_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0

def sharpe(returns: pd.Series, periods_per_year=52):
    if returns.std(ddof=0) == 0:
        return 0.0
    return (returns.mean() / returns.std(ddof=0)) * math.sqrt(periods_per_year)

def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def normalize_to_gross(weights: pd.Series, max_gross: float) -> pd.Series:
    gross = float(np.abs(weights).sum())
    if gross <= max_gross + 1e-12:
        return weights
    if gross == 0:
        return weights
    return weights * (max_gross / gross)

def make_equity_curve_chart(nav: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, mode="lines", name="NAV"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Week",
        yaxis_title="NAV",
        hovermode="x unified",
    )
    return fig

def make_drawdown_chart(nav: pd.Series):
    dd = dd_series(nav)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Week",
        yaxis_title="Drawdown",
        hovermode="x unified",
    )
    return fig

def make_prices_chart(prices: pd.DataFrame):
    fig = go.Figure()
    for c in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[c], mode="lines", name=c))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Week",
        yaxis_title="Price Index",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def make_positions_chart(weights: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weights.index.tolist(), y=weights.values.tolist(), name="Target weight"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Asset",
        yaxis_title="Weight (x NAV)",
    )
    return fig

# ---------------------------
# Game configuration
# ---------------------------
ASSETS = [
    "US Equities",
    "Semis",
    "Long Bonds",
    "Credit",
    "Gold",
    "Oil",
    "USDJPY",
    "BTC",
]

# Weekly vol assumptions (roughly)
BASE_VOL = pd.Series(
    {
        "US Equities": 0.020,
        "Semis": 0.030,
        "Long Bonds": 0.015,
        "Credit": 0.010,
        "Gold": 0.015,
        "Oil": 0.035,
        "USDJPY": 0.010,
        "BTC": 0.050,
    }
)

# Macro betas (how each asset reacts to macro factors)
# Factors: growth, inflation, liquidity, risk_aversion, usd_strength
BETA = pd.DataFrame(
    {
        "growth":     [ 0.70,  1.10, -0.50,  0.40,  0.10,  0.60,  0.10,  1.00],
        "inflation":  [ 0.10,  0.10, -0.60, -0.10,  0.70,  0.80,  0.20,  0.20],
        "liquidity":  [ 0.80,  1.20,  0.20,  0.40,  0.10,  0.20, -0.10,  1.30],
        "risk_av":    [-0.90, -1.30,  0.60, -0.60,  0.30, -0.80,  0.50, -1.40],
        "usd":        [-0.20, -0.30,  0.10,  0.05, -0.30, -0.15,  1.00, -0.60],
    },
    index=ASSETS
)

# Narrative events library
EVENTS = [
    {
        "name": "Fed: hawkish surprise",
        "shock": {"growth": -0.3, "inflation": -0.1, "liquidity": -0.6, "risk_av": +0.5, "usd": +0.4},
        "text": "Rates repriced higher. Financial conditions tighten. The crowd scrambles to de-risk."
    },
    {
        "name": "Fed: dovish pivot tease",
        "shock": {"growth": +0.2, "inflation":  0.0, "liquidity": +0.7, "risk_av": -0.5, "usd": -0.3},
        "text": "Liquidity impulse improves. Risk premia compress. Momentum gets oxygen."
    },
    {
        "name": "Hot CPI print",
        "shock": {"growth": -0.1, "inflation": +0.6, "liquidity": -0.3, "risk_av": +0.4, "usd": +0.2},
        "text": "Inflation narrative returns. Bonds wobble. Real rates matter again."
    },
    {
        "name": "Growth scare",
        "shock": {"growth": -0.7, "inflation": -0.3, "liquidity": +0.1, "risk_av": +0.8, "usd": +0.1},
        "text": "Forward expectations crack. Defensive flows dominate. The crowd sells what worked."
    },
    {
        "name": "AI capex boom headline",
        "shock": {"growth": +0.5, "inflation": +0.1, "liquidity": +0.2, "risk_av": -0.3, "usd": -0.1},
        "text": "Narrative bids up beta with a semi tilt. Reflexivity kicks in through price chasing."
    },
    {
        "name": "Geopolitical shock",
        "shock": {"growth": -0.3, "inflation": +0.3, "liquidity": -0.2, "risk_av": +0.9, "usd": +0.3},
        "text": "Risk off. Correlations go to one. Liquidity thins at the worst time."
    },
    {
        "name": "Credit accident rumor",
        "shock": {"growth": -0.4, "inflation": -0.1, "liquidity": -0.4, "risk_av": +1.0, "usd": +0.2},
        "text": "Tails get priced. Spreads widen. The crowd stops believing the soft landing story."
    },
    {
        "name": "Disinflation resumes",
        "shock": {"growth": +0.1, "inflation": -0.6, "liquidity": +0.3, "risk_av": -0.4, "usd": -0.2},
        "text": "Rates pressure eases. Duration breathes. Risk premia compress again."
    },
]

# Difficulty presets
DIFFICULTY = {
    "Classic": {"event_prob": 0.65, "crowd_alpha": 0.45, "flow_impact": 0.08, "crash_tail": 0.020},
    "Hard":    {"event_prob": 0.75, "crowd_alpha": 0.60, "flow_impact": 0.11, "crash_tail": 0.030},
    "Chaos":   {"event_prob": 0.85, "crowd_alpha": 0.75, "flow_impact": 0.14, "crash_tail": 0.045},
}

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.subheader("Game settings")
    colA, colB = st.columns(2)
    with colA:
        rounds = st.number_input("Weeks", min_value=12, max_value=60, value=26, step=1)
        max_gross = st.slider("Max gross (x NAV)", 0.5, 4.0, 2.0, 0.1)
    with colB:
        tc_bps = st.slider("Transaction cost (bps per $ traded)", 0, 50, 8, 1)
        difficulty = st.selectbox("Difficulty", list(DIFFICULTY.keys()), index=0)

    seed = st.number_input("Seed", min_value=1, max_value=999999, value=42069, step=1)
    st.caption("Tip: keep the seed stable to replay the same path and iterate your process.")

    c1, c2 = st.columns(2)
    reset = c1.button("Restart run", use_container_width=True)
    fast_forward = c2.button("Auto-play to end", use_container_width=True)

# ---------------------------
# Session state init / reset
# ---------------------------
def init_game():
    rng = np.random.default_rng(int(seed))
    start_nav = 1_000_000.0

    prices = pd.DataFrame(index=[0], columns=ASSETS, data=100.0)
    nav = pd.Series(index=[0], data=start_nav, dtype=float)
    port_rets = pd.Series(index=[0], data=0.0, dtype=float)

    # Player weights (x NAV)
    w0 = pd.Series(index=ASSETS, data=0.0, dtype=float)

    # Crowd positioning (x NAV equivalent), starts light risk-on
    crowd = pd.Series(index=ASSETS, data=0.0, dtype=float)
    crowd["US Equities"] = 0.35
    crowd["Semis"] = 0.25
    crowd["Long Bonds"] = -0.05
    crowd["BTC"] = 0.10
    crowd = normalize_to_gross(crowd, max_gross=1.2)

    # Macro state
    macro = {
        "growth": 0.15,
        "inflation": 0.05,
        "liquidity": 0.10,
        "risk_av": 0.10,
        "usd": 0.10,
        "vix": 18.0,
        "ust10y": 4.25,
        "fc": -0.10,  # financial conditions proxy (lower is easier)
    }

    log = []

    st.session_state.game = {
        "rng": rng,
        "t": 0,
        "rounds": int(rounds),
        "start_nav": float(start_nav),
        "nav": nav,
        "port_rets": port_rets,
        "prices": prices,
        "weights": w0,
        "prev_weights": w0.copy(),
        "crowd": crowd,
        "macro": macro,
        "last_event": None,
        "log": log,
        "max_gross": float(max_gross),
        "tc_bps": int(tc_bps),
        "difficulty": difficulty,
    }

if reset or "game" not in st.session_state:
    init_game()

G = st.session_state.game

# If user changes settings mid-run, keep the current run stable and only apply max gross dynamically
G["max_gross"] = float(max_gross)
G["tc_bps"] = int(tc_bps)
G["difficulty"] = difficulty

# ---------------------------
# Core simulation
# ---------------------------
def simulate_one_step(target_weights: pd.Series):
    rng = G["rng"]
    t = G["t"]
    max_g = G["max_gross"]
    tc = G["tc_bps"] / 10_000.0
    diff = DIFFICULTY[G["difficulty"]]

    target_weights = target_weights.reindex(ASSETS).fillna(0.0).astype(float)
    target_weights = normalize_to_gross(target_weights, max_gross=max_g)

    prev_w = G["weights"].copy()

    # Transaction costs based on turnover in gross notional terms
    turnover = float(np.abs(target_weights - prev_w).sum())
    tc_cost = turnover * tc

    # Macro mean reversion with noise
    m = G["macro"].copy()
    for k in ["growth", "inflation", "liquidity", "risk_av", "usd"]:
        m[k] = 0.85 * m[k] + 0.15 * float(rng.normal(0, 0.20))

    # Event draw
    event = None
    if rng.uniform() < diff["event_prob"]:
        event = EVENTS[int(rng.integers(0, len(EVENTS)))]
        for k, v in event["shock"].items():
            m[k] = float(m[k] + v)

    # Derive market observables
    # VIX up with risk aversion, down with liquidity
    m["vix"] = float(clamp(16.0 + 10.0 * m["risk_av"] - 5.0 * m["liquidity"] + rng.normal(0, 1.5), 10.0, 60.0))
    # 10y yield rises with growth and inflation, falls with risk aversion
    m["ust10y"] = float(clamp(3.75 + 0.45 * m["growth"] + 0.55 * m["inflation"] - 0.30 * m["risk_av"] + rng.normal(0, 0.08), 0.5, 8.0))
    # Financial conditions proxy: tighter with higher VIX and stronger USD, easier with liquidity
    m["fc"] = float(clamp(0.05 * (m["vix"] - 18) + 0.8 * m["usd"] - 0.9 * m["liquidity"], -3.0, 3.0))

    # Crowd behavior: chases last week winners but panics when risk av rises
    prices = G["prices"]
    last_ret = pd.Series(index=ASSETS, data=0.0)
    if t > 0:
        last_ret = prices.loc[t, ASSETS] / prices.loc[t - 1, ASSETS] - 1.0

    crowd = G["crowd"].copy()
    chase = softmax(last_ret.values / (last_ret.std() + 1e-6))
    chase = pd.Series(chase, index=ASSETS)

    crowd_alpha = diff["crowd_alpha"]
    panic = clamp(m["risk_av"], 0.0, 2.0)
    crowd_target = 0.7 * crowd + crowd_alpha * (chase - chase.mean())

    # Panic rotation: crowd cuts beta and adds duration/gold/usd
    crowd_target["US Equities"] -= 0.25 * panic
    crowd_target["Semis"] -= 0.30 * panic
    crowd_target["BTC"] -= 0.30 * panic
    crowd_target["Long Bonds"] += 0.20 * panic
    crowd_target["Gold"] += 0.15 * panic
    crowd_target["USDJPY"] += 0.12 * panic  # stronger USD vs JPY in risk off
    crowd_target = normalize_to_gross(crowd_target, max_gross=1.4)

    # Reflexive flow impact: crowd changes push prices, which then feed back
    flow = (crowd_target - crowd)
    flow_impact = diff["flow_impact"]
    flow_term = flow * flow_impact

    # Base returns from macro factors + idiosyncratic vol
    macro_vec = np.array([m["growth"], m["inflation"], m["liquidity"], m["risk_av"], m["usd"]], dtype=float)
    base = (BETA.values @ macro_vec) * 0.010  # scale to weekly
    base = pd.Series(base, index=ASSETS)

    eps = pd.Series(rng.normal(0, 1.0, size=len(ASSETS)), index=ASSETS)
    vol = BASE_VOL.copy()

    # Tail risk: occasional crash that hits crowded beta
    crash = 0.0
    if rng.uniform() < diff["crash_tail"]:
        crash = float(abs(rng.normal(0.08, 0.03)))
        base["US Equities"] -= 0.6 * crash
        base["Semis"] -= 0.9 * crash
        base["BTC"] -= 1.1 * crash
        base["Credit"] -= 0.7 * crash
        base["Oil"] -= 0.5 * crash

    asset_rets = base + flow_term + eps * vol

    # Keep USDJPY return direction intuitive: positive means USD up vs JPY
    # Already handled via usd factor and panic rotation; no extra adjustment here.

    # Portfolio return and NAV update
    gross = float(np.abs(target_weights).sum())
    if gross > max_g + 1e-9:
        target_weights = normalize_to_gross(target_weights, max_gross=max_g)

    port_ret = float((target_weights * asset_rets).sum()) - tc_cost
    new_nav = float(G["nav"].loc[t] * (1.0 + port_ret))

    # Update state series
    new_t = t + 1
    new_prices_row = (prices.loc[t, ASSETS] * (1.0 + asset_rets)).to_frame().T
    new_prices_row.index = [new_t]

    G["prices"] = pd.concat([prices, new_prices_row], axis=0)

    new_nav_s = pd.Series(index=[new_t], data=[new_nav], dtype=float)
    new_ret_s = pd.Series(index=[new_t], data=[port_ret], dtype=float)
    G["nav"] = pd.concat([G["nav"], new_nav_s], axis=0)
    G["port_rets"] = pd.concat([G["port_rets"], new_ret_s], axis=0)

    G["prev_weights"] = prev_w
    G["weights"] = target_weights
    G["crowd"] = crowd_target
    G["macro"] = m
    G["t"] = new_t
    G["last_event"] = event

    # Log
    G["log"].append(
        {
            "week": new_t,
            "event": None if event is None else event["name"],
            "vix": m["vix"],
            "ust10y": m["ust10y"],
            "fc": m["fc"],
            "turnover": turnover,
            "tc_cost": tc_cost,
            "port_ret": port_ret,
            "nav": new_nav,
            "crash": crash,
            **{f"w_{a}": float(target_weights[a]) for a in ASSETS},
            **{f"r_{a}": float(asset_rets[a]) for a in ASSETS},
            **{f"crowd_{a}": float(crowd_target[a]) for a in ASSETS},
        }
    )

def score_run():
    nav = G["nav"]
    rets = G["port_rets"].iloc[1:] if len(G["port_rets"]) > 1 else G["port_rets"]
    dd = dd_series(nav)
    max_dd = float(dd.min()) if len(dd) else 0.0
    sh = float(sharpe(rets)) if len(rets) > 5 else 0.0
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)

    # Scoring: reward return and risk-adjusted behavior, penalize deep drawdowns
    # Score is intentionally simple and gamey
    score = (total * 100.0) + (sh * 6.0) + (max_dd * 140.0)
    return {
        "total_return": total,
        "sharpe": sh,
        "max_dd": max_dd,
        "score": score,
        "end_nav": float(nav.iloc[-1]),
    }

# ---------------------------
# Game header status
# ---------------------------
t = G["t"]
R = G["rounds"]

left, mid, right = st.columns([1.1, 1.3, 1.2])
with left:
    st.subheader(f"Week {t} of {R}")
    nav_now = float(G["nav"].iloc[-1])
    st.metric("NAV", fmt_money(nav_now), None)

with mid:
    m = G["macro"]
    st.subheader("Macro tape")
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX", f"{m['vix']:.1f}")
    c2.metric("UST 10Y", f"{m['ust10y']:.2f}%")
    c3.metric("Fin conditions", f"{m['fc']:+.2f}")

with right:
    s = score_run()
    st.subheader("Risk and score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", fmt_pct(s["total_return"]))
    c2.metric("Max DD", fmt_pct(s["max_dd"]))
    c3.metric("Score", f"{s['score']:.1f}")

# Event box
event = G["last_event"]
if event is None and t == 0:
    st.info("You start flat. Set positions, submit, then see the tape update. Prices move with macro, news, and crowd flows.")
elif event is None:
    st.info("No headline this week. Tape drifts. Flows still matter.")
else:
    st.warning(f"{event['name']}: {event['text']}")

# End conditions
start_nav = G["start_nav"]
if nav_now <= 0.55 * start_nav:
    st.error("Margin call. NAV fell below the survival threshold. Restart and try a different process.")
    st.stop()

if t >= R:
    st.success("Run complete.")
    st.write("Final stats update live as you replay. If you want a version with levels, options, or a full macro dashboard, tell me what style you want.")
    st.stop()

# ---------------------------
# Position input
# ---------------------------
st.markdown("### Your book")
st.caption("Weights are in x NAV. Positive is long, negative is short. Gross leverage is the sum of absolute weights.")

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

def apply_preset(name):
    w = pd.Series(index=ASSETS, data=0.0, dtype=float)
    if name == "Risk on":
        w["US Equities"] = 0.60
        w["Semis"] = 0.55
        w["BTC"] = 0.25
        w["Credit"] = 0.35
        w["Long Bonds"] = -0.15
        w["USDJPY"] = -0.10
    if name == "Risk off":
        w["Long Bonds"] = 0.75
        w["Gold"] = 0.40
        w["USDJPY"] = 0.25
        w["Credit"] = -0.25
        w["Semis"] = -0.35
    if name == "Balanced":
        w["US Equities"] = 0.35
        w["Semis"] = 0.20
        w["Long Bonds"] = 0.25
        w["Gold"] = 0.15
        w["Credit"] = 0.20
        w["USDJPY"] = 0.05
    if name == "Mean reversion":
        # Fade what the crowd owns
        crowd = G["crowd"].copy()
        w = -0.9 * crowd
        # Add a small duration anchor
        w["Long Bonds"] += 0.15
    return normalize_to_gross(w, max_gross=G["max_gross"])

if preset_col1.button("Risk on", use_container_width=True):
    G["weights"] = apply_preset("Risk on")
if preset_col2.button("Risk off", use_container_width=True):
    G["weights"] = apply_preset("Risk off")
if preset_col3.button("Balanced", use_container_width=True):
    G["weights"] = apply_preset("Balanced")
if preset_col4.button("Mean reversion", use_container_width=True):
    G["weights"] = apply_preset("Mean reversion")

w_in = G["weights"].copy()

edit_df = pd.DataFrame({"asset": ASSETS, "weight_x_nav": [float(w_in[a]) for a in ASSETS]})
edited = st.data_editor(
    edit_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "asset": st.column_config.TextColumn("Asset", disabled=True),
        "weight_x_nav": st.column_config.NumberColumn(
            "Weight (x NAV)",
            help="Example: 0.50 means 50% long. -0.50 means 50% short.",
            step=0.05,
            format="%.2f",
        ),
    },
)

w_target = pd.Series(index=ASSETS, data=0.0, dtype=float)
for _, row in edited.iterrows():
    w_target[str(row["asset"])] = float(row["weight_x_nav"])

gross = float(np.abs(w_target).sum())
net = float(w_target.sum())
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Gross", f"{gross:.2f}x")
info_col2.metric("Net", f"{net:.2f}x")
est_tc = float(np.abs(w_target - G["weights"]).sum()) * (G["tc_bps"] / 10_000.0)
info_col3.metric("Est TC", fmt_pct(est_tc))

if gross > G["max_gross"] + 1e-9:
    st.warning(f"Gross exceeds limit. It will be scaled down automatically to {G['max_gross']:.2f}x when you submit.")

submit_col1, submit_col2, submit_col3 = st.columns([1.2, 1.0, 1.0])
submit = submit_col1.button("Submit trades and advance 1 week", type="primary", use_container_width=True)
peek = submit_col2.checkbox("Show crowd positioning", value=False)
show_log = submit_col3.checkbox("Show run log", value=False)

if submit:
    simulate_one_step(w_target)

# Auto-play
if fast_forward and G["t"] < G["rounds"]:
    # Move quickly but safely: cap iterations to avoid UI lockups
    # Streamlit reruns per click, so we just step to end here in one run
    steps_left = G["rounds"] - G["t"]
    steps_left = int(clamp(steps_left, 0, 80))
    w_ff = normalize_to_gross(w_target, max_gross=G["max_gross"])
    for _ in range(steps_left):
        if float(G["nav"].iloc[-1]) <= 0.55 * G["start_nav"]:
            break
        if G["t"] >= G["rounds"]:
            break
        simulate_one_step(w_ff)

# ---------------------------
# Charts
# ---------------------------
st.markdown("### Tape")
cL, cR = st.columns([1.2, 1.0])
with cL:
    st.plotly_chart(make_prices_chart(G["prices"]), use_container_width=True)
with cR:
    st.plotly_chart(make_equity_curve_chart(G["nav"]), use_container_width=True)
    st.plotly_chart(make_drawdown_chart(G["nav"]), use_container_width=True)

st.markdown("### Current exposures")
st.plotly_chart(make_positions_chart(normalize_to_gross(w_target, max_gross=G["max_gross"])), use_container_width=True)

if peek:
    st.markdown("### Crowd book (what the market is leaning into)")
    st.plotly_chart(make_positions_chart(G["crowd"]), use_container_width=True)

# ---------------------------
# Stats + log
# ---------------------------
st.markdown("### Run stats")
rets = G["port_rets"].iloc[1:] if len(G["port_rets"]) > 1 else G["port_rets"]
dd = dd_series(G["nav"])
stats = {
    "End NAV": fmt_money(float(G["nav"].iloc[-1])),
    "Total return": fmt_pct(float(G["nav"].iloc[-1] / G["nav"].iloc[0] - 1.0)),
    "Max drawdown": fmt_pct(float(dd.min()) if len(dd) else 0.0),
    "Weekly Sharpe (annualized)": f"{sharpe(rets):.2f}" if len(rets) > 5 else "0.00",
    "Avg weekly return": fmt_pct(float(rets.mean()) if len(rets) else 0.0),
    "Weekly vol": fmt_pct(float(rets.std(ddof=0)) if len(rets) else 0.0),
    "Score": f"{score_run()['score']:.1f}",
}
st.dataframe(pd.DataFrame(stats, index=["Value"]).T, use_container_width=True)

if show_log:
    st.markdown("### Run log")
    if len(G["log"]) == 0:
        st.write("No steps yet.")
    else:
        df_log = pd.DataFrame(G["log"])
        keep_cols = ["week", "event", "vix", "ust10y", "fc", "turnover", "tc_cost", "port_ret", "nav", "crash"]
        st.dataframe(df_log[keep_cols].set_index("week"), use_container_width=True)

st.caption("Design note: this game is built around feedback loops. If you want a version where you can trade options, set stops, or run a multi-strategy book with factor constraints, say what instruments you want and what rules you want enforced.")
