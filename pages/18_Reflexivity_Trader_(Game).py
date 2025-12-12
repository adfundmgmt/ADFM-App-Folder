# app.py
# Visual Reflexivity Arena (Streamlit financial game)
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Reflexivity Arena", layout="wide")

ASSETS = ["SPX", "SEMIS", "BONDS", "GOLD", "BTC", "USD"]
BASE_VOL_W = pd.Series({"SPX": 0.020, "SEMIS": 0.030, "BONDS": 0.015, "GOLD": 0.015, "BTC": 0.055, "USD": 0.010})

# Factor betas: growth, inflation, liquidity, risk_aversion, usd_strength
BETA = pd.DataFrame(
    {
        "growth":     [ 0.70,  1.10, -0.45,  0.10,  1.00,  0.10],
        "inflation":  [ 0.10,  0.05, -0.55,  0.70,  0.20,  0.25],
        "liquidity":  [ 0.85,  1.25,  0.15,  0.10,  1.35, -0.10],
        "risk_av":    [-0.95, -1.35,  0.65,  0.25, -1.55,  0.55],
        "usd":        [-0.20, -0.30,  0.10, -0.25, -0.70,  1.00],
    },
    index=ASSETS,
)

EVENTS = [
    ("Fed hawkish surprise", {"growth": -0.25, "inflation": -0.05, "liquidity": -0.55, "risk_av": +0.55, "usd": +0.35},
     "Rates reprice higher. Financial conditions tighten. Crowds cut beta."),
    ("Fed dovish leak", {"growth": +0.15, "inflation":  0.00, "liquidity": +0.65, "risk_av": -0.55, "usd": -0.30},
     "Liquidity impulse improves. Risk premia compress. Momentum re-ignites."),
    ("Hot CPI", {"growth": -0.05, "inflation": +0.60, "liquidity": -0.25, "risk_av": +0.35, "usd": +0.20},
     "Inflation risk returns. Duration gets hit. Cross-asset correlations jump."),
    ("Growth scare", {"growth": -0.70, "inflation": -0.30, "liquidity": +0.10, "risk_av": +0.80, "usd": +0.05},
     "Forward expectations crack. Defensive flows dominate. Leaders get sold."),
    ("AI capex boom", {"growth": +0.55, "inflation": +0.10, "liquidity": +0.20, "risk_av": -0.30, "usd": -0.10},
     "Narrative bids up beta with a SEMIS tilt. Price action pulls flows in."),
    ("Geopolitical shock", {"growth": -0.30, "inflation": +0.30, "liquidity": -0.15, "risk_av": +0.95, "usd": +0.30},
     "Risk off. Liquidity thins. Correlations go to one at the worst moment."),
    ("Credit accident rumor", {"growth": -0.45, "inflation": -0.05, "liquidity": -0.40, "risk_av": +1.00, "usd": +0.20},
     "Tails get priced. Spreads widen. Crowds stop believing the soft landing story."),
    ("Disinflation resumes", {"growth": +0.10, "inflation": -0.65, "liquidity": +0.30, "risk_av": -0.45, "usd": -0.20},
     "Rates pressure eases. Duration breathes. Risk premia compress again."),
]

DIFFICULTY = {
    "Classic": {"event_prob": 0.65, "flow_impact": 0.10, "crowd_chase": 0.45, "tail_prob": 0.020},
    "Hard":    {"event_prob": 0.75, "flow_impact": 0.13, "crowd_chase": 0.60, "tail_prob": 0.030},
    "Chaos":   {"event_prob": 0.85, "flow_impact": 0.16, "crowd_chase": 0.75, "tail_prob": 0.045},
}

ACTIONS = {
    "Risk On":  {"SPX": 0.70, "SEMIS": 0.70, "BTC": 0.25, "BONDS": -0.20, "GOLD": 0.05, "USD": -0.10},
    "Risk Off": {"SPX": -0.25, "SEMIS": -0.30, "BTC": -0.10, "BONDS": 0.90, "GOLD": 0.35, "USD": 0.20},
    "Barbell":  {"SPX": 0.35, "SEMIS": 0.20, "BTC": 0.10, "BONDS": 0.45, "GOLD": 0.20, "USD": 0.00},
    "Fade Crowd": None,  # computed from crowd
    "Flat":     {"SPX": 0.00, "SEMIS": 0.00, "BTC": 0.00, "BONDS": 0.00, "GOLD": 0.00, "USD": 0.00},
}

# ----------------------------
# Styling (visual game UI)
# ----------------------------
st.markdown(
    """
    <style>
      .appview-container { background: radial-gradient(1200px 600px at 20% 0%, rgba(40,80,180,0.22), transparent 60%),
                                        radial-gradient(1000px 600px at 80% 20%, rgba(220,120,60,0.14), transparent 55%),
                                        linear-gradient(180deg, rgba(10,12,18,1) 0%, rgba(8,10,14,1) 100%) !important; }
      h1, h2, h3, p, div, span { color: rgba(245,245,250,0.92); }
      .panel {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 14px 14px 12px 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      }
      .ticker {
        white-space: nowrap;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(0,0,0,0.25);
        border-radius: 14px;
        padding: 10px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 13px;
      }
      .ticker span {
        display: inline-block;
        padding-left: 100%;
        animation: scroll 18s linear infinite;
      }
      @keyframes scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
      }
      .bigbtn button {
        height: 58px !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.06) !important;
      }
      .bigbtn button:hover { background: rgba(255,255,255,0.10) !important; }
      .danger { color: #ff6b6b; font-weight: 700; }
      .good { color: #5CFFB0; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_to_gross(w: pd.Series, max_gross: float) -> pd.Series:
    w = w.reindex(ASSETS).fillna(0.0).astype(float)
    g = float(np.abs(w).sum())
    if g <= max_gross + 1e-12 or g == 0:
        return w
    return w * (max_gross / g)

def dd_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0

def make_arena_chart(df_prices: pd.DataFrame, nav: pd.Series):
    fig = go.Figure()
    for c in df_prices.columns:
        fig.add_trace(go.Scatter(x=df_prices.index, y=df_prices[c], mode="lines", name=c))
    fig.update_layout(
        height=440,
        margin=dict(l=12, r=12, t=30, b=10),
        xaxis_title="Week",
        yaxis_title="Index",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="rgba(245,245,250,0.92)"),
    )

    nav_norm = 100.0 * nav / nav.iloc[0]
    fig.add_trace(go.Scatter(x=nav_norm.index, y=nav_norm.values, mode="lines", name="YOU (NAV)", line=dict(width=4)))
    return fig

def score_snapshot(nav: pd.Series, rets: pd.Series):
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    dd = dd_series(nav)
    max_dd = float(dd.min()) if len(dd) else 0.0
    if rets.std(ddof=0) == 0:
        sh = 0.0
    else:
        sh = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(52)) if len(rets) > 6 else 0.0
    score = (total * 100.0) + (sh * 7.0) + (max_dd * 160.0)
    return total, max_dd, sh, score

# ----------------------------
# Game state
# ----------------------------
def init_game(seed: int, weeks: int, max_gross: float, tc_bps: int, difficulty: str):
    rng = np.random.default_rng(int(seed))
    prices = pd.DataFrame(index=[0], data={a: 100.0 for a in ASSETS})
    nav = pd.Series(index=[0], data=[1_000_000.0], dtype=float)
    rets = pd.Series(index=[0], data=[0.0], dtype=float)

    crowd = pd.Series(index=ASSETS, data=0.0, dtype=float)
    crowd["SPX"] = 0.35
    crowd["SEMIS"] = 0.25
    crowd["BTC"] = 0.10
    crowd["BONDS"] = -0.05
    crowd = normalize_to_gross(crowd, max_gross=1.2)

    macro = {"growth": 0.15, "inflation": 0.05, "liquidity": 0.10, "risk_av": 0.10, "usd": 0.10, "vix": 18.0, "ust10y": 4.25}

    st.session_state.G = {
        "rng": rng,
        "t": 0,
        "weeks": int(weeks),
        "max_gross": float(max_gross),
        "tc_bps": int(tc_bps),
        "difficulty": difficulty,
        "prices": prices,
        "nav": nav,
        "rets": rets,
        "crowd": crowd,
        "macro": macro,
        "last_event": None,
        "last_event_text": None,
        "last_asset_rets": pd.Series(index=ASSETS, data=0.0),
        "last_action": "Flat",
        "log": [],
    }

def simulate_week(action_name: str):
    G = st.session_state.G
    rng = G["rng"]
    diff = DIFFICULTY[G["difficulty"]]
    t = G["t"]
    max_gross = G["max_gross"]
    tc = G["tc_bps"] / 10_000.0

    # Build player weights
    if action_name == "Fade Crowd":
        w = -0.95 * G["crowd"].copy()
        w["BONDS"] += 0.20
    else:
        w = pd.Series(ACTIONS[action_name], dtype=float)
    w = normalize_to_gross(w, max_gross=max_gross)

    # Simple transaction costs vs prior action weights (tracked in log)
    prev_w = pd.Series(G["log"][-1]["weights"], index=ASSETS) if len(G["log"]) else pd.Series(index=ASSETS, data=0.0)
    turnover = float(np.abs(w - prev_w).sum())
    tc_cost = turnover * tc

    # Macro evolves (mean reversion + noise)
    m = G["macro"].copy()
    for k in ["growth", "inflation", "liquidity", "risk_av", "usd"]:
        m[k] = float(0.85 * m[k] + 0.15 * rng.normal(0, 0.20))

    # Headline event
    event_name, event_shock, event_text = (None, None, None)
    if rng.uniform() < diff["event_prob"]:
        e = EVENTS[int(rng.integers(0, len(EVENTS)))]
        event_name, event_shock, event_text = e
        for k, v in event_shock.items():
            m[k] = float(m[k] + v)

    # Market observables
    m["vix"] = float(clamp(16.0 + 10.0 * m["risk_av"] - 5.0 * m["liquidity"] + rng.normal(0, 1.5), 10.0, 70.0))
    m["ust10y"] = float(clamp(3.75 + 0.45 * m["growth"] + 0.55 * m["inflation"] - 0.30 * m["risk_av"] + rng.normal(0, 0.08), 0.5, 8.0))

    # Crowd chases winners, then panics as risk_av rises
    prices = G["prices"]
    last_ret = pd.Series(index=ASSETS, data=0.0)
    if t > 0:
        last_ret = prices.loc[t, ASSETS] / prices.loc[t - 1, ASSETS] - 1.0

    crowd = G["crowd"].copy()
    std = float(last_ret.std()) if float(last_ret.std()) > 1e-9 else 1e-9
    chase_signal = (last_ret / std).clip(-2, 2)
    crowd_target = 0.75 * crowd + diff["crowd_chase"] * chase_signal

    panic = clamp(m["risk_av"], 0.0, 2.0)
    crowd_target["SPX"] -= 0.20 * panic
    crowd_target["SEMIS"] -= 0.28 * panic
    crowd_target["BTC"] -= 0.30 * panic
    crowd_target["BONDS"] += 0.22 * panic
    crowd_target["GOLD"] += 0.12 * panic
    crowd_target["USD"] += 0.14 * panic
    crowd_target = normalize_to_gross(crowd_target, max_gross=1.4)

    # Reflexive flow term
    flow = crowd_target - crowd
    flow_term = flow * diff["flow_impact"]

    # Base returns from factors
    macro_vec = np.array([m["growth"], m["inflation"], m["liquidity"], m["risk_av"], m["usd"]], dtype=float)
    base = (BETA.values @ macro_vec) * 0.010
    base = pd.Series(base, index=ASSETS)

    # Tail shock
    crash = 0.0
    if rng.uniform() < diff["tail_prob"]:
        crash = float(abs(rng.normal(0.09, 0.03)))
        base["SPX"] -= 0.55 * crash
        base["SEMIS"] -= 0.85 * crash
        base["BTC"] -= 1.10 * crash

    eps = pd.Series(rng.normal(0, 1.0, size=len(ASSETS)), index=ASSETS)
    asset_rets = base + flow_term + eps * BASE_VOL_W

    # Portfolio return
    port_ret = float((w * asset_rets).sum()) - tc_cost
    new_nav = float(G["nav"].iloc[-1] * (1.0 + port_ret))

    # Update time series
    new_t = t + 1
    new_prices = (prices.loc[t, ASSETS] * (1.0 + asset_rets)).to_frame().T
    new_prices.index = [new_t]

    G["prices"] = pd.concat([prices, new_prices], axis=0)
    G["nav"] = pd.concat([G["nav"], pd.Series([new_nav], index=[new_t], dtype=float)], axis=0)
    G["rets"] = pd.concat([G["rets"], pd.Series([port_ret], index=[new_t], dtype=float)], axis=0)

    G["crowd"] = crowd_target
    G["macro"] = m
    G["t"] = new_t
    G["last_event"] = event_name
    G["last_event_text"] = event_text
    G["last_asset_rets"] = asset_rets
    G["last_action"] = action_name

    G["log"].append(
        {
            "week": new_t,
            "action": action_name,
            "event": event_name,
            "turnover": turnover,
            "tc_cost": tc_cost,
            "port_ret": port_ret,
            "nav": new_nav,
            "crash": crash,
            "weights": {a: float(w[a]) for a in ASSETS},
            "asset_rets": {a: float(asset_rets[a]) for a in ASSETS},
            "macro": {k: float(m[k]) for k in m},
            "crowd": {a: float(crowd_target[a]) for a in ASSETS},
        }
    )

# ----------------------------
# Sidebar settings
# ----------------------------
with st.sidebar:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Run setup")
    seed = st.number_input("Seed", min_value=1, max_value=999999, value=42069, step=1)
    weeks = st.slider("Weeks", min_value=12, max_value=60, value=26, step=1)
    difficulty = st.selectbox("Difficulty", list(DIFFICULTY.keys()), index=0)
    max_gross = st.slider("Max gross (x NAV)", 0.5, 4.0, 2.0, 0.1)
    tc_bps = st.slider("Transaction cost (bps per $ traded)", 0, 50, 8, 1)
    colA, colB = st.columns(2)
    new_run = colA.button("New run", use_container_width=True)
    autoplay = colB.button("Auto-play", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if new_run or "G" not in st.session_state:
    init_game(seed=seed, weeks=weeks, max_gross=max_gross, tc_bps=tc_bps, difficulty=difficulty)

# Keep limits live without corrupting history
st.session_state.G["max_gross"] = float(max_gross)
st.session_state.G["tc_bps"] = int(tc_bps)
st.session_state.G["difficulty"] = difficulty

G = st.session_state.G

# ----------------------------
# Top status bar (visual)
# ----------------------------
t = G["t"]
wks = G["weeks"]
nav_now = float(G["nav"].iloc[-1])

rets_live = G["rets"].iloc[1:] if len(G["rets"]) > 1 else G["rets"]
total, max_dd, sh, score = score_snapshot(G["nav"], rets_live)

ticker_parts = []
last = G["last_asset_rets"]
for a in ASSETS:
    r = float(last.get(a, 0.0))
    tag = "good" if r >= 0 else "danger"
    ticker_parts.append(f"{a} {r*100:+.2f}%")

ticker = "   |   ".join(ticker_parts) if t > 0 else "Make a move. Pick an action. Advance the tape."

st.markdown(
    f"<div class='ticker'><span>WEEK {t}/{wks}   |   NAV ${nav_now:,.0f}   |   TOTAL {total*100:+.2f}%   |   MAX DD {max_dd*100:.2f}%   |   SHARPE {sh:.2f}   |   {ticker}</span></div>",
    unsafe_allow_html=True,
)

# Survival / end checks
start_nav = float(G["nav"].iloc[0])
if nav_now <= 0.55 * start_nav:
    st.error("Margin call. You are out. New run and tighten leverage into high-risk regimes.")
    st.stop()
if t >= wks:
    st.success("Run complete. Hit New run to replay the same seed or change it to see a different market.")
    st.stop()

# ----------------------------
# Main layout
# ----------------------------
left, center, right = st.columns([0.9, 1.8, 0.9])

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Boss event")
    if G["last_event"] is None and t == 0:
        st.write("No tape yet. First decision sets your process.")
    elif G["last_event"] is None:
        st.write("No headline this week. The crowd still moves the tape.")
    else:
        st.write(f"**{G['last_event']}**")
        st.caption(G["last_event_text"])

    st.divider()
    st.subheader("Crowd heat")
    crowd_gross = float(np.abs(G["crowd"]).sum())
    heat = clamp(crowd_gross / 1.4, 0.0, 1.0)
    st.progress(heat, text=f"Crowd gross: {crowd_gross:.2f}x")

    st.subheader("Your drawdown")
    dd = float(dd_series(G["nav"]).iloc[-1])
    dd_bar = clamp(abs(dd) / 0.35, 0.0, 1.0)
    st.progress(dd_bar, text=f"DD: {dd*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Market arena")
    fig = make_arena_chart(G["prices"][ASSETS], G["nav"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption("The tape is reflexive: prices pull flows, flows push prices, and regime shifts change correlations.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Meters")
    m = G["macro"]
    st.metric("VIX", f"{m['vix']:.1f}")
    st.metric("10Y", f"{m['ust10y']:.2f}%")

    # Liquidity meter is a function of macro liquidity state
    liq_meter = clamp((m["liquidity"] + 1.2) / 2.4, 0.0, 1.0)
    st.progress(liq_meter, text=f"Liquidity: {m['liquidity']:+.2f}")

    risk_meter = clamp((m["risk_av"] + 1.2) / 2.4, 0.0, 1.0)
    st.progress(risk_meter, text=f"Risk aversion: {m['risk_av']:+.2f}")

    usd_meter = clamp((m["usd"] + 1.2) / 2.4, 0.0, 1.0)
    st.progress(usd_meter, text=f"USD strength: {m['usd']:+.2f}")

    st.divider()
    st.subheader("Score")
    st.metric("Total", f"{total*100:+.2f}%")
    st.metric("Max DD", f"{max_dd*100:.2f}%")
    st.metric("Sharpe", f"{sh:.2f}")
    st.metric("Score", f"{score:.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Action buttons (game controls)
# ----------------------------
st.markdown("### Choose your move")
b1, b2, b3, b4, b5 = st.columns(5)

def big_button(col, label):
    with col:
        st.markdown("<div class='bigbtn'>", unsafe_allow_html=True)
        clicked = st.button(label, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    return clicked

clicked = None
if big_button(b1, "Risk On"):
    clicked = "Risk On"
if big_button(b2, "Risk Off"):
    clicked = "Risk Off"
if big_button(b3, "Barbell"):
    clicked = "Barbell"
if big_button(b4, "Fade Crowd"):
    clicked = "Fade Crowd"
if big_button(b5, "Flat"):
    clicked = "Flat"

colX, colY = st.columns([1.2, 1.0])
with colX:
    st.caption(f"Last action: {G['last_action']}")
with colY:
    debug = st.toggle("Diagnostics", value=False)

# Advance one week if clicked
if clicked is not None:
    try:
        simulate_week(clicked)
        st.rerun()
    except Exception as e:
        st.error("Simulation error. Expand Diagnostics to see details.")
        st.session_state._last_exception = str(e)

# Auto-play
if autoplay:
    try:
        # keep it bounded so Streamlit does not feel stuck
        steps = int(clamp(G["weeks"] - G["t"], 0, 60))
        for _ in range(steps):
            if G["t"] >= G["weeks"]:
                break
            if float(G["nav"].iloc[-1]) <= 0.55 * float(G["nav"].iloc[0]):
                break
            simulate_week("Barbell")
        st.rerun()
    except Exception as e:
        st.error("Autoplay error. Expand Diagnostics to see details.")
        st.session_state._last_exception = str(e)

# ----------------------------
# Bottom: last week recap (visual)
# ----------------------------
st.markdown("### Last week recap")
if t == 0:
    st.write("No recap yet. Pick a move.")
else:
    last_log = G["log"][-1]
    a = last_log["action"]
    ev = last_log["event"] or "No headline"
    pr = last_log["port_ret"]
    crash = last_log["crash"]
    badge = "good" if pr >= 0 else "danger"
    crash_txt = f" | Tail shock: {crash*100:.1f}%" if crash and crash > 0 else ""

    st.markdown(
        f"<div class='panel'>"
        f"<div><b>Action:</b> {a} &nbsp;&nbsp; <b>Headline:</b> {ev} &nbsp;&nbsp; "
        f"<b>Week PnL:</b> <span class='{badge}'>{pr*100:+.2f}%</span>{crash_txt}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ----------------------------
# Diagnostics
# ----------------------------
if debug:
    with st.expander("Diagnostics", expanded=True):
        st.write("If something breaks, this is what I need to fix it in one pass.")
        st.write("Last exception:", st.session_state.get("_last_exception", "None"))
        st.write("State snapshot:")
        st.json(
            {
                "week": G["t"],
                "weeks": G["weeks"],
                "difficulty": G["difficulty"],
                "max_gross": G["max_gross"],
                "tc_bps": G["tc_bps"],
                "nav": float(G["nav"].iloc[-1]),
                "last_event": G["last_event"],
                "last_action": G["last_action"],
            }
        )
        if len(G["log"]):
            st.write("Last log row:")
            st.json(G["log"][-1])
