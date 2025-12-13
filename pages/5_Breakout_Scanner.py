import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None

import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import timedelta

plt.style.use("default")

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("Breakout Scanner")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Screen a custom ticker list for fresh breakouts, momentum regime confirmation, and actionability.

        • Uses Yahoo Finance daily data over a fixed 2 year lookback  
        • Checks for closes at or above rolling 20 / 50 / 100 / 200 day highs  
        • Adds distance to breakout levels (%), and days since breakout trigger  
        • Computes daily RSI(7 / 14 / 21) from Typical Price ((H+L+C)/3) and smooths with EMA(3)  
        • Adds ATR(20) as % of price and ATR compression percentile over ~1 year  
        • Builds a composite “Actionability Score” to rank the best setups  
        • Adds an invalidation proxy: downside to nearest broken level  

        Use the per-ticker chart at the bottom to inspect levels, RSI structure, and risk.
        """,
        unsafe_allow_html=True,
    )

# ── Inputs ─────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma separated):",
    "NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, AVGO, TSM"
).upper()

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# Fixed parameters
RSI_WINDOWS = [7, 14, 21]
RSI_SMOOTH_SPAN = 3
MIN_BARS = 200
LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"

WINDOWS_HI = (20, 50, 100, 200)
ATR_WINDOW = 20
ATR_PCTL_LOOKBACK = 252  # ~1 year of trading days

# ── Data Fetch (Batch + Cached) ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_batch(ticks, period=LOOKBACK_PERIOD, interval=INTERVAL):
    try:
        raw = yf.download(
            ticks,
            period=period,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=False
        )
    except Exception:
        return {}

    def extract(field):
        out = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in ticks:
                try:
                    s = raw[(t, field)].dropna()
                    if not s.empty:
                        out[t] = s
                except Exception:
                    continue
        else:
            if field in raw.columns:
                t0 = ticks[0] if isinstance(ticks, list) and len(ticks) == 1 else "TICKER"
                out[t0] = raw[field].dropna()
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out).sort_index().dropna(how="all")

    adj_close = extract("Adj Close")
    close = extract("Close")
    high = extract("High")
    low = extract("Low")
    return {"Adj Close": adj_close, "Close": close, "High": high, "Low": low}

data = fetch_batch(tickers)

# Fallback to Close if Adj Close missing
prices = data.get("Adj Close", pd.DataFrame())
if prices.empty:
    prices = data.get("Close", pd.DataFrame())

if prices.empty:
    st.error("No valid price data. Check tickers or connectivity.")
    st.stop()

# Ensure enough bars
sufficient_len = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_BARS]
prices = prices[sufficient_len]
if prices.empty:
    st.info("No tickers have at least 200 daily observations.")
    st.stop()

prices = prices.dropna(how="all").copy()

# ── Helpers ────────────────────────────────────────────────────────────────
def rsi_wilder(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def typical_price(sym: str) -> pd.Series:
    H = data.get("High", pd.DataFrame()).get(sym)
    L = data.get("Low", pd.DataFrame()).get(sym)
    C = data.get("Close", pd.DataFrame()).get(sym)
    if H is None or L is None or C is None:
        return prices[sym]
    idx = H.index.intersection(L.index).intersection(C.index)
    if len(idx) == 0:
        return prices[sym]
    tp = (H.loc[idx] + L.loc[idx] + C.loc[idx]) / 3.0
    return tp.dropna()

def compute_atr_pct(sym: str, window: int = ATR_WINDOW) -> pd.Series:
    H = data.get("High", pd.DataFrame()).get(sym)
    L = data.get("Low", pd.DataFrame()).get(sym)
    C = data.get("Close", pd.DataFrame()).get(sym)

    if H is None or L is None or C is None:
        # Fallback using close-only proxy: abs returns
        s = prices[sym].dropna()
        tr = s.pct_change().abs()
        atr = tr.rolling(window).mean()
        return atr * 100

    idx = H.index.intersection(L.index).intersection(C.index)
    if len(idx) == 0:
        s = prices[sym].dropna()
        tr = s.pct_change().abs()
        atr = tr.rolling(window).mean()
        return atr * 100

    H = H.loc[idx].astype(float)
    L = L.loc[idx].astype(float)
    C = C.loc[idx].astype(float)

    prev_close = C.shift(1)
    tr = pd.concat(
        [
            (H - L).abs(),
            (H - prev_close).abs(),
            (L - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(window).mean()
    atr_pct = (atr / C) * 100.0
    return atr_pct.dropna()

def last_breakout_days(close: pd.Series, rolling_high: pd.Series) -> float:
    # Trigger: today closes >= rolling high, yesterday was below yesterday rolling high
    cond_now = close >= rolling_high
    cond_prev = close.shift(1) < rolling_high.shift(1)
    trigger = (cond_now & cond_prev).fillna(False)

    if not trigger.any():
        return np.nan

    last_date = trigger[trigger].index[-1]
    days_since = (close.index[-1] - last_date).days
    return float(days_since)

def pct_distance_to_level(price: float, level: float) -> float:
    if level is None or np.isnan(level) or level == 0:
        return np.nan
    return (price / level - 1.0) * 100.0

def pct_downside_to_level(price: float, level: float) -> float:
    if level is None or np.isnan(level) or level == 0:
        return np.nan
    return ((price - level) / price) * 100.0

def safe_round(x, n=2):
    try:
        if pd.isna(x):
            return np.nan
        return round(float(x), n)
    except Exception:
        return np.nan

# ── Build Signal Table ──────────────────────────────────────────────────────
records = []

for sym in prices.columns:
    s = prices[sym].dropna()
    if s.shape[0] < MIN_BARS:
        continue

    latest = float(s.iloc[-1])

    # Rolling highs and breakouts
    rolling_highs = {w: s.rolling(w).max() for w in WINDOWS_HI}
    highs_latest = {w: float(rolling_highs[w].iloc[-1]) for w in WINDOWS_HI}
    breakout_flags = {w: bool(latest >= highs_latest[w]) for w in WINDOWS_HI}
    dist_to_highs = {w: pct_distance_to_level(latest, highs_latest[w]) for w in WINDOWS_HI}
    days_since_break = {w: last_breakout_days(s, rolling_highs[w]) for w in WINDOWS_HI}

    # ATR% and compression percentile
    atr_pct_series = compute_atr_pct(sym, ATR_WINDOW)
    atr_pct_latest = float(atr_pct_series.iloc[-1]) if not atr_pct_series.empty else np.nan
    if atr_pct_series.shape[0] >= min(ATR_PCTL_LOOKBACK, 60):
        look = atr_pct_series.dropna().iloc[-ATR_PCTL_LOOKBACK:]
        atr_pctl = float(pd.Series(look).rank(pct=True).iloc[-1] * 100.0) if look.shape[0] > 5 else np.nan
    else:
        atr_pctl = np.nan

    # RSI (Typical Price) + smoothing
    tp = typical_price(sym).dropna()
    rsi_smoothed = {}
    rsi_raw = {}
    for w in RSI_WINDOWS:
        if tp.shape[0] >= w:
            r = rsi_wilder(tp, w)
            rsi_raw[w] = r
            rsi_smoothed[w] = r.ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean()

    rsi_latest = {w: float(rsi_smoothed[w].iloc[-1]) for w in rsi_smoothed.keys()}

    # RSI regime flags
    r7 = rsi_latest.get(7, np.nan)
    r14 = rsi_latest.get(14, np.nan)
    r21 = rsi_latest.get(21, np.nan)

    rsi_aligned = bool(np.isfinite(r7) and np.isfinite(r14) and np.isfinite(r21) and (r7 >= r14 >= r21))
    rsi_above_50 = bool(np.isfinite(r7) and np.isfinite(r14) and np.isfinite(r21) and (r7 >= 50 and r14 >= 50 and r21 >= 50))

    def slope_up(r_series: pd.Series, lookback_days: int = 5) -> bool:
        if r_series is None or r_series.dropna().shape[0] < lookback_days + 1:
            return False
        rr = r_series.dropna()
        return bool(rr.iloc[-1] > rr.iloc[-(lookback_days + 1)])

    rsi_rising = bool(
        slope_up(rsi_smoothed.get(7), 5)
        and slope_up(rsi_smoothed.get(14), 5)
        and slope_up(rsi_smoothed.get(21), 5)
    )

    rsi_regime = "Strong" if (rsi_aligned and rsi_above_50 and rsi_rising) else ("OK" if (rsi_above_50 or rsi_rising) else "Weak")

    # Nearest broken level for invalidation proxy
    broken_levels = [highs_latest[w] for w in WINDOWS_HI if breakout_flags[w]]
    nearest_level = max(broken_levels) if broken_levels else np.nan
    downside_to_nearest = pct_downside_to_level(latest, nearest_level) if np.isfinite(nearest_level) else np.nan

    # Composite Actionability Score
    score = 0.0

    # Breakout breadth and quality
    weights = {20: 10, 50: 15, 100: 20, 200: 25}
    for w in WINDOWS_HI:
        if breakout_flags[w]:
            score += weights[w]

    # Prefer fresh breakouts
    for w in WINDOWS_HI:
        ds = days_since_break.get(w, np.nan)
        if np.isfinite(ds):
            if ds <= 3:
                score += 8
            elif ds <= 10:
                score += 4

    # Prefer early extension, penalize blow-off extension
    for w in WINDOWS_HI:
        d = dist_to_highs.get(w, np.nan)
        if np.isfinite(d):
            if 0.0 <= d <= 2.0:
                score += 4
            if d >= 6.0:
                score -= 4

    # RSI regime contribution
    if rsi_regime == "Strong":
        score += 18
    elif rsi_regime == "OK":
        score += 8
    else:
        score -= 6

    # ATR compression: low percentile is tighter
    if np.isfinite(atr_pctl):
        if atr_pctl <= 30:
            score += 12
        elif atr_pctl >= 80:
            score -= 8

    # Risk proxy: tighter stop distance is more actionable
    if np.isfinite(downside_to_nearest):
        if downside_to_nearest <= 2.0:
            score += 6
        elif downside_to_nearest >= 6.0:
            score -= 4

    rec = {
        "Ticker": sym,
        "Price": safe_round(latest, 2),
        "Score": safe_round(score, 1),
        "RSI Regime": rsi_regime,
        "ATR% (20D)": safe_round(atr_pct_latest, 2),
        "ATR%ile (~1Y)": safe_round(atr_pctl, 0),
        "Nearest Broken Level": safe_round(nearest_level, 2),
        "Downside to Level %": safe_round(downside_to_nearest, 2),
        "Breakout Count": int(sum(breakout_flags.values())),
    }

    for w in WINDOWS_HI:
        rec[f"{w}D High"] = safe_round(highs_latest[w], 2)
        rec[f"Breakout {w}D"] = breakout_flags[w]
        rec[f"Dist to {w}D High %"] = safe_round(dist_to_highs[w], 2)
        rec[f"Days Since {w}D Break"] = safe_round(days_since_break[w], 0)

    for w in RSI_WINDOWS:
        rec[f"RSI ({w})"] = safe_round(rsi_latest.get(w, np.nan), 1)

    records.append(rec)

df = pd.DataFrame(records)
if df.empty:
    st.info("No signals computed or insufficient data.")
    st.stop()

# Rank by Score, then breakout breadth, then strongest breakouts
break_cols = [c for c in df.columns if c.startswith("Breakout ")]
df = df.sort_values(
    by=["Score", "Breakout Count"] + break_cols + ["Price"],
    ascending=False
).reset_index(drop=True)

# ── Topline Summary ─────────────────────────────────────────────────────────
total = df.shape[0]
b20 = int(df["Breakout 20D"].sum()) if "Breakout 20D" in df.columns else 0
b50 = int(df["Breakout 50D"].sum()) if "Breakout 50D" in df.columns else 0
b100 = int(df["Breakout 100D"].sum()) if "Breakout 100D" in df.columns else 0
b200 = int(df["Breakout 200D"].sum()) if "Breakout 200D" in df.columns else 0

strong_rsi = int((df["RSI Regime"] == "Strong").sum()) if "RSI Regime" in df.columns else 0
tight_atr = int((df["ATR%ile (~1Y)"].fillna(999) <= 30).sum()) if "ATR%ile (~1Y)" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickers Scanned", f"{total}")
c2.metric("20D High Breaks", f"{b20}")
c3.metric("50D High Breaks", f"{b50}")
c4.metric("200D High Breaks", f"{b200}")
c5.metric("Strong RSI Regime", f"{strong_rsi}")

st.caption(f"ATR compression candidates (ATR%ile <= 30): {tight_atr}")

# ── Table Styling ───────────────────────────────────────────────────────────
def styled_table(dfin: pd.DataFrame):
    df_disp = dfin.copy()

    # Replace booleans with icons for breakout columns
    for col in break_cols:
        df_disp[col] = df_disp[col].map(lambda x: "✅" if bool(x) else "")

    # Formatting
    money_cols = [c for c in df_disp.columns if ("High" in c) or (c in ["Price", "Nearest Broken Level"])]
    pct_cols = [c for c in df_disp.columns if c.endswith("%") or c.endswith("%ile (~1Y)") or c.endswith("Level %")]
    rsi_cols = [c for c in df_disp.columns if c.startswith("RSI (")]
    days_cols = [c for c in df_disp.columns if c.startswith("Days Since")]

    for c in money_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{float(x):,.2f}" if pd.notna(x) else "")
    for c in pct_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{float(x):,.2f}%" if pd.notna(x) else "")
    for c in rsi_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
    for c in days_cols:
        df_disp[c] = df_disp[c].map(lambda x: f"{int(float(x))}" if pd.notna(x) else "")

    # Conditional coloring
    def color_rsi(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 80:
            return "color: #d62728; font-weight: bold;"
        if v <= 20:
            return "color: #1f77b4; font-weight: bold;"
        if 55 <= v < 80:
            return "color: #2ca02c;"
        return ""

    def color_regime(val):
        if val == "Strong":
            return "color: #2ca02c; font-weight: bold;"
        if val == "OK":
            return "color: #ff7f0e; font-weight: bold;"
        if val == "Weak":
            return "color: #d62728; font-weight: bold;"
        return ""

    def color_score(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 70:
            return "background-color: rgba(44, 160, 44, 0.15); font-weight: bold;"
        if v <= 25:
            return "background-color: rgba(214, 39, 40, 0.10);"
        return ""

    # We display Score already formatted as numeric, but styling expects raw values.
    styled = (
        df_disp.style
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "13px"), ("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("font-size", "13px"), ("text-align", "center")]},
        ])
        .applymap(color_rsi, subset=rsi_cols)
        .applymap(color_regime, subset=["RSI Regime"])
        .applymap(color_score, subset=["Score"])
        .set_properties(**{"background-color": "#f8fafc"}, subset=pd.IndexSlice[:, ["Price"]])
        .set_properties(**{"border": "1.2px solid #e0e0e0"})
    )
    return styled

# Curate columns for actionability
core_cols = [
    "Ticker", "Score", "Price",
    "Breakout Count",
    "RSI Regime", "RSI (7)", "RSI (14)", "RSI (21)",
    "ATR% (20D)", "ATR%ile (~1Y)",
    "Nearest Broken Level", "Downside to Level %"
]
level_cols = []
for w in WINDOWS_HI:
    level_cols += [f"Breakout {w}D", f"Dist to {w}D High %", f"Days Since {w}D Break", f"{w}D High"]

display_cols = [c for c in core_cols + level_cols if c in df.columns]
df_view = df[display_cols].copy()

st.markdown("### Actionable Breakout Table")
st.dataframe(styled_table(df_view), use_container_width=True, height=520)
st.download_button("Download as CSV", df.to_csv(index=False), file_name="breakout_signals_actionable.csv")

# ── Per-Ticker Charts ───────────────────────────────────────────────────────
sel = st.selectbox("Select ticker to chart:", df["Ticker"], index=0)

s_close = prices[sel].dropna()
if s_close.shape[0] < MIN_BARS:
    st.info(f"{sel} does not have enough data to draw rolling highs.")
else:
    # Rolling highs
    rh = {w: s_close.rolling(w).max() for w in WINDOWS_HI}

    # Nearest broken level for annotation
    latest = float(s_close.iloc[-1])
    highs_latest = {w: float(rh[w].iloc[-1]) for w in WINDOWS_HI}
    breakout_flags = {w: bool(latest >= highs_latest[w]) for w in WINDOWS_HI}
    broken_levels = [highs_latest[w] for w in WINDOWS_HI if breakout_flags[w]]
    nearest_level = max(broken_levels) if broken_levels else np.nan
    downside_to_nearest = pct_downside_to_level(latest, nearest_level) if np.isfinite(nearest_level) else np.nan

    # RSI
    base = typical_price(sel).dropna()
    rsi_lines = {}
    for w in RSI_WINDOWS:
        if base.shape[0] >= w:
            rsi_lines[w] = rsi_wilder(base, w).ewm(span=RSI_SMOOTH_SPAN, adjust=False).mean()

    # Days since breakout per window
    days_since = {w: last_breakout_days(s_close, rh[w]) for w in WINDOWS_HI}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), sharex=False, constrained_layout=True)

    # Price and rolling highs
    ax1.plot(s_close.index, s_close, label="Close", color="black", linewidth=2)

    colors = {20: "#1f77b4", 50: "#ff7f0e", 100: "#2ca02c", 200: "#d62728"}
    for w in WINDOWS_HI:
        ax1.plot(s_close.index, rh[w], lw=1.15, color=colors[w], label=f"{w}D High")

    # Shade recent consolidation range (last 40 days)
    lookback_shade = min(40, s_close.shape[0])
    recent = s_close.iloc[-lookback_shade:]
    recent_low = float(recent.min())
    recent_high = float(recent.max())
    ax1.fill_between(
        recent.index,
        recent_low,
        recent_high,
        alpha=0.08
    )

    # Nearest broken level line for invalidation proxy
    if np.isfinite(nearest_level):
        ax1.axhline(nearest_level, lw=1.1, linestyle="--", color="gray")
        ax1.text(
            s_close.index[-1],
            nearest_level,
            f"  Nearest level {nearest_level:,.2f} | downside {downside_to_nearest:.2f}%",
            va="center",
            fontsize=9,
            color="gray"
        )

    title_bits = []
    for w in WINDOWS_HI:
        ds = days_since.get(w, np.nan)
        if np.isfinite(ds) and ds <= 10:
            title_bits.append(f"{w}D:{int(ds)}d")

    title_suffix = (" | fresh " + ", ".join(title_bits)) if title_bits else ""
    ax1.set_title(f"{sel} Price, Levels, and Risk{title_suffix}", fontweight="bold")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)
    ax1.set_ylabel("Price")
    ax1.margins(x=0)
    ax1.set_xlim(s_close.index.min(), s_close.index.max() + timedelta(days=4))

    # RSI panel
    for w in RSI_WINDOWS:
        if w in rsi_lines:
            ax2.plot(rsi_lines[w].index, rsi_lines[w], label=f"RSI({w})", linewidth=1.5)

    ax2.axhline(80, ls="--", color="gray", lw=0.9)
    ax2.axhline(20, ls="--", color="gray", lw=0.9)
    ax2.axhline(50, ls=":", color="gray", lw=0.9)
    ax2.set_title(f"{sel} RSI Structure", fontweight="bold")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(alpha=0.25)
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.margins(x=0)
    ax2.set_xlim(base.index.min(), base.index.max() + timedelta(days=4))
    ax2.set_ylim(0, 100)

    st.pyplot(fig, use_container_width=True)

st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
