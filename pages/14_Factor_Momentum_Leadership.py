import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# Page config and styling
# =========================================================
st.set_page_config(page_title="Factor Momentum and Basket Rotation", layout="wide")

TITLE = "Factor Momentum and Basket Rotation"
SUBTITLE = "Factor momentum dashboard with Pro vs Anti ADFM basket rotation views."

TEXT = "#222222"
SUBTLE = "#666666"
GRID = "#e6e6e6"
BORDER = "#dddddd"
BG = "#ffffff"
CARD_BG = "#fafafa"

PASTELS = [
    "#6FB9C3",
    "#E07A2F",
    "#6FA85A",
    "#F2B874",
    "#8CC7F2",
    "#A889C7",
    "#C7E29E",
    "#D89AD3",
    "#8FE3A1",
    "#E8CFC3",
]

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.1px;
    }
    .stPlotlyChart {
        background: #ffffff;
    }
    .js-plotly-plot .table .cell {
        font-size: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CACHE_DIR = Path("data_cache_factor_rotation")
CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# Config
# =========================================================
FACTOR_ETFS: Dict[str, Tuple[str, Optional[str]]] = {
    "Growth vs Value": ("VUG", "VTV"),
    "Quality vs Junk": ("QUAL", "JNK"),
    "High Beta vs Low Vol": ("SPHB", "SPLV"),
    "Small vs Large": ("IWM", "SPY"),
    "Tech vs Broad": ("XLK", "SPY"),
    "Cyclicals vs Defensives": ("XLY", "XLP"),
    "US vs World": ("SPY", "VEA"),
    "Momentum": ("MTUM", "SPY"),
    "Equal Weight vs Cap": ("RSP", "SPY"),
}
BENCH = "SPY"

WINDOW_MAP_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Growth & Innovation": {
        "Semis ETFs": ["SMH", "SOXX", "XSD"],
        "Semis Compute and Accelerators": ["NVDA", "AMD", "INTC", "ARM", "AVGO", "MRVL"],
        "Semis Analog and Power": ["TXN", "ADI", "MCHP", "NXPI", "MPWR", "ON", "STM", "IFNNY", "WOLF"],
        "Semis RF and Connectivity": ["QCOM", "SWKS", "QRVO", "MTSI", "AVNW"],
        "Semis Memory and Storage": ["MU", "WDC", "STX", "SKM"],
        "Semis Foundry and OSAT": ["TSM", "UMC", "GFS", "ASX"],
        "Semis Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO", "AEIS", "ACMR"],
        "Semis EDA and IP": ["SNPS", "CDNS", "ANSS", "ARM"],
        "AI Infrastructure Leaders": [
            "NVDA", "AMD", "AVGO", "TSM", "ASML",
            "ANET", "SMCI", "DELL", "HPE",
            "AMAT", "LRCX", "KLAC", "TER",
            "MRVL", "MU", "WDC", "STX", "NTAP",
            "ORCL", "MSFT", "AMZN", "GOOGL"
        ],
        "Hyperscalers and Cloud": [
            "MSFT", "AMZN", "GOOGL", "META", "ORCL", "IBM",
            "NOW", "CRM", "DDOG", "SNOW", "MDB", "NET", "ZS", "OKTA"
        ],
        "Quality SaaS": ["ADBE", "CRM", "NOW", "INTU", "TEAM", "HUBS", "DDOG", "NET", "MDB", "SNOW"],
        "Cybersecurity": ["PANW", "FTNT", "CRWD", "ZS", "OKTA", "TENB", "S", "CYBR", "CHKP", "NET"],
        "Digital Payments": ["V", "MA", "PYPL", "SQ", "FI", "FIS", "GPN", "AXP", "COF", "DFS", "ADYEY", "MELI"],
        "China Tech ADRs": ["BABA", "BIDU", "JD", "PDD", "BILI", "NTES", "TCEHY"],
        "Net-Cash Compounders": ["AAPL", "MSFT", "GOOGL", "META", "ORCL", "ADBE", "INTU", "V"],
    },
    "Energy and Hard Assets": {
        "Energy Majors": ["XOM", "CVX", "COP", "SHEL", "BP", "TTE", "EQNR", "ENB", "PBR"],
        "US Shale and E&Ps": ["EOG", "DVN", "FANG", "MRO", "OXY", "APA", "AR", "RRC", "SWN", "CHK", "CTRA"],
        "Natural Gas and LNG": ["LNG", "EQNR", "KMI", "WMB", "EPD", "ET"],
        "Oilfield Services": ["SLB", "HAL", "BKR", "NOV"],
        "Gold and Silver Miners": ["GDX", "GDXJ", "NEM", "AEM", "GOLD", "KGC", "AG", "PAAS", "WPM"],
        "Metals and Mining": ["BHP", "RIO", "VALE", "FCX", "NEM", "TECK", "SCCO", "AA"],
    },
    "Regime Diagnostics": {
        "Long-Duration Equities": ["ARKK", "IPO", "IGV", "SNOW", "NET", "DDOG", "MDB", "SHOP"],
        "Short-Duration Cash Flow": ["BRK-B", "PGR", "CB", "ICE", "CME", "NDAQ", "SPGI", "MSCI"],
        "Yield Proxies": ["XLU", "VZ", "T", "KMI", "EPD", "ENB"],
        "Financial Conditions Sensitive": ["IWM", "XLY", "KRE", "HYG", "ARKK"],
        "Dollar-Down Beneficiaries": ["XME", "GDX", "EEM", "EWZ"],
        "Commodity FX Equities": ["EWC", "EWA", "EWZ", "EWW"],
        "EM Domestic Demand": ["EEM", "INDA", "EWW", "EWZ", "EIDO"],
        "Equity Credit Stress Proxies": ["HYG", "JNK", "LQD"],
    },
    "Defensives and Staples": {
        "Staples and Beverages": ["PG", "KO", "PEP", "PM", "MO", "MDLZ"],
        "Telecom and Cable": ["T", "VZ", "TMUS", "CHTR", "CMCSA"],
        "Utilities Defensive": ["DUK", "SO", "AEP", "XEL", "EXC", "ED"],
    },
    "Financials and Credit": {
        "Money-Center and IBs": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
        "Regional Banks": ["KRE", "TFC", "FITB", "CFG", "RF", "KEY", "PNC", "USB", "MTB"],
    },
}

ALL_BASKETS = {bk: tks for cat in CATEGORIES.values() for bk, tks in cat.items()}

FACTOR_TO_BASKETS: Dict[str, Dict[str, List[str]]] = {
    "US vs World": {
        "pro": ["Hyperscalers and Cloud", "Net-Cash Compounders", "Cybersecurity", "Semis Compute and Accelerators"],
        "anti": ["China Tech ADRs", "EM Domestic Demand", "Commodity FX Equities", "Dollar-Down Beneficiaries"],
    },
    "Cyclicals vs Defensives": {
        "pro": ["Energy Majors", "US Shale and E&Ps", "Metals and Mining", "Regional Banks"],
        "anti": ["Utilities Defensive", "Staples and Beverages", "Telecom and Cable", "Yield Proxies"],
    },
    "Growth vs Value": {
        "pro": ["Long-Duration Equities", "Quality SaaS", "Hyperscalers and Cloud", "AI Infrastructure Leaders"],
        "anti": ["Energy Majors", "Metals and Mining", "Staples and Beverages", "Utilities Defensive"],
    },
    "Small vs Large": {
        "pro": ["Financial Conditions Sensitive", "Regional Banks", "US Shale and E&Ps"],
        "anti": ["Net-Cash Compounders", "Hyperscalers and Cloud", "Short-Duration Cash Flow"],
    },
    "High Beta vs Low Vol": {
        "pro": ["Long-Duration Equities", "Financial Conditions Sensitive", "AI Infrastructure Leaders"],
        "anti": ["Staples and Beverages", "Utilities Defensive", "Yield Proxies"],
    },
    "Momentum": {
        "pro": ["AI Infrastructure Leaders", "Hyperscalers and Cloud", "Energy Majors", "Gold and Silver Miners"],
        "anti": ["Equity Credit Stress Proxies", "Yield Proxies"],
    },
    "Tech vs Broad": {
        "pro": ["Hyperscalers and Cloud", "AI Infrastructure Leaders", "Cybersecurity", "Quality SaaS"],
        "anti": ["Energy Majors", "Metals and Mining", "Staples and Beverages"],
    },
    "Quality vs Junk": {
        "pro": ["Short-Duration Cash Flow", "Net-Cash Compounders", "Staples and Beverages"],
        "anti": ["Equity Credit Stress Proxies", "Financial Conditions Sensitive", "Long-Duration Equities"],
    },
    "Equal Weight vs Cap": {
        "pro": ["Financial Conditions Sensitive", "Regional Banks", "US Shale and E&Ps"],
        "anti": ["Net-Cash Compounders", "Hyperscalers and Cloud"],
    },
}

# =========================================================
# UI helpers
# =========================================================
def card_box(inner_html: str) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid {BORDER};
            border-radius:10px;
            padding:14px;
            background:{CARD_BG};
            color:{TEXT};
            font-size:14px;
            line-height:1.4;">
            {inner_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, help_text: Optional[str] = None) -> None:
    extra = f'<div style="margin-top:4px; color:{SUBTLE}; font-size:12px;">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div style="
            border:1px solid {BORDER};
            border-radius:12px;
            padding:12px 14px;
            background:{BG};
            min-height:84px;">
            <div style="font-size:12px; color:{SUBTLE}; margin-bottom:6px;">{label}</div>
            <div style="font-size:26px; color:{TEXT}; font-weight:700;">{value}</div>
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_")

def _safe_basket_list(names: List[str]) -> List[str]:
    return [b for b in names if b in ALL_BASKETS]

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def cache_path_for_tickers(tickers: List[str]) -> Path:
    key = "_".join(sorted(set([str(t).upper() for t in tickers if t])))
    safe_key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)
    return CACHE_DIR / f"{safe_key[:220]}.parquet"

# =========================================================
# Math helpers
# =========================================================
def pct_change_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-(days + 1)] - 1.0)

def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < max(2, win):
        return np.nan
    return float(r.rolling(win).mean().iloc[-1])

def rs(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    out = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    out.name = f"{series_a.name}_vs_{series_b.name}"
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def trend_class(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 50:
        return "Neutral"
    e1 = ema(s, 10).iloc[-1]
    e2 = ema(s, 20).iloc[-1]
    e3 = ema(s, 40).iloc[-1]
    if e1 > e2 > e3:
        return "Up"
    if e1 < e2 < e3:
        return "Down"
    return "Neutral"

def inflection(short_mom: float, long_mom: float) -> str:
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if abs(short_mom) > abs(long_mom):
        return "Strengthening"
    return "Weakening"

def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c].dropna()
        if s.empty:
            out[c] = np.nan
        else:
            out[c] = out[c] / s.iloc[0] * 100.0
    return out

def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    s = series.dropna()
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def momentum_label(hist: pd.Series, lookback: int = 5, z_window: int = 63) -> str:
    h = hist.dropna()
    if h.shape[0] < max(lookback + 1, z_window):
        return "Neutral"
    latest = float(h.iloc[-1])
    ref = float(h.iloc[-(lookback + 1)])
    base = "Positive" if latest > 0 else ("Negative" if latest < 0 else "Neutral")
    if base == "Neutral":
        return "Neutral"
    window = h.iloc[-z_window:]
    std = float(window.std(ddof=0))
    z = (latest - float(window.mean())) / std if std and not np.isnan(std) else 0.0
    accel = "Accelerating" if (latest - ref) > 0 else "Decelerating"
    strength = "Strong" if abs(z) > 1 else "Weak"
    return f"{base} | {accel} | {strength}"

def pct_since(levels: pd.Series, start_ts: pd.Timestamp) -> float:
    s = levels.dropna()
    if s.empty:
        return np.nan
    sub = s[s.index >= start_ts]
    if sub.shape[0] < 2:
        return np.nan
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)

# =========================================================
# Commentary helpers
# =========================================================
def bucket_breadth(breadth: float) -> str:
    if breadth < 10:
        return "extremely narrow"
    if breadth < 25:
        return "narrow and selective"
    if breadth < 40:
        return "tilted to a small group of styles"
    if breadth < 60:
        return "balanced across factors"
    if breadth < 75:
        return "broadening out across styles"
    return "very broad and inclusive"

def bucket_regime(regime_score: float) -> str:
    if regime_score < 25:
        return "deeply defensive and stress driven"
    if regime_score < 40:
        return "defensive and risk averse"
    if regime_score < 55:
        return "roughly neutral with a mild defensive lean"
    if regime_score < 70:
        return "constructive and risk friendly"
    return "high beta and aggressively risk on"

def build_commentary(mom_df: pd.DataFrame, breadth: float, regime_score: float) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))

    established_leaders = mom_df[
        (mom_df["Short"] > 0) & (mom_df["Long"] > 0)
    ].sort_values("Short", ascending=False).index.tolist()
    new_rotations = mom_df[mom_df["Inflection"] == "Turning Up"].index.tolist()
    fading_leaders = mom_df[mom_df["Inflection"] == "Turning Down"].index.tolist()

    leaders_text = ", ".join(established_leaders[:4]) if established_leaders else "no factor pair in a clean dual-horizon uptrend"
    rotations_text = ", ".join(new_rotations[:4]) if new_rotations else "no factor is clearly turning up yet"
    fading_text = ", ".join(fading_leaders[:4]) if fading_leaders else "no obvious factor is rolling over from strength"

    conclusion = (
        f"Factor tape is {bucket_breadth(breadth)} and currently {bucket_regime(regime_score)}. "
        f"Leadership is anchored in {leaders_text}, with emerging rotation showing up in {rotations_text}, "
        f"while pressure is building in {fading_text}."
    )

    why_matters = (
        "This grid is the style map for the equity tape. It shows which factors are being rewarded, "
        "how durable that preference looks across short and long windows, and whether the market is pressing existing leadership or searching for a new one."
    )

    top_short = ", ".join(mom_df.sort_values("Short", ascending=False).index.tolist()[:5])

    drivers = [
        f"{up_count} factors are in uptrends and {down_count} are in downtrends using the 10, 20, and 40-day EMA stack.",
        f"Short-window strength is concentrated in {top_short}.",
    ]
    if new_rotations:
        drivers.append(f"Inflection signals flag {', '.join(new_rotations)} as turning up.")
    if fading_leaders:
        drivers.append(f"{', '.join(fading_leaders)} are turning down against stronger long-window history.")

    key_stats = f"Breadth index {breadth:.1f}%. Regime score {regime_score:.1f} on a 0 to 100 scale."

    body = (
        '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f"<div>{conclusion}</div>"
        '<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f"<div>{why_matters}</div>"
        '<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        '<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers)
        + "</ul>"
        '<div style="font-weight:700; margin:10px 0 6px;">Key stats</div>'
        f"<div>{key_stats}</div>"
    )
    return body

# =========================================================
# Data fetching
# =========================================================
def _download_close_batch(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    batch = [str(x).upper() for x in batch if x]
    if not batch:
        return pd.DataFrame()

    try:
        if len(batch) == 1:
            t = batch[0]
            df = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col is None:
                return pd.DataFrame()
            s = df[col].copy()
            s.name = t
            return s.to_frame()

        df = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = set(df.columns.get_level_values(0))
            if "Close" in lvl0:
                close = df["Close"].copy()
            elif "Adj Close" in lvl0:
                close = df["Adj Close"].copy()
            else:
                return pd.DataFrame()
            close.columns = [str(c).upper() for c in close.columns]
            return close

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_levels(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, chunk_size: int = 40) -> pd.DataFrame:
    uniq = sorted({str(t).upper() for t in tickers if t})
    cache_file = cache_path_for_tickers(uniq)

    frames: List[pd.DataFrame] = []

    for batch in _chunk(uniq, chunk_size):
        out = pd.DataFrame()
        for _ in range(3):
            out = _download_close_batch(batch, start, end)
            if not out.empty:
                break
            time.sleep(0.5)
        if not out.empty:
            frames.append(out)

    if frames:
        wide = pd.concat(frames, axis=1)
        wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

        if not wide.empty:
            bidx = pd.bdate_range(wide.index.min(), wide.index.max(), name=wide.index.name)
            wide = wide.reindex(bidx).ffill()

            try:
                wide.to_parquet(cache_file)
            except Exception:
                pass

            return wide

    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()
            cached = cached[(cached.index >= start) & (cached.index <= end)]
            if not cached.empty:
                return cached
        except Exception:
            pass

    return pd.DataFrame()

# =========================================================
# Basket engine
# =========================================================
def ew_rets_from_levels(levels: pd.DataFrame, baskets: Dict[str, List[str]], stale_days: int = 30) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()

    rets = levels.pct_change()
    out = {}
    last_idx = levels.index.max()

    for basket_name, tickers in baskets.items():
        valid_cols = []
        for t in tickers:
            t_u = str(t).upper()
            if t_u not in rets.columns:
                continue
            s = levels[t_u].dropna()
            if s.empty:
                continue
            if s.index.max() < last_idx - pd.Timedelta(days=int(stale_days)):
                continue
            valid_cols.append(t_u)

        if not valid_cols:
            continue

        if len(valid_cols) == 1:
            out[basket_name] = rets[valid_cols[0]]
        else:
            out[basket_name] = rets[valid_cols].mean(axis=1, skipna=True)

    return pd.DataFrame(out).dropna(how="all")

def build_all_basket_panels(
    basket_returns: pd.DataFrame,
    bench_rets: pd.Series,
    window_start: pd.Timestamp,
    window_label: str,
) -> pd.DataFrame:
    if basket_returns.empty or bench_rets.empty:
        return pd.DataFrame()

    bench_rets = bench_rets.dropna()
    basket_returns = basket_returns.copy().dropna(how="all")

    basket_levels = 100.0 * (1.0 + basket_returns).cumprod()
    dyn_col = f"↓ %{window_label}"
    rows = []

    for b in basket_levels.columns:
        lvl = basket_levels[b].dropna()
        if lvl.shape[0] < 20:
            continue

        r5d = pct_change_window(lvl, 5)
        r1m = pct_since(lvl, lvl.index.max() - pd.DateOffset(months=1))
        r_dyn = pct_since(lvl, window_start)

        hist = macd_hist(lvl, 12, 26, 9)
        macd_m = momentum_label(hist, lookback=5, z_window=63)

        corr_spy = np.nan
        merged = pd.concat([basket_returns[b], bench_rets], axis=1, join="inner").dropna()
        if merged.shape[0] >= 63:
            corr_spy = float(merged.iloc[:, 0].rolling(63).corr(merged.iloc[:, 1]).iloc[-1])

        rows.append({
            "Basket": b,
            "%5D": round(r5d * 100, 1) if pd.notna(r5d) else np.nan,
            "%1M": round(r1m * 100, 1) if pd.notna(r1m) else np.nan,
            dyn_col: round(r_dyn * 100, 1) if pd.notna(r_dyn) else np.nan,
            "MACD Momentum": macd_m,
            "Corr(63D)": round(corr_spy, 2) if pd.notna(corr_spy) else np.nan,
        })

    if not rows:
        return pd.DataFrame(columns=["Basket", "%5D", "%1M", dyn_col, "MACD Momentum", "Corr(63D)"]).set_index("Basket")

    return pd.DataFrame(rows).set_index("Basket").sort_values(dyn_col, ascending=False)

# =========================================================
# Coloring for tables
# =========================================================
def color_ret(x):
    if pd.isna(x):
        return "white"
    if x >= 0:
        s = min(abs(x) / 20.0, 1.0)
        g = int(255 - 90 * s)
        return f"rgb({int(240 - 120 * s)},{g},{int(240 - 120 * s)})"
    s = min(abs(x) / 20.0, 1.0)
    r = int(255 - 90 * s)
    return f"rgb({r},{int(240 - 120 * s)},{int(240 - 120 * s)})"

def color_macd(tag):
    if not isinstance(tag, str):
        return "white"
    if tag.startswith("Positive"):
        if "Accelerating" in tag and "Strong" in tag:
            return "rgb(190,235,190)"
        if "Accelerating" in tag:
            return "rgb(204,238,204)"
        return "rgb(225,246,225)"
    if tag.startswith("Negative"):
        if "Accelerating" in tag and "Strong" in tag:
            return "rgb(255,190,190)"
        if "Accelerating" in tag:
            return "rgb(255,210,210)"
        return "rgb(255,228,228)"
    return "rgb(230,236,245)"

def color_corr(x):
    if pd.isna(x):
        return "white"
    v = abs(x)
    if v >= 0.8:
        return "rgb(210,230,255)"
    if v >= 0.5:
        return "rgb(220,235,255)"
    return "rgb(230,240,255)"

# =========================================================
# Plot helpers
# =========================================================
def plot_rotation_table(panel_df: pd.DataFrame, title: str, key: str):
    st.markdown(f"**{title}**")
    if panel_df.empty:
        st.info("No baskets passed the data checks for this window.")
        return

    dyn_col = panel_df.columns[2]
    headers = ["Basket", "%5D", "%1M", dyn_col, "MACD Momentum", "Corr(63D)"]
    values = [
        panel_df.index.tolist(),
        panel_df["%5D"].tolist(),
        panel_df["%1M"].tolist(),
        panel_df[dyn_col].tolist(),
        panel_df["MACD Momentum"].tolist(),
        panel_df["Corr(63D)"].tolist(),
    ]
    fill_colors = [
        ["white"] * len(panel_df),
        [color_ret(v) for v in panel_df["%5D"].tolist()],
        [color_ret(v) for v in panel_df["%1M"].tolist()],
        [color_ret(v) for v in panel_df[dyn_col].tolist()],
        [color_macd(v) for v in panel_df["MACD Momentum"].tolist()],
        [color_corr(v) for v in panel_df["Corr(63D)"].tolist()],
    ]

    fig_tbl = go.Figure(
        data=[go.Table(
            columnwidth=[340, 90, 90, 110, 260, 110],
            header=dict(
                values=headers,
                fill_color="white",
                line_color="rgb(230,230,230)",
                font=dict(color="black", size=13),
                align="left",
                height=32
            ),
            cells=dict(
                values=values,
                fill_color=fill_colors,
                line_color="rgb(240,240,240)",
                font=dict(color="black", size=12),
                align="left",
                height=26,
                format=[None, ".1f", ".1f", ".1f", None, ".2f"]
            )
        )]
    )
    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=6, b=0),
        height=min(520, 64 + 26 * max(3, len(panel_df)))
    )
    st.plotly_chart(fig_tbl, use_container_width=True, key=key)

def plot_side_cumulative(
    basket_returns: pd.DataFrame,
    baskets: List[str],
    title: str,
    benchmark_rets: pd.Series,
    key: str,
):
    if basket_returns.empty or not baskets:
        st.info("Insufficient data to render chart.")
        return

    common = basket_returns.index.intersection(benchmark_rets.index)
    if common.empty:
        st.info("No overlapping dates.")
        return

    use = [b for b in baskets if b in basket_returns.columns]
    if not use:
        st.info("No baskets loaded for this side.")
        return

    cum_pct = ((1 + basket_returns.loc[common, use]).cumprod() - 1.0) * 100.0
    bm_cum = ((1 + benchmark_rets.loc[common]).cumprod() - 1.0) * 100.0

    fig = go.Figure()
    for i, b in enumerate(use):
        fig.add_trace(go.Scatter(
            x=cum_pct.index,
            y=cum_pct[b],
            mode="lines",
            line=dict(width=2, color=PASTELS[i % len(PASTELS)]),
            name=b,
            hovertemplate=f"{b}<br>Cumulative: %{{y:.1f}}%<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=bm_cum.index,
        y=bm_cum.values,
        mode="lines",
        line=dict(width=2, dash="dash", color="#888888"),
        name="SPY",
        hovertemplate="SPY<br>Cumulative: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        title=dict(text=title, x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="Cumulative return, %",
        xaxis=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_factor_timeseries_plotly(factor_df: pd.DataFrame):
    norm = normalize_to_100(factor_df)
    fig = go.Figure()

    for i, col in enumerate(norm.columns):
        s = norm[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=col,
            line=dict(width=2, color=PASTELS[i % len(PASTELS)]),
            hovertemplate=f"{col}<br>Index: %{{y:.1f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Normalized factor relative-strength series, start = 100", x=0, xanchor="left"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_leadership_map_plotly(mom_df: pd.DataFrame):
    short_vals = mom_df["Short"] * 100.0
    long_vals = mom_df["Long"] * 100.0

    x_max = max(abs(short_vals.min()), abs(short_vals.max()), 1.0)
    y_max = max(abs(long_vals.min()), abs(long_vals.max()), 1.0)
    pad_x = x_max * 0.18
    pad_y = y_max * 0.18

    x0, x1 = -x_max - pad_x, x_max + pad_x
    y0, y1 = -y_max - pad_y, y_max + pad_y

    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, x1=x1, y0=0, y1=y1, fillcolor="rgba(225,245,224,0.55)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0, y1=y1, fillcolor="rgba(255,249,196,0.55)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0, fillcolor="rgba(253,224,220,0.55)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0, fillcolor="rgba(255,233,179,0.55)", line=dict(width=0), layer="below")

    fig.add_shape(type="line", x0=0, x1=0, y0=y0, y1=y1, line=dict(color="#888888", width=1))
    fig.add_shape(type="line", x0=x0, x1=x1, y0=0, y1=0, line=dict(color="#888888", width=1))

    for i, factor in enumerate(mom_df.index):
        fig.add_trace(go.Scatter(
            x=[short_vals.loc[factor]],
            y=[long_vals.loc[factor]],
            mode="markers+text",
            text=[factor],
            textposition="top center",
            marker=dict(size=13, color=PASTELS[i % len(PASTELS)], line=dict(color="#444444", width=1)),
            name=factor,
            hovertemplate=f"{factor}<br>Short: %{{x:.1f}}%<br>Long: %{{y:.1f}}%<extra></extra>",
            showlegend=False
        ))

    fig.add_annotation(x=x1 * 0.62, y=y1 * 0.72, text="Established leaders", showarrow=False, font=dict(size=12, color="#333333"))
    fig.add_annotation(x=x0 * 0.62, y=y1 * 0.72, text="Mean reversion", showarrow=False, font=dict(size=12, color="#333333"))
    fig.add_annotation(x=x0 * 0.62, y=y0 * 0.72, text="Persistent laggards", showarrow=False, font=dict(size=12, color="#333333"))
    fig.add_annotation(x=x1 * 0.62, y=y0 * 0.72, text="New rotations", showarrow=False, font=dict(size=12, color="#333333"))

    fig.update_layout(
        title=dict(text="Leadership map: short vs long momentum", x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Short window return %", range=[x0, x1], showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Long window return %", range=[y0, y1], showgrid=True, gridcolor=GRID, zeroline=False),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Sidebar
# =========================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("Settings")
    history_start = st.date_input("History start", datetime(2018, 1, 1))
    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    top_n = st.slider("Top baskets per side", 3, 10, 6)
    expand_all = st.checkbox("Expand all factor sections", value=False)
    st.caption("Data source: Yahoo Finance. Internal use only.")

# =========================================================
# Window selection
# =========================================================
today = date.today()
window_end = pd.Timestamp(today) + pd.Timedelta(days=1)

if window_choice == "YTD":
    window_start = pd.Timestamp(date(datetime.now().year, 1, 1))
    requested_trading_days = None
else:
    requested_trading_days = WINDOW_MAP_DAYS[window_choice]
    window_start = pd.Timestamp(today) - pd.Timedelta(days=int(requested_trading_days * 1.6))

# =========================================================
# Build ticker universe
# =========================================================
factor_tickers = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t is not None} | {BENCH})

mapped_baskets = []
for factor_name in FACTOR_ETFS.keys():
    mapping = FACTOR_TO_BASKETS.get(factor_name, {})
    mapped_baskets.extend(_safe_basket_list(mapping.get("pro", [])))
    mapped_baskets.extend(_safe_basket_list(mapping.get("anti", [])))
mapped_baskets = sorted(set(mapped_baskets))

basket_tickers = set()
for b in mapped_baskets:
    basket_tickers.update([str(x).upper() for x in ALL_BASKETS.get(b, [])])

need = sorted(set(factor_tickers) | set(basket_tickers) | {BENCH})

# =========================================================
# Fetch
# =========================================================
levels = fetch_daily_levels(
    need,
    start=pd.Timestamp(history_start),
    end=window_end,
)

if levels.empty:
    st.error("No data returned from Yahoo Finance or local cache.")
    st.stop()

if BENCH not in levels.columns or levels[BENCH].dropna().empty:
    st.error("SPY data missing or empty.")
    st.stop()

# =========================================================
# Factor construction
# =========================================================
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    up = up.upper()
    down_u = down.upper() if down is not None else None

    if down_u is None:
        if up in levels.columns:
            factor_levels_full[name] = levels[up]
        continue

    if up in levels.columns and down_u in levels.columns:
        rel = rs(levels[up], levels[down_u])
        if not rel.empty:
            factor_levels_full[name] = rel

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")
if factor_df_full.empty:
    st.error("No factor series could be constructed.")
    st.stop()

if requested_trading_days is not None:
    factor_df = factor_df_full.tail(min(requested_trading_days, len(factor_df_full))).copy()
    if not factor_df.empty:
        window_start = factor_df.index.min()
else:
    factor_df = factor_df_full[factor_df_full.index >= window_start].copy()

if factor_df.empty:
    st.error("No data available for the selected window.")
    st.stop()

# =========================================================
# Momentum snapshot
# =========================================================
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < max(lookback_long + 1, 15):
        continue

    r5 = pct_change_window(s, 5)
    r_short = pct_change_window(s, lookback_short)
    r_long = pct_change_window(s, lookback_long)
    mom_val = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)

    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors passed data checks for this window.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
num_down = int(trend_counts.get("Down", 0))
breadth = num_up / len(mom_df) * 100.0

raw_score = (
    0.4 * mom_df["Short"].mean()
    + 0.3 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.3 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

# =========================================================
# Summary cards
# =========================================================
st.subheader(f"Factor Tape Summary ({window_choice})")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Breadth", f"{breadth:.1f}%")
with c2:
    metric_card("Regime score", f"{regime_score:.1f}")
with c3:
    metric_card("Uptrends", str(num_up))
with c4:
    metric_card("Downtrends", str(num_down))

summary_html = build_commentary(mom_df, breadth, regime_score)
card_box(summary_html)

# =========================================================
# Factor charts
# =========================================================
st.subheader(f"Factor Time Series ({window_choice})")
plot_factor_timeseries_plotly(factor_df)

# =========================================================
# Snapshot table
# =========================================================
st.subheader("Factor Momentum Snapshot")

display_df = mom_df.copy()
for col in ["%5D", "Short", "Long", "Momentum"]:
    display_df[col] = display_df[col] * 100.0

display_df = display_df[["%5D", "Short", "Long", "Momentum", "Trend", "Inflection"]]

st.dataframe(
    display_df.style.format(
        {
            "%5D": "{:.1f}%",
            "Short": "{:.1f}%",
            "Long": "{:.1f}%",
            "Momentum": "{:.2f}%",
        }
    ),
    use_container_width=True,
)

# =========================================================
# Leadership map
# =========================================================
st.subheader("Leadership Map (Short vs Long Momentum)")
plot_leadership_map_plotly(mom_df)

# =========================================================
# Basket rotation
# =========================================================
st.subheader("Factor Rotation into ADFM Baskets (Pro vs Anti)")

mapped_basket_dict = {b: ALL_BASKETS[b] for b in mapped_baskets if b in ALL_BASKETS}
basket_rets = ew_rets_from_levels(levels, mapped_basket_dict, stale_days=30)
bench_rets = levels[BENCH].pct_change().dropna()

if basket_rets.empty:
    st.error("Basket return series not loaded. Check basket tickers and Yahoo Finance availability.")
    st.stop()

rot_df = basket_rets[basket_rets.index >= window_start].copy()
bench_rets_win = bench_rets[bench_rets.index >= window_start].copy()

if rot_df.empty or bench_rets_win.empty:
    st.error("Window slice produced empty basket or SPY series. Try a longer window or earlier history start.")
    st.stop()

all_panel_stats = build_all_basket_panels(
    basket_returns=rot_df,
    bench_rets=bench_rets_win,
    window_start=window_start,
    window_label=window_choice,
)

if all_panel_stats.empty:
    st.error("Basket panel statistics could not be built.")
    st.stop()

for i, factor_name in enumerate(FACTOR_ETFS.keys()):
    factor_slug = _slug(factor_name)
    pro_list = _safe_basket_list(FACTOR_TO_BASKETS.get(factor_name, {}).get("pro", []))
    anti_list = _safe_basket_list(FACTOR_TO_BASKETS.get(factor_name, {}).get("anti", []))

    with st.expander(factor_name, expanded=expand_all if i > 0 else True):
        if not pro_list and not anti_list:
            st.info("No basket mapping defined for this factor yet.")
            continue

        pro_panel = all_panel_stats.loc[all_panel_stats.index.intersection(pro_list)].head(top_n)
        anti_panel = all_panel_stats.loc[all_panel_stats.index.intersection(anti_list)].head(top_n)

        c1, c2 = st.columns(2)
        with c1:
            plot_rotation_table(pro_panel, "Pro side baskets", key=f"rot_tbl_pro_{factor_slug}")
        with c2:
            plot_rotation_table(anti_panel, "Anti side baskets", key=f"rot_tbl_anti_{factor_slug}")

        c3, c4 = st.columns(2)
        with c3:
            plot_side_cumulative(
                rot_df,
                pro_panel.index.tolist(),
                f"{factor_name} (Pro) vs SPY",
                bench_rets_win,
                key=f"rot_ch_pro_{factor_slug}",
            )
        with c4:
            plot_side_cumulative(
                rot_df,
                anti_panel.index.tolist(),
                f"{factor_name} (Anti) vs SPY",
                bench_rets_win,
                key=f"rot_ch_anti_{factor_slug}",
            )

st.caption("© 2026 AD Fund Management LP")
