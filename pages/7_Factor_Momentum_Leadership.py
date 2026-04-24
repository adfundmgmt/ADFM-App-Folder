import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Factor Momentum Leadership", layout="wide")
plt.style.use("default")

# =========================================================
# Config
# =========================================================
TITLE = "Factor Momentum Leadership"
SUBTITLE = "Relative factor leadership, regime pressure, rotation, and data-quality diagnostics."

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"
PASTEL_BLUE = "#6FB9C3"
PASTEL_ORANGE = "#E07A2F"
PASTEL_PURPLE = "#A889C7"
PASTEL_YELLOW = "#F2B874"
PASTEL_TEAL = "#58b4ae"

PALETTE = [
    PASTEL_BLUE,
    PASTEL_ORANGE,
    PASTEL_GREEN,
    PASTEL_YELLOW,
    "#8CC7F2",
    PASTEL_PURPLE,
    "#C7E29E",
    "#D89AD3",
    "#8FE3A1",
    "#E8CFC3",
    PASTEL_TEAL,
    "#C0A36E",
]

TEXT = "#222222"
SUBTLE = "#666666"
GRID = "#E6E6E6"
BORDER = "#E0E0E0"
CARD_BG = "#FAFAFA"

CACHE_DIR = Path(".adfm_factor_cache")
CACHE_DIR.mkdir(exist_ok=True)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.15px;
        color: #222222;
    }

    div[data-testid="stCaptionContainer"] {
        color: #666666;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# Factor universe
# =========================================================
@dataclass(frozen=True)
class FactorPair:
    name: str
    numerator: str
    denominator: Optional[str]
    category: str
    interpretation: str

    @property
    def expression(self) -> str:
        if self.denominator:
            return f"{self.numerator} / {self.denominator}"
        return self.numerator


FACTOR_PAIRS: List[FactorPair] = [
    FactorPair(
        name="Growth vs Value",
        numerator="VUG",
        denominator="VTV",
        category="Equity Style",
        interpretation="Growth leadership versus value leadership.",
    ),
    FactorPair(
        name="Quality vs Spec Growth",
        numerator="QUAL",
        denominator="ARKK",
        category="Equity Style",
        interpretation="Quality balance sheets versus speculative long-duration growth.",
    ),
    FactorPair(
        name="High Beta vs Low Vol",
        numerator="SPHB",
        denominator="SPLV",
        category="Risk Appetite",
        interpretation="High-beta equities versus low-volatility defensives.",
    ),
    FactorPair(
        name="Small vs Large",
        numerator="IWM",
        denominator="SPY",
        category="Market Breadth",
        interpretation="Small-cap participation versus S&P 500 leadership.",
    ),
    FactorPair(
        name="Equal Weight vs Cap Weight",
        numerator="RSP",
        denominator="SPY",
        category="Market Breadth",
        interpretation="Average-stock participation versus cap-weighted concentration.",
    ),
    FactorPair(
        name="Momentum vs Market",
        numerator="MTUM",
        denominator="SPY",
        category="Equity Style",
        interpretation="Price momentum factor versus broad market.",
    ),
    FactorPair(
        name="Dividend Quality vs Market",
        numerator="SCHD",
        denominator="SPY",
        category="Equity Style",
        interpretation="Dividend quality and cash-return equities versus broad market.",
    ),
    FactorPair(
        name="Tech vs Broad Market",
        numerator="XLK",
        denominator="SPY",
        category="Sector Leadership",
        interpretation="Technology sector leadership versus the broad market.",
    ),
    FactorPair(
        name="Semis vs Tech",
        numerator="SMH",
        denominator="XLK",
        category="Sector Leadership",
        interpretation="Semiconductor leadership versus the broader technology sector.",
    ),
    FactorPair(
        name="Cyclicals vs Staples",
        numerator="XLY",
        denominator="XLP",
        category="Cycle Signal",
        interpretation="Consumer cyclicals versus staples defensiveness.",
    ),
    FactorPair(
        name="Industrials vs Utilities",
        numerator="XLI",
        denominator="XLU",
        category="Cycle Signal",
        interpretation="Industrial cyclicality versus utility defensiveness.",
    ),
    FactorPair(
        name="Financials vs Utilities",
        numerator="XLF",
        denominator="XLU",
        category="Rates and Cycle",
        interpretation="Financials versus rate-sensitive defensives.",
    ),
    FactorPair(
        name="US vs Ex-US",
        numerator="SPY",
        denominator="VXUS",
        category="Global Equity",
        interpretation="US equity leadership versus non-US equities.",
    ),
    FactorPair(
        name="EM vs DM Ex-US",
        numerator="EEM",
        denominator="VEA",
        category="Global Equity",
        interpretation="Emerging markets versus developed ex-US equities.",
    ),
    FactorPair(
        name="High Yield Credit vs Treasuries",
        numerator="HYG",
        denominator="IEF",
        category="Credit Risk",
        interpretation="Credit risk appetite versus intermediate Treasuries.",
    ),
    FactorPair(
        name="Long Duration vs Bills",
        numerator="TLT",
        denominator="SHY",
        category="Rates",
        interpretation="Long-duration Treasury performance versus short bills.",
    ),
    FactorPair(
        name="Copper Miners vs Gold Miners",
        numerator="COPX",
        denominator="GDX",
        category="Hard Assets",
        interpretation="Cyclical metal miners versus monetary metal miners.",
    ),
]

BENCH = "SPY"

WINDOW_MAP_DAYS: Dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

# =========================================================
# General helpers
# =========================================================
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _safe_upper(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    out = str(value).upper().strip()
    return out if out else None


def _hash_key(prefix: str, payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]
    return f"{prefix}_{digest}"


def _cache_path(name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    return CACHE_DIR / f"{safe_name}.pkl"


def _cache_meta_path(name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    return CACHE_DIR / f"{safe_name}.json"


def _load_last_good_cache(name: str) -> Tuple[pd.DataFrame, Dict]:
    path = _cache_path(name)
    meta_path = _cache_meta_path(name)

    if not path.exists():
        return pd.DataFrame(), {}

    try:
        df = pd.read_pickle(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        return df, meta

    except Exception:
        return pd.DataFrame(), {}


def _save_last_good_cache(name: str, df: pd.DataFrame, meta: Dict) -> None:
    if df is None or df.empty:
        return

    try:
        df.to_pickle(_cache_path(name))
        _cache_meta_path(name).write_text(json.dumps(meta, indent=2, default=str))
    except Exception:
        pass


def _last_valid_date(series: pd.Series) -> Optional[pd.Timestamp]:
    s = series.dropna()

    if s.empty:
        return None

    return pd.Timestamp(s.index.max()).normalize()


def _format_date(value: Optional[pd.Timestamp]) -> str:
    if value is None or pd.isna(value):
        return "n/a"

    return pd.Timestamp(value).strftime("%Y-%m-%d")

# =========================================================
# Math helpers
# =========================================================
def pct_change_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    days = int(days)

    if days <= 0 or len(s) <= days:
        return np.nan

    base = s.iloc[-(days + 1)]

    if pd.isna(base) or base == 0:
        return np.nan

    return float(s.iloc[-1] / base - 1.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    s = series.dropna()
    return s.ewm(span=span, adjust=False).mean()


def trend_class(series: pd.Series) -> str:
    s = series.dropna()

    if len(s) < 50:
        return "Neutral"

    e10 = ema(s, 10).iloc[-1]
    e20 = ema(s, 20).iloc[-1]
    e40 = ema(s, 40).iloc[-1]

    if e10 > e20 > e40:
        return "Up"

    if e10 < e20 < e40:
        return "Down"

    return "Neutral"


def inflection(short_ret: float, long_ret: float) -> str:
    if pd.isna(short_ret) or pd.isna(long_ret):
        return "Neutral"

    if short_ret > 0 and long_ret < 0:
        return "Turning Up"

    if short_ret < 0 and long_ret > 0:
        return "Turning Down"

    if short_ret > 0 and long_ret > 0:
        return "Confirmed Up"

    if short_ret < 0 and long_ret < 0:
        return "Confirmed Down"

    return "Mixed"


def slope_zscore(series: pd.Series, lookback: int = 20, history: int = 252) -> float:
    s = series.dropna()
    lookback = int(max(5, lookback))
    history = int(max(40, history))

    if len(s) < lookback + 25:
        return np.nan

    r = s.pct_change().dropna()
    roll = r.rolling(lookback).mean().dropna()
    ref = roll.tail(history)

    if len(ref) < max(20, lookback):
        return np.nan

    mu = ref.mean()
    sd = ref.std(ddof=0)

    if sd == 0 or pd.isna(sd):
        return 0.0

    return float((roll.iloc[-1] - mu) / sd)


def trend_strength(series: pd.Series) -> float:
    s = series.dropna()

    if len(s) < 50:
        return np.nan

    e10 = ema(s, 10)
    e40 = ema(s, 40)

    if e40.iloc[-1] == 0 or pd.isna(e40.iloc[-1]):
        return np.nan

    return float((e10.iloc[-1] - e40.iloc[-1]) / e40.iloc[-1])


def normalized_series(series: pd.Series) -> pd.Series:
    s = series.dropna()

    if s.empty or s.iloc[0] == 0:
        return s

    return s / s.iloc[0] * 100.0


def robust_z_cross_section(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)

    if s.dropna().empty:
        return pd.Series(0.0, index=s.index)

    median = s.median(skipna=True)
    mad = (s - median).abs().median(skipna=True)

    if pd.isna(mad) or mad == 0:
        std = s.std(skipna=True, ddof=0)

        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)

        z = (s - s.mean(skipna=True)) / std
        return z.clip(-3, 3).fillna(0.0)

    z = 0.6745 * (s - median) / mad
    return z.clip(-3, 3).fillna(0.0)


def latest_return_percentile(series: pd.Series, days: int, min_samples: int = 60) -> float:
    s = series.dropna()
    days = int(days)

    if days <= 0 or len(s) < days + min_samples:
        return np.nan

    returns = (s / s.shift(days) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < min_samples:
        return np.nan

    current = returns.iloc[-1]

    if pd.isna(current):
        return np.nan

    return float((returns <= current).mean() * 100.0)


def percentile_signal(value: float) -> float:
    if pd.isna(value):
        return 0.0

    return float(np.clip((value - 50.0) / 50.0, -1.0, 1.0))


def build_relative_series(a: pd.Series, b: pd.Series, min_obs: int = 60) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()

    if len(aligned) < min_obs:
        return pd.Series(dtype=float)

    denom = aligned.iloc[:, 1].replace(0, np.nan)
    out = aligned.iloc[:, 0] / denom
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out.name = f"{a.name}_vs_{b.name}"

    return out

# =========================================================
# Data download and normalization
# =========================================================
def _normalize_yf_download(df: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
    requested = [_safe_upper(x) for x in requested if x]
    requested = [x for x in requested if x]

    if df is None or df.empty:
        return pd.DataFrame()

    out: Optional[pd.DataFrame] = None

    if not isinstance(df.columns, pd.MultiIndex):
        cols_lower = {str(c).lower(): c for c in df.columns}
        price_col = None

        for candidate in ("close", "adj close"):
            if candidate in cols_lower:
                price_col = cols_lower[candidate]
                break

        if price_col is not None:
            s = df[price_col].copy()
            ticker = requested[0] if requested else "TICKER"
            s.name = ticker
            out = s.to_frame()

    if out is None and isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        level1 = [str(x) for x in df.columns.get_level_values(1)]

        if "Close" in level0 or "Adj Close" in level0:
            field = "Close" if "Close" in level0 else "Adj Close"

            try:
                tmp = df[field].copy()

                if isinstance(tmp, pd.Series):
                    tmp = tmp.to_frame()

                tmp.columns = [_safe_upper(c) for c in tmp.columns]
                out = tmp

            except Exception:
                out = None

        if out is None:
            candidate_fields = {"Close", "Adj Close"}

            if any(f in level1 for f in candidate_fields):
                frames = {}

                for ticker in requested:
                    for field in ("Close", "Adj Close"):
                        key = (ticker, field)

                        if key in df.columns:
                            frames[ticker] = df[key]
                            break

                if frames:
                    out = pd.DataFrame(frames)

    if out is None or out.empty:
        return pd.DataFrame()

    out = out.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    out.columns = [_safe_upper(c) for c in out.columns]

    valid_cols = [c for c in requested if c in out.columns]

    if not valid_cols:
        return pd.DataFrame()

    return out[valid_cols]


def _download_batch_once(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    batch = [_safe_upper(x) for x in batch if x]
    batch = [x for x in batch if x]

    if not batch:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=batch if len(batch) > 1 else batch[0],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        return _normalize_yf_download(data, batch)

    except Exception:
        return pd.DataFrame()


def _download_one_by_one(batch: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []

    for ticker in batch:
        out = pd.DataFrame()

        for attempt in range(3):
            out = _download_batch_once([ticker], start, end)

            if not out.empty and ticker in out.columns and out[ticker].dropna().shape[0] > 0:
                break

            time.sleep(0.6 + attempt * 0.2)

        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame()

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()].sort_index()

    return wide


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_daily_levels(
    tickers: Tuple[str, ...],
    start_str: str,
    end_str: str,
    chunk_size: int = 25,
) -> Tuple[pd.DataFrame, Dict]:
    uniq = sorted({_safe_upper(t) for t in tickers if t})
    uniq = [t for t in uniq if t]

    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)

    if not uniq:
        return pd.DataFrame(), {"mode": "empty", "message": "No tickers requested."}

    cache_key = _hash_key(
        "levels",
        {
            "tickers": uniq,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
    )

    cached_df, cached_meta = _load_last_good_cache(cache_key)

    frames: List[pd.DataFrame] = []
    live_found = set()

    for batch in _chunk(uniq, chunk_size):
        out = pd.DataFrame()

        for attempt in range(3):
            out = _download_batch_once(batch, start, end)

            if not out.empty:
                break

            time.sleep(0.6 + attempt * 0.25)

        if not out.empty:
            frames.append(out)
            live_found.update([c for c in out.columns if out[c].dropna().shape[0] > 0])

    missing_after_batch = [t for t in uniq if t not in live_found]

    if missing_after_batch:
        rescue = _download_one_by_one(missing_after_batch, start, end)

        if not rescue.empty:
            frames.append(rescue)
            live_found.update([c for c in rescue.columns if rescue[c].dropna().shape[0] > 0])

    if frames:
        wide = pd.concat(frames, axis=1)
        wide = wide.loc[:, ~wide.columns.duplicated()]
        wide = wide.sort_index()
        wide = wide[[c for c in uniq if c in wide.columns]]
        wide = wide.ffill(limit=3)

        cache_rescue = []

        if not cached_df.empty:
            for ticker in uniq:
                if ticker not in wide.columns and ticker in cached_df.columns:
                    wide[ticker] = cached_df[ticker]
                    cache_rescue.append(ticker)

            wide = wide[[c for c in uniq if c in wide.columns]]
            wide = wide.sort_index()

        missing = [t for t in uniq if t not in wide.columns or wide[t].dropna().empty]
        mode = "live_plus_cache" if cache_rescue else "live"

        meta = {
            "mode": mode,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "live_tickers": sorted(live_found),
            "cache_rescue_tickers": cache_rescue,
            "missing_tickers": missing,
            "last_index": _format_date(pd.Timestamp(wide.index.max()) if not wide.empty else None),
            "prior_cache_fetched_at": cached_meta.get("fetched_at"),
        }

        if not wide.empty:
            _save_last_good_cache(cache_key, wide, meta)

        return wide, meta

    if not cached_df.empty:
        missing = [t for t in uniq if t not in cached_df.columns or cached_df[t].dropna().empty]

        cached_df = cached_df[[c for c in uniq if c in cached_df.columns]].sort_index()

        meta = {
            "mode": "cache",
            "fetched_at": cached_meta.get("fetched_at", "unknown"),
            "cache_loaded_at": datetime.now().isoformat(timespec="seconds"),
            "live_tickers": [],
            "cache_rescue_tickers": [c for c in cached_df.columns if c in uniq],
            "missing_tickers": missing,
            "last_index": _format_date(pd.Timestamp(cached_df.index.max()) if not cached_df.empty else None),
        }

        return cached_df, meta

    return pd.DataFrame(), {
        "mode": "empty",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "missing_tickers": uniq,
        "message": "Yahoo returned no usable close data and no last-good cache was available.",
    }

# =========================================================
# State and styling helpers
# =========================================================
def factor_state(score: float) -> str:
    if pd.isna(score):
        return "Unscored"

    if score >= 70:
        return "Leader"

    if score >= 57:
        return "Positive"

    if score > 43:
        return "Neutral"

    if score > 30:
        return "Weak"

    return "Laggard"


def composite_color(score: float) -> str:
    if pd.isna(score):
        return PASTEL_GREY

    if score >= 57:
        return PASTEL_GREEN

    if score <= 43:
        return PASTEL_RED

    return PASTEL_GREY


def style_state(value: str) -> str:
    if value in ("Leader", "Positive"):
        return f"background-color: {PASTEL_GREEN}; color: white; font-weight: 600;"

    if value in ("Weak", "Laggard"):
        return f"background-color: {PASTEL_RED}; color: white; font-weight: 600;"

    return f"background-color: {PASTEL_GREY}; color: white; font-weight: 600;"


def style_pct_value(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return ""

    if v > 0:
        return f"color: {PASTEL_GREEN}; font-weight: 600;"

    if v < 0:
        return f"color: {PASTEL_RED}; font-weight: 600;"

    return "color: #555555;"

# =========================================================
# Plot helpers
# =========================================================
def plot_factor_grid(
    plot_df: pd.DataFrame,
    mom_df: pd.DataFrame,
    normalize_charts: bool,
    show_ema: bool,
) -> Optional[plt.Figure]:
    if plot_df.empty or len(plot_df.columns) == 0:
        return None

    n_factors = len(plot_df.columns)
    ncols = 3
    nrows = int(np.ceil(n_factors / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.1 * nrows), squeeze=False)
    axes = axes.ravel()

    if len(plot_df.index) > 1:
        span_days = (plot_df.index[-1] - plot_df.index[0]).days
    else:
        span_days = 0

    if span_days <= 120:
        locator = mdates.WeekdayLocator(interval=2)
        formatter = mdates.DateFormatter("%b %d")
    elif span_days <= 420:
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter("%b")
    else:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")

    for i, factor_name in enumerate(plot_df.columns):
        ax = axes[i]
        s = plot_df[factor_name].dropna()

        if s.empty:
            ax.axis("off")
            continue

        if factor_name in mom_df.index:
            color = composite_color(mom_df.loc[factor_name, "Composite"])
        else:
            color = PALETTE[i % len(PALETTE)]

        ax.plot(s.index, s.values, color=color, linewidth=2.1)

        if show_ema and len(s) >= 20 and not normalize_charts:
            e20 = ema(s, 20)
            ax.plot(e20.index, e20.values, color="#777777", linewidth=1.0, alpha=0.9)

        latest = s.iloc[-1]

        if factor_name in mom_df.index:
            latest_short = mom_df.loc[factor_name, "Short"] * 100.0
            latest_long = mom_df.loc[factor_name, "Long"] * 100.0
            composite = mom_df.loc[factor_name, "Composite"]
            title = f"{factor_name}\nScore {composite:.1f} | S {latest_short:.1f}% | L {latest_long:.1f}%"
        else:
            title = factor_name

        ax.set_title(title, color=TEXT, fontsize=10.5, pad=8)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.scatter(s.index[-1], latest, s=20, color=color, zorder=3)

    for j in range(n_factors, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    return fig


def plot_composite_bar(mom_df: pd.DataFrame) -> plt.Figure:
    df = mom_df.sort_values("Composite", ascending=True).copy()
    height = max(4.5, 0.42 * len(df))

    fig, ax = plt.subplots(figsize=(10.5, height))

    colors = [composite_color(v) for v in df["Composite"]]
    ax.barh(df.index, df["Composite"], color=colors, edgecolor="#FFFFFF", linewidth=0.8)

    ax.axvline(50, color="#777777", linewidth=1.0, alpha=0.9)
    ax.axvline(57, color=PASTEL_GREEN, linewidth=0.8, alpha=0.7)
    ax.axvline(43, color=PASTEL_RED, linewidth=0.8, alpha=0.7)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Composite leadership score", color=TEXT)
    ax.set_title("Factor Ranking", color=TEXT, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    for y, value in enumerate(df["Composite"]):
        ax.text(min(value + 1.2, 97.0), y, f"{value:.1f}", va="center", fontsize=8, color="#333333")

    fig.tight_layout()

    return fig


def plot_leadership_map(mom_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))

    short_vals = mom_df["Short"] * 100.0
    long_vals = mom_df["Long"] * 100.0

    x_abs = max(abs(short_vals.min()), abs(short_vals.max()), 1.0)
    y_abs = max(abs(long_vals.min()), abs(long_vals.max()), 1.0)

    pad_x = x_abs * 0.18
    pad_y = y_abs * 0.18

    ax.set_xlim(-x_abs - pad_x, x_abs + pad_x)
    ax.set_ylim(-y_abs - pad_y, y_abs + pad_y)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.fill_between([0, x_max], 0, y_max, color="#DDF3E7", alpha=0.8)
    ax.fill_between([x_min, 0], 0, y_max, color="#FFF2C2", alpha=0.8)
    ax.fill_between([x_min, 0], y_min, 0, color="#FAD7D7", alpha=0.8)
    ax.fill_between([0, x_max], y_min, 0, color="#FFE3B5", alpha=0.8)

    ax.axvline(0, color="#777777", linewidth=1)
    ax.axhline(0, color="#777777", linewidth=1)

    ax.text(
        x_max * 0.58,
        y_max * 0.78,
        "Short up / Long up\nEstablished leaders",
        fontsize=9,
        ha="center",
        va="center",
        color="#333333",
    )
    ax.text(
        x_min * 0.58,
        y_max * 0.78,
        "Short down / Long up\nMean reversion risk",
        fontsize=9,
        ha="center",
        va="center",
        color="#333333",
    )
    ax.text(
        x_min * 0.58,
        y_min * 0.78,
        "Short down / Long down\nPersistent laggards",
        fontsize=9,
        ha="center",
        va="center",
        color="#333333",
    )
    ax.text(
        x_max * 0.58,
        y_min * 0.78,
        "Short up / Long down\nNew rotations",
        fontsize=9,
        ha="center",
        va="center",
        color="#333333",
    )

    for factor_name in mom_df.index:
        x = short_vals.loc[factor_name]
        y = long_vals.loc[factor_name]
        score = mom_df.loc[factor_name, "Composite"]

        ax.scatter(
            x,
            y,
            s=80,
            color=composite_color(score),
            edgecolor="#444444",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            factor_name,
            xy=(x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8.5,
            va="center",
            color="#111111",
        )

    ax.set_xlabel("Short-window return %", color=TEXT)
    ax.set_ylabel("Long-window return %", color=TEXT)
    ax.set_title("Short vs Long Momentum", color=TEXT, pad=10)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    return fig

# =========================================================
# Sidebar
# =========================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Monitor factor leadership using ETF-relative price ratios, absolute return percentiles, trend structure, and rotation pressure.

        **How to read it**
        - Composite score above 57 means positive factor pressure.
        - Composite score below 43 means negative factor pressure.
        - Short breadth shows near-term participation.
        - Long breadth shows durability.
        - Conflict shows factors where short and long windows disagree.

        **Data source:** Yahoo Finance via `yfinance`. Signals are ETF price-based and do not include holdings-level flows, options data, or valuation data.
        """
    )

    st.markdown("---")
    st.header("Settings")

    history_start_input = st.date_input("History start", datetime(2015, 1, 1))

    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )

    lookback_short = st.slider("Short momentum window, trading days", 10, 60, 20)

    long_min = max(30, lookback_short + 5)
    lookback_long = st.slider("Long momentum window, trading days", long_min, 180, max(60, long_min))

    normalize_charts = st.checkbox("Normalize factor charts to 100", value=False)
    show_ema = st.checkbox("Show 20-day EMA on factor charts", value=True)

    min_overlap_obs = st.slider("Minimum overlap observations per factor pair", 40, 252, 60, 5)
    stale_threshold_days = st.slider("Stale data threshold in calendar days", 3, 10, 5)

    st.caption("Internal use only. Cached fallback is shown explicitly when used.")

# =========================================================
# Window setup
# =========================================================
today = pd.Timestamp(date.today()).normalize()
history_start = pd.Timestamp(history_start_input).normalize()

if window_choice == "YTD":
    window_start = pd.Timestamp(date(datetime.now().year, 1, 1))
    requested_days = None
else:
    requested_days = WINDOW_MAP_DAYS[window_choice]
    window_start = today - pd.Timedelta(days=int(requested_days * 1.65))

window_end = today + pd.Timedelta(days=1)

# =========================================================
# Fetch data
# =========================================================
factor_tickers = sorted(
    {
        ticker
        for pair in FACTOR_PAIRS
        for ticker in (pair.numerator, pair.denominator)
        if ticker is not None
    }
    | {BENCH}
)

with st.spinner("Loading ETF price data..."):
    levels, fetch_meta = fetch_daily_levels(
        tuple(factor_tickers),
        start_str=history_start.strftime("%Y-%m-%d"),
        end_str=window_end.strftime("%Y-%m-%d"),
        chunk_size=25,
    )

if levels.empty:
    st.error("Price download failed for all requested ETFs. Yahoo returned no usable close data and no last-good cache was available.")
    st.stop()

levels = levels.sort_index()
levels = levels.loc[:, ~levels.columns.duplicated()]

if BENCH not in levels.columns or levels[BENCH].dropna().empty:
    st.error("SPY data is missing or empty after fetch normalization.")
    st.stop()

# =========================================================
# Freshness diagnostics
# =========================================================
freshness_rows = []
stale_tickers = []

for ticker in factor_tickers:
    if ticker not in levels.columns or levels[ticker].dropna().empty:
        freshness_rows.append(
            {
                "Ticker": ticker,
                "Last Date": "n/a",
                "Age Days": np.nan,
                "Status": "Missing",
            }
        )
        continue

    last_dt = _last_valid_date(levels[ticker])
    age_days = int((today - last_dt).days) if last_dt is not None else np.nan
    status = "Stale" if pd.notna(age_days) and age_days > stale_threshold_days else "OK"

    if status == "Stale":
        stale_tickers.append(ticker)

    freshness_rows.append(
        {
            "Ticker": ticker,
            "Last Date": _format_date(last_dt),
            "Age Days": age_days,
            "Status": status,
        }
    )

freshness_df = pd.DataFrame(freshness_rows).sort_values(["Status", "Ticker"])
as_of_date = _format_date(pd.Timestamp(levels.index.max()))

# =========================================================
# Build factor series
# =========================================================
factor_meta = {pair.name: pair for pair in FACTOR_PAIRS}

factor_levels_full: Dict[str, pd.Series] = {}
pair_diagnostics: List[Dict[str, object]] = []

for pair in FACTOR_PAIRS:
    up = _safe_upper(pair.numerator)
    down = _safe_upper(pair.denominator)

    if up not in levels.columns or levels[up].dropna().empty:
        pair_diagnostics.append(
            {
                "Factor": pair.name,
                "Category": pair.category,
                "Expression": pair.expression,
                "Status": "Skipped",
                "Observations": 0,
                "Reason": f"Missing usable data for {up}",
            }
        )
        continue

    if down is None:
        s = levels[up].dropna().copy()
        s.name = pair.name
        factor_levels_full[pair.name] = s

        pair_diagnostics.append(
            {
                "Factor": pair.name,
                "Category": pair.category,
                "Expression": pair.expression,
                "Status": "OK",
                "Observations": len(s),
                "Reason": f"Using standalone series {up}",
            }
        )
        continue

    if down not in levels.columns or levels[down].dropna().empty:
        pair_diagnostics.append(
            {
                "Factor": pair.name,
                "Category": pair.category,
                "Expression": pair.expression,
                "Status": "Skipped",
                "Observations": 0,
                "Reason": f"Missing usable data for {down}",
            }
        )
        continue

    rel = build_relative_series(levels[up], levels[down], min_obs=min_overlap_obs)
    overlap = pd.concat([levels[up], levels[down]], axis=1).dropna().shape[0]

    if rel.empty:
        pair_diagnostics.append(
            {
                "Factor": pair.name,
                "Category": pair.category,
                "Expression": pair.expression,
                "Status": "Skipped",
                "Observations": overlap,
                "Reason": f"Overlap too short: {overlap} observations",
            }
        )
        continue

    rel.name = pair.name
    factor_levels_full[pair.name] = rel

    pair_diagnostics.append(
        {
            "Factor": pair.name,
            "Category": pair.category,
            "Expression": pair.expression,
            "Status": "OK",
            "Observations": len(rel),
            "Reason": pair.interpretation,
        }
    )

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")

if factor_df_full.empty:
    st.error("No factor series could be constructed from the available ETF data.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

if requested_days is None:
    factor_df = factor_df_full[factor_df_full.index >= window_start].copy()
else:
    factor_df = factor_df_full.tail(min(requested_days, len(factor_df_full))).copy()

    if not factor_df.empty:
        window_start = factor_df.index.min()

if factor_df.empty:
    st.error("No data is available for the selected window.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

# =========================================================
# Momentum snapshot
# =========================================================
rows = []

for factor_name in factor_df.columns:
    s_win = factor_df[factor_name].dropna()
    s_full = factor_df_full[factor_name].dropna()

    if len(s_win) < 15:
        continue

    eff_short = min(lookback_short, max(5, len(s_win) - 2))
    eff_long = min(lookback_long, max(eff_short + 1, len(s_win) - 2))

    if eff_long <= eff_short:
        eff_long = min(len(s_win) - 2, eff_short + 1)

    if len(s_win) <= eff_long or eff_long <= 0:
        continue

    r5 = pct_change_window(s_win, min(5, len(s_win) - 2))
    r_short = pct_change_window(s_win, eff_short)
    r_long = pct_change_window(s_win, eff_long)

    if pd.notna(r_long):
        expected_short_from_long_pace = r_long * (eff_short / eff_long)
    else:
        expected_short_from_long_pace = np.nan

    if pd.notna(r_short) and pd.notna(expected_short_from_long_pace):
        accel = r_short - expected_short_from_long_pace
    else:
        accel = np.nan

    short_pctile = latest_return_percentile(s_full, eff_short)
    long_pctile = latest_return_percentile(s_full, eff_long)
    tclass = trend_class(s_win)
    infl = inflection(r_short, r_long)
    slope_z = slope_zscore(s_full, lookback=min(20, max(10, len(s_full) // 20)), history=252)
    t_strength = trend_strength(s_win)

    pair = factor_meta.get(factor_name)

    rows.append(
        {
            "Factor": factor_name,
            "Category": pair.category if pair else "n/a",
            "Expression": pair.expression if pair else "n/a",
            "Interpretation": pair.interpretation if pair else "n/a",
            "%5D": r5,
            "Short": r_short,
            "Long": r_long,
            "Accel": accel,
            "Short Pctl": short_pctile,
            "Long Pctl": long_pctile,
            "Trend": tclass,
            "Inflection": infl,
            "Eff Short": eff_short,
            "Eff Long": eff_long,
            "Slope Z": slope_z,
            "Trend Strength": t_strength,
            "Obs Window": len(s_win),
            "Obs Full": len(s_full),
        }
    )

mom_df = pd.DataFrame(rows)

if mom_df.empty:
    st.error("No factors passed data checks for this window. Try a longer analysis window or reduce the short and long lookbacks.")
    with st.expander("Diagnostics", expanded=True):
        st.dataframe(pd.DataFrame(pair_diagnostics), use_container_width=True, hide_index=True)
    st.stop()

mom_df = mom_df.set_index("Factor")

# =========================================================
# Composite scores
# =========================================================
short_abs = mom_df["Short Pctl"].apply(percentile_signal)
long_abs = mom_df["Long Pctl"].apply(percentile_signal)

accel_cs = robust_z_cross_section(mom_df["Accel"]).clip(-3, 3) / 3.0
slope_component = mom_df["Slope Z"].fillna(0.0).clip(-3, 3) / 3.0

trend_signal = mom_df["Trend"].map({"Up": 1.0, "Neutral": 0.0, "Down": -1.0}).fillna(0.0)

inflection_signal = mom_df["Inflection"].map(
    {
        "Confirmed Up": 1.0,
        "Turning Up": 0.65,
        "Mixed": 0.0,
        "Neutral": 0.0,
        "Turning Down": -0.65,
        "Confirmed Down": -1.0,
    }
).fillna(0.0)

score_raw = (
    0.24 * short_abs
    + 0.20 * long_abs
    + 0.18 * accel_cs
    + 0.18 * trend_signal
    + 0.10 * inflection_signal
    + 0.10 * slope_component
)

mom_df["Composite"] = (50.0 + 35.0 * score_raw).clip(0.0, 100.0)
mom_df["State"] = mom_df["Composite"].apply(factor_state)
mom_df = mom_df.sort_values("Composite", ascending=False)

ok_pairs = [d for d in pair_diagnostics if d["Status"] == "OK"]
skipped_pairs = [d for d in pair_diagnostics if d["Status"] != "OK"]
missing_tickers = fetch_meta.get("missing_tickers", []) or []

# =========================================================
# Main layout
# =========================================================
st.caption(
    f"As of {as_of_date}. Window starts {window_start.strftime('%Y-%m-%d')}. "
    f"Fetch mode: {fetch_meta.get('mode', 'unknown')}. Last fetch timestamp: {fetch_meta.get('fetched_at', 'unknown')}."
)

# =========================================================
# Factor charts first
# =========================================================
st.subheader(f"Factor Time Series ({window_choice})")

plot_df = factor_df.copy()
plot_df = plot_df[[c for c in mom_df.index if c in plot_df.columns]]

if normalize_charts:
    for col in plot_df.columns:
        plot_df[col] = normalized_series(plot_df[col])

fig_ts = plot_factor_grid(
    plot_df=plot_df,
    mom_df=mom_df,
    normalize_charts=normalize_charts,
    show_ema=show_ema,
)

if fig_ts is not None:
    st.pyplot(fig_ts, clear_figure=True)
    plt.close(fig_ts)
else:
    st.info("No factor charts are available for the selected window.")

# =========================================================
# Ranking view below factor charts
# =========================================================
st.subheader("All-Factor Ranking View")

fig_rank = plot_composite_bar(mom_df)
st.pyplot(fig_rank, clear_figure=True)
plt.close(fig_rank)

ranking_df = mom_df.copy()

for col in ["%5D", "Short", "Long", "Accel", "Trend Strength"]:
    ranking_df[col] = ranking_df[col] * 100.0

ranking_df = ranking_df[
    [
        "State",
        "Category",
        "Expression",
        "Composite",
        "%5D",
        "Short",
        "Long",
        "Accel",
        "Short Pctl",
        "Long Pctl",
        "Trend",
        "Inflection",
        "Slope Z",
        "Trend Strength",
        "Obs Window",
        "Obs Full",
        "Interpretation",
    ]
]

format_map = {
    "Composite": "{:.1f}",
    "%5D": "{:.1f}%",
    "Short": "{:.1f}%",
    "Long": "{:.1f}%",
    "Accel": "{:.1f}%",
    "Short Pctl": "{:.0f}",
    "Long Pctl": "{:.0f}",
    "Slope Z": "{:.2f}",
    "Trend Strength": "{:.2f}%",
    "Obs Window": "{:.0f}",
    "Obs Full": "{:.0f}",
}

styled_ranking = ranking_df.style.format(format_map, na_rep="n/a")

try:
    styled_ranking = styled_ranking.map(style_state, subset=["State"])
    styled_ranking = styled_ranking.map(
        style_pct_value,
        subset=["%5D", "Short", "Long", "Accel", "Trend Strength"],
    )
except AttributeError:
    styled_ranking = styled_ranking.applymap(style_state, subset=["State"])
    styled_ranking = styled_ranking.applymap(
        style_pct_value,
        subset=["%5D", "Short", "Long", "Accel", "Trend Strength"],
    )

st.dataframe(styled_ranking, use_container_width=True)

# =========================================================
# Leadership map
# =========================================================
st.subheader("Leadership Map")

fig_lead = plot_leadership_map(mom_df)
st.pyplot(fig_lead, clear_figure=True)
plt.close(fig_lead)

# =========================================================
# Data quality and diagnostics
# =========================================================
st.subheader("Data Quality and Diagnostics")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Fetch status**")

    status_rows = [
        {"Field": "Mode", "Value": fetch_meta.get("mode", "unknown")},
        {"Field": "Fetched At", "Value": fetch_meta.get("fetched_at", "unknown")},
        {"Field": "Loaded Cache At", "Value": fetch_meta.get("cache_loaded_at", "n/a")},
        {"Field": "Last Index", "Value": fetch_meta.get("last_index", as_of_date)},
        {"Field": "Live Tickers", "Value": len(fetch_meta.get("live_tickers", []) or [])},
        {
            "Field": "Cache Rescue Tickers",
            "Value": ", ".join(fetch_meta.get("cache_rescue_tickers", []) or []) or "none",
        },
        {"Field": "Missing Tickers", "Value": ", ".join(missing_tickers) or "none"},
    ]

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**Ticker freshness**")
    st.dataframe(freshness_df, use_container_width=True, hide_index=True)

with st.expander(f"Factor diagnostics: {len(ok_pairs)} built, {len(skipped_pairs)} skipped", expanded=False):
    diag_df = pd.DataFrame(pair_diagnostics)
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

st.caption("© 2026 AD Fund Management LP")
