import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

from adfm_core.palette import PASTEL, PASTEL_20
from adfm_core.ui import PageHeader, render_footer, render_kpi_cards, render_page_header
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

PASTEL_GREEN = PASTEL["sage"]
PASTEL_RED = PASTEL["rose"]
PASTEL_BLUE = PASTEL["blue"]
PASTEL_ORANGE = PASTEL["coral"]
PASTEL_PURPLE = PASTEL["lavender"]

PALETTE = list(PASTEL_20)

TEXT = "#222222"
GRID = "#E6E6E6"

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
        category="Style",
        interpretation="Growth leadership versus value leadership.",
    ),
    FactorPair(
        name="Momentum vs Min Vol",
        numerator="MTUM",
        denominator="USMV",
        category="Style",
        interpretation="Price momentum leadership versus minimum-volatility equities.",
    ),
    FactorPair(
        name="GARP vs Pure Growth",
        numerator="SPGP",
        denominator="IWF",
        category="Style",
        interpretation="Growth-at-a-reasonable-price leadership versus pure large-cap growth.",
    ),
    FactorPair(
        name="Small Value vs Small Growth",
        numerator="IWN",
        denominator="IWO",
        category="Style",
        interpretation="Small-cap value leadership versus small-cap growth.",
    ),
    FactorPair(
        name="High Beta vs Low Vol",
        numerator="SPHB",
        denominator="SPLV",
        category="Structure",
        interpretation="High-beta equities versus low-volatility defensives.",
    ),
    FactorPair(
        name="Small vs Large",
        numerator="IWM",
        denominator="SPY",
        category="Structure",
        interpretation="Small-cap participation versus S&P 500 leadership.",
    ),
    FactorPair(
        name="Equal Weight vs Cap Weight",
        numerator="RSP",
        denominator="SPY",
        category="Structure",
        interpretation="Average-stock participation versus cap-weighted concentration.",
    ),
    FactorPair(
        name="Microcap vs Market",
        numerator="IWC",
        denominator="SPY",
        category="Structure",
        interpretation="Microcap participation and liquidity reach versus the broad market.",
    ),
    FactorPair(
        name="Quality vs Market",
        numerator="QUAL",
        denominator="SPY",
        category="Fundamentals",
        interpretation="Profitable, higher-quality companies versus the broad market.",
    ),
    FactorPair(
        name="Free Cash Flow Yield",
        numerator="COWZ",
        denominator="SPY",
        category="Fundamentals",
        interpretation="High free-cash-flow-yield companies versus the broad market.",
    ),
    FactorPair(
        name="Buybacks vs Market",
        numerator="PKW",
        denominator="SPY",
        category="Fundamentals",
        interpretation="Companies returning capital through buybacks versus the broad market.",
    ),
    FactorPair(
        name="Dividend Quality vs High Yield",
        numerator="NOBL",
        denominator="SPYD",
        category="Fundamentals",
        interpretation="Durable dividend growth versus the highest-yielding S&P 500 equities.",
    ),
    FactorPair(
        name="Innovation vs Quality",
        numerator="ARKK",
        denominator="QUAL",
        category="Positioning & Liquidity",
        interpretation="Speculative long-duration innovation versus profitable quality.",
    ),
    FactorPair(
        name="Hedge Fund Crowding",
        numerator="GVIP",
        denominator="SPY",
        category="Positioning & Liquidity",
        interpretation="Concentrated hedge-fund favorite longs versus the broad market.",
    ),
    FactorPair(
        name="IPO Risk Appetite",
        numerator="IPO",
        denominator="SPY",
        category="Positioning & Liquidity",
        interpretation="Recently listed companies versus the broad market.",
    ),
    FactorPair(
        name="Biotech Risk Appetite",
        numerator="XBI",
        denominator="XLV",
        category="Positioning & Liquidity",
        interpretation="Early-stage and equal-weight biotech versus established healthcare.",
    ),
    FactorPair(
        name="Tech vs Broad Market",
        numerator="XLK",
        denominator="SPY",
        category="Macro & Industry",
        interpretation="Technology sector leadership versus the broad market.",
    ),
    FactorPair(
        name="Semis vs Tech",
        numerator="SMH",
        denominator="XLK",
        category="Macro & Industry",
        interpretation="Semiconductor leadership versus the broader technology sector.",
    ),
    FactorPair(
        name="Semis vs Software",
        numerator="SMH",
        denominator="IGV",
        category="Macro & Industry",
        interpretation="Semiconductor and AI hardware leadership versus software leadership.",
    ),
    FactorPair(
        name="Cyclicals vs Staples",
        numerator="XLY",
        denominator="XLP",
        category="Macro & Industry",
        interpretation="Consumer cyclicals versus staples defensiveness.",
    ),
    FactorPair(
        name="Industrials vs Materials",
        numerator="XLI",
        denominator="XLB",
        category="Macro & Industry",
        interpretation="Industrial capex and production leadership versus materials and input-cost exposure.",
    ),
    FactorPair(
        name="Regional Banks vs REITs",
        numerator="KRE",
        denominator="XLRE",
        category="Macro & Industry",
        interpretation="Regional-bank credit and curve sensitivity versus real-estate duration sensitivity.",
    ),
    FactorPair(
        name="US vs Ex-US",
        numerator="SPY",
        denominator="VXUS",
        category="Macro & Industry",
        interpretation="US equity leadership versus non-US equities.",
    ),
    FactorPair(
        name="EM vs DM Ex-US",
        numerator="EEM",
        denominator="VEA",
        category="Macro & Industry",
        interpretation="Emerging markets versus developed ex-US equities.",
    ),
    FactorPair(
        name="High Yield Credit vs Treasuries",
        numerator="HYG",
        denominator="IEF",
        category="Macro & Industry",
        interpretation="Credit risk appetite versus intermediate Treasuries.",
    ),
    FactorPair(
        name="Long Duration vs Bills",
        numerator="TLT",
        denominator="SHY",
        category="Macro & Industry",
        interpretation="Long-duration Treasury performance versus short bills.",
    ),
]

SYNTHETIC_FACTORS: Dict[str, Dict[str, object]] = {
    "Speculative Liquidity Composite": {
        "category": "Positioning & Liquidity",
        "components": [
            "High Beta vs Low Vol",
            "Microcap vs Market",
            "Innovation vs Quality",
            "IPO Risk Appetite",
            "Biotech Risk Appetite",
        ],
        "interpretation": "Equal-weight signal for how far liquidity is reaching into high-beta, microcap, IPO, innovation, and biotech risk.",
    },
    "Fundamental Quality Composite": {
        "category": "Fundamentals",
        "components": [
            "Quality vs Market",
            "Free Cash Flow Yield",
            "Buybacks vs Market",
            "Dividend Quality vs High Yield",
        ],
        "interpretation": "Equal-weight market confirmation from quality, cash-flow yield, buybacks, and dividend durability.",
    },
}

FACTOR_FAMILIES = [
    "Style",
    "Structure",
    "Fundamentals",
    "Positioning & Liquidity",
    "Macro & Industry",
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


def _format_percent(value: object, decimals: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if not np.isfinite(number):
        return "n/a"

    return f"{number:.{decimals}f}%"

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


def build_equal_weight_index(source: pd.DataFrame, components: List[str]) -> pd.Series:
    available = [name for name in components if name in source.columns]

    if len(available) < 2:
        return pd.Series(dtype=float)

    returns = source[available] / source[available].shift(1) - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan)
    min_components = max(2, int(np.ceil(len(available) * 0.6)))
    daily = returns.mean(axis=1, skipna=True).where(returns.count(axis=1) >= min_components)
    daily = daily.clip(lower=-0.5, upper=0.5)

    first_valid = daily.first_valid_index()

    if first_valid is None:
        return pd.Series(dtype=float)

    index = (1.0 + daily.loc[first_valid:].fillna(0.0)).cumprod() * 100.0
    index.name = "equal_weight_index"
    return index


def _centered_cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.rank(axis=1, pct=True, method="average")
    centered = 2.0 * (ranked - 0.5)
    return centered.where(frame.notna())


def build_historical_leadership(
    factor_levels: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a no-look-ahead cross-sectional score and percentile history."""
    levels = factor_levels.replace([np.inf, -np.inf], np.nan).ffill(limit=2)
    short_window = max(5, int(short_window))
    long_window = max(short_window + 1, int(long_window))

    r_short = levels / levels.shift(short_window) - 1.0
    r_long = levels / levels.shift(long_window) - 1.0
    expected_short = r_long * (short_window / long_window)
    acceleration = r_short - expected_short

    short_signal = _centered_cross_sectional_rank(r_short)
    long_signal = _centered_cross_sectional_rank(r_long)
    accel_signal = _centered_cross_sectional_rank(acceleration)

    e10 = levels.ewm(span=10, adjust=False, min_periods=10).mean()
    e20 = levels.ewm(span=20, adjust=False, min_periods=20).mean()
    e40 = levels.ewm(span=40, adjust=False, min_periods=40).mean()
    trend_signal = pd.DataFrame(0.0, index=levels.index, columns=levels.columns)
    trend_signal = trend_signal.mask((e10 > e20) & (e20 > e40), 1.0)
    trend_signal = trend_signal.mask((e10 < e20) & (e20 < e40), -1.0)

    inflection_signal = pd.DataFrame(0.0, index=levels.index, columns=levels.columns)
    inflection_signal = inflection_signal.mask((r_short > 0) & (r_long > 0), 1.0)
    inflection_signal = inflection_signal.mask((r_short > 0) & (r_long < 0), 0.65)
    inflection_signal = inflection_signal.mask((r_short < 0) & (r_long > 0), -0.65)
    inflection_signal = inflection_signal.mask((r_short < 0) & (r_long < 0), -1.0)

    daily_returns = levels / levels.shift(1) - 1.0
    rolling_slope = daily_returns.rolling(20, min_periods=10).mean()
    slope_mean = rolling_slope.rolling(252, min_periods=60).mean()
    slope_std = rolling_slope.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan)
    slope_signal = ((rolling_slope - slope_mean) / slope_std).clip(-3, 3) / 3.0

    raw = (
        0.24 * short_signal
        + 0.20 * long_signal
        + 0.18 * accel_signal
        + 0.18 * trend_signal
        + 0.10 * inflection_signal
        + 0.10 * slope_signal.fillna(0.0)
    )
    valid = r_short.notna() & r_long.notna()
    score = (50.0 + 35.0 * raw).clip(0.0, 100.0).where(valid)
    rank_percentile = score.rank(axis=1, pct=True, method="average") * 100.0
    return score, rank_percentile


def _current_true_streak(series: pd.Series) -> int:
    values = series.dropna().astype(bool).tolist()
    streak = 0

    for value in reversed(values):
        if not value:
            break
        streak += 1

    return streak


def build_leadership_stats(
    score_history: pd.DataFrame,
    rank_percentile: pd.DataFrame,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    scores = score_history.dropna(how="all")

    if scores.empty:
        return pd.DataFrame()

    rank_number = scores.rank(axis=1, ascending=False, method="min")
    current = rank_number.iloc[-1]
    prior_5 = rank_number.iloc[max(0, len(rank_number) - 6)]
    prior_20 = rank_number.iloc[max(0, len(rank_number) - 21)]
    displayed = rank_percentile[rank_percentile.index >= window_start]

    rows = []
    for factor in scores.columns:
        factor_ranks = displayed[factor].dropna() if factor in displayed else pd.Series(dtype=float)
        top_quartile = factor_ranks >= 75.0
        rows.append(
            {
                "Factor": factor,
                "Rank": current.get(factor, np.nan),
                "5D Rank Change": prior_5.get(factor, np.nan) - current.get(factor, np.nan),
                "20D Rank Change": prior_20.get(factor, np.nan) - current.get(factor, np.nan),
                "Top-Quartile Days": int(top_quartile.sum()) if not top_quartile.empty else 0,
                "Leadership Streak": _current_true_streak(top_quartile),
            }
        )

    return pd.DataFrame(rows).set_index("Factor").sort_values("Rank")


def build_structure_history(factor_levels: pd.DataFrame) -> pd.DataFrame:
    levels = factor_levels.replace([np.inf, -np.inf], np.nan).ffill(limit=2)
    momentum_20 = levels / levels.shift(20) - 1.0
    dispersion = (momentum_20.quantile(0.80, axis=1) - momentum_20.quantile(0.20, axis=1)) * 100.0

    positive = momentum_20.clip(lower=0.0)
    positive_total = positive.sum(axis=1).replace(0, np.nan)
    top_three = positive.apply(lambda row: row.nlargest(min(3, row.notna().sum())).sum(), axis=1)
    concentration = top_three / positive_total * 100.0

    daily_returns = levels / levels.shift(1) - 1.0
    correlation = pd.Series(index=levels.index, dtype=float)

    for end_pos in range(59, len(daily_returns), 5):
        sample = daily_returns.iloc[end_pos - 59 : end_pos + 1].dropna(axis=1, thresh=40)

        if sample.shape[1] < 2:
            continue

        matrix = sample.corr().to_numpy(dtype=float)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        upper = upper[np.isfinite(upper)]

        if upper.size:
            correlation.iloc[end_pos] = float(upper.mean() * 100.0)

    correlation = correlation.ffill(limit=5)
    return pd.DataFrame(
        {
            "Factor Dispersion": dispersion,
            "Average Correlation": correlation,
            "Top-3 Leadership Share": concentration,
        }
    )

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


def plot_metric_timeseries(
    metric_df: pd.DataFrame,
    selected_factors: List[str],
    metric_name: str,
    show_ema: bool,
) -> Optional[plt.Figure]:
    selected = [factor for factor in selected_factors if factor in metric_df.columns]

    if not selected:
        return None

    chart = metric_df[selected].copy().dropna(how="all")

    if chart.empty:
        return None

    if metric_name == "Relative Performance":
        for factor in chart.columns:
            chart[factor] = normalized_series(chart[factor])

    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    color_map = {factor: PALETTE[i % len(PALETTE)] for i, factor in enumerate(selected)}

    for factor in selected:
        series = chart[factor].dropna()

        if series.empty:
            continue

        color = color_map[factor]
        ax.plot(series.index, series.values, color=color, linewidth=2.0, alpha=0.95)

        if show_ema and metric_name == "Relative Performance" and len(series) >= 20:
            smoothed = series.ewm(span=20, adjust=False).mean()
            ax.plot(smoothed.index, smoothed.values, color=color, linewidth=0.9, alpha=0.35)

    if metric_name == "Composite Score":
        ax.axhline(50, color="#777777", linewidth=1.0, alpha=0.8)
        ax.axhline(57, color=PASTEL_GREEN, linewidth=0.8, alpha=0.55)
        ax.axhline(43, color=PASTEL_RED, linewidth=0.8, alpha=0.55)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Historical leadership score")
    elif metric_name == "Cross-Sectional Rank":
        ax.axhspan(75, 100, color=PASTEL_GREEN, alpha=0.08)
        ax.axhspan(0, 25, color=PASTEL_RED, alpha=0.08)
        ax.axhline(50, color="#777777", linewidth=1.0, alpha=0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Cross-sectional percentile")
    else:
        ax.axhline(100, color="#777777", linewidth=1.0, alpha=0.6)
        ax.set_ylabel("Rebased ratio level")

    span = chart.index.max() - chart.index.min()
    label_extension = max(pd.Timedelta(days=5), span * 0.14)
    label_x = chart.index.max() + label_extension * 0.08
    ax.set_xlim(chart.index.min(), chart.index.max() + label_extension)

    latest_values = {
        factor: chart[factor].dropna().iloc[-1]
        for factor in selected
        if not chart[factor].dropna().empty
    }
    y_min, y_max = ax.get_ylim()
    min_gap = max((y_max - y_min) * 0.035, 0.01)
    ordered = sorted(latest_values, key=latest_values.get)
    adjusted_values: Dict[str, float] = {}
    cursor = y_min

    for factor in ordered:
        adjusted = max(float(latest_values[factor]), cursor)
        adjusted_values[factor] = adjusted
        cursor = adjusted + min_gap

    if ordered and adjusted_values[ordered[-1]] > y_max:
        shift = adjusted_values[ordered[-1]] - y_max
        adjusted_values = {factor: value - shift for factor, value in adjusted_values.items()}

    if ordered and adjusted_values[ordered[0]] < y_min:
        shift = y_min - adjusted_values[ordered[0]]
        adjusted_values = {factor: value + shift for factor, value in adjusted_values.items()}

    for factor in selected:
        series = chart[factor].dropna()

        if series.empty:
            continue

        ax.scatter(series.index[-1], series.iloc[-1], s=18, color=color_map[factor], zorder=4)
        ax.annotate(
            factor,
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(label_x, adjusted_values.get(factor, series.iloc[-1])),
            textcoords="data",
            fontsize=8.2,
            fontweight=600,
            color=color_map[factor],
            va="center",
            arrowprops={"arrowstyle": "-", "color": color_map[factor], "lw": 0.7, "alpha": 0.7},
            clip_on=False,
        )

    if len(chart.index) > 1:
        span_days = (chart.index[-1] - chart.index[0]).days
    else:
        span_days = 0

    if span_days <= 120:
        locator = mdates.WeekdayLocator(interval=2)
        formatter = mdates.DateFormatter("%b %d")
    elif span_days <= 540:
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter("%b")
    else:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"{metric_name} Through Time", color=TEXT, pad=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def plot_leadership_history(
    rank_percentile: pd.DataFrame,
    factor_order: List[str],
    window_start: pd.Timestamp,
) -> Optional[plt.Figure]:
    available = [factor for factor in factor_order if factor in rank_percentile.columns]

    if not available:
        return None

    displayed = rank_percentile[rank_percentile.index >= window_start][available]
    weekly = displayed.resample("W-FRI").last().dropna(how="all")

    if weekly.empty:
        return None

    matrix = weekly.T
    height = max(6.0, 0.34 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(15.5, height))
    cmap = LinearSegmentedColormap.from_list(
        "adfm_leadership",
        [PASTEL_RED, "#F2F2F2", PASTEL_GREEN],
    )
    image = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=100)

    tick_count = min(12, len(matrix.columns))
    tick_positions = np.unique(np.linspace(0, len(matrix.columns) - 1, tick_count, dtype=int))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [pd.Timestamp(matrix.columns[i]).strftime("%b %d") for i in tick_positions],
        rotation=0,
        fontsize=8.5,
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8.5)
    ax.set_title("Weekly Cross-Sectional Leadership Percentile", color=TEXT, pad=10)
    ax.set_xlabel("Green marks persistent leadership; red marks persistent lagging pressure.")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Leadership percentile", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def plot_structure_diagnostics(
    structure_history: pd.DataFrame,
    window_start: pd.Timestamp,
) -> Optional[plt.Figure]:
    displayed = structure_history[structure_history.index >= window_start].dropna(how="all")

    if displayed.empty:
        return None

    specs = [
        ("Factor Dispersion", "20D top-minus-bottom quintile spread", PASTEL_PURPLE),
        ("Average Correlation", "Rolling 60D average factor correlation", PASTEL_BLUE),
        ("Top-3 Leadership Share", "Share of positive 20D momentum from top three", PASTEL_ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.6), squeeze=False)

    for ax, (column, title, color) in zip(axes.ravel(), specs):
        series = displayed[column].dropna()

        if series.empty:
            ax.axis("off")
            continue

        ax.plot(series.index, series.values, color=color, linewidth=2.0)
        ax.fill_between(series.index, series.values, color=color, alpha=0.10)
        latest = series.iloc[-1]
        ax.scatter(series.index[-1], latest, s=22, color=color, zorder=3)
        ax.set_title(title, fontsize=9.3, color=TEXT, pad=8)
        ax.text(
            0.98,
            0.93,
            f"{latest:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            fontweight=700,
            color=color,
        )
        ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="both", labelsize=8)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig

# =========================================================
# Sidebar
# =========================================================
render_page_header(
    PageHeader(
        title=TITLE,
        description=SUBTITLE,
        eyebrow="ADFM Equity Leadership",
    )
)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Monitor factor leadership using ETF-relative price ratios, absolute return percentiles, trend structure, and rotation pressure.

        **How to read it**
        - The current composite identifies leaders, laggards, and inflections.
        - The history heatmap separates persistent leadership from fresh rotation.
        - Dispersion shows whether factor selection is becoming more consequential.
        - Correlation shows whether factors are differentiating or trading as one market.
        - Leadership concentration shows whether a small number of factors dominate the tape.

        **Data source:** Yahoo Finance via `yfinance`. Signals are ETF price-based. The two synthetic composites equal-weight their underlying relative-price signals and do not claim holdings-level fundamental or estimate data.
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

    selected_families = st.multiselect(
        "Factor families",
        options=FACTOR_FAMILIES,
        default=FACTOR_FAMILIES,
        help="Controls the time-series, leadership history, and structure diagnostics.",
    )

    lookback_short = st.slider("Short momentum window, trading days", 10, 60, 20)

    long_min = max(30, lookback_short + 5)
    lookback_long = st.slider("Long momentum window, trading days", long_min, 180, max(60, long_min))

    show_ema = st.checkbox("Show 20-day EMA on relative-performance chart", value=False)

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

base_factor_df = pd.DataFrame(factor_levels_full).dropna(how="all")

for synthetic_name, config in SYNTHETIC_FACTORS.items():
    components = [str(name) for name in config.get("components", [])]
    synthetic = build_equal_weight_index(base_factor_df, components)

    if synthetic.empty:
        pair_diagnostics.append(
            {
                "Factor": synthetic_name,
                "Category": config.get("category", "Synthetic"),
                "Expression": "Equal-weight composite",
                "Status": "Skipped",
                "Observations": 0,
                "Reason": "Fewer than two component series were available.",
            }
        )
        continue

    synthetic.name = synthetic_name
    factor_levels_full[synthetic_name] = synthetic
    factor_meta[synthetic_name] = FactorPair(
        name=synthetic_name,
        numerator="Equal-weight component signals",
        denominator=None,
        category=str(config.get("category", "Synthetic")),
        interpretation=str(config.get("interpretation", "Equal-weight synthetic factor.")),
    )
    pair_diagnostics.append(
        {
            "Factor": synthetic_name,
            "Category": config.get("category", "Synthetic"),
            "Expression": "EW(" + ", ".join(components) + ")",
            "Status": "OK",
            "Observations": len(synthetic),
            "Reason": config.get("interpretation", "Equal-weight synthetic factor."),
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

if not selected_families:
    st.warning("Select at least one factor family in the sidebar.")
    st.stop()

all_scored_factors = [factor for factor in mom_df.index if factor in factor_df_full.columns]
score_history, rank_history = build_historical_leadership(
    factor_df_full[all_scored_factors],
    short_window=lookback_short,
    long_window=lookback_long,
)

score_dates = score_history.dropna(how="all")
if not score_dates.empty:
    latest_score_date = score_dates.index[-1]
    current_common = [factor for factor in mom_df.index if factor in score_history.columns]
    score_history.loc[latest_score_date, current_common] = mom_df.loc[current_common, "Composite"]
    rank_history.loc[latest_score_date, current_common] = (
        score_history.loc[latest_score_date, current_common].rank(pct=True, method="average") * 100.0
    )

active_factor_names = [
    factor
    for factor in mom_df.index
    if mom_df.loc[factor, "Category"] in selected_families and factor in factor_df_full.columns
]

if not active_factor_names:
    st.warning("No scored factors are available for the selected families and analysis window.")
    st.stop()

active_mom_df = mom_df.loc[active_factor_names].copy()
active_score_history = score_history[[c for c in active_factor_names if c in score_history.columns]]
active_rank_history = rank_history[[c for c in active_factor_names if c in rank_history.columns]]
leadership_stats = build_leadership_stats(active_score_history, active_rank_history, window_start)
structure_history = build_structure_history(factor_df_full[active_factor_names])

# =========================================================
# Main layout
# =========================================================
st.caption(
    f"As of {as_of_date}. Window starts {window_start.strftime('%Y-%m-%d')}. "
    f"Fetch mode: {fetch_meta.get('mode', 'unknown')}. Last fetch timestamp: {fetch_meta.get('fetched_at', 'unknown')}."
)

leader = active_mom_df.index[0]
laggard = active_mom_df.index[-1]
acceleration_values = active_mom_df["Accel"].dropna()
accelerating = acceleration_values.idxmax() if not acceleration_values.empty else leader
breadth = float((active_mom_df["Composite"] > 50.0).mean() * 100.0)
latest_structure = structure_history.dropna(how="all").iloc[-1] if not structure_history.dropna(how="all").empty else pd.Series(dtype=float)

render_kpi_cards(
    [
        ("Current Leader", leader, f"Score {active_mom_df.loc[leader, 'Composite']:.1f}"),
        ("Current Laggard", laggard, f"Score {active_mom_df.loc[laggard, 'Composite']:.1f}"),
        (
            "Fastest Acceleration",
            accelerating,
            f"{active_mom_df.loc[accelerating, 'Accel'] * 100.0:+.1f}%"
            if pd.notna(active_mom_df.loc[accelerating, "Accel"])
            else "n/a",
        ),
        ("Positive Breadth", f"{breadth:.0f}%", "Share of selected factors above 50"),
        (
            "Factor Dispersion",
            _format_percent(latest_structure.get("Factor Dispersion", np.nan)),
            "20D top-minus-bottom quintile",
        ),
        (
            "Average Correlation",
            _format_percent(latest_structure.get("Average Correlation", np.nan)),
            "Rolling 60D factor correlation",
        ),
    ]
)

st.subheader(f"Factor Time Series ({window_choice})")
control_a, control_b = st.columns([1, 2.4])

with control_a:
    metric_choice = st.selectbox(
        "Metric",
        ["Relative Performance", "Composite Score", "Cross-Sectional Rank"],
        index=0,
    )

default_factors = list(active_mom_df.head(min(5, len(active_mom_df))).index)
default_factors += [
    factor for factor in active_mom_df.tail(min(5, len(active_mom_df))).index if factor not in default_factors
]

with control_b:
    selected_chart_factors = st.multiselect(
        "Displayed factors",
        options=active_factor_names,
        default=default_factors,
        help="Defaults to the five strongest and five weakest selected factors.",
    )

if metric_choice == "Relative Performance":
    chart_metric = factor_df_full[factor_df_full.index >= window_start]
elif metric_choice == "Composite Score":
    chart_metric = active_score_history[active_score_history.index >= window_start]
else:
    chart_metric = active_rank_history[active_rank_history.index >= window_start]

fig_ts = plot_metric_timeseries(
    metric_df=chart_metric,
    selected_factors=selected_chart_factors,
    metric_name=metric_choice,
    show_ema=show_ema,
)

if fig_ts is not None:
    st.pyplot(fig_ts, clear_figure=True)
    plt.close(fig_ts)
else:
    st.info("Select at least one factor with usable history.")

st.subheader("Leadership History")
st.caption(
    "Weekly percentile history shows persistence and rotation. Current rank changes are positive when a factor moved up the leadership table."
)

factor_order = list(active_mom_df.index)
fig_history = plot_leadership_history(active_rank_history, factor_order, window_start)

if fig_history is not None:
    st.pyplot(fig_history, clear_figure=True)
    plt.close(fig_history)

if not leadership_stats.empty:
    transition_table = leadership_stats.reset_index()
    transition_table["Category"] = transition_table["Factor"].map(active_mom_df["Category"])
    transition_table = transition_table[
        [
            "Factor",
            "Category",
            "Rank",
            "5D Rank Change",
            "20D Rank Change",
            "Top-Quartile Days",
            "Leadership Streak",
        ]
    ]
    for column in [
        "Rank",
        "5D Rank Change",
        "20D Rank Change",
        "Top-Quartile Days",
        "Leadership Streak",
    ]:
        transition_table[column] = pd.to_numeric(transition_table[column], errors="coerce").round(0).astype("Int64")
    st.dataframe(
        transition_table,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Factor Market Structure")
st.caption(
    "Dispersion measures the opportunity set, correlation measures differentiation, and top-three share measures leadership concentration."
)
fig_structure = plot_structure_diagnostics(structure_history, window_start)

if fig_structure is not None:
    st.pyplot(fig_structure, clear_figure=True)
    plt.close(fig_structure)

with st.expander("Current factor detail", expanded=False):
    detail = active_mom_df.copy()
    detail = detail.join(leadership_stats, how="left")
    for col in ["%5D", "Short", "Long", "Accel", "Trend Strength"]:
        detail[col] = detail[col] * 100.0
    detail = detail.reset_index()[
        [
            "Factor",
            "State",
            "Category",
            "Expression",
            "Composite",
            "%5D",
            "Short",
            "Long",
            "Accel",
            "Trend",
            "Inflection",
            "Slope Z",
            "Rank",
            "5D Rank Change",
            "20D Rank Change",
            "Interpretation",
        ]
    ]
    detail = detail.round(
        {
            "Composite": 1,
            "%5D": 1,
            "Short": 1,
            "Long": 1,
            "Accel": 1,
            "Slope Z": 2,
            "Rank": 0,
            "5D Rank Change": 0,
            "20D Rank Change": 0,
        }
    )
    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
    )

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

render_footer()
