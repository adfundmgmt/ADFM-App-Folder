import html
import os
import re
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ---------------- Config ----------------

TITLE = "Fed Balance Sheet & Liquidity Tracker"

H41_DDP_OUTPUT_URL = "https://www.federalreserve.gov/datadownload/Output.aspx"
H41_CURRENT_URL = "https://www.federalreserve.gov/releases/h41/current/h41.htm"

CACHE_DIR = "data"
CACHE_PATH = os.path.join(CACHE_DIR, "fed_liquidity_cache.csv")

REQUEST_TIMEOUT = (4, 14)
MAX_RETRIES = 2
BACKOFF_SECONDS = 0.8

DEFAULT_LOOKBACK = "3y"
DEFAULT_SMOOTH = 1

# These package hashes come from the Fed H.4.1 DDP preformatted/review pages.
# Package 1 captures the condition statement, including total assets.
# Package 2 captures reserve-balance factors, including TGA, RRP, and reserves.
DDP_PACKAGES = {
    "condition_statement": "3ab1b33ad80c27bc5cc4f8122b7a6440",
    "reserve_factors": "2704b6bc9b50bc034baf9660364dfb26",
}

# H.4.1 DDP series identifiers. Units are millions of dollars.
SERIES_PATTERNS = {
    "Fed Assets": [
        "RESPPMA_N.WW",
        "total assets (less eliminations from consolidation)",
        "assets: total assets: total assets (less eliminations",
    ],
    "Fed Assets Gross": [
        "RESPPA_N.WW",
        "assets: total assets: total assets: wednesday level",
    ],
    "Securities Held Outright": [
        "RESPPALG_N.WW",
        "securities held outright: securities held outright: wednesday level",
    ],
    "UST Holdings": [
        "RESPPALGUM_N.WW",
        "u.s. treasury securities: all: wednesday level",
        "u.s. treasury securities: wednesday level",
    ],
    "MBS Holdings": [
        "RESPPALGASMO_N.WW",
        "mortgage-backed securities: wednesday level",
    ],
    "TGA": [
        "RESPPLLDT_N.WW",
        "u.s. treasury, general account: wednesday level",
    ],
    "RRP Total": [
        "RESPPLLR_N.WW",
        "reverse repurchase agreements: wednesday level",
        "liabilities: reverse repurchase agreements: wednesday level",
    ],
    "RRP Foreign Official": [
        "RESPPLLRF_N.WW",
        "reverse repurchase agreements: foreign official and international accounts: wednesday level",
    ],
    "RRP Others": [
        "RESPPLLRD_N.WW",
        "reverse repurchase agreements: others: wednesday level",
    ],
    "Reserve Balances": [
        "RESH4R_N.WW",
        "reserve balances with federal reserve banks: wednesday level",
    ],
}

DISPLAY_COLUMNS = [
    "Fed Assets",
    "TGA",
    "RRP Others",
    "RRP Total",
    "Reserve Balances",
    "Securities Held Outright",
    "UST Holdings",
    "MBS Holdings",
    "Net Liquidity",
]

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

# ---------------- CSS ----------------

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1580px;
    }

    .adfm-title {
        font-size: 1.9rem;
        line-height: 1.15;
        font-weight: 780;
        color: #111827;
        margin-bottom: 0.15rem;
        letter-spacing: -0.025em;
    }

    .adfm-subtitle {
        font-size: 0.96rem;
        color: #6b7280;
        margin-bottom: 1.05rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 760;
        color: #111827;
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .section-subtitle {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 13px 15px 11px 15px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.035);
        min-height: 94px;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.45rem;
        white-space: nowrap;
    }

    .metric-value {
        font-size: 1.36rem;
        font-weight: 760;
        color: #111827;
        line-height: 1.08;
        white-space: nowrap;
    }

    .metric-footnote {
        font-size: 0.74rem;
        color: #9ca3af;
        margin-top: 0.43rem;
        line-height: 1.25;
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.85rem;
        color: #111827;
        font-size: 0.91rem;
        line-height: 1.45;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 0.9rem;
        color: #7c2d12;
        font-size: 0.9rem;
        line-height: 1.42;
    }

    .good-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 0.9rem;
        color: #111827;
        font-size: 0.9rem;
        line-height: 1.42;
    }

    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 650;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Helpers ----------------

def norm_text(x: object) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def parse_number(x: object) -> float:
    if x is None:
        return np.nan

    s = html.unescape(str(x))
    s = s.replace("\xa0", " ")
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("−", "-")
    s = re.sub(r"\s+", "", s)

    if s in {"", ".", "...", "nan", "None"}:
        return np.nan

    paren_negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")

    try:
        value = float(s)
        return -value if paren_negative else value
    except Exception:
        return np.nan


def fmt_b(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"${x:,.0f}B"


def fmt_delta_b(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}${x:,.0f}B"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}%"


def fmt_plain(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.0f}"


def metric_card(label: str, value: str, footnote: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fetch_text(url: str, params: Optional[Dict[str, str]] = None) -> str:
    headers = {
        "User-Agent": "ADFM-Fed-Liquidity-Tracker/2.0",
        "Accept": "text/csv,text/html,application/xhtml+xml,*/*;q=0.8",
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            text = r.text

            if not text or len(text.strip()) < 50:
                raise ValueError("Empty response")

            return text

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Fetch failed after {MAX_RETRIES} attempts: {last_error}")


def clean_ddp_csv_text(text: str) -> str:
    if "<html" in text[:500].lower() or "application is temporarily unavailable" in text.lower():
        raise ValueError("Fed DDP returned HTML instead of CSV")

    return text.replace("\ufeff", "")


def read_ddp_csv_flexible(text: str) -> pd.DataFrame:
    text = clean_ddp_csv_text(text)
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        raise ValueError("Fed DDP CSV response was empty")

    best: Optional[pd.DataFrame] = None
    best_score = -1

    for skip in range(0, min(35, len(lines))):
        candidate_text = "\n".join(lines[skip:])

        try:
            candidate = pd.read_csv(StringIO(candidate_text), dtype=str)
        except Exception:
            continue

        if candidate.empty or candidate.shape[1] < 2:
            continue

        candidate.columns = [str(c).strip() for c in candidate.columns]

        for col in candidate.columns[:4]:
            dates = pd.to_datetime(candidate[col], errors="coerce")
            score = int(dates.notna().sum())

            if score > best_score:
                tmp = candidate.copy()
                tmp["_parsed_date_"] = dates
                tmp = tmp.dropna(subset=["_parsed_date_"])
                tmp = tmp.drop(columns=[col])
                tmp = tmp.set_index("_parsed_date_").sort_index()
                tmp.index.name = "Date"

                if not tmp.empty and tmp.shape[1] >= 1:
                    best = tmp
                    best_score = score

    if best is None or best.empty:
        raise ValueError("Could not locate a usable date column in Fed DDP CSV")

    best = best.loc[:, ~best.columns.duplicated()]
    return best


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_ddp_package(package_hash: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_iso)
    end_dt = pd.to_datetime(end_iso)

    params = {
        "filetype": "csv",
        "from": start_dt.strftime("%m/%d/%Y"),
        "to": end_dt.strftime("%m/%d/%Y"),
        "label": "include",
        "lastobs": "",
        "layout": "seriescolumn",
        "rel": "H41",
        "series": package_hash,
    }

    text = fetch_text(H41_DDP_OUTPUT_URL, params=params)
    return read_ddp_csv_flexible(text)


def locate_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    if df.empty:
        return None

    normalized_cols = {col: norm_text(col) for col in df.columns}
    normalized_patterns = [norm_text(p) for p in patterns]

    for pattern in normalized_patterns:
        for col, ncol in normalized_cols.items():
            if pattern in ncol:
                return col

    return None


def build_working_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)

    for display_name, patterns in SERIES_PATTERNS.items():
        col = locate_column(raw, patterns)
        if col is not None:
            out[display_name] = raw[col].map(parse_number) / 1000.0

    # Prefer less-eliminations total assets. Fall back to gross total assets if needed.
    if "Fed Assets" not in out.columns and "Fed Assets Gross" in out.columns:
        out["Fed Assets"] = out["Fed Assets Gross"]

    # For the net-liquidity proxy, use RRP Others when available.
    # RRP Total includes foreign official and international accounts, which can distort the market-liquidity read.
    if "Fed Assets" in out.columns and "TGA" in out.columns:
        if "RRP Others" in out.columns:
            out["Net Liquidity"] = out["Fed Assets"] - out["TGA"] - out["RRP Others"]
        elif "RRP Total" in out.columns:
            out["Net Liquidity"] = out["Fed Assets"] - out["TGA"] - out["RRP Total"]

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Drop rows where the core series is missing.
    if "Fed Assets" in out.columns:
        out = out.dropna(subset=["Fed Assets"])

    return out


def load_ddp_history(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}

    start_iso = start.strftime("%Y-%m-%d")
    end_iso = end.strftime("%Y-%m-%d")

    for package_name, package_hash in DDP_PACKAGES.items():
        try:
            package_df = fetch_ddp_package(package_hash, start_iso, end_iso)
            frames.append(package_df)
        except Exception as exc:
            errors[package_name] = str(exc)

    if not frames:
        return pd.DataFrame(), errors

    raw = pd.concat(frames, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    working = build_working_df(raw)
    return working, errors


def load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame()

    try:
        cached = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
        cached = cached.set_index("Date").sort_index()
        cached = cached[~cached.index.duplicated(keep="last")]
        return cached
    except Exception:
        return pd.DataFrame()


def save_cache(df: pd.DataFrame) -> None:
    if df.empty:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    save = df.copy()
    save.index.name = "Date"
    save = save.sort_index()
    save = save[~save.index.duplicated(keep="last")]
    save.to_csv(CACHE_PATH)


def merge_live_and_cache(live: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
    if cached.empty and live.empty:
        return pd.DataFrame()

    if cached.empty:
        return live.copy()

    if live.empty:
        return cached.copy()

    all_cols = sorted(set(cached.columns).union(set(live.columns)))
    cached2 = cached.reindex(columns=all_cols)
    live2 = live.reindex(columns=all_cols)

    merged = pd.concat([cached2, live2], axis=0)
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    return merged


def html_to_lines(raw_html: str) -> List[str]:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|td|th|li|h1|h2|h3|h4)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if clean:
            lines.append(clean)

    return lines


def numeric_tokens_after(lines: List[str], label: str, start: int = 0, max_scan: int = 40) -> List[float]:
    target = norm_text(label)

    for i in range(start, len(lines)):
        if norm_text(lines[i]) == target or target in norm_text(lines[i]):
            nums: List[float] = []
            for j in range(i + 1, min(i + 1 + max_scan, len(lines))):
                val = parse_number(lines[j])
                if pd.notna(val):
                    nums.append(val)
                if len(nums) >= 6:
                    break
            return nums

    return []


def current_h41_snapshot() -> pd.DataFrame:
    """
    Fallback parser for the current H.4.1 HTML page.
    This gives the latest observation only. It is used to keep the app alive
    even when the DDP service is temporarily unavailable.
    """
    try:
        raw_html = fetch_text(H41_CURRENT_URL)
    except Exception:
        return pd.DataFrame()

    lines = html_to_lines(raw_html)

    release_date = None
    joined = "\n".join(lines)
    release_match = re.search(r"Release Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", joined)
    if release_match:
        release_date = pd.to_datetime(release_match.group(1), errors="coerce")

    obs_date = None
    for i, line in enumerate(lines):
        if norm_text(line) == "wednesday" and i + 1 < len(lines):
            dt = pd.to_datetime(lines[i + 1], errors="coerce")
            if pd.notna(dt):
                obs_date = dt
                break

    if obs_date is None or pd.isna(obs_date):
        obs_date = release_date

    if obs_date is None or pd.isna(obs_date):
        return pd.DataFrame()

    row: Dict[str, float] = {}

    # Table 5 total assets has an eliminations line before the asset figure.
    total_assets_nums = numeric_tokens_after(lines, "Total assets", start=0, max_scan=20)
    if total_assets_nums:
        big_nums = [x for x in total_assets_nums if abs(x) > 1000000]
        if big_nums:
            row["Fed Assets"] = big_nums[0] / 1000.0

    # Table 1 reserve-factor lines.
    tga_nums = numeric_tokens_after(lines, "U.S. Treasury, General Account", start=0, max_scan=20)
    if len(tga_nums) >= 4:
        row["TGA"] = tga_nums[3] / 1000.0
    elif tga_nums:
        row["TGA"] = tga_nums[0] / 1000.0

    reserves_nums = numeric_tokens_after(lines, "Reserve balances with Federal Reserve Banks", start=0, max_scan=20)
    if len(reserves_nums) >= 4:
        row["Reserve Balances"] = reserves_nums[3] / 1000.0
    elif reserves_nums:
        row["Reserve Balances"] = reserves_nums[0] / 1000.0

    rrp_total_nums = numeric_tokens_after(lines, "Reverse repurchase agreements", start=0, max_scan=20)
    if len(rrp_total_nums) >= 4:
        row["RRP Total"] = rrp_total_nums[3] / 1000.0
    elif rrp_total_nums:
        row["RRP Total"] = rrp_total_nums[0] / 1000.0

    # Get "Others" after the first reverse-repo line, which is closer to ON RRP than total RRP.
    rrp_start = 0
    for i, line in enumerate(lines):
        if "reverse repurchase agreements" in norm_text(line):
            rrp_start = i
            break

    rrp_others_nums = numeric_tokens_after(lines, "Others", start=rrp_start, max_scan=20)
    if len(rrp_others_nums) >= 4:
        row["RRP Others"] = rrp_others_nums[3] / 1000.0
    elif rrp_others_nums:
        row["RRP Others"] = rrp_others_nums[0] / 1000.0

    if "Fed Assets" in row and "TGA" in row:
        if "RRP Others" in row:
            row["Net Liquidity"] = row["Fed Assets"] - row["TGA"] - row["RRP Others"]
        elif "RRP Total" in row:
            row["Net Liquidity"] = row["Fed Assets"] - row["TGA"] - row["RRP Total"]

    if not row or "Fed Assets" not in row:
        return pd.DataFrame()

    out = pd.DataFrame([row], index=[pd.to_datetime(obs_date).normalize()])
    out.index.name = "Date"
    return out


def change_obs(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def pct_change_obs(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan

    base = s.iloc[-1 - periods]
    if base == 0 or pd.isna(base):
        return np.nan

    return float((s.iloc[-1] / base - 1.0) * 100.0)


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()


def rebase(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(index=s.index, dtype="float64")

    base = valid.iloc[0]
    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, dtype="float64")

    return (s / base) * 100.0


def annualized_pace(series: pd.Series, periods: int = 13) -> float:
    move = change_obs(series, periods)
    if pd.isna(move):
        return np.nan
    return move / periods * 52.0


def liquidity_regime(asset_13w_ann: float, net_13w_ann: float) -> Tuple[str, str]:
    primary = net_13w_ann if pd.notna(net_13w_ann) else asset_13w_ann

    if pd.isna(primary):
        return "Insufficient history", "#6b7280"

    if primary > 250:
        return "Strong liquidity expansion", "#166534"
    if primary > 75:
        return "Moderate liquidity expansion", "#15803d"
    if primary < -250:
        return "Strong liquidity drain", "#991b1b"
    if primary < -75:
        return "Moderate liquidity drain", "#b91c1c"

    return "Flat liquidity impulse", "#92400e"


def filter_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df.empty:
        return df

    latest_date = df.index.max()

    if lookback == "6m":
        start = latest_date - pd.DateOffset(months=6)
    elif lookback == "1y":
        start = latest_date - pd.DateOffset(years=1)
    elif lookback == "2y":
        start = latest_date - pd.DateOffset(years=2)
    elif lookback == "3y":
        start = latest_date - pd.DateOffset(years=3)
    elif lookback == "5y":
        start = latest_date - pd.DateOffset(years=5)
    elif lookback == "10y":
        start = latest_date - pd.DateOffset(years=10)
    else:
        start = df.index.min()

    return df[df.index >= start].copy()


def infer_source_status(live: pd.DataFrame, cached: pd.DataFrame, current: pd.DataFrame) -> str:
    parts = []

    if not live.empty:
        parts.append("Fed DDP live")
    if not current.empty:
        parts.append("Current H.4.1 fallback")
    if not cached.empty:
        parts.append("local cache")

    if not parts:
        return "No data source available"

    return " + ".join(parts)


# ---------------- Header ----------------

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Fed H.4.1 balance-sheet level, liquidity impulse, TGA, reverse repos, reserves, and rate of change. FRED is not used.</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        """
This page avoids FRED entirely.

Primary source is the Federal Reserve H.4.1 Data Download Program. If the DDP endpoint is unavailable, the app falls back to the current H.4.1 release and the local CSV cache.

Core proxy:

**Net Liquidity = Fed Assets - TGA - RRP Others**

RRP Others is used because total reverse repo includes foreign official and international accounts.
        """
    )

    st.divider()
    st.markdown("### Controls")

    lookback = st.selectbox(
        "Lookback",
        ["6m", "1y", "2y", "3y", "5y", "10y", "max"],
        index=["6m", "1y", "2y", "3y", "5y", "10y", "max"].index(DEFAULT_LOOKBACK),
    )

    smooth_window = st.number_input(
        "Smoothing window, weekly observations",
        min_value=1,
        max_value=12,
        value=DEFAULT_SMOOTH,
        step=1,
    )

    show_current_fallback = st.checkbox(
        "Use current H.4.1 page as fallback",
        value=True,
    )

    show_components = st.checkbox(
        "Show components chart",
        value=True,
    )

    show_raw_table = st.checkbox(
        "Show raw data table",
        value=True,
    )

    st.caption("Source: Federal Reserve H.4.1 DDP and current H.4.1 release. No FRED dependency.")

# ---------------- Data Load ----------------

today = pd.Timestamp.today().normalize()
start_for_fetch = today - pd.DateOffset(years=12)

with st.spinner("Loading Federal Reserve H.4.1 data"):
    cached_df = load_cache()

    live_df, live_errors = load_ddp_history(start_for_fetch, today)

    current_df = pd.DataFrame()
    if show_current_fallback:
        current_df = current_h41_snapshot()

    combined_live = merge_live_and_cache(live_df, current_df)
    full_df = merge_live_and_cache(combined_live, cached_df)

    if not combined_live.empty:
        save_cache(full_df)

if full_df.empty or "Fed Assets" not in full_df.columns:
    st.error("No usable Fed balance-sheet data loaded.")
    st.markdown(
        """
        <div class="warning-box">
        The Fed DDP endpoint did not return usable data, the current H.4.1 fallback did not parse,
        and there is no local cache available. Once the app successfully loads one time, it will keep a local cache
        and avoid failing on temporary source outages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if live_errors:
        with st.expander("Source error details", expanded=True):
            for k, v in live_errors.items():
                st.write(f"**{k}:** {v}")

    st.stop()

full_df = full_df.sort_index()
full_df = full_df[~full_df.index.duplicated(keep="last")]

# Keep only columns we care about, preserving any columns that exist.
available_display_cols = [c for c in DISPLAY_COLUMNS if c in full_df.columns]
full_df = full_df[available_display_cols].copy()

# Filter lookback for display only.
df = filter_lookback(full_df, lookback)

if df.empty:
    st.error("Lookback filter removed all data.")
    st.stop()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

latest = df.dropna(subset=["Fed Assets"]).iloc[-1]
latest_date = df.dropna(subset=["Fed Assets"]).index[-1]

source_status = infer_source_status(live_df, cached_df, current_df)

# ---------------- Derived Metrics ----------------

assets_1w = change_obs(df["Fed Assets"], 1)
assets_4w = change_obs(df["Fed Assets"], 4)
assets_13w = change_obs(df["Fed Assets"], 13)
assets_52w = change_obs(df["Fed Assets"], 52)
assets_13w_ann = annualized_pace(df["Fed Assets"], 13)
assets_13w_pct = pct_change_obs(df["Fed Assets"], 13)

if "Net Liquidity" in df.columns:
    net_1w = change_obs(df["Net Liquidity"], 1)
    net_4w = change_obs(df["Net Liquidity"], 4)
    net_13w = change_obs(df["Net Liquidity"], 13)
    net_52w = change_obs(df["Net Liquidity"], 52)
    net_13w_ann = annualized_pace(df["Net Liquidity"], 13)
else:
    net_1w = net_4w = net_13w = net_52w = net_13w_ann = np.nan

if "TGA" in df.columns:
    tga_1w = change_obs(df["TGA"], 1)
    tga_4w = change_obs(df["TGA"], 4)
else:
    tga_1w = tga_4w = np.nan

rrp_col = "RRP Others" if "RRP Others" in df.columns else "RRP Total" if "RRP Total" in df.columns else None
if rrp_col is not None:
    rrp_1w = change_obs(df[rrp_col], 1)
    rrp_4w = change_obs(df[rrp_col], 4)
else:
    rrp_1w = rrp_4w = np.nan

regime_label, regime_color = liquidity_regime(assets_13w_ann, net_13w_ann)

# ---------------- Snapshot ----------------

st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='section-subtitle'>Latest available H.4.1 observation: {latest_date.strftime('%b %d, %Y')}. Values are billions of dollars.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    metric_card("Fed Assets", fmt_b(latest.get("Fed Assets")), "Total assets, H.4.1")

with c2:
    metric_card("1W Asset Change", fmt_delta_b(assets_1w), "Week over week")

with c3:
    metric_card("4W Asset Change", fmt_delta_b(assets_4w), "Monthly impulse")

with c4:
    metric_card("13W Ann. Pace", fmt_delta_b(assets_13w_ann), "Annualized balance-sheet pace")

with c5:
    metric_card("Net Liquidity", fmt_b(latest.get("Net Liquidity")), "Assets - TGA - RRP")

with c6:
    metric_card("Net 13W Ann.", fmt_delta_b(net_13w_ann), "Annualized net-liquidity pace")

c7, c8, c9, c10, c11, c12 = st.columns(6)

with c7:
    metric_card("TGA", fmt_b(latest.get("TGA")), "Treasury cash balance")

with c8:
    metric_card("TGA 1W", fmt_delta_b(tga_1w), "Higher TGA drains liquidity")

with c9:
    metric_card("RRP", fmt_b(latest.get(rrp_col)) if rrp_col else "N/A", rrp_col or "Unavailable")

with c10:
    metric_card("RRP 1W", fmt_delta_b(rrp_1w), "Lower RRP releases liquidity")

with c11:
    metric_card("Reserve Balances", fmt_b(latest.get("Reserve Balances")), "Bank reserves")

with c12:
    metric_card("13W Asset %", fmt_pct(assets_13w_pct), "Fed assets rate of change")

# ---------------- Read Through ----------------

st.markdown("<div class='section-title'>Liquidity Read</div>", unsafe_allow_html=True)

left, right = st.columns([1.22, 0.78])

with left:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Balance-sheet impulse:</b><br><br>
        Fed assets are <b>{fmt_b(latest.get("Fed Assets"))}</b>. The one-week move is <b>{fmt_delta_b(assets_1w)}</b>,
        the four-week move is <b>{fmt_delta_b(assets_4w)}</b>, and the thirteen-week annualized pace is
        <b>{fmt_delta_b(assets_13w_ann)}</b>. The regime read is
        <b style="color:{regime_color};">{regime_label}</b>.
        <br><br>
        The market-sensitive question is whether balance-sheet runoff is being offset or amplified by Treasury cash rebuilding
        and residual reverse-repo absorption. A rising TGA drains reserves. A falling RRP releases liquidity back into the system.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="good-box">
        <b>Data status</b><br><br>
        Active source stack: <b>{source_status}</b>.
        <br><br>
        FRED is not called anywhere in this page. If the Fed DDP endpoint is down, the app uses the current H.4.1 release and the local cache instead of failing.
        <br><br>
        Smoothing: <b>{smooth_window}</b> weekly observation(s). Lookback: <b>{lookback}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

if live_errors:
    st.warning("One or more Fed DDP package pulls failed. The dashboard remained alive using the available source stack.")
    with st.expander("DDP error details"):
        for package_name, message in live_errors.items():
            st.write(f"**{package_name}:** {message}")

# ---------------- Charts ----------------

st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Fed assets, net liquidity, funding drains, reserves, and rate of change.</div>",
    unsafe_allow_html=True,
)

plot_df = df.copy()
for col in plot_df.columns:
    plot_df[f"{col}_s"] = smooth_series(plot_df[col], int(smooth_window))

fig = make_subplots(
    rows=4 if show_components else 3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    subplot_titles=(
        "Fed Balance Sheet Level",
        "Liquidity Proxy",
        "Rate of Change",
        "Funding Plumbing",
    ) if show_components else (
        "Fed Balance Sheet Level",
        "Liquidity Proxy",
        "Rate of Change",
    ),
)

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["Fed Assets_s"],
        name="Fed Assets",
        mode="lines",
        line=dict(width=2.8, color="#111827"),
        hovertemplate="%{x|%Y-%m-%d}<br>Fed Assets: %{y:,.0f}B<extra></extra>",
    ),
    row=1,
    col=1,
)

if "Net Liquidity" in plot_df.columns:
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Net Liquidity_s"],
            name="Net Liquidity",
            mode="lines",
            line=dict(width=2.8, color="#2563eb"),
            hovertemplate="%{x|%Y-%m-%d}<br>Net Liquidity: %{y:,.0f}B<extra></extra>",
        ),
        row=2,
        col=1,
    )

asset_roc = plot_df["Fed Assets"].diff(13) / 13 * 52
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af", row=3, col=1)
fig.add_trace(
    go.Scatter(
        x=asset_roc.index,
        y=asset_roc,
        name="Fed Assets 13W Annualized Change",
        mode="lines",
        line=dict(width=2.3, color="#7c2d12"),
        hovertemplate="%{x|%Y-%m-%d}<br>13W Ann. Change: %{y:,.0f}B<extra></extra>",
    ),
    row=3,
    col=1,
)

if "Net Liquidity" in plot_df.columns:
    net_roc = plot_df["Net Liquidity"].diff(13) / 13 * 52
    fig.add_trace(
        go.Scatter(
            x=net_roc.index,
            y=net_roc,
            name="Net Liquidity 13W Annualized Change",
            mode="lines",
            line=dict(width=2.3, color="#9333ea"),
            hovertemplate="%{x|%Y-%m-%d}<br>Net 13W Ann. Change: %{y:,.0f}B<extra></extra>",
        ),
        row=3,
        col=1,
    )

if show_components:
    if "TGA" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["TGA_s"],
                name="TGA",
                mode="lines",
                line=dict(width=2.1, color="#059669"),
                hovertemplate="%{x|%Y-%m-%d}<br>TGA: %{y:,.0f}B<extra></extra>",
            ),
            row=4,
            col=1,
        )

    if rrp_col is not None:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[f"{rrp_col}_s"],
                name=rrp_col,
                mode="lines",
                line=dict(width=2.1, color="#dc2626"),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{rrp_col}: %{{y:,.0f}}B<extra></extra>",
            ),
            row=4,
            col=1,
        )

    if "Reserve Balances" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["Reserve Balances_s"],
                name="Reserve Balances",
                mode="lines",
                line=dict(width=2.1, color="#f59e0b"),
                hovertemplate="%{x|%Y-%m-%d}<br>Reserves: %{y:,.0f}B<extra></extra>",
            ),
            row=4,
            col=1,
        )

fig.update_layout(
    template="plotly_white",
    height=1040 if show_components else 850,
    margin=dict(l=55, r=25, t=58, b=35),
    legend=dict(
        orientation="h",
        x=0,
        y=1.04,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.78)",
    ),
    hovermode="x unified",
)

fig.update_annotations(font=dict(size=13, color="#111827"))

fig.update_yaxes(title_text="Billions", showgrid=True, gridcolor="#edf0f4", zeroline=False, row=1, col=1)
fig.update_yaxes(title_text="Billions", showgrid=True, gridcolor="#edf0f4", zeroline=False, row=2, col=1)
fig.update_yaxes(title_text="Billions Ann.", showgrid=True, gridcolor="#edf0f4", zeroline=False, row=3, col=1)

if show_components:
    fig.update_yaxes(title_text="Billions", showgrid=True, gridcolor="#edf0f4", zeroline=False, row=4, col=1)
    fig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date", row=4, col=1)
else:
    fig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Component Rebase ----------------

if show_components:
    st.markdown("<div class='section-title'>Component Pressure Map</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Rebased view to show which balance-sheet plumbing variable is driving the liquidity impulse.</div>",
        unsafe_allow_html=True,
    )

    rebase_cols = [c for c in ["Fed Assets", "TGA", "RRP Others", "RRP Total", "Reserve Balances"] if c in df.columns]

    rebased = pd.DataFrame(index=df.index)
    for col in rebase_cols:
        rebased[col] = rebase(df[col])

    fig2 = go.Figure()

    color_map = {
        "Fed Assets": "#111827",
        "TGA": "#059669",
        "RRP Others": "#dc2626",
        "RRP Total": "#b91c1c",
        "Reserve Balances": "#f59e0b",
    }

    for col in rebase_cols:
        fig2.add_trace(
            go.Scatter(
                x=rebased.index,
                y=smooth_series(rebased[col], int(smooth_window)),
                mode="lines",
                name=col,
                line=dict(width=2.2, color=color_map.get(col, "#374151")),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col} Index: %{{y:.1f}}<extra></extra>",
            )
        )

    fig2.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=55, r=25, t=25, b=35),
        legend=dict(
            orientation="h",
            x=0,
            y=1.08,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.78)",
        ),
        hovermode="x unified",
    )

    fig2.update_yaxes(title_text="Index, first obs = 100", showgrid=True, gridcolor="#edf0f4", zeroline=False)
    fig2.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date")

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tables and Download ----------------

left, right = st.columns([1.25, 0.75])

with left:
    if show_raw_table:
        st.markdown("<div class='section-title'>Cleaned Data</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Latest observations from the cleaned weekly H.4.1 dataset. Values are billions.</div>",
            unsafe_allow_html=True,
        )

        table = df.copy()
        table.index.name = "Date"

        display_table = table.tail(75).copy()
        for col in display_table.columns:
            display_table[col] = display_table[col].map(lambda x: np.nan if pd.isna(x) else round(float(x), 1))

        st.dataframe(display_table, use_container_width=True, height=430)

with right:
    st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the cleaned dataset used by the dashboard.</div>",
        unsafe_allow_html=True,
    )

    export = full_df.copy()
    export.index.name = "Date"
    csv = export.to_csv(index=True).encode("utf-8")

    st.download_button(
        "Download Fed Liquidity CSV",
        data=csv,
        file_name="fed_liquidity_tracker.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
        <b>Primary source</b><br>
        Federal Reserve H.4.1 Data Download Program.
        <br><br>
        <b>Fallback</b><br>
        Current H.4.1 release page plus local CSV cache.
        <br><br>
        <b>Core proxy</b><br>
        Net Liquidity = Fed Assets - TGA - RRP Others.
        <br><br>
        <b>Units</b><br>
        H.4.1 data are reported in millions. The app converts everything to billions.
        <br><br>
        <b>Rate of change</b><br>
        Weekly H.4.1 observations are used directly. 13-week annualized pace = 13-week change / 13 * 52.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("© 2026 AD Fund Management LP")
