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

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

TITLE = "Fed Balance Sheet & Liquidity Tracker"

H41_CURRENT_URL = "https://www.federalreserve.gov/releases/h41/current/"
H41_ARCHIVE_URL_TEMPLATE = "https://www.federalreserve.gov/releases/h41/{yyyymmdd}/"

CACHE_DIR = "data"
CACHE_PATH = os.path.join(CACHE_DIR, "fed_h41_liquidity_cache.csv")

REQUEST_TIMEOUT = (4, 12)
MAX_RETRIES = 2
BACKOFF_SECONDS = 0.75

DEFAULT_LOOKBACK = "2y"
DEFAULT_SMOOTH = 1

LOOKBACK_TO_WEEKS = {
    "6m": 30,
    "1y": 60,
    "2y": 115,
    "3y": 170,
    "5y": 275,
    "10y": 540,
}

DISPLAY_COLUMNS = [
    "Fed Assets",
    "Net Liquidity",
    "TGA",
    "RRP Others",
    "RRP Total",
    "Reserve Balances",
    "Securities Held Outright",
    "UST Holdings",
    "MBS Holdings",
]

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm_text(x: object) -> str:
    text = html.unescape(str(x))
    text = text.replace("\xa0", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_cell(x: object) -> str:
    if pd.isna(x):
        return ""
    text = html.unescape(str(x))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(x: object) -> float:
    if x is None or pd.isna(x):
        return np.nan

    s = html.unescape(str(x))
    s = s.replace("\xa0", " ")
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = re.sub(r"\s+", "", s)

    if s in {"", ".", "...", "nan", "None", "N/A"}:
        return np.nan

    is_negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")

    try:
        val = float(s)
        return -val if is_negative else val
    except Exception:
        return np.nan


def extract_numeric_values(row: pd.Series, min_abs: float = 1000.0) -> List[float]:
    vals: List[float] = []

    for item in row.tolist():
        val = parse_number(item)
        if pd.notna(val) and abs(val) >= min_abs:
            vals.append(float(val))

    return vals


def row_text(row: pd.Series) -> str:
    return norm_text(" ".join(clean_cell(x) for x in row.tolist() if clean_cell(x)))


def row_cells(row: pd.Series) -> List[str]:
    return [norm_text(x) for x in row.tolist() if clean_cell(x)]


def value_from_row(row: pd.Series, min_abs: float = 1000.0) -> float:
    nums = extract_numeric_values(row, min_abs=min_abs)
    if not nums:
        return np.nan

    positive_nums = [x for x in nums if x > 0]
    if positive_nums:
        return max(positive_nums)

    return nums[0]


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


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_text_cached(url: str) -> str:
    headers = {
        "User-Agent": "ADFM-H41-Liquidity-Tracker/3.0",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            text = response.text
            if not text or len(text.strip()) < 500:
                raise ValueError("Empty or truncated H.4.1 response")

            if "Factors Affecting Reserve Balances" not in text and "H.4.1" not in text:
                raise ValueError("Unexpected H.4.1 page structure")

            return text

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_release_date(page_html: str, fallback: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    patterns = [
        r"Release Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Last Update:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_html, flags=re.IGNORECASE)
        if match:
            dt = pd.to_datetime(match.group(1), errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt).normalize()

    if fallback is not None and pd.notna(fallback):
        return pd.Timestamp(fallback).normalize()

    return pd.Timestamp.today().normalize()


def read_h41_tables(page_html: str) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(StringIO(page_html), displayed_only=False)
    except Exception:
        return []

    cleaned: List[pd.DataFrame] = []

    for table in tables:
        if table.empty:
            continue

        table = table.copy()

        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                " ".join(str(x) for x in col if str(x) != "nan").strip()
                for col in table.columns
            ]
        else:
            table.columns = [str(c).strip() for c in table.columns]

        table = table.dropna(how="all")
        table = table.loc[:, ~table.columns.duplicated()]

        if not table.empty:
            cleaned.append(table)

    return cleaned


def find_value_by_row_text(
    tables: List[pd.DataFrame],
    required_phrases: List[str],
    excluded_phrases: Optional[List[str]] = None,
    min_abs: float = 1000.0,
) -> float:
    required = [norm_text(x) for x in required_phrases]
    excluded = [norm_text(x) for x in (excluded_phrases or [])]

    candidates: List[float] = []

    for table in tables:
        for _, row in table.iterrows():
            txt = row_text(row)

            if not txt:
                continue

            if all(p in txt for p in required) and not any(p in txt for p in excluded):
                val = value_from_row(row, min_abs=min_abs)
                if pd.notna(val):
                    candidates.append(float(val))

    if not candidates:
        return np.nan

    return max(candidates)


def find_exact_label_value(
    tables: List[pd.DataFrame],
    labels: List[str],
    min_abs: float = 1000.0,
) -> float:
    labels_norm = [norm_text(x) for x in labels]
    candidates: List[float] = []

    for table in tables:
        for _, row in table.iterrows():
            cells = row_cells(row)

            if any(cell in labels_norm for cell in cells):
                val = value_from_row(row, min_abs=min_abs)
                if pd.notna(val):
                    candidates.append(float(val))

    if not candidates:
        return np.nan

    return max(candidates)


def find_child_after_parent(
    tables: List[pd.DataFrame],
    parent_phrase: str,
    child_label: str,
    max_rows_after_parent: int = 10,
    min_abs: float = 1000.0,
) -> float:
    parent_norm = norm_text(parent_phrase)
    child_norm = norm_text(child_label)

    candidates: List[float] = []

    for table in tables:
        seen_parent = False
        rows_after = 0

        for _, row in table.iterrows():
            txt = row_text(row)
            cells = row_cells(row)

            if parent_norm in txt:
                seen_parent = True
                rows_after = 0
                continue

            if seen_parent:
                rows_after += 1

                if child_norm in cells or txt.startswith(child_norm + " "):
                    val = value_from_row(row, min_abs=min_abs)
                    if pd.notna(val):
                        candidates.append(float(val))
                        break

                if rows_after > max_rows_after_parent:
                    seen_parent = False
                    rows_after = 0

    if not candidates:
        return np.nan

    return max(candidates)


def parse_h41_page(page_html: str, fallback_date: Optional[pd.Timestamp] = None) -> pd.Series:
    release_date = parse_release_date(page_html, fallback=fallback_date)
    tables = read_h41_tables(page_html)

    if not tables:
        return pd.Series(dtype="float64", name=release_date)

    data: Dict[str, float] = {}

    # H.4.1 data are reported in millions. Convert to billions at the end.
    fed_assets_m = find_exact_label_value(
        tables,
        labels=[
            "Total assets",
            "Total assets (less eliminations from consolidation)",
        ],
        min_abs=1_000_000.0,
    )

    if pd.isna(fed_assets_m):
        fed_assets_m = find_value_by_row_text(
            tables,
            required_phrases=["total assets"],
            excluded_phrases=["memoranda"],
            min_abs=1_000_000.0,
        )

    tga_m = find_exact_label_value(
        tables,
        labels=["U.S. Treasury, General Account"],
        min_abs=10_000.0,
    )

    rrp_total_m = find_exact_label_value(
        tables,
        labels=["Reverse repurchase agreements"],
        min_abs=1_000.0,
    )

    rrp_others_m = find_child_after_parent(
        tables,
        parent_phrase="Reverse repurchase agreements",
        child_label="Others",
        max_rows_after_parent=8,
        min_abs=1_000.0,
    )

    reserves_m = find_exact_label_value(
        tables,
        labels=["Reserve balances with Federal Reserve Banks"],
        min_abs=100_000.0,
    )

    securities_m = find_exact_label_value(
        tables,
        labels=["Securities held outright"],
        min_abs=1_000_000.0,
    )

    ust_m = find_value_by_row_text(
        tables,
        required_phrases=["u.s. treasury securities"],
        excluded_phrases=["inflation compensation", "tips", "strips", "securities lent"],
        min_abs=100_000.0,
    )

    mbs_m = find_value_by_row_text(
        tables,
        required_phrases=["mortgage-backed securities"],
        excluded_phrases=["commitments", "memorandum", "securities lent"],
        min_abs=100_000.0,
    )

    if pd.notna(fed_assets_m):
        data["Fed Assets"] = fed_assets_m / 1000.0

    if pd.notna(tga_m):
        data["TGA"] = tga_m / 1000.0

    if pd.notna(rrp_others_m):
        data["RRP Others"] = rrp_others_m / 1000.0

    if pd.notna(rrp_total_m):
        data["RRP Total"] = rrp_total_m / 1000.0

    if pd.notna(reserves_m):
        data["Reserve Balances"] = reserves_m / 1000.0

    if pd.notna(securities_m):
        data["Securities Held Outright"] = securities_m / 1000.0

    if pd.notna(ust_m):
        data["UST Holdings"] = ust_m / 1000.0

    if pd.notna(mbs_m):
        data["MBS Holdings"] = mbs_m / 1000.0

    if "Fed Assets" in data and "TGA" in data:
        if "RRP Others" in data:
            data["Net Liquidity"] = data["Fed Assets"] - data["TGA"] - data["RRP Others"]
        elif "RRP Total" in data:
            data["Net Liquidity"] = data["Fed Assets"] - data["TGA"] - data["RRP Total"]

    s = pd.Series(data, name=release_date)
    return s


def previous_thursday(date_value: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(date_value).normalize()
    while d.weekday() != 3:
        d -= pd.Timedelta(days=1)
    return d


def generate_release_dates(latest_release_date: pd.Timestamp, weeks: int) -> List[pd.Timestamp]:
    latest = previous_thursday(latest_release_date)
    dates = [latest - pd.Timedelta(weeks=i) for i in range(weeks)]
    return dates


def archive_url_for_date(date_value: pd.Timestamp) -> str:
    return H41_ARCHIVE_URL_TEMPLATE.format(yyyymmdd=pd.Timestamp(date_value).strftime("%Y%m%d"))


def load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception:
        return pd.DataFrame()


def save_cache(df: pd.DataFrame) -> None:
    if df.empty:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    out = df.copy()
    out.index.name = "Date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out.to_csv(CACHE_PATH)


def merge_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    valid = [x for x in frames if x is not None and not x.empty]

    if not valid:
        return pd.DataFrame()

    merged = pd.concat(valid, axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    columns = [c for c in DISPLAY_COLUMNS if c in merged.columns]
    remaining = [c for c in merged.columns if c not in columns]
    merged = merged[columns + remaining]

    return merged


def fetch_h41_history(
    requested_weeks: int,
    force_backfill: bool,
) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    errors: Dict[str, str] = {}

    cached = load_cache()

    try:
        current_html = fetch_text_cached(H41_CURRENT_URL)
        current_release_date = parse_release_date(current_html)
        current_series = parse_h41_page(current_html, fallback_date=current_release_date)

        current_df = pd.DataFrame()
        if not current_series.empty:
            current_df = current_series.to_frame().T
            current_df.index = pd.to_datetime(current_df.index)
            current_df.index.name = "Date"

    except Exception as exc:
        errors["current"] = str(exc)
        current_df = pd.DataFrame()
        current_release_date = pd.Timestamp.today().normalize()

    release_dates = generate_release_dates(current_release_date, requested_weeks)

    rows: List[pd.Series] = []

    missing_dates = []
    cached_index = set(pd.to_datetime(cached.index).normalize()) if not cached.empty else set()

    for dt in release_dates:
        normalized_dt = pd.Timestamp(dt).normalize()

        if not force_backfill and normalized_dt in cached_index:
            continue

        if not current_df.empty and normalized_dt in set(pd.to_datetime(current_df.index).normalize()):
            continue

        missing_dates.append(normalized_dt)

    if missing_dates:
        progress = st.progress(0)
        status = st.empty()

        for i, dt in enumerate(missing_dates, start=1):
            yyyymmdd = dt.strftime("%Y%m%d")
            url = archive_url_for_date(dt)

            status.caption(f"Backfilling H.4.1 release {yyyymmdd}")

            try:
                page_html = fetch_text_cached(url)
                s = parse_h41_page(page_html, fallback_date=dt)

                if not s.empty and pd.notna(s.get("Fed Assets", np.nan)):
                    rows.append(s)
                else:
                    errors[yyyymmdd] = "Parsed page but could not find Fed Assets"

            except Exception as exc:
                errors[yyyymmdd] = str(exc)

            progress.progress(i / len(missing_dates))

        progress.empty()
        status.empty()

    archive_df = pd.DataFrame()
    if rows:
        archive_df = pd.DataFrame(rows)
        archive_df.index = pd.to_datetime(archive_df.index)
        archive_df.index.name = "Date"

    merged = merge_frames([cached, archive_df, current_df])

    if not merged.empty:
        save_cache(merged)

    if not current_df.empty:
        status_text = "current H.4.1 + archive pages + local cache"
    elif not cached.empty:
        status_text = "local cache only"
    else:
        status_text = "no working source"

    return merged, errors, status_text


def filter_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df.empty:
        return df

    if lookback == "max":
        return df.copy()

    latest = df.index.max()

    if lookback == "6m":
        start = latest - pd.DateOffset(months=6)
    elif lookback == "1y":
        start = latest - pd.DateOffset(years=1)
    elif lookback == "2y":
        start = latest - pd.DateOffset(years=2)
    elif lookback == "3y":
        start = latest - pd.DateOffset(years=3)
    elif lookback == "5y":
        start = latest - pd.DateOffset(years=5)
    elif lookback == "10y":
        start = latest - pd.DateOffset(years=10)
    else:
        start = df.index.min()

    return df[df.index >= start].copy()


def obs_change(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) <= periods:
        return np.nan

    return float(s.iloc[-1] - s.iloc[-1 - periods])


def obs_pct_change(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) <= periods:
        return np.nan

    base = s.iloc[-1 - periods]
    if pd.isna(base) or base == 0:
        return np.nan

    return float((s.iloc[-1] / base - 1.0) * 100.0)


def annualized_pace(series: pd.Series, periods: int = 13) -> float:
    move = obs_change(series, periods)

    if pd.isna(move):
        return np.nan

    return float(move / periods * 52.0)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if window <= 1:
        return s

    return s.rolling(window, min_periods=1).mean()


def rebase(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(index=s.index, dtype="float64")

    base = valid.iloc[0]

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, dtype="float64")

    return s / base * 100.0


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


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Fed H.4.1 balance-sheet level, liquidity impulse, TGA, reverse repos, reserves, and rate of change. FRED is not used.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        """
This page avoids FRED entirely.

It pulls the Fed's H.4.1 release pages directly, parses the balance-sheet tables, stores observations in a local CSV cache, and keeps running even when one historical release page fails.

Core proxy:

**Net Liquidity = Fed Assets - TGA - RRP Others**

If RRP Others is unavailable, the app falls back to total reverse repos.
        """
    )

    st.divider()
    st.markdown("### Controls")

    lookback = st.selectbox(
        "Lookback",
        ["6m", "1y", "2y", "3y", "5y", "10y", "max"],
        index=["6m", "1y", "2y", "3y", "5y", "10y", "max"].index(DEFAULT_LOOKBACK),
    )

    base_weeks = LOOKBACK_TO_WEEKS.get(lookback, 170)

    backfill_weeks = st.slider(
        "Archive releases to cache",
        min_value=30,
        max_value=540,
        value=min(max(base_weeks, 60), 540),
        step=10,
        help="First run builds this local cache from Fed H.4.1 archive pages. Later runs use the cache.",
    )

    smooth_window = st.number_input(
        "Smoothing window, weekly observations",
        min_value=1,
        max_value=12,
        value=DEFAULT_SMOOTH,
        step=1,
    )

    force_backfill = st.checkbox(
        "Force archive refresh",
        value=False,
        help="Leave off unless the local cache is stale or corrupted.",
    )

    show_components = st.checkbox("Show components chart", value=True)
    show_raw_table = st.checkbox("Show cleaned data table", value=True)

    st.caption("Source: Federal Reserve H.4.1 current and historical release pages. No FRED dependency.")

# ------------------------------------------------------------
# Data Load
# ------------------------------------------------------------

with st.spinner("Loading Fed H.4.1 data"):
    full_df, source_errors, source_status = fetch_h41_history(
        requested_weeks=int(backfill_weeks),
        force_backfill=bool(force_backfill),
    )

if full_df.empty or "Fed Assets" not in full_df.columns:
    st.error("No usable Fed balance-sheet data loaded.")
    st.markdown(
        """
        <div class="warning-box">
        The dashboard did not crash, but it could not locate a valid Fed Assets row from the current H.4.1 page,
        the archive pages, or the local cache. This usually means the Fed page structure changed or the runtime
        cannot reach federalreserve.gov.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_errors:
        with st.expander("Source error details", expanded=True):
            for key, message in list(source_errors.items())[:50]:
                st.write(f"**{key}:** {message}")

    st.stop()

full_df = full_df.sort_index()
full_df = full_df[~full_df.index.duplicated(keep="last")]

for col in full_df.columns:
    full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

available_cols = [col for col in DISPLAY_COLUMNS if col in full_df.columns]
full_df = full_df[available_cols].copy()

df = filter_lookback(full_df, lookback)

if df.empty:
    st.error("No data available for the selected lookback.")
    st.stop()

core_df = df.dropna(subset=["Fed Assets"])

if core_df.empty:
    st.error("Fed Assets exists as a column, but every value is blank after parsing.")
    st.markdown(
        """
        <div class="warning-box">
        This is the condition that caused the prior IndexError. The revised page stops here instead of calling iloc[-1]
        on an empty dataframe. Check the parsed data table and source errors below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(df.tail(30), use_container_width=True)

    if source_errors:
        with st.expander("Source error details", expanded=True):
            for key, message in list(source_errors.items())[:50]:
                st.write(f"**{key}:** {message}")

    st.stop()

latest = core_df.iloc[-1]
latest_date = core_df.index[-1]

# ------------------------------------------------------------
# Derived Metrics
# ------------------------------------------------------------

assets_1w = obs_change(df["Fed Assets"], 1)
assets_4w = obs_change(df["Fed Assets"], 4)
assets_13w = obs_change(df["Fed Assets"], 13)
assets_52w = obs_change(df["Fed Assets"], 52)
assets_13w_ann = annualized_pace(df["Fed Assets"], 13)
assets_13w_pct = obs_pct_change(df["Fed Assets"], 13)

if "Net Liquidity" in df.columns:
    net_1w = obs_change(df["Net Liquidity"], 1)
    net_4w = obs_change(df["Net Liquidity"], 4)
    net_13w = obs_change(df["Net Liquidity"], 13)
    net_52w = obs_change(df["Net Liquidity"], 52)
    net_13w_ann = annualized_pace(df["Net Liquidity"], 13)
else:
    net_1w = net_4w = net_13w = net_52w = net_13w_ann = np.nan

if "TGA" in df.columns:
    tga_1w = obs_change(df["TGA"], 1)
    tga_4w = obs_change(df["TGA"], 4)
else:
    tga_1w = tga_4w = np.nan

rrp_col = None
if "RRP Others" in df.columns and df["RRP Others"].notna().any():
    rrp_col = "RRP Others"
elif "RRP Total" in df.columns and df["RRP Total"].notna().any():
    rrp_col = "RRP Total"

if rrp_col is not None:
    rrp_1w = obs_change(df[rrp_col], 1)
    rrp_4w = obs_change(df[rrp_col], 4)
else:
    rrp_1w = rrp_4w = np.nan

regime_label, regime_color = liquidity_regime(assets_13w_ann, net_13w_ann)

# ------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------

st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='section-subtitle'>Latest parsed H.4.1 release: {latest_date.strftime('%b %d, %Y')}. Values are billions of dollars.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    metric_card("Fed Assets", fmt_b(latest.get("Fed Assets")), "Total assets")

with c2:
    metric_card("1W Asset Change", fmt_delta_b(assets_1w), "Weekly impulse")

with c3:
    metric_card("4W Asset Change", fmt_delta_b(assets_4w), "Monthly impulse")

with c4:
    metric_card("13W Ann. Pace", fmt_delta_b(assets_13w_ann), "Annualized pace")

with c5:
    metric_card("Net Liquidity", fmt_b(latest.get("Net Liquidity")), "Assets - TGA - RRP")

with c6:
    metric_card("Net 13W Ann.", fmt_delta_b(net_13w_ann), "Annualized net pace")

c7, c8, c9, c10, c11, c12 = st.columns(6)

with c7:
    metric_card("TGA", fmt_b(latest.get("TGA")), "Treasury cash")

with c8:
    metric_card("TGA 1W", fmt_delta_b(tga_1w), "Higher drains liquidity")

with c9:
    metric_card("RRP", fmt_b(latest.get(rrp_col)) if rrp_col else "N/A", rrp_col or "Unavailable")

with c10:
    metric_card("RRP 1W", fmt_delta_b(rrp_1w), "Lower releases liquidity")

with c11:
    metric_card("Reserve Balances", fmt_b(latest.get("Reserve Balances")), "Bank reserves")

with c12:
    metric_card("13W Asset %", fmt_pct(assets_13w_pct), "Fed assets ROC")

# ------------------------------------------------------------
# Liquidity Read
# ------------------------------------------------------------

st.markdown("<div class='section-title'>Liquidity Read</div>", unsafe_allow_html=True)

left, right = st.columns([1.22, 0.78])

with left:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Balance-sheet impulse:</b><br><br>
        Fed assets are <b>{fmt_b(latest.get("Fed Assets"))}</b>. The one-week move is
        <b>{fmt_delta_b(assets_1w)}</b>, the four-week move is <b>{fmt_delta_b(assets_4w)}</b>,
        and the thirteen-week annualized pace is <b>{fmt_delta_b(assets_13w_ann)}</b>.
        The regime read is <b style="color:{regime_color};">{regime_label}</b>.
        <br><br>
        The pressure point is whether Treasury cash rebuilding and reverse-repo absorption are offsetting or amplifying
        the Fed balance-sheet path. Rising TGA drains liquidity. Falling RRP releases liquidity. Reserve balances show
        where the plumbing pressure actually settles.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Data status</b><br><br>
        Source stack: <b>{source_status}</b>.
        <br><br>
        FRED is not called anywhere in this page. The app parses Fed H.4.1 pages directly and stores a local CSV cache.
        <br><br>
        Lookback: <b>{lookback}</b>. Cached archive releases targeted: <b>{backfill_weeks}</b>.
        Smoothing: <b>{smooth_window}</b> weekly observation(s).
        </div>
        """,
        unsafe_allow_html=True,
    )

if source_errors:
    with st.expander("Source issues, non-fatal"):
        for key, message in list(source_errors.items())[:80]:
            st.write(f"**{key}:** {message}")

# ------------------------------------------------------------
# Chartbook
# ------------------------------------------------------------

st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Fed assets, liquidity proxy, funding drains, reserves, and rate of change.</div>",
    unsafe_allow_html=True,
)

plot_df = df.copy()

for col in plot_df.columns:
    plot_df[f"{col}_s"] = smooth_series(plot_df[col], int(smooth_window))

rows = 4 if show_components else 3

fig = make_subplots(
    rows=rows,
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

if "Net Liquidity" in plot_df.columns and plot_df["Net Liquidity"].notna().any():
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
        hovertemplate="%{x|%Y-%m-%d}<br>Assets 13W Ann.: %{y:,.0f}B<extra></extra>",
    ),
    row=3,
    col=1,
)

if "Net Liquidity" in plot_df.columns and plot_df["Net Liquidity"].notna().any():
    net_roc = plot_df["Net Liquidity"].diff(13) / 13 * 52

    fig.add_trace(
        go.Scatter(
            x=net_roc.index,
            y=net_roc,
            name="Net Liquidity 13W Annualized Change",
            mode="lines",
            line=dict(width=2.3, color="#9333ea"),
            hovertemplate="%{x|%Y-%m-%d}<br>Net 13W Ann.: %{y:,.0f}B<extra></extra>",
        ),
        row=3,
        col=1,
    )

if show_components:
    if "TGA" in plot_df.columns and plot_df["TGA"].notna().any():
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

    if "Reserve Balances" in plot_df.columns and plot_df["Reserve Balances"].notna().any():
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

# ------------------------------------------------------------
# Component Pressure Map
# ------------------------------------------------------------

if show_components:
    component_cols = [
        col for col in ["Fed Assets", "Net Liquidity", "TGA", rrp_col, "Reserve Balances"]
        if col is not None and col in df.columns and df[col].notna().any()
    ]

    if component_cols:
        st.markdown("<div class='section-title'>Component Pressure Map</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Rebased view showing which balance-sheet plumbing variable is driving the liquidity impulse.</div>",
            unsafe_allow_html=True,
        )

        rebased = pd.DataFrame(index=df.index)

        for col in component_cols:
            rebased[col] = rebase(df[col])

        fig2 = go.Figure()

        color_map = {
            "Fed Assets": "#111827",
            "Net Liquidity": "#2563eb",
            "TGA": "#059669",
            "RRP Others": "#dc2626",
            "RRP Total": "#b91c1c",
            "Reserve Balances": "#f59e0b",
        }

        for col in component_cols:
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

# ------------------------------------------------------------
# Table and Download
# ------------------------------------------------------------

left, right = st.columns([1.25, 0.75])

with left:
    if show_raw_table:
        st.markdown("<div class='section-title'>Cleaned H.4.1 Data</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Latest parsed observations. Values are billions.</div>",
            unsafe_allow_html=True,
        )

        table = df.copy()
        table.index.name = "Date"

        display_table = table.tail(90).copy()

        for col in display_table.columns:
            display_table[col] = display_table[col].map(
                lambda x: np.nan if pd.isna(x) else round(float(x), 1)
            )

        st.dataframe(display_table, use_container_width=True, height=430)

with right:
    st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the cleaned cached dataset used by the dashboard.</div>",
        unsafe_allow_html=True,
    )

    export = full_df.copy()
    export.index.name = "Date"
    csv = export.to_csv(index=True).encode("utf-8")

    st.download_button(
        "Download Fed Liquidity CSV",
        data=csv,
        file_name="fed_h41_liquidity_tracker.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
        <b>Primary source</b><br>
        Federal Reserve H.4.1 current and historical release pages.
        <br><br>
        <b>Core proxy</b><br>
        Net Liquidity = Fed Assets - TGA - RRP Others.
        <br><br>
        <b>Fallback</b><br>
        If RRP Others is unavailable, total reverse repos are used. If archive pages fail, the app uses the local cache.
        <br><br>
        <b>Units</b><br>
        H.4.1 figures are reported in millions. The app converts them to billions.
        <br><br>
        <b>Rate of change</b><br>
        13-week annualized pace = 13-week change / 13 * 52.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("© 2026 AD Fund Management LP")
