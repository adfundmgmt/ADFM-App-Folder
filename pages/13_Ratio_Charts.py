import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("default")
pd.options.mode.chained_assignment = None

# ============================== Page config ==============================
st.set_page_config(layout="wide", page_title="Ratio Charts")
st.title("Ratio Charts")

# ============================== Defaults =================================
SHOW_MA50 = True
SHOW_MA200 = True
SHOW_RSI = True
RSI_WINDOW = 14

# ============================== Sidebar ==================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Ratio chart workspace for cross-asset, global relative strength, and leadership dynamics.

        What it covers
        • Cross asset relationships that help frame macro regime shifts
        • US versus ex-US relative moves across regions and markets
        • Cross sector and cross ticker leadership dynamics inside equities

        Data source
        • Yahoo Finance adjusted daily price history
        """
    )

    st.markdown("---")
    st.header("Lookback")
    spans = {
        "3 Months": 90,
        "6 Months": 180,
        "9 Months": 270,
        "YTD": None,
        "1 Year": 365,
        "3 Years": 365 * 3,
        "5 Years": 365 * 5,
        "10 Years": 365 * 10,
        "20 Years": 365 * 20,
    }
    span_key = st.selectbox(
        "Period",
        list(spans.keys()),
        index=list(spans.keys()).index("5 Years"),
    )

    st.markdown("---")
    st.header("Custom Ratio")
    custom_t1 = st.text_input("Ticker 1", "GLD").strip().upper()
    custom_t2 = st.text_input("Ticker 2", "TLT").strip().upper()

# ============================== Dates ====================================
now = datetime.today()
hist_start = now - timedelta(days=365 * 25)

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_start = pd.Timestamp(now - timedelta(days=spans[span_key]))

# ============================== Definitions ==============================
CYCLICALS = ["XLK", "XLI", "XLF", "XLY", "XLC", "XLB", "XLE"]
DEFENSIVES = ["XLP", "XLV", "XLU", "XLRE"]

PRESET_GROUPS: Dict[str, List[Tuple[str, str, str]]] = {
    "Cross Asset": [
        ("SPY", "TLT", "Equities / Long Treasuries"),
        ("HYG", "LQD", "High Yield / Investment Grade Credit"),
        ("GLD", "TLT", "Gold / Long Treasuries"),
        ("TIP", "TLT", "Inflation Linkers / Long Treasuries"),
        ("DBC", "TLT", "Broad Commodities / Long Treasuries"),
        ("UUP", "GLD", "Dollar / Gold"),
        ("XLE", "TLT", "Energy / Long Treasuries"),
    ],
    "US vs Ex-US": [
        ("VXUS", "SPY", "Ex-US Equities / US Equities"),
        ("EFA", "SPY", "Developed ex-US / US Equities"),
        ("EWJ", "SPY", "Japan / US Equities"),
        ("EZU", "SPY", "Europe / US Equities"),
        ("EEM", "SPY", "Emerging Markets / US Equities"),
        ("FXI", "SPY", "China Large Caps / US Equities"),
    ],
    "Cross Sector / Cross Ticker": [
        ("XLY", "XLP", "Consumer Discretionary / Staples"),
        ("XLF", "XLU", "Financials / Utilities"),
        ("XLI", "XLU", "Industrials / Utilities"),
        ("QQQ", "IWM", "Large Growth / Small Caps"),
        ("SMH", "QQQ", "Semis / Nasdaq 100"),
        ("SMH", "IGV", "Semis / Software"),
        ("NVDA", "SMH", "NVIDIA / Semis ETF"),
        ("AMZN", "XLY", "Amazon / Consumer Discretionary"),
        ("META", "QQQ", "Meta / Nasdaq 100"),
    ],
}

GROUP_DESCRIPTIONS = {
    "Cross Asset": "Cross-asset regime relationships",
    "US vs Ex-US": "Relative leadership between US and international markets",
    "Cross Sector / Cross Ticker": "Equity internal leadership and sector rotation",
}

ALL_PRESETS: List[Tuple[str, str, str, str]] = []
for group_name, pairs in PRESET_GROUPS.items():
    ALL_PRESETS.extend([(a, b, label, group_name) for a, b, label in pairs])

STATIC_TICKERS = sorted(
    set(CYCLICALS + DEFENSIVES + [t for a, b, _, _ in ALL_PRESETS for t in (a, b)])
)

# ============================== Data fetch ================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    def _normalize(raw: pd.DataFrame, tickers_list: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers_list:
                try:
                    s = raw[(t, "Close")].copy()
                    s.name = t
                    out[t] = s
                    continue
                except Exception:
                    pass
                try:
                    s = raw[("Close", t)].copy()
                    s.name = t
                    out[t] = s
                except Exception:
                    continue
        else:
            if len(tickers_list) == 1 and "Close" in raw.columns:
                out[tickers_list[0]] = raw["Close"].copy()

        if out.empty:
            return pd.DataFrame()

        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.ffill()
        out = out.dropna(how="all")
        return out

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        out = _normalize(raw, tickers)
        if not out.empty:
            return out
    except Exception:
        pass

    try:
        raw = yf.download(
            tickers=tickers,
            period="max",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return _normalize(raw, tickers)
    except Exception:
        return pd.DataFrame()

# ============================== Helpers ===================================
def compute_cumrets(df: pd.DataFrame) -> pd.DataFrame:
    out = 1.0 + df.pct_change()
    out = out.fillna(1.0).cumprod()
    return out

def last_valid_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()
    if s.empty:
        return pd.NaT, np.nan
    try:
        sub = s.loc[:ts]
        if not sub.empty:
            return sub.index[-1], float(sub.iloc[-1])
    except Exception:
        pass
    return s.index[0], float(s.iloc[0])

def rebase_series(series: pd.Series, base_date: pd.Timestamp, base: float = 100.0) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    base_ts, base_val = last_valid_on_or_before(s, base_date)
    if pd.isna(base_ts) or not np.isfinite(base_val) or base_val == 0:
        return pd.Series(dtype=float)
    return s / base_val * base

def compute_price_ratio(
    s1: pd.Series,
    s2: pd.Series,
    base_date: pd.Timestamp,
    base: float = 100.0,
) -> pd.Series:
    s1, s2 = s1.align(s2, join="inner")
    raw_ratio = (s1 / s2).replace([np.inf, -np.inf], np.nan).dropna()
    return rebase_series(raw_ratio, base_date=base_date, base=base)

def compute_basket_ratio(
    cum_1: pd.Series,
    cum_2: pd.Series,
    base_date: pd.Timestamp,
    base: float = 100.0,
) -> pd.Series:
    cum_1, cum_2 = cum_1.align(cum_2, join="inner")
    raw_ratio = (cum_1 / cum_2).replace([np.inf, -np.inf], np.nan).dropna()
    return rebase_series(raw_ratio, base_date=base_date, base=base)

def rsi_wilder(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def make_display_title(ticker_1: str, ticker_2: str, label: str) -> str:
    return f"{label} ({ticker_1}/{ticker_2})"

def make_fig(
    ratio: pd.Series,
    title: str,
    y_label: str,
    display_start: pd.Timestamp,
    show_ma50_flag: bool,
    show_ma200_flag: bool,
    show_rsi_flag: bool,
    rsi_len: int,
):
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if ratio.empty:
        fig, ax = plt.subplots(figsize=(16, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return fig

    ratio_view = ratio.loc[display_start:].copy()
    if ratio_view.empty:
        ratio_view = ratio.copy()

    ma50 = ratio.rolling(50, min_periods=1).mean()
    ma200 = ratio.rolling(200, min_periods=1).mean()
    rsi = rsi_wilder(ratio, window=rsi_len)

    nrows = 2 if show_rsi_flag else 1
    height_ratios = [4, 1.2] if show_rsi_flag else [1]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=True,
        figsize=(18, 5.0 if show_rsi_flag else 3.2),
        gridspec_kw={"height_ratios": height_ratios},
    )

    if show_rsi_flag:
        ax1, ax2 = axes
    else:
        ax1 = axes
        ax2 = None

    x_start = ratio_view.index.min()
    x_end = ratio_view.index.max()

    ax1.plot(ratio_view.index, ratio_view, color="black", lw=1.5)

    if show_ma50_flag:
        ma50_view = ma50.loc[x_start:x_end].dropna()
        ax1.plot(ma50_view.index, ma50_view, color="#1f77b4", lw=1.0, label="50 DMA")

    if show_ma200_flag:
        ma200_view = ma200.loc[x_start:x_end].dropna()
        ax1.plot(ma200_view.index, ma200_view, color="#d62728", lw=1.0, label="200 DMA")

    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel(y_label)
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.set_xlim(x_start, x_end)
    ax1.margins(x=0)

    y_parts = [ratio_view]
    if show_ma50_flag:
        y_parts.append(ma50.loc[x_start:x_end])
    if show_ma200_flag:
        y_parts.append(ma200.loc[x_start:x_end])

    y_all = pd.concat(y_parts).replace([np.inf, -np.inf], np.nan).dropna()
    if not y_all.empty:
        ymin, ymax = y_all.min(), y_all.max()
        pad = (ymax - ymin) * 0.06 if ymax != ymin else max(abs(ymin) * 0.05, 1.0)
        ax1.set_ylim(ymin - pad, ymax + pad)

    if show_ma50_flag or show_ma200_flag:
        ax1.legend(loc="upper left", fontsize=8, frameon=False)

    if show_rsi_flag and ax2 is not None:
        rsi_view = rsi.loc[x_start:x_end].dropna()
        ax2.axhspan(30, 70, alpha=0.08, color="gray")
        ax2.plot(rsi_view.index, rsi_view, color="black", lw=1.0)
        ax2.axhline(70, ls=":", lw=0.9, color="red")
        ax2.axhline(30, ls=":", lw=0.9, color="green")
        if not rsi_view.empty:
            ax2.scatter(rsi_view.index[-1], rsi_view.iloc[-1], s=18, color="black", zorder=3)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.grid(True, linestyle="--", alpha=0.25)
        ax2.set_xlim(x_start, x_end)
        ax2.margins(x=0)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.08, hspace=0.10)
    return fig

def render_ratio_block(
    ratio: pd.Series,
    title: str,
    display_start: pd.Timestamp,
    rsi_len: int,
):
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        st.warning(f"No usable data for {title}.")
        return

    fig = make_fig(
        ratio=ratio,
        title=title,
        y_label="Ratio Index",
        display_start=display_start,
        show_ma50_flag=SHOW_MA50,
        show_ma200_flag=SHOW_MA200,
        show_rsi_flag=SHOW_RSI,
        rsi_len=rsi_len,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("---")

# ============================== Fetch static data =========================
closes_static = fetch_closes(STATIC_TICKERS, hist_start, now)
if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

# ============================== Headline basket ===========================
cum_static = compute_cumrets(closes_static)

available_cyc = [t for t in CYCLICALS if t in cum_static.columns]
available_def = [t for t in DEFENSIVES if t in cum_static.columns]

st.subheader("Headline Basket")

if available_cyc and available_def:
    eq_cyc = cum_static[available_cyc].mean(axis=1)
    eq_def = cum_static[available_def].mean(axis=1)
    cyc_def_ratio = compute_basket_ratio(eq_cyc, eq_def, base_date=disp_start, base=100.0)

    render_ratio_block(
        cyc_def_ratio,
        "Cyclicals / Defensives (Equal Weight)",
        disp_start,
        RSI_WINDOW,
    )
else:
    st.warning("Unable to build the cyclicals / defensives basket with current data.")

# ============================== Preset ratios =============================
st.subheader("Preset Ratio Charts")

failed_pairs = []
current_group = None

for a, b, label, group_name in ALL_PRESETS:
    if group_name != current_group:
        current_group = group_name
        st.markdown(f"### {group_name}")
        st.caption(GROUP_DESCRIPTIONS.get(group_name, ""))

    if a in closes_static.columns and b in closes_static.columns:
        ratio = compute_price_ratio(
            closes_static[a],
            closes_static[b],
            base_date=disp_start,
            base=100.0,
        )
        if ratio.empty:
            failed_pairs.append(f"{a}/{b}")
            continue

        display_title = make_display_title(a, b, label)

        render_ratio_block(
            ratio,
            display_title,
            disp_start,
            RSI_WINDOW,
        )
    else:
        failed_pairs.append(f"{a}/{b}")

if failed_pairs:
    st.caption("Unavailable this session: " + ", ".join(sorted(set(failed_pairs))))

# ============================== Custom ratio ==============================
st.subheader("Custom Ratio")

if custom_t1 and custom_t2:
    closes_custom = fetch_closes([custom_t1, custom_t2], hist_start, now)

    if (
        not closes_custom.empty
        and custom_t1 in closes_custom.columns
        and custom_t2 in closes_custom.columns
    ):
        custom_ratio = compute_price_ratio(
            closes_custom[custom_t1],
            closes_custom[custom_t2],
            base_date=disp_start,
            base=100.0,
        )

        if custom_ratio.empty:
            st.warning(f"No usable aligned data for {custom_t1}/{custom_t2}.")
        else:
            custom_title = make_display_title(custom_t1, custom_t2, f"{custom_t1} / {custom_t2}")
            render_ratio_block(
                custom_ratio,
                custom_title,
                disp_start,
                RSI_WINDOW,
            )
    else:
        st.warning(f"Data not available for {custom_t1}/{custom_t2}.")

st.caption("© 2026 AD Fund Management LP")
