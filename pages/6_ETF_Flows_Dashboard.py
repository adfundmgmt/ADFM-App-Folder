import time
from datetime import datetime, date
from typing import Dict, Tuple, List
from io import BytesIO

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="ETF Net Flows", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.6rem;
        max-width: 1700px;
    }
    h1, h2, h3 {
        letter-spacing: 0.1px;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 10px 12px;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.84rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Slightly crisper global Matplotlib defaults
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 260
plt.rcParams["text.antialiased"] = True
plt.rcParams["font.size"] = 9

# =========================================================
# GLOBALS
# =========================================================
TZ = pytz.timezone("US/Eastern")


# =========================================================
# TIME HELPERS
# =========================================================
def now_et() -> datetime:
    return datetime.now(TZ)


def as_naive_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts.replace(tzinfo=None))


def ytd_days(as_of: datetime) -> int:
    start_ytd = TZ.localize(datetime(as_of.year, 1, 1))
    return max((as_of - start_ytd).days, 1)


def calc_start_date(days: int, as_of: datetime) -> date:
    padding = 21
    return (as_of - pd.Timedelta(days=days + padding)).date()


def last_friday(on_or_before: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(on_or_before).normalize()
    wd = ts.weekday()
    days_back = wd - 4 if wd >= 4 else wd + 3
    return ts - pd.Timedelta(days=int(days_back))


def week_start_monday(week_ending_friday: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(week_ending_friday).normalize() - pd.tseries.offsets.BDay(4)).normalize()


# =========================================================
# RUNTIME
# =========================================================
as_of_dt = now_et()
as_of_date = as_of_dt.date()
as_of_dt_naive = as_naive_ts(as_of_dt)
as_of_ts = pd.Timestamp(as_of_dt_naive).normalize()

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": ytd_days(as_of_dt),
}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** ETF flow-pressure monitor using a stable price-volume proxy workflow.

        **What this tab shows**
        - Horizontal ranking bars for the selected lookback.
        - Full-table coverage with lookback total, latest week, and week-to-date values.
        - Transparent handling of missing rows to preserve coverage context.

        **Data source**
        - Yahoo Finance OHLCV data used as a directional flow-pressure proxy.
        """
    )
    st.markdown("---")
    period_label = st.radio("Lookback Window", list(lookback_dict.keys()), index=0)
    st.caption(f"As of: {as_of_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

period_days = int(lookback_dict[period_label])

# =========================================================
# ETF COVERAGE
# =========================================================
etf_info = {
    "VTI": ("US Total Market", "Total US equity market"),
    "VUG": ("US Growth", "Large-cap growth"),
    "VTV": ("US Value", "Large-cap value"),
    "MTUM": ("US Momentum", "Momentum factor"),
    "QUAL": ("US Quality", "Quality factor"),
    "USMV": ("US Min Vol", "Minimum volatility"),
    "SCHD": ("US Dividends", "Dividend growth and yield"),
    "XLB": ("US Materials", "S&P 500 Materials"),
    "XLC": ("US Communication Services", "S&P 500 Communication Services"),
    "XLE": ("US Energy", "S&P 500 Energy"),
    "XLF": ("US Financials", "S&P 500 Financials"),
    "XLI": ("US Industrials", "S&P 500 Industrials"),
    "XLK": ("US Technology", "S&P 500 Technology"),
    "XLP": ("US Staples", "S&P 500 Consumer Staples"),
    "XLRE": ("US Real Estate", "S&P 500 Real Estate"),
    "XLU": ("US Utilities", "S&P 500 Utilities"),
    "XLV": ("US Healthcare", "S&P 500 Healthcare"),
    "XLY": ("US Discretionary", "S&P 500 Consumer Discretionary"),
    "SMH": ("Semiconductors", "Global semiconductor equities"),
    "IGV": ("Software", "US application software"),
    "SKYY": ("Cloud", "Cloud infrastructure and services"),
    "VT": ("Global Equity", "Total world equity market"),
    "ACWI": ("Global Equity ex-Frontier", "All-country world equity"),
    "VEA": ("Developed ex-US", "Developed markets ex-US"),
    "VWO": ("Emerging Markets", "Emerging markets equity"),
    "EXUS": ("Non-US Equity", "Global equity ex-US"),
    "EWG": ("Germany", "Germany equities"),
    "EWQ": ("France", "France equities"),
    "EWU": ("United Kingdom", "UK equities"),
    "EWI": ("Italy", "Italy equities"),
    "EWP": ("Spain", "Spain equities"),
    "EWL": ("Switzerland", "Switzerland equities"),
    "EWN": ("Netherlands", "Netherlands equities"),
    "EWD": ("Sweden", "Sweden equities"),
    "EWO": ("Austria", "Austria equities"),
    "EWK": ("Belgium", "Belgium equities"),
    "EWJ": ("Japan", "Japan equities"),
    "EWY": ("South Korea", "Korea equities"),
    "ASHR": ("China A-Shares", "Onshore China equities"),
    "FXI": ("China Large-Cap", "China offshore large caps"),
    "EWT": ("Taiwan", "Taiwan equities"),
    "INDA": ("India", "India equities"),
    "EWS": ("Singapore", "Singapore equities"),
    "EWA": ("Australia", "Australia equities"),
    "EWH": ("Hong Kong", "Hong Kong equities"),
    "EPHE": ("Philippines", "Philippines equities"),
    "EWM": ("Malaysia", "Malaysia equities"),
    "IDX": ("Indonesia", "Indonesia equities"),
    "THD": ("Thailand", "Thailand equities"),
    "VNM": ("Vietnam", "Vietnam equities"),
    "EWZ": ("Brazil", "Brazil equities"),
    "EWW": ("Mexico", "Mexico equities"),
    "EWC": ("Canada", "Canada equities"),
    "EPU": ("Peru", "Peru equities"),
    "ECH": ("Chile", "Chile equities"),
    "ARGT": ("Argentina", "Argentina equities"),
    "GXG": ("Colombia", "Colombia equities"),
    "SGOV": ("UST Bills", "0-3 month Treasuries"),
    "SHY": ("UST 1-3y", "Short-term Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate Treasuries"),
    "TLT": ("UST 20y+", "Long-duration Treasuries"),
    "TIP": ("TIPS", "Inflation-linked Treasuries"),
    "LQD": ("IG Credit", "Investment-grade corporates"),
    "VCIT": ("IG Credit Duration", "Intermediate IG corporates"),
    "HYG": ("High Yield", "High-yield credit"),
    "BKLN": ("Floating-Rate Credit", "Senior loans"),
    "EMB": ("EM Debt", "USD EM sovereign debt"),
    "BND": ("US Aggregate", "Total US bond market"),
    "GLD": ("Gold", "Gold bullion"),
    "SLV": ("Silver", "Silver bullion"),
    "CPER": ("Copper", "Industrial copper"),
    "USO": ("Crude Oil", "WTI crude oil"),
    "DBC": ("Broad Commodities", "Commodity basket"),
    "PDBC": ("Broad Commodities Alt", "Rules-based commodities"),
    "URA": ("Uranium", "Nuclear fuel cycle"),
    "VXX": ("Equity Volatility", "Front-end VIX futures"),
    "UUP": ("USD", "US Dollar Index"),
    "FXE": ("EURUSD", "Euro vs USD"),
    "FXY": ("JPYUSD", "Japanese yen vs USD"),
    "FXF": ("CHFUSD", "Swiss franc vs USD"),
    "CEW": ("EM FX", "Emerging market currencies"),
    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETH": ("Ethereum", "Ethereum proxy ticker as originally listed"),
}
etf_tickers = tuple(etf_info.keys())

# =========================================================
# FORMATTERS
# =========================================================
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"${x / 1e12:,.2f}T"
    if ax >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):,.2f}%"


def fmt_price(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.2f}"


def fmt_score(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def axis_fmt(x, _pos=None) -> str:
    return fmt_compact_cur(x)


# =========================================================
# RETRY
# =========================================================
def retry(n=3, delay=0.8):
    def deco(fn):
        def wrap(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (i + 1))
            if last:
                raise last
        return wrap
    return deco


# =========================================================
# DATA HELPERS
# =========================================================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    idx = pd.to_datetime(out.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    out.index = idx

    out = out.dropna(subset=["Close"]).sort_index()
    return out


@retry()
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prices(
    tickers: Tuple[str, ...],
    start_date: date,
    as_of_date: date,
) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=as_of_date + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )

    out: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tk in raw.columns.get_level_values(0):
                    df = raw[tk].copy()
                else:
                    df = pd.DataFrame()
            else:
                df = raw.copy() if len(tickers) == 1 else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

        out[tk] = normalize_ohlcv(df)

    return out


# =========================================================
# FLOW PROXY ENGINE
# =========================================================
def compute_money_flow_proxy(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfm = mfm.fillna(0.0)

    typical = (high + low + close) / 3.0
    money_flow_value = (mfm * vol * typical).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    money_flow_value.index = pd.to_datetime(money_flow_value.index, errors="coerce")
    try:
        money_flow_value.index = money_flow_value.index.tz_localize(None)
    except Exception:
        pass

    return money_flow_value.sort_index()


def compute_pressure_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan

    mfv = compute_money_flow_proxy(df)
    if mfv.empty:
        return np.nan

    traded_value = (
        pd.to_numeric(df["Close"], errors="coerce")
        * pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    denom = float(traded_value.sum())
    if denom <= 0:
        return np.nan

    return float(mfv.sum() / denom)


def compute_window_sum(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if daily is None or daily.empty:
        return np.nan

    s = daily.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    window = s.loc[(s.index >= start) & (s.index <= end)]
    if window.empty:
        return np.nan
    return float(window.sum())


# =========================================================
# BUILD TABLE
# =========================================================
@st.cache_data(show_spinner=True, ttl=300)
def build_table(
    tickers: Tuple[str, ...],
    period_days: int,
    as_of_date: date,
    as_of_dt_naive: pd.Timestamp,
) -> pd.DataFrame:
    as_of_dt_local = as_of_dt_naive.to_pydatetime().replace(tzinfo=None)
    as_of_dt_et = TZ.localize(as_of_dt_local)
    start_date = calc_start_date(period_days, as_of_dt_et)

    price_map = fetch_prices(tickers, start_date, as_of_date)
    cutoff = as_of_dt_naive - pd.Timedelta(days=period_days)

    latest_complete_week_end = last_friday(as_of_dt_naive - pd.Timedelta(days=1))
    latest_complete_week_start = week_start_monday(latest_complete_week_end)

    current_week_anchor = last_friday(as_of_dt_naive)
    current_week_start = week_start_monday(current_week_anchor)
    current_day_end = pd.Timestamp(as_of_dt_naive).normalize()

    rows: List[Dict] = []

    for tk in tickers:
        cat, desc = etf_info.get(tk, ("", ""))
        px = price_map.get(tk, pd.DataFrame()).copy()

        lookback_px = pd.DataFrame()
        if px is not None and not px.empty:
            lookback_px = px.loc[px.index >= cutoff].copy()

        full_daily_proxy = (
            compute_money_flow_proxy(px)
            if px is not None and not px.empty
            else pd.Series(dtype="float64")
        )

        lookback_proxy = np.nan
        pressure_score = np.nan
        ret = np.nan
        last_price = np.nan
        adv = np.nan

        if not lookback_px.empty:
            lookback_daily_proxy = compute_money_flow_proxy(lookback_px)
            lookback_proxy = float(lookback_daily_proxy.sum()) if not lookback_daily_proxy.empty else np.nan
            pressure_score = compute_pressure_score(lookback_px)

            close = pd.to_numeric(lookback_px["Close"], errors="coerce").dropna()
            if len(close) >= 2 and close.iloc[0] != 0:
                ret = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
            if len(close) >= 1:
                last_price = float(close.iloc[-1])

            adv_series = (
                pd.to_numeric(lookback_px["Close"], errors="coerce")
                * pd.to_numeric(lookback_px["Volume"], errors="coerce").fillna(0.0)
            ).replace([np.inf, -np.inf], np.nan)
            if adv_series.notna().any():
                adv = float(adv_series.mean())

        latest_complete_week_proxy = compute_window_sum(
            full_daily_proxy,
            latest_complete_week_start,
            latest_complete_week_end,
        )

        week_to_date_proxy = compute_window_sum(
            full_daily_proxy,
            current_week_start,
            current_day_end,
        )

        rows.append(
            {
                "Ticker": tk,
                "Label": f"{cat} ({tk})",
                "Category": cat,
                "Description": desc,
                "Last Price": last_price,
                f"{period_label} Flow Pressure": lookback_proxy,
                "Latest Complete Week": latest_complete_week_proxy,
                "Week to Date": week_to_date_proxy,
                "Pressure Score": pressure_score,
                f"{period_label} Return %": ret,
                "Avg Daily Dollar Vol": adv,
                "Data Status": "OK" if not px.empty else "Missing",
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# HIGH-RES IMAGE RENDER
# =========================================================
def render_matplotlib_high_res(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=280,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="white",
    )
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# HEADER
# =========================================================
st.title("ETF Net Flows")

# =========================================================
# DATA BUILD
# =========================================================
df = build_table(etf_tickers, period_days, as_of_date, as_of_dt_naive)

# =========================================================
# VIEW SELECTOR FOR CHART
# =========================================================
bar_view = st.radio(
    label="",
    options=[f"{period_label} Flow Pressure", "Latest Complete Week", "Week to Date"],
    horizontal=True,
    key="bar_view",
    label_visibility="collapsed",
)

# =========================================================
# SUMMARY
# =========================================================
missing_count = int((df["Data Status"] == "Missing").sum())
ok_count = int((df["Data Status"] == "OK").sum())

net_lookback = pd.to_numeric(df[f"{period_label} Flow Pressure"], errors="coerce").sum(min_count=1)
net_lcw = pd.to_numeric(df["Latest Complete Week"], errors="coerce").sum(min_count=1)
net_wtd = pd.to_numeric(df["Week to Date"], errors="coerce").sum(min_count=1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickers Listed", f"{len(df)}")
c2.metric("Data Returned", f"{ok_count}")
c3.metric("Missing", f"{missing_count}")
c4.metric(f"{period_label} Net", fmt_compact_cur(net_lookback) if pd.notna(net_lookback) else "")
c5.metric("WTD Net", fmt_compact_cur(net_wtd) if pd.notna(net_wtd) else "")

# =========================================================
# BAR CHART
# =========================================================
st.markdown("---")

chart_df = df[["Ticker", "Label", bar_view, "Data Status"]].copy()
chart_df[bar_view] = pd.to_numeric(chart_df[bar_view], errors="coerce")
chart_df = chart_df.dropna(subset=[bar_view]).sort_values(bar_view, ascending=False)

if chart_df.empty:
    st.info("No chartable values available for this view.")
else:
    vals = chart_df[bar_view].fillna(0.0)
    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.055 * span

    n = len(chart_df)

    # Harder packing: less height per row, taller bars, smaller gaps
    fig_h = max(6.9, min(15.2, 0.165 * n + 1.2))
    bar_height = 0.88 if n >= 60 else 0.84
    y_font = 7.1 if n >= 70 else 7.8
    value_font = 6.8 if n >= 70 else 7.5

    colors = ["#2e7d32" if v > 0 else "#c0392b" if v < 0 else "#808080" for v in vals]

    fig, ax = plt.subplots(figsize=(16.2, fig_h), dpi=220)

    bars = ax.barh(
        chart_df["Label"],
        vals,
        color=colors,
        alpha=0.97,
        height=bar_height,
        linewidth=0,
    )

    ax.set_xlabel("Estimated Flow Pressure ($)", fontsize=9)
    ax.set_title(bar_view, fontsize=10.5, pad=5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))
    ax.tick_params(axis="y", labelsize=y_font, pad=0.6, length=0)
    ax.tick_params(axis="x", labelsize=8.2)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x_min - pad, x_max + pad)

    text_pad = 0.0065 * span
    for bar, raw in zip(bars, vals):
        label_txt = fmt_compact_cur(raw) if pd.notna(raw) else "$0"
        x = bar.get_width()

        if raw > 0:
            x_text, ha = x + text_pad, "left"
        elif raw < 0:
            x_text, ha = x - text_pad, "right"
        else:
            x_text, ha = 0.0, "center"

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            label_txt,
            va="center",
            ha=ha,
            fontsize=value_font,
            color="#333333",
            clip_on=True,
        )

    fig.subplots_adjust(left=0.26, right=0.985, top=0.94, bottom=0.085)

    chart_png = render_matplotlib_high_res(fig)
    st.image(chart_png, use_container_width=True)
    plt.close(fig)

# =========================================================
# TABLE PREP
# =========================================================
display_df = df[
    [
        "Ticker",
        "Category",
        "Description",
        "Last Price",
        f"{period_label} Flow Pressure",
        "Latest Complete Week",
        "Week to Date",
        "Pressure Score",
        f"{period_label} Return %",
        "Avg Daily Dollar Vol",
        "Data Status",
    ]
].copy()

display_df["Last Price"] = display_df["Last Price"].apply(fmt_price)
display_df[f"{period_label} Flow Pressure"] = display_df[f"{period_label} Flow Pressure"].apply(fmt_compact_cur)
display_df["Latest Complete Week"] = display_df["Latest Complete Week"].apply(fmt_compact_cur)
display_df["Week to Date"] = display_df["Week to Date"].apply(fmt_compact_cur)
display_df["Pressure Score"] = display_df["Pressure Score"].apply(fmt_score)
display_df[f"{period_label} Return %"] = display_df[f"{period_label} Return %"].apply(fmt_pct)
display_df["Avg Daily Dollar Vol"] = display_df["Avg Daily Dollar Vol"].apply(fmt_compact_cur)

# =========================================================
# OUTPUT TABLE
# =========================================================
st.markdown("---")
st.subheader("Full Coverage Table")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=900,
)

# =========================================================
# FOOTNOTE
# =========================================================
st.caption(
    f"Last refresh: {as_of_dt_naive} | Source: Yahoo Finance OHLCV via yfinance | "
    f"Method: price-volume flow-pressure proxy using public OHLCV data | "
    f"Every original ticker remains in the table even if chart data are missing | "
    f"© 2026 AD Fund Management LP"
)
