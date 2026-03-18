import time
from datetime import datetime, date
from typing import Dict, Tuple

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
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    h1, h2, h3 {
        letter-spacing: 0.1px;
        font-weight: 650;
    }
    .stMetric {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .adfm-card {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .adfm-muted {
        color: #666;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
    padding = 15
    return (as_of - pd.Timedelta(days=days + padding)).date()


def last_friday(on_or_before: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(on_or_before).normalize()
    wd = ts.weekday()
    days_back = wd - 4 if wd >= 4 else wd + 3
    return ts - pd.Timedelta(days=int(days_back))


def week_start_monday(week_ending_friday: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(week_ending_friday).normalize() - pd.tseries.offsets.BDay(4)).normalize()


# =========================================================
# SIDEBAR
# =========================================================
as_of_dt = now_et()
as_of_date = as_of_dt.date()
as_of_dt_naive = as_naive_ts(as_of_dt)

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": ytd_days(as_of_dt),
}

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard tracks ETF flow pressure across major equity, rates, credit, commodity, FX, and crypto sleeves using public Yahoo Finance OHLCV data.

        It does not rely on Yahoo shares outstanding history, which has become inconsistent for ETFs and is the main reason many older ETF flow scripts stopped working.

        The output here is a flow-pressure proxy, built from money flow mechanics using price and volume. It is useful for relative ranking, trend detection, and tape reading, but it is not the same thing as official issuer-reported creations and redemptions.
        """
    )
    st.markdown("---")
    period_label = st.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
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
    "ETHA": ("Ethereum", "Spot Ethereum ETF"),
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
        return f"${x/1e12:,.2f}T"
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.2f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


def fmt_num(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):,.2f}"


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

    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
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
        pd.to_numeric(df["Close"], errors="coerce") *
        pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    denom = float(traded_value.sum())
    if denom <= 0:
        return np.nan

    return float(mfv.sum() / denom)


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

    rows = []
    for tk in tickers:
        px = price_map.get(tk, pd.DataFrame()).copy()
        if not px.empty:
            px = px.loc[px.index >= cutoff]

        if px.empty:
            rows.append(
                {
                    "Ticker": tk,
                    "Category": etf_info.get(tk, ("", ""))[0],
                    "Description": etf_info.get(tk, ("", ""))[1],
                    "Proxy Flow ($)": np.nan,
                    "Pressure Score": np.nan,
                    "Last Price": np.nan,
                    "Return %": np.nan,
                    "Avg Daily Dollar Vol": np.nan,
                    "_daily_proxy": pd.Series(dtype="float64"),
                }
            )
            continue

        daily_proxy = compute_money_flow_proxy(px)
        proxy_flow = float(daily_proxy.sum()) if not daily_proxy.empty else np.nan
        pressure_score = compute_pressure_score(px)

        close = pd.to_numeric(px["Close"], errors="coerce").dropna()
        ret = np.nan
        if len(close) >= 2 and close.iloc[0] != 0:
            ret = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)

        adv = (
            pd.to_numeric(px["Close"], errors="coerce") *
            pd.to_numeric(px["Volume"], errors="coerce").fillna(0.0)
        ).replace([np.inf, -np.inf], np.nan)

        rows.append(
            {
                "Ticker": tk,
                "Category": etf_info.get(tk, ("", ""))[0],
                "Description": etf_info.get(tk, ("", ""))[1],
                "Proxy Flow ($)": proxy_flow,
                "Pressure Score": pressure_score,
                "Last Price": float(close.iloc[-1]) if not close.empty else np.nan,
                "Return %": ret,
                "Avg Daily Dollar Vol": float(adv.mean()) if adv.notna().any() else np.nan,
                "_daily_proxy": daily_proxy,
            }
        )

    df = pd.DataFrame(rows)
    df["Label"] = df["Category"] + " (" + df["Ticker"] + ")"
    return df


def compute_week_view_value_from_daily(daily: pd.Series, view: str, as_of_ts: pd.Timestamp) -> float:
    if daily is None or daily.empty:
        return np.nan

    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    this_week_end = last_friday(as_of_ts)
    this_week_start = week_start_monday(this_week_end)

    if view == "Week to date":
        start = this_week_start
        end = pd.Timestamp(as_of_ts).normalize()
        return float(daily.loc[(daily.index >= start) & (daily.index <= end)].sum())

    last_week_end = this_week_end - pd.Timedelta(days=7)
    last_week_start = week_start_monday(last_week_end)
    return float(daily.loc[(daily.index >= last_week_start) & (daily.index <= last_week_end)].sum())


# =========================================================
# BUILD DATA
# =========================================================
st.title("ETF Net Flows")
st.markdown(
    f"""
    <div class="adfm-card">
        <div><strong>Selected window:</strong> {period_label}</div>
        <div class="adfm-muted">This version plots every ticker you listed. Nothing is trimmed, sampled, or hidden.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = build_table(etf_tickers, period_days, as_of_date, as_of_dt_naive)

bar_view = st.radio(
    label="",
    options=["Lookback total", "Last complete week", "Week to date"],
    horizontal=True,
    key="bar_view",
    label_visibility="collapsed",
)

as_of_ts = pd.Timestamp(as_of_dt_naive).normalize()

chart_rows = []
for _, row in df.iterrows():
    daily = row.get("_daily_proxy", pd.Series(dtype="float64"))

    if bar_view == "Lookback total":
        value = row["Proxy Flow ($)"]
    else:
        value = compute_week_view_value_from_daily(daily, bar_view, as_of_ts)

    chart_rows.append(
        {
            "Ticker": row["Ticker"],
            "Label": row["Label"],
            "Value": value,
            "Pressure Score": row["Pressure Score"],
        }
    )

chart_df = pd.DataFrame(chart_rows)
chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
chart_df = chart_df.dropna(subset=["Value"]).sort_values("Value", ascending=False)

if chart_df.empty:
    st.info("No values available for this view.")
else:
    vals = chart_df["Value"].fillna(0.0)
    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.08 * span

    n = len(chart_df)
    fig_h = max(10.0, 0.34 * n + 2.5)

    colors = ["#1f7a1f" if v > 0 else "#b22222" if v < 0 else "#808080" for v in vals]

    fig, ax = plt.subplots(figsize=(16, fig_h))
    bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.92, height=0.82)

    ax.set_xlabel("Estimated Net Flow Pressure ($)")
    ax.set_title(f"ETF Flow Pressure Proxy | {bar_view} | {period_label}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x_min - pad, x_max + pad)

    text_pad = 0.012 * span
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
            fontsize=9,
            color="black",
            clip_on=True,
        )

    fig.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close(fig)

# =========================================================
# SUMMARY METRICS
# =========================================================
valid = df.dropna(subset=["Proxy Flow ($)"]).copy()

if not valid.empty:
    total_proxy = float(valid["Proxy Flow ($)"].sum())
    pos_count = int((valid["Proxy Flow ($)"] > 0).sum())
    neg_count = int((valid["Proxy Flow ($)"] < 0).sum())
    median_score = float(valid["Pressure Score"].median()) if valid["Pressure Score"].notna().any() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Flow Pressure", fmt_compact_cur(total_proxy))
    c2.metric("Positive Sleeves", f"{pos_count}")
    c3.metric("Negative Sleeves", f"{neg_count}")
    c4.metric("Median Pressure Score", fmt_num(median_score))

# =========================================================
# TOP INFLOWS / OUTFLOWS
# =========================================================
st.markdown("---")
st.markdown("#### Top Inflows and Outflows")

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Value"] = valid["Proxy Flow ($)"].apply(fmt_compact_cur)
    show_cols = ["Value", "Return %", "Avg Daily Dollar Vol"]
    valid["Return %"] = valid["Return %"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")
    valid["Avg Daily Dollar Vol"] = valid["Avg Daily Dollar Vol"].apply(fmt_compact_cur)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top Inflows**")
        inflows = (
            valid[valid["Proxy Flow ($)"] > 0]
            .nlargest(5, "Proxy Flow ($)")
            .set_index("Label")[show_cols]
        )
        st.table(inflows)

    with col2:
        st.write("**Top Outflows**")
        outflows = (
            valid[valid["Proxy Flow ($)"] < 0]
            .nsmallest(5, "Proxy Flow ($)")
            .set_index("Label")[show_cols]
        )
        st.table(outflows)

# =========================================================
# DETAIL TABLE
# =========================================================
st.markdown("---")
st.markdown("#### Full Coverage Table")

table_df = valid[
    [
        "Ticker",
        "Category",
        "Description",
        "Proxy Flow ($)",
        "Pressure Score",
        "Return %",
        "Last Price",
        "Avg Daily Dollar Vol",
    ]
].copy()

table_df["Proxy Flow ($)"] = table_df["Proxy Flow ($)"].apply(fmt_compact_cur)
table_df["Pressure Score"] = table_df["Pressure Score"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
table_df["Return %"] = table_df["Return %"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")
table_df["Last Price"] = table_df["Last Price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
table_df["Avg Daily Dollar Vol"] = table_df["Avg Daily Dollar Vol"].apply(fmt_compact_cur)

st.dataframe(table_df, use_container_width=True, hide_index=True)

# =========================================================
# FOOTNOTE
# =========================================================
st.caption(
    f"Last refresh: {as_of_dt_naive} | Source: Yahoo Finance OHLCV via yfinance | "
    f"Method: money-flow proxy based on price and volume, not issuer-reported ETF creations/redemptions | © 2026 AD Fund Management LP"
)
