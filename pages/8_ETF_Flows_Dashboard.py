import time
from datetime import datetime, date
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Plotly (used only for Section 2 to match your screenshot UI)
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Optional: click events on Plotly charts
try:
    from streamlit_plotly_events import plotly_events  # pip install streamlit-plotly-events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="ETF Net Flows", layout="wide")

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Estimate primary market money moving in and out of key ETFs over a chosen window.

        Methodology
        • First pass: primary flows from Δ shares outstanding × daily close (close as NAV proxy)
        • Fallback: CMF-style turnover proxy when historical shares data are missing or incomplete
        • Sign convention: positive = net creations (inflows), negative = net redemptions (outflows)

        Notes
        • `get_shares_full` from Yahoo Finance is not available for every ETF or every date
        • Close is used as a proxy for daily NAV where needed
        • All flows are aggregated over the selected lookback window and shown in USD
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Lookback")

TZ = pytz.timezone("US/Eastern")


def now_et() -> datetime:
    return datetime.now(TZ)


def as_naive_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts.replace(tzinfo=None))


def ytd_days(as_of: datetime) -> int:
    start_ytd = TZ.localize(datetime(as_of.year, 1, 1))
    return (as_of - start_ytd).days


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
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
period_days = int(lookback_dict[period_label])

# --------------------------- ETF COVERAGE ---------------------------
# Added: SPY, QQQ, IWM, EEM to match your screenshot set.
# Also tightened some category labels to resemble the UI in the image.
etf_info = {
    "SPY": ("US Large Cap", "S&P 500"),
    "QQQ": ("US Tech", "Nasdaq 100"),
    "IWM": ("US Small Cap", "Russell 2000"),
    "EEM": ("EM Equity", "Emerging markets equity"),

    "VTI": ("US Total Market", "Total US equity market"),
    "VUG": ("US Growth", "Large-cap growth"),
    "VTV": ("US Value", "Large-cap value"),
    "MTUM": ("US Momentum", "Momentum factor"),
    "QUAL": ("US Quality", "Quality factor"),
    "USMV": ("US Min Vol", "Minimum volatility"),
    "SCHD": ("US Dividends", "Dividend growth and yield"),

    "XLB": ("Materials", "S&P 500 Materials"),
    "XLC": ("Comm Services", "S&P 500 Communication Services"),
    "XLE": ("Energy", "S&P 500 Energy"),
    "XLF": ("Financials", "S&P 500 Financials"),
    "XLI": ("Industrials", "S&P 500 Industrials"),
    "XLK": ("Technology", "S&P 500 Technology"),
    "XLP": ("Staples", "S&P 500 Consumer Staples"),
    "XLRE": ("Real Estate", "S&P 500 Real Estate"),
    "XLU": ("Utilities", "S&P 500 Utilities"),
    "XLV": ("Healthcare", "S&P 500 Healthcare"),
    "XLY": ("Discretionary", "S&P 500 Consumer Discretionary"),

    "SMH": ("Semis", "Global semiconductor equities"),
    "IGV": ("Software", "US application software"),
    "SKYY": ("Cloud", "Cloud infrastructure and services"),

    "VT": ("Global Equity", "Total world equity market"),
    "ACWI": ("Global Equity", "All-country world equity"),
    "VEA": ("Dev ex-US", "Developed markets ex-US"),
    "VWO": ("EM Equity", "Emerging markets equity"),
    "EXUS": ("Non-US Equity", "Global equity ex-US"),

    "EWG": ("Germany", "Germany equities"),
    "EWQ": ("France", "France equities"),
    "EWU": ("UK", "UK equities"),
    "EWI": ("Italy", "Italy equities"),
    "EWP": ("Spain", "Spain equities"),
    "EWL": ("Switzerland", "Switzerland equities"),
    "EWN": ("Netherlands", "Netherlands equities"),
    "EWD": ("Sweden", "Sweden equities"),
    "EWO": ("Austria", "Austria equities"),
    "EWK": ("Belgium", "Belgium equities"),
    "EWJ": ("Japan", "Japan equities"),
    "EWY": ("Korea", "Korea equities"),
    "ASHR": ("China A", "Onshore China equities"),
    "FXI": ("China", "China offshore large caps"),
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

    "SGOV": ("T-Bills", "0-3 month Treasuries"),
    "SHY": ("UST 1-3y", "Short-term Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate Treasuries"),
    "TLT": ("Long Bonds", "UST 20y+"),
    "TIP": ("TIPS", "Inflation-linked Treasuries"),

    "LQD": ("IG Credit", "Investment-grade corporates"),
    "VCIT": ("IG Credit", "Intermediate IG corporates"),
    "HYG": ("High Yield", "High-yield credit"),
    "BKLN": ("Float Credit", "Senior loans"),
    "EMB": ("EM Debt", "USD EM sovereign debt"),
    "BND": ("Agg Bonds", "Total US bond market"),

    "GLD": ("Gold", "Gold bullion"),
    "SLV": ("Silver", "Silver bullion"),
    "CPER": ("Copper", "Industrial copper"),
    "USO": ("Crude Oil", "WTI crude oil"),
    "DBC": ("Commodities", "Commodity basket"),
    "PDBC": ("Commodities", "Rules-based commodities"),
    "URA": ("Uranium", "Nuclear fuel cycle"),

    "VXX": ("Volatility", "Front-end VIX futures"),
    "UUP": ("USD", "US Dollar Index"),
    "FXE": ("EURUSD", "Euro vs USD"),
    "FXY": ("USDJPY", "Japanese Yen"),
    "FXF": ("CHFUSD", "Swiss franc"),
    "CEW": ("EM FX", "Emerging market currencies"),

    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETH": ("Ethereum", "Spot Ethereum ETF"),
}

etf_tickers = list(etf_info.keys())

# --------------------------- HELPERS ---------------------------
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:,.0f}T"
    if ax >= 1e9:
        return f"${x/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


def fmt_compact_flow(x) -> str:
    """Heatmap/line labels like the screenshot: no $, allow 1dp in billions."""
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    sgn = "-" if x < 0 else ""
    if ax >= 1e9:
        return f"{sgn}{ax/1e9:.1f}B"
    if ax >= 1e6:
        return f"{sgn}{ax/1e6:.0f}M"
    if ax >= 1e3:
        return f"{sgn}{ax/1e3:.0f}K"
    return f"{x:.0f}"


def axis_fmt(x, _pos=None) -> str:
    return fmt_compact_cur(x)


def retry(n=2, delay=0.5):
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


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx
    return df.dropna(subset=["Close"])


def _calc_start_date(days: int, as_of: datetime) -> date:
    padding = 7
    return (as_of - pd.Timedelta(days=days + padding)).date()


def _quality_flag(method: str, shares_obs_window: int, nonzero_delta_days: int) -> str:
    if method == "cmf_proxy":
        return "Proxy"
    if shares_obs_window >= 6 and nonzero_delta_days >= 2:
        return "High"
    if shares_obs_window >= 2:
        return "Medium"
    return "Low"


# --------------------------- DATA ---------------------------
@retry()
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prices(tickers: Tuple[str, ...], start_date: date, as_of_date: date) -> Dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=as_of_date + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[tk].copy()
            else:
                df = data.copy()
        except Exception:
            df = pd.DataFrame()
        out[tk] = _normalize_ohlcv(df)
    return out


@retry()
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_one_shares_series_cached(ticker: str, start_date: date) -> pd.Series:
    t = yf.Ticker(ticker)
    try:
        s = t.get_shares_full(start=start_date)
    except Exception:
        s = None
    if s is None or (isinstance(s, pd.DataFrame) and s.empty):
        return pd.Series(dtype="float64")
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")
    s = pd.to_numeric(s, errors="coerce")
    idx = pd.to_datetime(s.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    s.index = idx
    return s.dropna()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_shares_series_parallel(
    tickers: Tuple[str, ...],
    start_date: date,
    timeout_sec: float = 10.0,
) -> Dict[str, pd.Series]:
    results: Dict[str, pd.Series] = {tk: pd.Series(dtype="float64") for tk in tickers}
    max_workers = min(10, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one_shares_series_cached, tk, start_date): tk for tk in tickers}
        start_time = time.time()
        for fut in as_completed(futures):
            tk = futures[fut]
            remaining = max(0.01, timeout_sec - (time.time() - start_time))
            try:
                results[tk] = fut.result(timeout=remaining)
            except FuturesTimeoutError:
                results[tk] = pd.Series(dtype="float64")
            except Exception:
                results[tk] = pd.Series(dtype="float64")
            if (time.time() - start_time) >= timeout_sec:
                break
    return results


def compute_daily_flows_with_stats(
    close: pd.Series,
    shares: pd.Series,
    window_index: pd.DatetimeIndex
) -> Tuple[pd.Series, Dict[str, float]]:
    stats = {"shares_obs_window": 0, "nonzero_delta_days": 0}
    if close is None or shares is None or close.empty or shares.empty or window_index is None or len(window_index) == 0:
        return pd.Series(dtype="float64"), stats

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index, errors="coerce")
    try:
        close.index = close.index.tz_localize(None)
    except Exception:
        pass

    sh = pd.to_numeric(shares, errors="coerce").dropna()
    sh.index = pd.to_datetime(sh.index, errors="coerce")
    try:
        sh.index = sh.index.tz_localize(None)
    except Exception:
        pass

    sh_in = sh.loc[(sh.index >= window_index.min()) & (sh.index <= window_index.max())]
    stats["shares_obs_window"] = int(sh_in.shape[0])

    idx = pd.DatetimeIndex(window_index)
    sh_daily = sh.sort_index().resample("B").ffill().reindex(idx).ffill()
    delta_shares = sh_daily.diff().fillna(0.0)
    stats["nonzero_delta_days"] = int((delta_shares != 0).sum())

    close_aligned = close.reindex(idx).ffill()
    flows = (delta_shares * close_aligned).astype(float)
    flows = flows.replace([np.inf, -np.inf], np.nan).dropna()
    return flows, stats


def compute_cmf_turnover_proxy(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfm = mfm.fillna(0.0)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return float((mfm * typical * vol).sum())


# --------------------------- BUILD TABLE ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def build_table(tickers: Tuple[str, ...], period_days: int, as_of_date: date, as_of_dt_naive: pd.Timestamp) -> pd.DataFrame:
    as_of_dt_local = as_of_dt_naive.to_pydatetime().replace(tzinfo=None)
    as_of_dt_et = TZ.localize(as_of_dt_local)
    start_date = _calc_start_date(period_days, as_of_dt_et)

    price_map = fetch_prices(tuple(tickers), start_date, as_of_date)
    shares_map = fetch_shares_series_parallel(tuple(tickers), start_date, timeout_sec=10.0)

    cutoff = as_of_dt_naive - pd.Timedelta(days=period_days)

    rows = []
    for tk in tickers:
        px = price_map.get(tk, pd.DataFrame())
        if not px.empty:
            px = px.loc[px.index >= cutoff]

        flow_usd_sum = np.nan
        method = "flows"
        quality = ""
        shares_obs_window = 0
        nonzero_delta_days = 0
        daily_flows_series = pd.Series(dtype="float64")

        if not px.empty:
            window_index = pd.DatetimeIndex(px.index)
            shares = shares_map.get(tk, pd.Series(dtype="float64"))

            if shares is not None and not shares.empty:
                daily_flows, stats = compute_daily_flows_with_stats(px["Close"], shares, window_index)
                shares_obs_window = int(stats.get("shares_obs_window", 0))
                nonzero_delta_days = int(stats.get("nonzero_delta_days", 0))

                if not daily_flows.empty and daily_flows.replace(0.0, np.nan).dropna().size > 0:
                    flow_usd_sum = float(daily_flows.sum())
                    method = "flows"
                    daily_flows_series = daily_flows
                else:
                    flow_usd_sum = compute_cmf_turnover_proxy(px)
                    method = "cmf_proxy"
            else:
                flow_usd_sum = compute_cmf_turnover_proxy(px)
                method = "cmf_proxy"

            quality = _quality_flag(method, shares_obs_window, nonzero_delta_days)

        cat, desc = etf_info.get(tk, ("", ""))
        rows.append({
            "Ticker": tk,
            "Category": cat,
            "Flow ($)": flow_usd_sum,
            "Method": method,
            "Quality": quality,
            "Shares Obs (window)": shares_obs_window,
            "Nonzero Δ days": nonzero_delta_days,
            "Description": desc,
            "_daily_flows": daily_flows_series,
        })

    df = pd.DataFrame(rows)

    def label_for(t: str) -> str:
        cat = etf_info.get(t, ("", ""))[0]
        return f"{cat} ({t})"

    df["Label"] = [label_for(t) for t in df["Ticker"]]
    return df


# --------------------------- PROGRESSION HELPERS ---------------------------
def resample_flows(daily: pd.Series, granularity: str) -> pd.Series:
    if daily is None or daily.empty:
        return pd.Series(dtype="float64")
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    if granularity == "Daily":
        return daily
    if granularity == "Weekly":
        return daily.resample("W-FRI").sum()
    return daily.resample("ME").sum()


def buckets_for_granularity(granularity: str) -> int:
    if granularity == "Daily":
        return 20
    if granularity == "Weekly":
        return 4
    return 6


def bucket_labels(granularity: str, n: int) -> List[str]:
    if granularity == "Daily":
        return [f"D-{k}" for k in range(n, 0, -1)]
    if granularity == "Weekly":
        return [f"Wk-{k}" for k in range(n, 0, -1)]
    return [f"Mo-{k}" for k in range(n, 0, -1)]


def build_progression_matrix(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    rows=tickers, columns=bucket end timestamps, values=flow per bucket.
    Only includes tickers with real daily flow data (Method == 'flows').
    """
    rows = {}
    for _, row in df.iterrows():
        tk = row["Ticker"]
        daily = row.get("_daily_flows", pd.Series(dtype="float64"))
        if isinstance(daily, pd.Series) and (not daily.empty) and row["Method"] == "flows":
            bucketed = resample_flows(daily, granularity)
            bucketed = bucketed.sort_index()
            if not bucketed.empty:
                rows[tk] = bucketed

    if not rows:
        return pd.DataFrame()

    mat = pd.DataFrame(rows).T
    mat = mat.reindex(sorted(mat.columns), axis=1)
    mat = mat.fillna(0.0)
    return mat


# --------------------------- SECTION 2 RENDERERS (Plotly) ---------------------------
def _calc_vmax(vals: np.ndarray) -> float:
    vals = vals.astype(float)
    nonzero = np.abs(vals[vals != 0])
    if nonzero.size == 0:
        return 1.0
    return float(np.nanpercentile(nonzero, 95))


def render_heatmap_plotly(mat: pd.DataFrame, granularity: str, selected_tickers: List[str]) -> "go.Figure":
    n_rows = mat.shape[0]
    n_cols = mat.shape[1]

    tickers = mat.index.tolist()
    col_labels = bucket_labels(granularity, n_cols)

    z = mat.values.astype(float)
    vmax = _calc_vmax(z.flatten())
    zmin, zmax = -vmax, vmax

    # Text labels in cells (compact, like screenshot)
    text = np.empty_like(z, dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            v = z[i, j]
            text[i, j] = "" if abs(v) < 1e-9 else fmt_compact_flow(v)

    y_vals = list(range(n_rows))
    x_vals = list(range(n_cols))

    # Dark theme colorscale: red -> dark -> green
    colorscale = [
        [0.00, "rgba(232,68,90,0.90)"],
        [0.50, "rgba(20,23,32,1.00)"],
        [1.00, "rgba(46,204,138,0.90)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_vals,
            y=y_vals,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            showscale=False,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#e2e6f0"},
            hovertemplate="%{y}<br>%{x}<br>%{z:.3e}<extra></extra>",
        )
    )

    y_ticktext = [f"{etf_info.get(tk, ('', ''))[0]} ({tk})" for tk in tickers]

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_vals,
        ticktext=col_labels,
        tickfont={"color": "#6b7491", "size": 11},
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_vals,
        ticktext=y_ticktext,
        tickfont={"color": "#e2e6f0", "size": 11},
        showgrid=False,
        zeroline=False,
    )

    # Blue row outline for selected tickers
    for tk in selected_tickers:
        if tk in tickers:
            i = tickers.index(tk)
            fig.add_shape(
                type="rect",
                x0=-0.5,
                x1=n_cols - 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line={"color": "#4f7cff", "width": 2},
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141720",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=max(380, 28 * n_rows + 110),
    )
    return fig


def render_cumulative_lines_plotly(df: pd.DataFrame, tickers: List[str], granularity: str) -> "go.Figure":
    palette = ["#4f7cff", "#2ecc8a", "#f5c842", "#e8445a", "#b57bff", "#4ecdc4"]

    n_buckets = buckets_for_granularity(granularity)
    x_labels = bucket_labels(granularity, n_buckets)

    fig = go.Figure()
    plotted = 0

    y_all = []

    for i, tk in enumerate(tickers):
        row = df[df["Ticker"] == tk]
        if row.empty:
            continue
        daily = row.iloc[0].get("_daily_flows", pd.Series(dtype="float64"))
        if not isinstance(daily, pd.Series) or daily.empty:
            continue

        bucketed = resample_flows(daily, granularity).sort_index()
        if bucketed.empty:
            continue

        bucketed = bucketed.iloc[-n_buckets:]
        if bucketed.empty:
            continue

        cumulative = bucketed.cumsum()
        y = cumulative.values.astype(float)
        y_all.append(y)

        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))),
                y=y,
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 7},
                name=tk,
                hovertemplate=f"{tk}<br>%{{y}}<extra></extra>",
            )
        )
        plotted += 1

    if plotted == 0:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Select ETFs with flow data (Method = flows)",
            showarrow=False,
            font={"color": "#6b7491", "size": 12},
        )

    # Compact y-axis tick labels
    if y_all:
        y_concat = np.concatenate(y_all) if len(y_all) > 1 else y_all[0]
        y_min = float(np.nanmin(y_concat))
        y_max = float(np.nanmax(y_concat))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        tickvals = np.linspace(y_min, y_max, 5)
        ticktext = [fmt_compact_flow(v) for v in tickvals]
        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals.tolist(),
            ticktext=ticktext,
            gridcolor="rgba(26,30,42,0.7)",
            zeroline=True,
            zerolinecolor="rgba(46,52,71,1)",
            tickfont={"color": "#6b7491", "size": 11},
        )
    else:
        fig.update_yaxes(
            gridcolor="rgba(26,30,42,0.7)",
            tickfont={"color": "#6b7491", "size": 11},
        )

    # X labels: show fewer ticks on daily for readability
    if granularity == "Daily":
        # map last N points to D-N .. D-1; show every 3rd tick
        n = n_buckets
        tickvals = list(range(n))
        ticktext = x_labels[:n]
        keep = set([0, n - 1] + list(range(3, n, 3)))
        tickvals = [i for i in tickvals if i in keep]
        ticktext = [ticktext[i] for i in tickvals]
    else:
        n = n_buckets
        tickvals = list(range(n))
        ticktext = x_labels[:n]

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont={"color": "#6b7491", "size": 11},
        gridcolor="rgba(26,30,42,0.45)",
        zeroline=False,
    )

    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141720",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0.0,
            "font": {"color": "#e2e6f0", "size": 11},
        },
    )
    return fig


# --------------------------- MAIN ---------------------------
st.title("ETF Net Flows")
st.caption(
    f"Estimated net creations/redemptions over: {period_label}. "
    f"Quality uses shares-observations and nonzero share-change days when shares data exist."
)

df = build_table(tuple(etf_tickers), period_days, as_of_date, as_of_dt_naive)

# ================================================================
# SECTION 1: ORIGINAL BAR CHART (unchanged)
# ================================================================
chart_df = df.dropna(subset=["Flow ($)"]).sort_values("Flow ($)", ascending=False).copy()
vals = pd.to_numeric(chart_df["Flow ($)"], errors="coerce").fillna(0.0)

if vals.empty:
    st.info("No values computed. Try a different period.")
else:
    x_min = float(vals.min())
    x_max = float(vals.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.08 * span

    n = len(chart_df)
    fig_h = max(6.0, min(20.0, 0.30 * n + 2.0))
    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in vals]

    fig, ax = plt.subplots(figsize=(15, fig_h))
    bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.9, height=0.82)

    ax.set_xlabel("Estimated Net Flow ($)")
    ax.set_title(f"ETF Net Flows - {period_label}")
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

    text_pad = 0.012 * (span if span > 0 else 1.0)
    for bar, raw in zip(bars, vals):
        label = fmt_compact_cur(raw) if pd.notna(raw) else "$0"
        x = bar.get_width()
        if raw > 0:
            x_text, ha = x + text_pad, "left"
        elif raw < 0:
            x_text, ha = x - text_pad, "right"
        else:
            x_text, ha = 0.0, "center"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha=ha, fontsize=10, color="black", clip_on=True)

    fig.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
# SECTION 2: HYBRID FLOW PROGRESSION (updated to match screenshot)
# ================================================================
st.markdown("---")

st.markdown(
    """
<style>
.fp-title {
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #6b7491;
  margin: 0 0 6px 0;
}
.fp-rule { height: 1px; background: #252a38; margin: 2px 0 14px 0; }
.fp-sub {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #6b7491;
  margin: 0 0 8px 0;
}
.fp-card {
  background: #141720;
  border: 1px solid #252a38;
  border-radius: 14px;
  padding: 12px 12px 8px 12px;
}
</style>
<div class="fp-title">Flow Progression · Hybrid View</div>
<div class="fp-rule"></div>
""",
    unsafe_allow_html=True,
)

# Granularity toggle (defaults to Weekly like your screenshot)
gran_col, _ = st.columns([2, 6])
with gran_col:
    st.markdown('<div class="fp-sub">Granularity</div>', unsafe_allow_html=True)
    granularity = st.radio(
        "Granularity",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        index=1,
        key="gran_toggle",
        label_visibility="collapsed",
    )

# Build progression matrix (only flows-method tickers)
prog_matrix = build_progression_matrix(df, granularity)

# Limit the matrix to the same fixed buckets as the screenshot-style view
n_buckets = buckets_for_granularity(granularity)
if not prog_matrix.empty:
    prog_matrix = prog_matrix.iloc[:, -n_buckets:].copy()

# Ticker multiselect defaults to top 4 inflow ETFs (flows method only)
flows_tickers = [
    row["Ticker"] for _, row in df.iterrows()
    if row["Method"] == "flows"
    and isinstance(row["_daily_flows"], pd.Series)
    and not row["_daily_flows"].empty
]

if flows_tickers:
    df_flows = df[df["Ticker"].isin(flows_tickers)].dropna(subset=["Flow ($)"]).copy()
    top_default = df_flows.nlargest(4, "Flow ($)")["Ticker"].tolist()
    top_default = top_default[:4] if top_default else flows_tickers[:4]
else:
    top_default = []

if "hybrid_multiselect" not in st.session_state or not isinstance(st.session_state.get("hybrid_multiselect"), list):
    st.session_state["hybrid_multiselect"] = top_default

selected_tickers = st.multiselect(
    "Select ETFs for drill-down (flows data only)",
    options=flows_tickers,
    default=st.session_state["hybrid_multiselect"],
    key="hybrid_multiselect",
)

hm_col, line_col = st.columns([3, 2])

with hm_col:
    st.markdown('<div class="fp-sub">Overview · Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-card">', unsafe_allow_html=True)

    if prog_matrix.empty:
        st.info("No daily flow series available. Shares data needed for progression view.")
    else:
        if HAS_PLOTLY:
            hm_fig = render_heatmap_plotly(prog_matrix, granularity, selected_tickers)

            # Click-to-select behavior (optional, matches screenshot instruction)
            if HAS_PLOTLY_EVENTS:
                clicks = plotly_events(
                    hm_fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=int(max(380, 28 * prog_matrix.shape[0] + 110)),
                    key="heatmap_clicks",
                )
                if clicks:
                    y_idx = clicks[0].get("y", None)
                    if y_idx is not None:
                        try:
                            y_idx = int(y_idx)
                            clicked_tk = prog_matrix.index.tolist()[y_idx]
                            cur = list(st.session_state.get("hybrid_multiselect", []))

                            if clicked_tk in cur:
                                cur = [t for t in cur if t != clicked_tk]
                            else:
                                cur = [clicked_tk] + cur
                                cur = cur[:6]

                            st.session_state["hybrid_multiselect"] = cur
                            st.rerun()
                        except Exception:
                            pass
            else:
                st.plotly_chart(hm_fig, use_container_width=True)

                st.caption(
                    "Tip: install streamlit-plotly-events to enable click-to-select on heatmap rows. "
                    "Multiselect above still controls the drill-down."
                )
        else:
            st.info("Plotly is required for the heatmap UI in Section 2.")
    st.markdown("</div>", unsafe_allow_html=True)

with line_col:
    st.markdown('<div class="fp-sub">Drill-down · Click heatmap row to select</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-card">', unsafe_allow_html=True)

    if not selected_tickers:
        st.info("Select at least one ETF above to view cumulative flows.")
    else:
        if HAS_PLOTLY:
            line_fig = render_cumulative_lines_plotly(df, selected_tickers, granularity)
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.info("Plotly is required for the drill-down line chart UI in Section 2.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# SECTION 3: QUALITY TABLE (original, unchanged)
# ================================================================
st.markdown("---")
st.markdown("#### Coverage and Data Quality")
view = df.copy()
view["Flow"] = pd.to_numeric(view["Flow ($)"], errors="coerce")
view["Flow (fmt)"] = view["Flow"].apply(fmt_compact_cur)
view = view.sort_values(["Quality", "Flow"], ascending=[True, False])

show_cols = [
    "Ticker",
    "Category",
    "Flow (fmt)",
    "Method",
    "Quality",
    "Shares Obs (window)",
    "Nonzero Δ days",
    "Description",
]
st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

# ================================================================
# SECTION 4: TOP INFLOWS / OUTFLOWS (original, unchanged)
# ================================================================
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Value"] = valid["Flow ($)"].apply(fmt_compact_cur)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**")
        st.table(
            valid[valid["Flow ($)"] > 0]
            .nlargest(5, "Flow ($)")
            .set_index("Label")[["Value", "Quality"]]
        )
    with col2:
        st.write("**Top Outflows**")
        st.table(
            valid[valid["Flow ($)"] < 0]
            .nsmallest(5, "Flow ($)")
            .set_index("Label")[["Value", "Quality"]]
        )

st.caption(f"Last refresh: {as_of_dt_naive}  |  © 2026 AD Fund Management LP")
