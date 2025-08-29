import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("default")

# -------------- Page config --------------
st.set_page_config(layout="wide", page_title="Ratio Charts")
st.title("Ratio Charts")

# -------------- Sidebar --------------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "Visualizes regime shifts via ratios.\n\n"
    "- Cyclicals vs Defensives (Eq Wt) normalized to base date\n"
    "- Presets: SMH/IGV, QQQ/IWM, HYG/LQD, HYG/IEF\n"
    "- Custom ratio between any two tickers"
)

st.sidebar.header("Look back")
SPANS = {
    "3 Months": 90,
    "6 Months": 180,
    "9 Months": 270,
    "YTD": None,
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5
}
span_key = st.sidebar.selectbox("", list(SPANS.keys()), index=list(SPANS.keys()).index("5 Years"))

st.sidebar.markdown("---")
st.sidebar.header("Options")
use_log = st.sidebar.checkbox("Log scale", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
custom_t1 = st.sidebar.text_input("Ticker 1", "NVDA").strip().upper()
custom_t2 = st.sidebar.text_input("Ticker 2", "SMH").strip().upper()

# -------------- Dates --------------
now = pd.Timestamp(datetime.today().date())
hist_start = now - timedelta(days=365 * 15)

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_start = pd.Timestamp(now - timedelta(days=SPANS[span_key]))

# Add lookback pad for indicators (RSI 14, MA 200)
INDICATOR_PAD_DAYS = 260
fetch_start = min(hist_start, disp_start - timedelta(days=INDICATOR_PAD_DAYS))
fetch_end = now + timedelta(days=1)  # yfinance end is exclusive

# -------------- Definitions --------------
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
PRESETS    = [("SMH", "IGV"), ("QQQ", "IWM"), ("HYG", "LQD"), ("HYG", "IEF")]
STATIC     = sorted(set(CYCLICALS + DEFENSIVES + [t for a, b in PRESETS for t in (a, b)]))

# -------------- Helpers --------------
def first_on_or_after_index(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    if len(idx) == 0:
        return None
    i = idx.searchsorted(ts, side="left")
    if i >= len(idx):
        return None
    return idx[i]

def last_on_or_before_value(s: pd.Series, ts: pd.Timestamp) -> float | None:
    if s.empty:
        return None
    i = s.index.searchsorted(ts, side="right") - 1
    if i < 0:
        return None
    v = float(s.iloc[i])
    return v if np.isfinite(v) else None

def rsi_wilder(s: pd.Series, window=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    ma_dn = dn.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def compute_cumrets(df: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + df.pct_change()).cumprod()

def compute_ratio_by_base(s1: pd.Series, s2: pd.Series, base_ts: pd.Timestamp, base=100.0) -> pd.Series:
    s1, s2 = s1.align(s2, join="inner")
    s1 = s1.replace([np.inf, -np.inf], np.nan).dropna()
    s2 = s2.replace([np.inf, -np.inf], np.nan).dropna()
    if s1.empty or s2.empty:
        return pd.Series(dtype=float)

    b1 = last_on_or_before_value(s1, base_ts)
    b2 = last_on_or_before_value(s2, base_ts)
    if b1 is None or b2 is None or not np.isfinite(b1) or not np.isfinite(b2) or b2 == 0:
        return pd.Series(dtype=float)

    n1 = s1 / b1 * base
    n2 = s2 / b2 * base
    ratio = n1 / n2
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return ratio

def plot_ratio(ratio: pd.Series, title: str, ylab: str, use_log_scale: bool = False):
    ratio = ratio.dropna()
    if ratio.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig, None

    # indicators across the full series then cut to view
    ma50  = ratio.rolling(50, min_periods=1).mean()
    ma200 = ratio.rolling(200, min_periods=1).mean()
    rsi = rsi_wilder(ratio)

    view = ratio.loc[disp_start:]
    ma50v = ma50.loc[disp_start:]
    ma200v = ma200.loc[disp_start:]
    rsiv = rsi.loc[disp_start:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 4),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    ax1.plot(view.index, view.values, lw=1.2, label=title)
    ax1.plot(ma50v.index, ma50v.values, lw=1.0, label="50 DMA")
    ax1.plot(ma200v.index, ma200v.values, lw=1.0, label="200 DMA")

    if use_log_scale:
        ax1.set_yscale("log")

    x_min = view.index.min()
    x_max = now
    ax1.set_xlim(left=x_min, right=x_max)
    ax1.margins(x=0)

    y_all = pd.concat([view, ma50v, ma200v]).replace([np.inf, -np.inf], np.nan).dropna()
    if not y_all.empty and not use_log_scale:
        mn, mx = y_all.min(), y_all.max()
        pad = (mx - mn) * 0.05 if mx != mn else (abs(mn) * 0.05 if mn != 0 else 1.0)
        ax1.set_ylim(mn - pad, mx + pad)

    ax1.set_ylabel(ylab)
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(rsiv.index, rsiv.values, lw=1.0)
    ax2.axhline(70, ls=":", lw=1)
    ax2.axhline(30, ls=":", lw=1)
    ax2.set_xlim(left=x_min, right=x_max)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # prepare CSV for download
    out = pd.DataFrame({"ratio": ratio}).loc[disp_start:]
    return fig, out

def to_download(name: str, df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode()

# -------------- Data fetch --------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]

    def _normalize(df, tlist):
        if df is None or len(tlist) == 0:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            cols = {}
            for t in tlist:
                try:
                    s = df[(t, "Close")].dropna()
                    if not s.empty:
                        cols[t] = s
                except Exception:
                    continue
            out = pd.DataFrame(cols)
        else:
            # single ticker shape
            if "Close" in df.columns:
                out = pd.DataFrame({tlist[0]: df["Close"].dropna()})
            else:
                # sometimes yf returns a Series
                if isinstance(df, pd.Series):
                    out = df.to_frame(name=tlist[0]).dropna()
                else:
                    out = pd.DataFrame()
        return out.sort_index().ffill().dropna(how="all")

    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, group_by="ticker", threads=True
        )
        out = _normalize(raw, tickers)
        if not out.empty:
            return out
    except Exception:
        pass

    # fallback with long period
    try:
        raw2 = yf.download(
            tickers, period="15y",
            auto_adjust=True, progress=False, group_by="ticker", threads=True
        )
        out2 = _normalize(raw2, tickers)
        return out2
    except Exception:
        return pd.DataFrame()

# -------------- Static block: Cyclicals vs Defensives --------------
closes_static = fetch_closes(STATIC, fetch_start, fetch_end)
if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

cumrets = compute_cumrets(closes_static)

# Equal weight baskets of cumulative returns
eq_cyc = cumrets[CYCLICALS].mean(axis=1, skipna=True)
eq_def = cumrets[DEFENSIVES].mean(axis=1, skipna=True)

ratio_cd = compute_ratio_by_base(eq_cyc, eq_def, base_ts=disp_start, base=100.0)
fig, df_download = plot_ratio(ratio_cd, "Cyclical vs Defensive (Eq Wt)", "Ratio", use_log_scale=use_log)
st.pyplot(fig, use_container_width=True)
if df_download is not None and not df_download.empty:
    st.download_button("Download Cyc/Def CSV", data=to_download("cyc_def.csv", df_download), file_name="cyc_def.csv", mime="text/csv")

st.markdown("---")

# -------------- Preset ratios --------------
for a, b in PRESETS:
    if a in cumrets.columns and b in cumrets.columns:
        r = compute_ratio_by_base(cumrets[a], cumrets[b], base_ts=disp_start, base=100.0)
        fig, dd = plot_ratio(r, f"{a}/{b}", f"{a}/{b}", use_log_scale=use_log)
        st.pyplot(fig, use_container_width=True)
        if dd is not None and not dd.empty:
            st.download_button(f"Download {a}_{b} CSV", data=to_download(f"{a}_{b}.csv", dd), file_name=f"{a}_{b}.csv", mime="text/csv")
        st.markdown('---')

# -------------- Custom ratio --------------
if custom_t1 and custom_t2:
    tickers = [custom_t1, custom_t2]
    closes_c = fetch_closes(tickers, fetch_start, fetch_end)
    if not closes_c.empty and all(t in closes_c.columns for t in tickers):
        cum_c = compute_cumrets(closes_c)
        r_c = compute_ratio_by_base(cum_c[custom_t1], cum_c[custom_t2], base_ts=disp_start, base=100.0)
        fig, dd = plot_ratio(r_c, f"{custom_t1}/{custom_t2}", f"{custom_t1}/{custom_t2}", use_log_scale=use_log)
        st.pyplot(fig, use_container_width=True)
        if dd is not None and not dd.empty:
            st.download_button(f"Download {custom_t1}_{custom_t2} CSV", data=to_download(f"{custom_t1}_{custom_t2}.csv", dd), file_name=f"{custom_t1}_{custom_t2}.csv", mime="text/csv")
    else:
        st.warning(f"Data not available for {custom_t1}/{custom_t2}.")
