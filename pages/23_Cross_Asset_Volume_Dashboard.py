from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="Cross-Asset Volume Dashboard", layout="wide")
plt.style.use("default")

# ===== Styling =====
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.10rem;
            padding-bottom: 1.15rem;
            max-width: 1580px;
        }
        .main-title {
            font-size: 2.05rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.10rem;
        }
        .sub-title {
            font-size: 1rem;
            color: #5f6368;
            margin-bottom: 1.00rem;
        }
        .section-title {
            font-size: 1.16rem;
            font-weight: 700;
            margin-top: 0.25rem;
            margin-bottom: 0.60rem;
        }
        .insight-card {
            padding: 0.95rem 1rem;
            border: 1px solid #E8E8E8;
            border-radius: 12px;
            background: #FAFAFA;
            margin-bottom: 0.70rem;
        }
        .muted {
            color: #6b7280;
            font-size: 0.86rem;
        }
        .tiny-note {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #ECECEC;
            border-radius: 12px;
            padding: 0.65rem 0.75rem 0.6rem 0.75rem;
            background: #FBFBFB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Universe =====
ASSET_MAP: Dict[str, List[Tuple[str, str]]] = {
    "Equities": [
        ("SPY", "S&P 500"),
        ("QQQ", "Nasdaq 100"),
        ("IWM", "Russell 2000"),
        ("SMH", "Semiconductors"),
        ("XLF", "Financials"),
        ("XLK", "Technology"),
        ("XLE", "Energy"),
        ("XLI", "Industrials"),
        ("ARKK", "High Beta Growth"),
    ],
    "Credit": [
        ("LQD", "IG Credit"),
        ("HYG", "High Yield"),
        ("JNK", "HY Alt"),
        ("TLT", "Long Duration"),
        ("IEF", "7-10Y Treasuries"),
        ("EMB", "EM Sovereign"),
        ("BKLN", "Senior Loans"),
        ("MBB", "Agency MBS"),
    ],
    "Commodities": [
        ("GLD", "Gold"),
        ("SLV", "Silver"),
        ("USO", "WTI Oil"),
        ("UNG", "Nat Gas"),
        ("CPER", "Copper"),
        ("DBA", "Agriculture"),
        ("DBC", "Broad Basket"),
    ],
    "Currencies": [
        ("UUP", "Dollar Index Proxy"),
        ("FXE", "Euro"),
        ("FXY", "Japanese Yen"),
        ("FXB", "British Pound"),
        ("FXA", "Australian Dollar"),
        ("FXC", "Canadian Dollar"),
        ("CEW", "EM FX Basket"),
        ("CYB", "Chinese Yuan"),
    ],
}


@dataclass
class SignalRow:
    asset_class: str
    ticker: str
    label: str
    close: float
    daily_pct: float
    vol_multiple: float
    vol_z: float
    dollar_vol_vs_avg: float
    breakout_20d: str
    trend_50d: str
    trend_200d: str
    atr_pct: float
    urgency: float
    setup: str
    setup_score: float
    trend_score: float
    participation_score: float
    price_location_score: float
    data_quality: str


def human_num(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"


def safe_zscore(series: pd.Series, lookback: int = 60) -> pd.Series:
    mean = series.rolling(lookback, min_periods=max(20, lookback // 2)).mean()
    std = series.rolling(lookback, min_periods=max(20, lookback // 2)).std().replace(0, np.nan)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def minmax_scale(series: pd.Series, low: float = 1.0, high: float = 5.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(low, index=series.index)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series((low + high) / 2, index=series.index)
    out = low + (s - smin) * (high - low) / (smax - smin)
    return out.clip(low, high)


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_data(tickers: List[str], period: str = "2y") -> tuple[Dict[str, pd.DataFrame], List[str]]:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    if raw is None or len(raw) == 0:
        return out, tickers

    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        tmp = raw.copy()
        tmp.columns = [str(c) for c in tmp.columns]
        tmp = tmp.dropna(how="all")
        if tmp.empty:
            missing.append(tickers[0])
        else:
            out[tickers[0]] = tmp
        return out, missing

    if not isinstance(raw.columns, pd.MultiIndex):
        return out, tickers

    top_level = set(raw.columns.get_level_values(0))
    for ticker in tickers:
        if ticker not in top_level:
            missing.append(ticker)
            continue
        tmp = raw[ticker].copy()
        tmp.columns = [str(c) for c in tmp.columns]
        tmp = tmp.dropna(how="all")
        if tmp.empty:
            missing.append(ticker)
        else:
            out[ticker] = tmp

    return out, missing


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["Open", "High", "Low", "Close", "Volume"]
    d = df[needed].copy()

    for col in needed:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d["Volume"] = d["Volume"].fillna(0.0)

    d["ret_1d"] = d["Close"].pct_change() * 100.0
    d["vol_avg_20"] = d["Volume"].rolling(20, min_periods=10).mean()
    d["vol_multiple"] = np.where(d["vol_avg_20"] > 0, d["Volume"] / d["vol_avg_20"], np.nan)
    d["vol_z"] = safe_zscore(d["Volume"], 60)

    d["dollar_volume"] = d["Close"] * d["Volume"]
    d["dollar_vol_avg_20"] = d["dollar_volume"].rolling(20, min_periods=10).mean()
    d["dollar_vol_vs_avg"] = np.where(d["dollar_vol_avg_20"] > 0, d["dollar_volume"] / d["dollar_vol_avg_20"], np.nan)

    d["ma_20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["ma_50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["ma_200"] = d["Close"].rolling(200, min_periods=150).mean()

    d["hh_20_prev"] = d["High"].shift(1).rolling(20, min_periods=20).max()
    d["ll_20_prev"] = d["Low"].shift(1).rolling(20, min_periods=20).min()

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr_14"] = d["tr"].rolling(14, min_periods=10).mean()
    d["atr_pct"] = np.where(d["Close"] > 0, d["atr_14"] / d["Close"] * 100.0, np.nan)

    d["line_weight"] = minmax_scale(d["vol_multiple"].fillna(1.0), 1.25, 5.25)
    d["line_color_score"] = d["vol_multiple"].clip(lower=0.5, upper=4.0)

    d["distance_vs_20d_high_pct"] = np.where(
        d["hh_20_prev"] > 0,
        (d["Close"] / d["hh_20_prev"] - 1.0) * 100.0,
        np.nan,
    )
    d["distance_vs_20d_low_pct"] = np.where(
        d["ll_20_prev"] > 0,
        (d["Close"] / d["ll_20_prev"] - 1.0) * 100.0,
        np.nan,
    )

    return d.dropna(how="all")


def classify_breakout(latest: pd.Series) -> str:
    if pd.notna(latest["hh_20_prev"]) and latest["Close"] > latest["hh_20_prev"]:
        return "20D Up"
    if pd.notna(latest["ll_20_prev"]) and latest["Close"] < latest["ll_20_prev"]:
        return "20D Down"
    return "Inside"


def classify_setup(latest: pd.Series, trend_50: str, trend_200: str, breakout: str) -> str:
    vol_mult = float(latest["vol_multiple"]) if pd.notna(latest["vol_multiple"]) else np.nan
    day_move = float(latest["ret_1d"]) if pd.notna(latest["ret_1d"]) else np.nan

    if pd.isna(vol_mult) or pd.isna(day_move):
        return "Unclear"

    if breakout == "20D Up" and vol_mult >= 1.6 and trend_50 == "Above":
        return "Bullish expansion"
    if breakout == "20D Down" and vol_mult >= 1.6 and trend_50 == "Below":
        return "Bearish expansion"
    if vol_mult >= 1.8 and abs(day_move) <= 0.75:
        return "Absorption / churn"
    if vol_mult >= 1.35 and trend_50 != trend_200:
        return "Transition regime"
    return "Watch"


def compute_signal_table(raw: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, str]]:
    rows: List[SignalRow] = []
    enriched: Dict[str, pd.DataFrame] = {}
    issues: Dict[str, str] = {}

    for asset_class, items in ASSET_MAP.items():
        for ticker, label in items:
            df = raw.get(ticker)
            if df is None or df.empty:
                issues[ticker] = "Missing download"
                continue

            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                issues[ticker] = "Missing required columns"
                continue

            d = enrich_df(df)
            if len(d) < 80:
                issues[ticker] = "Too little history"
                continue

            latest = d.iloc[-1]
            if pd.isna(latest.get("Close")) or pd.isna(latest.get("ret_1d")):
                issues[ticker] = "Missing latest price fields"
                continue

            breakout = classify_breakout(latest)
            trend_50 = "Above" if pd.notna(latest["ma_50"]) and latest["Close"] > latest["ma_50"] else "Below"
            trend_200 = "Above" if pd.notna(latest["ma_200"]) and latest["Close"] > latest["ma_200"] else "Below"

            participation_score = 0.0
            participation_score += min(max(float(latest.get("vol_multiple", np.nan) or np.nan), 0.0), 4.0) * 1.6 if pd.notna(latest.get("vol_multiple")) else 0.0
            participation_score += min(max(float(latest.get("dollar_vol_vs_avg", np.nan) or np.nan), 0.0), 4.0) * 1.2 if pd.notna(latest.get("dollar_vol_vs_avg")) else 0.0
            participation_score += min(max(float(latest.get("vol_z", np.nan) or 0.0), -1.0), 4.0) * 0.7 if pd.notna(latest.get("vol_z")) else 0.0

            trend_score = 0.0
            trend_score += 1.0 if trend_50 == "Above" else 0.0
            trend_score += 1.0 if trend_200 == "Above" else 0.0

            price_location_score = 0.0
            price_location_score += 1.25 if breakout in {"20D Up", "20D Down"} else 0.0
            price_location_score += min(abs(float(latest["ret_1d"])) / 2.5, 2.0)

            setup = classify_setup(latest, trend_50, trend_200, breakout)
            setup_score = {
                "Bullish expansion": 2.4,
                "Bearish expansion": 2.4,
                "Absorption / churn": 1.6,
                "Transition regime": 1.2,
                "Watch": 0.5,
                "Unclear": 0.0,
            }.get(setup, 0.0)

            urgency = participation_score + trend_score + price_location_score + setup_score

            vol_nonzero_ratio = (d["Volume"] > 0).mean() if "Volume" in d else 0.0
            if vol_nonzero_ratio >= 0.95:
                data_quality = "High"
            elif vol_nonzero_ratio >= 0.75:
                data_quality = "Medium"
            else:
                data_quality = "Low"

            rows.append(
                SignalRow(
                    asset_class=asset_class,
                    ticker=ticker,
                    label=label,
                    close=float(latest["Close"]),
                    daily_pct=float(latest["ret_1d"]),
                    vol_multiple=float(latest["vol_multiple"]) if pd.notna(latest["vol_multiple"]) else np.nan,
                    vol_z=float(latest["vol_z"]) if pd.notna(latest["vol_z"]) else np.nan,
                    dollar_vol_vs_avg=float(latest["dollar_vol_vs_avg"]) if pd.notna(latest["dollar_vol_vs_avg"]) else np.nan,
                    breakout_20d=breakout,
                    trend_50d=trend_50,
                    trend_200d=trend_200,
                    atr_pct=float(latest["atr_pct"]) if pd.notna(latest["atr_pct"]) else np.nan,
                    urgency=float(urgency),
                    setup=setup,
                    setup_score=float(setup_score),
                    trend_score=float(trend_score),
                    participation_score=float(participation_score),
                    price_location_score=float(price_location_score),
                    data_quality=data_quality,
                )
            )
            enriched[ticker] = d

    table = pd.DataFrame([r.__dict__ for r in rows])
    if table.empty:
        return pd.DataFrame(), enriched, issues

    table = table.sort_values(["urgency", "vol_multiple"], ascending=[False, False]).reset_index(drop=True)
    return table, enriched, issues


def explain_setup(setup: str) -> str:
    mapping = {
        "Bullish expansion": "Price cleared the 20-day range on strong volume and is trading above key trend levels.",
        "Bearish expansion": "Price broke down from the 20-day range on strong volume and is trading below short-term trend support.",
        "Absorption / churn": "Volume is elevated but price barely moved, which often points to ownership transfer rather than immediate follow-through.",
        "Transition regime": "Participation picked up while the 50-day and 200-day trends disagree, so the tape may be changing character.",
        "Watch": "Some activity is present, but confirmation is still limited.",
        "Unclear": "The latest data is incomplete, so the setup is hard to classify cleanly.",
    }
    return mapping.get(setup, "")


def summarize_market(table: pd.DataFrame) -> Dict[str, str]:
    if table.empty:
        return {
            "headline": "No qualifying setups under the current filters.",
            "breadth": "No breadth signal.",
            "equities": "No qualifying equities setup.",
            "credit": "No qualifying credit setup.",
            "commodities": "No qualifying commodities setup.",
            "currencies": "No qualifying currencies setup.",
        }

    bullish = int((table["setup"] == "Bullish expansion").sum())
    bearish = int((table["setup"] == "Bearish expansion").sum())
    churn = int((table["setup"] == "Absorption / churn").sum())
    transition = int((table["setup"] == "Transition regime").sum())
    lead = table.iloc[0]

    def leader_line(asset_class: str) -> str:
        sub = table[table["asset_class"] == asset_class].sort_values("urgency", ascending=False).head(1)
        if sub.empty:
            return f"No qualifying {asset_class.lower()} setup."
        r = sub.iloc[0]
        return (
            f"{r['ticker']} is the lead {asset_class.lower()} instrument, up {r['daily_pct']:+.2f}% on "
            f"{r['vol_multiple']:.2f}x relative volume with a {r['setup'].lower()} read."
        )

    return {
        "headline": (
            f"{lead['ticker']} is the highest-urgency setup, moving {lead['daily_pct']:+.2f}% on "
            f"{lead['vol_multiple']:.2f}x relative volume. Current read: {lead['setup']}."
        ),
        "breadth": f"Bullish {bullish} | Bearish {bearish} | Churn {churn} | Transition {transition}",
        "equities": leader_line("Equities"),
        "credit": leader_line("Credit"),
        "commodities": leader_line("Commodities"),
        "currencies": leader_line("Currencies"),
    }


def prep_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["Close"] = out["close"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    out["1D %"] = out["daily_pct"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}%")
    out["Vol / 20D"] = out["vol_multiple"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
    out["Vol Z"] = out["vol_z"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["$Vol / 20D"] = out["dollar_vol_vs_avg"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
    out["ATR %"] = out["atr_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    out["20D"] = out["breakout_20d"]
    out["50D"] = out["trend_50d"]
    out["200D"] = out["trend_200d"]
    out["Setup"] = out["setup"]
    out["Urgency"] = out["urgency"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["Quality"] = out["data_quality"]

    out = out[
        [
            "asset_class",
            "ticker",
            "label",
            "Close",
            "1D %",
            "Vol / 20D",
            "Vol Z",
            "$Vol / 20D",
            "ATR %",
            "20D",
            "50D",
            "200D",
            "Setup",
            "Urgency",
            "Quality",
        ]
    ]
    out.columns = [
        "Asset",
        "Ticker",
        "Name",
        "Close",
        "1D %",
        "Vol / 20D",
        "Vol Z",
        "$Vol / 20D",
        "ATR %",
        "20D State",
        "50D",
        "200D",
        "Setup",
        "Urgency",
        "Data Quality",
    ]
    return out


def signal_bg(setup: str) -> str:
    return {
        "Bullish expansion": "#dff5e7",
        "Bearish expansion": "#fde7e7",
        "Absorption / churn": "#fff4d7",
        "Transition regime": "#e8f0ff",
        "Watch": "#f4f4f5",
        "Unclear": "#f4f4f5",
    }.get(setup, "#f4f4f5")


def build_detail_read(row: pd.Series) -> str:
    setup = row["setup"]
    if setup == "Bullish expansion":
        return "Participation and price direction are aligned. This deserves follow-through monitoring over the next few sessions."
    if setup == "Bearish expansion":
        return "This is the kind of downside tape that can force de-risking, especially if volume stays elevated on follow-through."
    if setup == "Absorption / churn":
        return "Turnover is elevated but price has not resolved yet. That often shows transfer of ownership before the larger move."
    if setup == "Transition regime":
        return "The tape is trying to change character, but the trend stack is still mixed. Watch whether the 50-day and 200-day relationship resolves."
    if setup == "Watch":
        return "There is some attention here, but the tape has not delivered enough confirmation to treat it as a high-conviction event."
    return "The signal is incomplete because the latest data does not classify cleanly."


def plot_volume_weighted_panel(
    ticker: str,
    label: str,
    df: pd.DataFrame,
    lookback: int = 120,
    show_bars: bool = True,
    normalize_price: bool = False,
) -> None:
    if df.empty:
        st.warning("No chart data available for this instrument.")
        return

    view = df.tail(lookback).copy()
    if len(view) < 3:
        st.warning("Not enough history to draw the chart.")
        return

    if normalize_price:
        base = float(view["Close"].iloc[0])
        if base == 0:
            base = 1.0
        price_series = 100.0 * view["Close"] / base
        ma20 = 100.0 * view["ma_20"] / base
        ma50 = 100.0 * view["ma_50"] / base
        y_label = "Indexed Price"
    else:
        price_series = view["Close"]
        ma20 = view["ma_20"]
        ma50 = view["ma_50"]
        y_label = "Price"

    x = mdates.date2num(view.index.to_pydatetime())
    y = pd.to_numeric(price_series, errors="coerce").values.astype(float)

    if np.isnan(y).all():
        st.warning("Price series is empty after cleaning.")
        return

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    vol_for_segments = view["line_color_score"].iloc[1:].fillna(1.0).values.astype(float)
    width_for_segments = view["line_weight"].iloc[1:].fillna(2.0).values.astype(float)

    vmin = max(0.75, float(np.nanmin(vol_for_segments))) if len(vol_for_segments) else 0.75
    vmax = max(vmin + 0.1, float(np.nanmax(vol_for_segments))) if len(vol_for_segments) else 2.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    fig, ax1 = plt.subplots(figsize=(12.4, 5.1))
    ax2 = ax1.twinx()

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(vol_for_segments)
    lc.set_linewidth(width_for_segments)
    ax1.add_collection(lc)

    ax1.plot(view.index, ma20, linewidth=1.05, alpha=0.95, linestyle="--", label="20D MA")
    ax1.plot(view.index, ma50, linewidth=1.05, alpha=0.90, linestyle="-.", label="50D MA")
    ax1.scatter(view.index[-1], price_series.iloc[-1], s=28, zorder=5)

    if show_bars:
        ax2.bar(view.index, view["Volume"], alpha=0.15, width=1.8, label="Volume")
        ax2.plot(view.index, view["vol_avg_20"], linewidth=1.0, alpha=0.90, label="20D Avg Volume")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda val, _: human_num(val)))
        ax2.set_ylabel("Volume")

    ax1.set_title(f"{ticker} | {label} | Price Path Weighted by Relative Volume")
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.18)
    ax1.set_xlim(view.index.min(), view.index.max())

    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    if np.isfinite(ymin) and np.isfinite(ymax):
        if ymax > ymin:
            pad = (ymax - ymin) * 0.06
        else:
            pad = max(abs(ymin) * 0.03, 0.5)
        ax1.set_ylim(ymin - pad, ymax + pad)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.01, fraction=0.04)
    cbar.set_label("Relative Volume vs 20D")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if show_bars else ([], [])
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, ncol=2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def signal_cards(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if df.empty:
        st.info("No setups matched this filter.")
        return

    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="insight-card" style="background:{signal_bg(row['setup'])};">
                <div style="font-weight:700;">{row['ticker']} | {row['label']}</div>
                <div class="muted">{row['asset_class']} | {row['setup']}</div>
                <div style="margin-top:0.35rem;">
                    {row['daily_pct']:+.2f}% on {row['vol_multiple']:.2f}x relative volume and
                    {row['dollar_vol_vs_avg']:.2f}x dollar volume. 20-day state is {row['breakout_20d']},
                    while price is {row['trend_50d']} the 50-day and {row['trend_200d']} the 200-day.
                </div>
                <div class="tiny-note" style="margin-top:0.45rem;">{explain_setup(row['setup'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        """
        This dashboard is meant to answer one question: where is price moving with real participation behind it?

        **How to read the outputs**

        **Vol / 20D** compares today's volume to its own 20-day average.  
        1.0x means normal. 2.0x means twice normal.

        **$Vol / 20D** does the same with dollar volume, which helps compare instruments with different prices.

        **20D State** checks whether price broke above or below the prior 20-day range.  
        "Inside" means price stayed inside that range.

        **50D / 200D** show where price sits versus those moving averages.

        **Setup** is the tape label:
        bullish expansion, bearish expansion, absorption/churn, transition regime, or watch.

        **Urgency** is a ranking score. It is only a sorting tool. Treat it as a relative priority measure, not a forecast.

        **Chart color and thickness**
        darker and thicker line segments mean stronger relative volume on those days.
        """
    )

    with st.expander("Signal glossary", expanded=False):
        st.markdown(
            """
            **Bullish expansion**: upside range break with strong volume.

            **Bearish expansion**: downside range break with strong volume.

            **Absorption / churn**: heavy trading, small price move.

            **Transition regime**: activity rose while the trend stack is mixed.

            **Watch**: worth monitoring, but confirmation is limited.
            """
        )

    st.markdown("### Controls")
    selected_classes = st.multiselect(
        "Asset classes",
        options=list(ASSET_MAP.keys()),
        default=list(ASSET_MAP.keys()),
    )
    setup_filter = st.multiselect(
        "Setups",
        options=["Bullish expansion", "Bearish expansion", "Absorption / churn", "Transition regime", "Watch"],
        default=["Bullish expansion", "Bearish expansion", "Absorption / churn", "Transition regime", "Watch"],
    )
    min_vol_multiple = st.slider("Minimum relative volume", 1.0, 4.0, 1.1, 0.1)
    min_abs_move = st.slider("Minimum absolute 1D move %", 0.0, 5.0, 0.0, 0.05)
    top_n = st.slider("Rows to show", 5, 40, 16, 1)
    lookback = st.slider("Chart lookback", 60, 180, 120, 10)
    normalize_price = st.checkbox("Normalize price to 100", value=False)
    show_volume_bars = st.checkbox("Show volume bars", value=True)
    show_watchlist = st.checkbox("Include Watch setups in ranked table", value=True)

    selected_tickers: List[str] = []
    for asset_class in selected_classes:
        selected_tickers.extend([t for t, _ in ASSET_MAP[asset_class]])

st.markdown("<div class='main-title'>Cross-Asset Volume Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>A cross-asset monitor that ties price movement to participation, so the tape is easier to interpret.</div>",
    unsafe_allow_html=True,
)

if not selected_tickers:
    st.warning("Select at least one asset class.")
    st.stop()

with st.spinner("Loading market data..."):
    raw, missing_downloads = fetch_data(selected_tickers, period="2y")
    full_table, enriched, issues = compute_signal_table(raw)

if full_table.empty:
    st.error("No valid market data returned for the selected universe.")
    st.stop()

filtered = full_table[
    full_table["asset_class"].isin(selected_classes)
    & (full_table["vol_multiple"].fillna(0) >= min_vol_multiple)
    & (full_table["daily_pct"].abs().fillna(0) >= min_abs_move)
    & (full_table["setup"].isin(setup_filter))
].copy()

if not show_watchlist:
    filtered = filtered[filtered["setup"] != "Watch"].copy()

summary = summarize_market(filtered)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Visible setups", f"{len(filtered):,}")
m2.metric("Bullish expansions", f"{(filtered['setup'] == 'Bullish expansion').sum():,}")
m3.metric("Bearish expansions", f"{(filtered['setup'] == 'Bearish expansion').sum():,}")
m4.metric("Median vol / 20D", f"{filtered['vol_multiple'].median():.2f}x" if not filtered.empty else "n/a")

st.markdown("<div class='section-title'>Executive read</div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="insight-card">
        <div><strong>Headline:</strong> {summary['headline']}</div>
        <div style="margin-top:0.3rem;"><strong>Breadth:</strong> {summary['breadth']}</div>
        <div style="margin-top:0.3rem;"><strong>Equities:</strong> {summary['equities']}</div>
        <div style="margin-top:0.3rem;"><strong>Credit:</strong> {summary['credit']}</div>
        <div style="margin-top:0.3rem;"><strong>Commodities:</strong> {summary['commodities']}</div>
        <div style="margin-top:0.3rem;"><strong>Currencies:</strong> {summary['currencies']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
with c1:
    signal_cards(
        filtered[filtered["setup"].isin(["Bullish expansion", "Bearish expansion"])].head(5),
        "Highest-conviction setups",
    )
with c2:
    signal_cards(
        filtered[filtered["setup"].isin(["Absorption / churn", "Transition regime"])].head(5),
        "Secondary setups",
    )
with c3:
    st.markdown("<div class='section-title'>Data health</div>", unsafe_allow_html=True)
    valid_count = len(full_table)
    missing_count = len(missing_downloads)
    issue_count = len(issues)
    st.markdown(
        f"""
        <div class="insight-card">
            <div><strong>Universe requested:</strong> {len(selected_tickers)}</div>
            <div style="margin-top:0.25rem;"><strong>Valid instruments:</strong> {valid_count}</div>
            <div style="margin-top:0.25rem;"><strong>Missing downloads:</strong> {missing_count}</div>
            <div style="margin-top:0.25rem;"><strong>Processing issues:</strong> {issue_count}</div>
            <div class="tiny-note" style="margin-top:0.45rem;">
                FX entries are ETF proxies, so volume reflects listed proxy turnover rather than spot FX.
                Use this dashboard for participation clues, not exact cross-asset liquidity comparisons.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if filtered.empty:
    st.warning("No instruments match the current filters. Lower the volume threshold or include Watch setups.")
else:
    st.markdown("<div class='section-title'>Ranked monitor</div>", unsafe_allow_html=True)
    st.dataframe(prep_display(filtered.head(top_n)), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Drilldown</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.05, 2.15])

    with col1:
        drill_ticker = st.selectbox("Select instrument", filtered["ticker"].tolist(), index=0)
        row = filtered.loc[filtered["ticker"] == drill_ticker].iloc[0]

        st.markdown(
            f"""
            <div class="insight-card" style="background:{signal_bg(row['setup'])};">
                <div style="font-weight:700; font-size:1.03rem;">{row['ticker']} | {row['label']}</div>
                <div class="muted">{row['asset_class']} | {row['setup']}</div>
                <div style="margin-top:0.35rem;">Close: {row['close']:.2f}</div>
                <div>1D move: {row['daily_pct']:+.2f}%</div>
                <div>Relative volume: {row['vol_multiple']:.2f}x</div>
                <div>Dollar volume / 20D: {row['dollar_vol_vs_avg']:.2f}x</div>
                <div>20D state: {row['breakout_20d']}</div>
                <div>50D trend: {row['trend_50d']}</div>
                <div>200D trend: {row['trend_200d']}</div>
                <div>ATR: {row['atr_pct']:.2f}%</div>
                <div>Urgency: {row['urgency']:.2f}</div>
                <div>Data quality: {row['data_quality']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**What the signal means:** {explain_setup(row['setup'])}")
        st.markdown(f"**Read:** {build_detail_read(row)}")

    with col2:
        plot_volume_weighted_panel(
            drill_ticker,
            row["label"],
            enriched[drill_ticker],
            lookback=lookback,
            show_bars=show_volume_bars,
            normalize_price=normalize_price,
        )

    st.markdown("<div class='section-title'>Asset class internals</div>", unsafe_allow_html=True)
    tabs = st.tabs(selected_classes)

    for tab, asset_class in zip(tabs, selected_classes):
        with tab:
            sub = filtered[filtered["asset_class"] == asset_class].copy()
            if sub.empty:
                st.info("No setups under current filters.")
            else:
                st.dataframe(prep_display(sub.head(12)), use_container_width=True, hide_index=True)

with st.expander("Why outputs disappear or look odd", expanded=False):
    st.markdown(
        """
        The main reasons are simple:

        first, the filters can be too tight, especially minimum 1-day move and minimum relative volume.

        second, some Yahoo Finance downloads come back missing or partially stale.

        third, cross-asset volume is not fully apples-to-apples. Spot FX is represented with ETF proxies, and treasury ETFs trade differently from single-stock ETFs or commodity funds.

        fourth, urgency is a ranking score built from participation, trend, and price-location inputs. It helps sort the list, but it should not be treated as a target price or expected return.

        If you want this tool to feel more decision-grade, the next step is to add regime context, rolling percentiles, and instrument-specific thresholds rather than using one global threshold for every asset.
        """
    )

st.caption(
    "Data source: Yahoo Finance via yfinance. Currencies are represented by listed ETF proxies because "
    "spot FX does not have a single consolidated public volume tape."
)
