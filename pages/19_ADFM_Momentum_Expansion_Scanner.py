"""ADFM Momentum Expansion Scanner â€” Page 19."""

from __future__ import annotations

from html import escape
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from adfm_momentum_scanner import (
    GICS_TO_ADFM,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    SIGNAL_COLUMNS,
    adjusted_ohlcv,
    benchmark_calendar,
    diagnostic_table,
    fetch_market_metadata,
    fetch_ohlcv,
    load_equity_universe,
    market_cap_bucket,
    rank_change,
    score_scanner,
    sector_etf,
    sector_leadership,
    security_features,
    stale_sessions,
    valid_ohlcv_reason,
)
from adfm_sector_rotation_config import BENCHMARKS, MAJOR_SECTORS, SUBSECTOR_ROWS


st.set_page_config(page_title="ADFM Momentum Expansion Scanner", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {max-width: 1500px; padding-top: 1.25rem; padding-bottom: 1.5rem;}
    .scanner-about {font-size:.91rem; line-height:1.55; color:#64748b; max-width:1100px; margin:0 0 .7rem 0;}
    .scanner-status {font-size:.79rem; color:#64748b; margin-bottom:1.1rem;}
    .scanner-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(238px, 1fr)); gap:12px; align-items:stretch;}
    .scanner-card {background:#fff; border:1px solid rgba(49,51,63,.13); border-left:3px solid #94a3b8; border-radius:11px; padding:12px 13px 11px; box-shadow:0 1px 2px rgba(15,23,42,.04); min-height:335px;}
    .scanner-card.gold {border-left-color:#b8871a;} .scanner-card.blue {border-left-color:#2563eb;} .scanner-card.red {border-left-color:#dc2626;}
    .scanner-top {display:flex; justify-content:space-between; align-items:baseline; gap:8px;} .ticker {font-size:1.05rem; font-weight:800; color:#111827;} .price {font-size:.84rem; font-weight:700; color:#1f2937;}
    .cap {font-size:.62rem; letter-spacing:.06em; color:#2563eb; background:#eaf1ff; padding:2px 7px; border-radius:99px; font-weight:800;}
    .signal-row {margin-top:6px; display:flex; gap:7px; align-items:center; font-size:.83rem; font-weight:800;} .state {font-size:.69rem; padding:3px 7px; border-radius:99px; background:#f8edd2; color:#926f1b; font-weight:750;} .state.watch {background:#eaf1ff;color:#2555a5;} .state.bad {background:#fde8e7;color:#ae2722;}
    .trend {font-size:.75rem; color:#36765e; font-weight:700; margin:8px 0 7px;} .badges{display:flex; flex-wrap:wrap; gap:4px; min-height:61px;}.badge{font-size:.60rem; letter-spacing:.025em; border-radius:99px; padding:3px 6px; background:#f0f1f3; color:#9aa0a8; font-weight:750;}.badge.on{background:#e2f1e8;color:#356d51;}.badge.spring{background:#fff4d8;color:#92722b;}
    .rsline {font-size:.75rem; margin:9px 0 5px; color:#4f46a5;} .description {font-size:.73rem; color:#667085; line-height:1.48; min-height:52px;}.foot {font-size:.69rem; color:#4b5563; line-height:1.62; margin-top:8px;}.foot strong {color:#2e6b4d;}.warning {color:#b42318; font-weight:750;}
    @media (prefers-color-scheme: dark) {.scanner-card{background:rgba(15,23,42,.72); border-color:rgba(148,163,184,.26)}.ticker,.price{color:#f8fafc}.description,.scanner-about,.scanner-status{color:#cbd5e1}.foot{color:#cbd5e1}}
</style>
""", unsafe_allow_html=True)

st.title("ADFM Momentum Expansion Scanner")
st.markdown("<div class='scanner-about'><b>About This Tool.</b> Potential Next-Session Expansion identifies end-of-day technical conditions historically associated with a possible next-session gap, breakout, or momentum expansion. It cannot know whether an overnight catalyst will occur and does not guarantee an opening gap. Signals use the latest fully completed trading session only.</div>", unsafe_allow_html=True)


def _money(value: object) -> str:
    if value is None or pd.isna(value): return "â€”"
    value = float(value)
    return f"${value / 1e9:.1f}B" if value >= 1e9 else f"${value / 1e6:.0f}M"


def _pct(value: object) -> str:
    return "â€”" if value is None or pd.isna(value) else f"{float(value):+.1%}"


def _card_class(row: pd.Series) -> str:
    if row["Risk Warning"] or row["Setup State"] == "Deteriorating": return "red"
    if row["Setup State"] in {"Momentum Building", "Constructive"}: return "gold"
    if row["Setup State"] in {"Watch", "Early Turn"}: return "blue"
    return ""


def _render_card(row: pd.Series, all_signals: bool, benchmark: str) -> str:
    state_class = "bad" if row["Setup State"] == "Deteriorating" else "watch" if row["Setup State"] in {"Watch", "Early Turn"} else ""
    badges = []
    for signal in SIGNAL_COLUMNS + (["SPRING"] if all_signals else []):
        active = bool(row.get(signal, False))
        extra = "on" if active else ""
        if signal == "SPRING" and active: extra += " spring"
        badges.append(f"<span class='badge {extra}'>{signal}</span>")
    sector_text = "Sector ETF â€”" if pd.isna(row.get("Sector RS 21D")) else f"Sector ETF {'outperform' if row['Sector RS 21D'] > 0 else 'underperform'}"
    spy_text = "SPY â€”" if pd.isna(row.get("RS 21D")) else f"{benchmark} {'outperform' if row['RS 21D'] > 0 else 'underperform'}"
    warning = f"<div class='warning'>{escape(str(row['Risk Warning']))}</div>" if row["Risk Warning"] else ""
    return f"""
    <div class='scanner-card {_card_class(row)}'>
      <div class='scanner-top'><span class='ticker'>{escape(str(row['Ticker']))}</span><span class='price'>${float(row['Price']):,.2f}</span><span class='cap'>{escape(str(row['Market Cap Bucket']))}</span></div>
      <div class='signal-row'><span>{int(row['Signal Count'])}/12</span><span class='state {state_class}'>{escape(str(row['Setup State']))}</span></div>
      <div class='trend'>{escape(str(row['Trend Label']))}</div><div class='badges'>{''.join(badges)}</div>
      <div class='rsline'>â˜… {escape(str(row['RS Label']))} Â· Setup {int(row['Signal Count Change 5D']):+d} over 5D</div>
      <div class='description'>{escape(str(row['Setup Description']))}</div>
      <div class='foot'>RVOL <b>{float(row['RVOL']):.1f}x</b> &nbsp; ATR <b>{float(row['ATR %']):.2%}</b> &nbsp; {"âœ“" if row['Above SMA200'] else "Ã—"} 200EMA<br>â˜… 52W HI {_pct(row['Distance 52W High'])} &nbsp; { _money(row['Median Dollar Volume']) }<br><strong>{spy_text}</strong> Â· <strong>{sector_text}</strong>{warning}</div>
    </div>"""


with st.sidebar:
    st.header("Scanner settings")
    universe_choice = st.selectbox("Universe", ["S&P 500", "Nasdaq 100", "Russell 1000 liquid", "Combined liquid indices", "Custom watchlist"], index=0)
    custom_watchlist = st.text_area("Custom watchlist", placeholder="AAPL, MSFT, NVDA", height=68, disabled=universe_choice != "Custom watchlist")

universe, universe_errors = load_equity_universe(universe_choice, custom_watchlist)
if universe.empty:
    st.error("No scanner universe is available. " + " | ".join(universe_errors or ["Check the custom watchlist or constituent data source."]))
    st.stop()

with st.sidebar:
    sector_options = sorted(universe["ADFM Sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect("Sector", sector_options, default=sector_options)
    industry_options = sorted(universe["Industry"].dropna().astype(str).unique().tolist())
    selected_industries = st.multiselect("Industry", industry_options, default=industry_options)
    cap_range = st.selectbox("Market-cap range", ["All", "Mega", "Large", "Mid", "Small"], index=0)
    min_dv = st.number_input("Minimum median dollar volume", min_value=1_000_000, max_value=500_000_000, value=int(MIN_DOLLAR_VOLUME), step=5_000_000, format="%d")
    min_score = st.slider("Minimum Expansion Score", 0, 100, 50)
    benchmark = st.selectbox("Benchmark", list(BENCHMARKS), index=0, format_func=lambda x: f"{x} | {BENCHMARKS[x]}")
    max_cards = st.slider("Maximum cards", 4, 80, 24, 4)
    include_adrs = st.checkbox("Include identifiable ADRs", value=False)
    include_etfs = st.checkbox("Include ETFs", value=False)
    all_signals = st.radio("Signal display", ["Core signals only", "All signals"], index=0) == "All signals"

filtered_universe = universe.copy()
if selected_sectors: filtered_universe = filtered_universe[filtered_universe["ADFM Sector"].isin(selected_sectors)]
else: filtered_universe = filtered_universe.iloc[0:0]
if industry_options and selected_industries: filtered_universe = filtered_universe[filtered_universe["Industry"].isin(selected_industries) | filtered_universe["Industry"].isna()]
elif industry_options and not selected_industries: filtered_universe = filtered_universe.iloc[0:0]
pre_filter_drops: List[Dict[str, object]] = []
if not include_adrs:
    # Index constituent files do not consistently label all ADRs; only exclude
    # explicitly labelled ADR/ADS names rather than guessing from country data.
    adr_mask = filtered_universe["Name"].fillna("").str.contains(r"\bADR\b|\bADS\b", case=False, regex=True)
    pre_filter_drops = [{"Ticker": ticker, "Reason": "Identifiable ADR/ADS excluded by setting"} for ticker in filtered_universe.loc[adr_mask, "Ticker"]]
    filtered_universe = filtered_universe.loc[~adr_mask]

sector_tickers = list(MAJOR_SECTORS) + [row["Ticker"] for row in SUBSECTOR_ROWS]
requested = tuple(dict.fromkeys(filtered_universe["Ticker"].tolist() + [benchmark] + sector_tickers))
with st.spinner("Loading completed-session OHLCV data..."):
    downloaded = fetch_ohlcv(requested)
frames = downloaded.frames
calendar = benchmark_calendar(frames, benchmark)
if len(calendar) < 252:
    st.error(f"Benchmark {benchmark} has insufficient completed daily history.")
    st.stop()

dropped_rows: List[Dict[str, object]] = pre_filter_drops + downloaded.dropped.to_dict("records")
eligible_records: List[Dict[str, object]] = []
for record in filtered_universe.to_dict("records"):
    ticker = record["Ticker"]
    reason = valid_ohlcv_reason(frames.get(ticker), calendar)
    if reason:
        dropped_rows.append({"Ticker": ticker, "Reason": reason})
        continue
    adjusted = adjusted_ohlcv(frames[ticker]).dropna(subset=["Close", "Volume"])
    if adjusted["Close"].iloc[-1] < MIN_PRICE:
        dropped_rows.append({"Ticker": ticker, "Reason": f"Price below ${MIN_PRICE:.0f}"})
        continue
    median_dv = (adjusted["Close"] * adjusted["Volume"]).rolling(20, min_periods=20).median().iloc[-1]
    if pd.isna(median_dv) or median_dv < min_dv:
        dropped_rows.append({"Ticker": ticker, "Reason": "Median 20D dollar volume below filter"})
        continue
    eligible_records.append(record)

eligible_universe = pd.DataFrame(eligible_records)
if eligible_universe.empty:
    st.warning("No securities passed the data, price, and liquidity filters.")
    st.stop()
with st.spinner("Resolving available market-cap and security metadata..."):
    metadata = fetch_market_metadata(tuple(eligible_universe["Ticker"].tolist()))
eligible_universe = eligible_universe.merge(metadata, on="Ticker", how="left")
eligible_universe["Market Cap Bucket"] = eligible_universe["Market Cap"].apply(market_cap_bucket)
small_cap_mask = eligible_universe["Market Cap"].notna() & (eligible_universe["Market Cap"] < 1_000_000_000)
for ticker in eligible_universe.loc[small_cap_mask, "Ticker"]: dropped_rows.append({"Ticker": ticker, "Reason": "Market capitalization below $1B"})
eligible_universe = eligible_universe.loc[~small_cap_mask]
if not include_etfs:
    etf_mask = eligible_universe["Quote Type"].fillna("").str.upper().eq("ETF")
    for ticker in eligible_universe.loc[etf_mask, "Ticker"]: dropped_rows.append({"Ticker": ticker, "Reason": "ETF excluded by setting"})
    eligible_universe = eligible_universe.loc[~etf_mask]
if cap_range != "All": eligible_universe = eligible_universe[eligible_universe["Market Cap Bucket"] == cap_range]

rows: List[Dict[str, object]] = []
bench_close = adjusted_ohlcv(frames[benchmark])["Close"]
for record in eligible_universe.to_dict("records"):
    sector_ticker = sector_etf(record.get("ADFM Sector"))
    sector_close = adjusted_ohlcv(frames[sector_ticker])["Close"] if sector_ticker in frames else None
    row = security_features(record["Ticker"], frames[record["Ticker"]], bench_close, sector_close)
    rows.append({**record, **row, "Stale Sessions": stale_sessions(frames[record["Ticker"]], calendar)})
scanner = score_scanner(pd.DataFrame(rows)) if rows else pd.DataFrame()
if scanner.empty:
    st.warning("No securities have sufficient data to calculate all required scanner features.")
    st.stop()
passing = scanner[scanner["Expansion Score"] >= min_score]
latest_date = calendar[-1].date()
st.markdown(f"<div class='scanner-status'><b>Potential Next-Session Expansion</b> Â· Data through: {latest_date} Â· Eligible securities: {len(scanner)} Â· Passing score threshold: {len(passing)} Â· Benchmark: {benchmark}</div>", unsafe_allow_html=True)
for error in universe_errors: st.caption(error)

view = st.radio("Display", ["Card View", "Full Table"], horizontal=True, label_visibility="collapsed")
if view == "Card View":
    cards = passing.head(max_cards)
    if cards.empty: st.info("No securities meet the selected Expansion Score threshold.")
    else: st.markdown("<div class='scanner-grid'>" + "".join(_render_card(row, all_signals, benchmark) for _, row in cards.iterrows()) + "</div>", unsafe_allow_html=True)
else:
    st.dataframe(scanner, use_container_width=True, hide_index=True, height=720)

st.download_button("Download scanner CSV", scanner.to_csv(index=False).encode("utf-8"), "adfm_momentum_expansion_scanner.csv", "text/csv")
if dropped_rows:
    with st.expander(f"Dropped securities ({len(pd.DataFrame(dropped_rows).drop_duplicates())})"):
        st.dataframe(pd.DataFrame(dropped_rows).drop_duplicates(), use_container_width=True, hide_index=True)

st.subheader("Sector and Subsector Leadership")
with st.sidebar:
    core_only = st.checkbox("Sector table: Core subsectors only", value=True)
sector_df, subsector_df = sector_leadership(frames, benchmark, include_thematic=not core_only)
if not sector_df.empty:
    past_frames = {ticker: frame.iloc[:-5].copy() for ticker, frame in frames.items() if len(frame) > 5}
    past_sector, past_sub = sector_leadership(past_frames, benchmark, include_thematic=not core_only)
    if not past_sector.empty:
        sector_df = rank_change(sector_df, past_sector, key="Sector")
    display_cols = ["Rank", "Rank change over five sessions", "Sector", "Major ETF", "Leadership Score", "Leadership State", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS Slope 10D", "RS Acceleration", "Positive RS 1M", "Above EMA50 Breadth", "Above SMA200 Breadth", "Strongest Subsector", "Weakest Subsector", "Dispersion", "Participation State"]
    display = sector_df[[c for c in display_cols if c in sector_df]].copy()
    style_cols = [c for c in ["Leadership Score", "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS Slope 10D", "RS Acceleration", "Positive RS 1M", "Above EMA50 Breadth", "Above SMA200 Breadth"] if c in display]
    st.dataframe(display.style.format({c: "{:.1%}" for c in style_cols if c != "Leadership Score"}).format({"Leadership Score": "{:.1f}"}).background_gradient(subset=style_cols, cmap="RdYlGn"), use_container_width=True, hide_index=True, height=460)
    st.download_button("Download sector leadership CSV", sector_df.to_csv(index=False).encode("utf-8"), "adfm_sector_leadership.csv", "text/csv")
    with st.expander("Tracked subsectors", expanded=False):
        if not subsector_df.empty:
            if not past_sub.empty:
                subsector_df = rank_change(subsector_df, past_sub)
            st.dataframe(subsector_df, use_container_width=True, hide_index=True, height=500)
            st.download_button("Download subsector leadership CSV", subsector_df.to_csv(index=False).encode("utf-8"), "adfm_subsector_leadership.csv", "text/csv")
else:
    st.warning("Sector leadership is unavailable because required major-sector ETF data could not be loaded.")

with st.expander("Signal Diagnostics", expanded=False):
    st.caption("Two-year walk-forward observations use only the signal date for setup classification and subsequent sessions only for outcomes. These are descriptive diagnostics, not optimized validation.")
    run_diagnostics = st.checkbox("Run two-year diagnostics", value=False)
    if run_diagnostics:
        diagnostic_frames: Dict[str, pd.DataFrame] = {benchmark: frames[benchmark]}
        diagnostic_frames.update({ticker: frames[ticker] for ticker in scanner["Ticker"] if ticker in frames})
        with st.spinner("Calculating no-lookahead historical diagnostics..."):
            diagnostics = diagnostic_table(diagnostic_frames, benchmark, min_dollar_volume=float(min_dv))
        if diagnostics.empty: st.info("No eligible historical observations were available.")
        else:
            fmt = {c: "{:.2%}" for c in diagnostics.columns if c not in {"Signal Bucket", "Positive RS", "Observations"}}
            st.dataframe(diagnostics.style.format(fmt), use_container_width=True, hide_index=True)
            st.download_button("Download signal diagnostics CSV", diagnostics.to_csv(index=False).encode("utf-8"), "adfm_signal_diagnostics.csv", "text/csv")

st.caption(f"Data through: {latest_date} | Benchmark: {benchmark} | Raw OHLCV is never forward-filled for signals or pattern recognition.")
st.caption("Â© 2026 AD Fund Management LP")
