"""Reported ETF issuance and daily cross-asset trading pressure."""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from adfm_core.etf_pressure import ICI_URL, parse_ici_issuance, pressure_reading
from adfm_core.market_data import completed_daily_observations, configure_yfinance_cache, download_market_data
from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header, render_sidebar_about

st.set_page_config(page_title="ETF Flow Pressure", layout="wide")
configure_yfinance_cache()
inject_explorer_style(max_width_px=1700)

etf_info = {
    # -------------------------
    # US EQUITY: TACTICAL INDEX, STYLE, FACTOR
    # -------------------------
    "DIA": ("Dow Industrials", "Old-economy large-cap cyclicals"),
    "IWM": ("Russell 2000", "Small-cap equity risk appetite"),
    "IJR": ("S&P SmallCap 600", "Higher-quality small-cap proxy"),
    "MDY": ("S&P MidCap 400", "Mid-cap equity risk appetite"),
    "RSP": ("S&P 500 Equal Weight", "Equal-weight breadth versus cap-weight leadership"),
    "IWF": ("Russell 1000 Growth", "Large-cap growth factor"),
    "IWD": ("Russell 1000 Value", "Large-cap value factor"),
    "IWO": ("Russell 2000 Growth", "Small-cap growth factor"),
    "IWN": ("Russell 2000 Value", "Small-cap value factor"),
    "MTUM": ("US Momentum", "Momentum factor"),
    "QUAL": ("US Quality", "Quality factor"),
    "USMV": ("US Minimum Volatility", "Defensive low-volatility factor"),
    "VLUE": ("US Value Factor", "Value factor"),
    "SPHB": ("S&P 500 High Beta", "High-beta equity risk appetite"),
    "SPLV": ("S&P 500 Low Volatility", "Low-volatility defensive equity"),
    "SCHD": ("US Dividend Quality", "Dividend / quality / defensive income equity"),

    # -------------------------
    # US SECTORS, INDUSTRIES, AND CYCLICAL SIGNALS
    # -------------------------
    "XLK": ("US Technology", "S&P 500 technology sector"),
    "SMH": ("Semiconductors", "Large-cap semiconductor leadership"),
    "SOXX": ("Semiconductor Breadth", "Semiconductor industry breadth"),
    "XSD": ("Equal-Weight Semis", "Equal-weight semiconductor risk appetite"),
    "XLC": ("US Communication Services", "Communication services / platform equities"),
    "XLY": ("US Discretionary", "Consumer discretionary sector"),
    "XLP": ("US Staples", "Consumer staples defensive sector"),
    "XLF": ("US Financials", "Large-cap financials"),
    "KRE": ("Regional Banks", "Regional banking and credit sensitivity"),
    "XLE": ("US Energy", "Integrated energy and energy beta"),
    "XOP": ("Oil & Gas E&P", "Exploration and production equities"),
    "OIH": ("Oil Services", "Oilfield services / energy capex cycle"),
    "XLI": ("US Industrials", "Industrial cyclicals"),
    "IYT": ("Transports", "Transport cyclicality and goods movement"),
    "XLU": ("US Utilities", "Defensive rate-sensitive equity"),
    "XLV": ("US Healthcare", "Healthcare defensives"),
    "IBB": ("Biotech Large-Cap", "Large-cap biotech"),
    "XBI": ("Biotech Equal-Weight", "Speculative biotech risk appetite"),
    "IHF": ("Healthcare Providers", "Managed care and healthcare services"),
    "XLB": ("US Materials", "Materials cyclicality"),
    "XME": ("Metals & Mining", "Mining and reflation sensitivity"),
    "XRT": ("Retail", "Retail / consumer impulse"),
    "XHB": ("Homebuilders", "Housing cycle proxy"),
    "ITB": ("Home Construction", "Home construction and housing beta"),
    "IYR": ("US Real Estate", "REITs and rate-sensitive real estate"),
    "IGV": ("Software", "Software and application tech"),
    "ARKK": ("Speculative Innovation", "Long-duration speculative growth"),

    # -------------------------
    # INTERNATIONAL AND EM EQUITY SIGNALS
    # -------------------------
    "EEM": ("Emerging Markets", "Liquid EM equity trading proxy"),
    "VGK": ("Europe", "Developed Europe equities"),
    "FEZ": ("Eurozone Large-Cap", "Eurozone blue-chip equity"),
    "EWU": ("United Kingdom", "UK equities"),
    "EWG": ("Germany", "Germany equities"),
    "EWQ": ("France", "France equities"),
    "EWI": ("Italy", "Italy equities"),
    "EWP": ("Spain", "Spain equities"),
    "EWJ": ("Japan", "Japan equities"),
    "EWY": ("South Korea", "Korea equities"),
    "EWT": ("Taiwan", "Taiwan equities / semis supply chain"),
    "FXI": ("China Large-Cap", "China offshore large caps"),
    "MCHI": ("China Broad Equity", "Broad China equity"),
    "KWEB": ("China Internet", "China internet and platform equities"),
    "ASHR": ("China A-Shares", "Onshore China equities"),
    "INDA": ("India", "India equities"),
    "EWZ": ("Brazil", "Brazil equities"),
    "EWW": ("Mexico", "Mexico equities"),
    "ECH": ("Chile", "Chile equities / copper sensitivity"),
    "ARGT": ("Argentina", "Argentina equities"),

    # -------------------------
    # RATES, CREDIT, CASH, INFLATION
    # -------------------------
    "SGOV": ("UST Bills", "0-3 month Treasury bills"),
    "BIL": ("UST Bills Alt", "1-3 month Treasury bills"),
    "SHY": ("UST 1-3y", "Short-duration Treasuries"),
    "IEF": ("UST 7-10y", "Intermediate-duration Treasuries"),
    "TLT": ("UST 20y+", "Long-duration Treasuries"),
    "EDV": ("Extended Duration UST", "Long-duration zero-coupon Treasury exposure"),
    "TIP": ("TIPS", "Inflation-linked Treasuries"),
    "STIP": ("Short TIPS", "Short-duration inflation-linked Treasuries"),
    "LQD": ("IG Credit", "Investment-grade corporate credit"),
    "VCIT": ("Intermediate IG Credit", "Intermediate-duration investment-grade credit"),
    "HYG": ("High Yield", "High-yield corporate credit"),
    "JNK": ("High Yield Alt", "High-yield corporate credit alternative"),
    "BKLN": ("Senior Loans", "Floating-rate senior loans"),
    "MBB": ("Agency MBS", "Mortgage-backed securities"),
    "EMB": ("EM Debt", "USD emerging-market sovereign debt"),
    "MUB": ("Municipal Bonds", "Investment-grade municipal bonds"),
    "GOVT": ("UST Aggregate", "Broad Treasury curve exposure"),

    # -------------------------
    # COMMODITIES, CRYPTO, VOLATILITY, FX
    # -------------------------
    "GLD": ("Gold", "Gold bullion"),
    "SLV": ("Silver", "Silver bullion"),
    "GDX": ("Gold Miners", "Large-cap gold miners"),
    "GDXJ": ("Junior Gold Miners", "Junior gold miners"),
    "USO": ("Crude Oil", "WTI crude oil"),
    "UNG": ("Natural Gas", "Natural gas futures"),
    "DBC": ("Broad Commodities", "Diversified commodity basket"),
    "DBA": ("Agriculture", "Agricultural commodities"),
    "CPER": ("Copper", "Copper futures"),
    "URA": ("Uranium", "Uranium and nuclear fuel cycle equities"),
    "REMX": ("Rare Earths", "Rare earth and strategic metals equities"),
    "IBIT": ("Bitcoin", "Spot Bitcoin ETF"),
    "ETHA": ("Ethereum", "Spot Ethereum ETF"),
    "VIXY": ("Equity Volatility", "Front-end VIX futures ETF"),
    "UUP": ("US Dollar", "US Dollar Index bullish exposure"),
    "FXE": ("Euro", "Euro versus US dollar"),
    "FXY": ("Japanese Yen", "Japanese yen versus US dollar"),
    "FXF": ("Swiss Franc", "Swiss franc versus US dollar"),
    "CEW": ("EM FX", "Emerging-market currency basket"),
}

etf_tickers = tuple(etf_info.keys())

US_EQUITY_TICKERS = {
    "DIA", "IWM", "IJR", "MDY", "RSP",
    "IWF", "IWD", "IWO", "IWN",
    "MTUM", "QUAL", "USMV", "VLUE", "SPHB", "SPLV", "SCHD",
    "XLK", "SMH", "SOXX", "XSD", "XLC", "XLY", "XLP", "XLF", "KRE",
    "XLE", "XOP", "OIH", "XLI", "IYT", "XLU", "XLV", "IBB", "XBI", "IHF",
    "XLB", "XME", "XRT", "XHB", "ITB", "IYR", "IGV", "ARKK",
}

INTERNATIONAL_EQUITY_TICKERS = {
    "EEM", "VGK", "FEZ", "EWU", "EWG", "EWQ", "EWI", "EWP",
    "EWJ", "EWY", "EWT",
    "FXI", "MCHI", "KWEB", "ASHR", "INDA",
    "EWZ", "EWW", "ECH", "ARGT",
}

RATES_CREDIT_TICKERS = {
    "SGOV", "BIL", "SHY", "IEF", "TLT", "EDV",
    "TIP", "STIP",
    "LQD", "VCIT", "HYG", "JNK", "BKLN", "MBB", "EMB", "MUB", "GOVT",
}

COMMODITY_TICKERS = {
    "GLD", "SLV", "GDX", "GDXJ",
    "USO", "UNG", "DBC", "DBA", "CPER",
    "URA", "REMX",
}

FX_TICKERS = {
    "UUP", "FXE", "FXY", "FXF", "CEW",
}

CRYPTO_VOL_TICKERS = {
    "IBIT", "ETHA", "VIXY",
}


def infer_asset_class(ticker: str) -> str:
    if ticker in US_EQUITY_TICKERS:
        return "US Equity"
    if ticker in INTERNATIONAL_EQUITY_TICKERS:
        return "International Equity"
    if ticker in RATES_CREDIT_TICKERS:
        return "Rates and Credit"
    if ticker in COMMODITY_TICKERS:
        return "Commodities"
    if ticker in FX_TICKERS:
        return "FX"
    if ticker in CRYPTO_VOL_TICKERS:
        return "Crypto and Volatility"
    return "Other"


def strip_tz_from_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx, errors="coerce")
    try:
        if out.tz is not None:
            out = out.tz_localize(None)
    except Exception:
        try:
            out = out.tz_localize(None)
        except Exception:
            pass
    return pd.DatetimeIndex(out)

def chunked(seq: Sequence[str], size: int) -> List[Tuple[str, ...]]:
    return [tuple(seq[i:i + size]) for i in range(0, len(seq), size)]

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out.index = strip_tz_from_index(out.index)
    out = out[~out.index.isna()].dropna(subset=["Close"]).sort_index()

    return out[needed]

def extract_ticker_frame(raw: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)

        if ticker in lvl0:
            return raw[ticker].copy()

        if ticker in lvl1:
            return raw.xs(ticker, axis=1, level=1).copy()

        return pd.DataFrame()

    if batch_len == 1:
        return raw.copy()

    return pd.DataFrame()

def safe_yf_download(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    threads: bool = True,
    attempts: int = 3,
    delay: float = 0.9,
    deadline: float = None,
) -> pd.DataFrame:
    deadline = time.monotonic() + 25.0 if deadline is None else deadline
    for i in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = download_market_data(
                tickers=list(tickers),
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=threads,
                progress=False,
                recovery_budget_seconds=remaining,
                timeout=min(10.0, remaining),
            )

            if raw is not None and not raw.empty:
                return raw

        except Exception:
            pass

        if i < attempts - 1:
            time.sleep(min(delay * (i + 1), max(0.0, deadline - time.monotonic())))

    return pd.DataFrame()

def fetch_prices(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    cache_key: str,
    batch_size: int = 35,
) -> Dict[str, pd.DataFrame]:
    _ = cache_key
    deadline = time.monotonic() + 25.0

    out: Dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, batch_size):
        if time.monotonic() >= deadline:
            break
        raw = safe_yf_download(batch, start_date, end_date, threads=True, deadline=deadline)

        for tk in batch:
            df = normalize_ohlcv(extract_ticker_frame(raw, tk, len(batch)))

            if df.empty and len(batch) > 1 and time.monotonic() < deadline:
                raw_single = safe_yf_download((tk,), start_date, end_date, threads=False, attempts=2, deadline=deadline)
                df = normalize_ohlcv(extract_ticker_frame(raw_single, tk, 1))

            out[tk] = df

    for tk in tickers:
        if tk not in out:
            out[tk] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    return out

@st.cache_data(ttl=900, show_spinner=False, max_entries=4)
def load_price_history(tickers: tuple, bucket: str) -> dict:
    today = datetime.now(ZoneInfo('America/New_York')).date()
    frames = fetch_prices(tickers, today - timedelta(days=365 * 3 + 30), today + timedelta(days=1), bucket)
    return {ticker: completed_daily_observations(frame, ticker) for ticker, frame in frames.items()}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def load_issuance() -> tuple:
    error = ''
    try:
        response = requests.get(ICI_URL, timeout=10)
        response.raise_for_status()
        result = parse_ici_issuance(response.text)
        if result.columns.max().date() > datetime.now(ZoneInfo('America/New_York')).date():
            raise ValueError('Future report date')
        return result, 'ICI published report', ''
    except Exception as exc:
        error = type(exc).__name__
    try:
        snapshot = json.loads((Path(__file__).resolve().parents[1] / 'data/ici/etf_net_issuance.json').read_text())
        frame = pd.DataFrame(snapshot['values'], index=snapshot['categories'], columns=pd.to_datetime(snapshot['weeks']))
        return frame, 'Last saved ICI report', error
    except Exception:
        return pd.DataFrame(), 'Unavailable', error


def pressure_chart(frame: pd.DataFrame) -> go.Figure:
    positive = frame.loc[frame['Pressure (%)'] > 0].nlargest(10, 'Pressure (%)')
    negative = frame.loc[frame['Pressure (%)'] < 0].nsmallest(10, 'Pressure (%)')
    view = pd.concat([positive, negative]).sort_values('Pressure (%)', ascending=False)
    fig = go.Figure(go.Bar(
        x=view['Pressure (%)'], y=view['Exposure'] + ' (' + view['Ticker'] + ')', orientation='h',
        marker_color=[PASTEL['sage'] if value > 0 else PASTEL['rose'] for value in view['Pressure (%)']],
        text=[f'{value:+.1f}%' for value in view['Pressure (%)']], textposition='outside',
        customdata=view[['Return (%)', 'Activity (x)']].to_numpy(),
        hovertemplate='%{y}<br>Pressure %{x:+.1f}%<br>Return %{customdata[0]:+.2f}%<br>Activity %{customdata[1]:.2f}x<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='#94a3b8', line_width=1)
    fig.update_layout(
        height=max(330, 25 * len(view) + 95), margin=dict(l=10, r=65, t=15, b=40),
        template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        xaxis=dict(range=[-115, 115], title='Normalized trading pressure (%)', ticksuffix='%'),
        yaxis=dict(autorange='reversed'), font=dict(size=12, color='#1f2937'),
    )
    return fig


def tint(value):
    if pd.isna(value):
        return ''
    return f"color: {'#28734c' if value > 0 else '#a33d4b' if value < 0 else '#64748b'}"


with st.sidebar:
    render_sidebar_about('14_ETF_Flow_Pressure_Proxy.py')
    window_label = st.selectbox('Trading window', ['5 sessions', '21 sessions', '63 sessions', '252 sessions', 'YTD'], index=1)
    asset_filter = st.selectbox('Asset class', ['All'] + sorted({infer_asset_class(ticker) for ticker in etf_tickers}))

render_page_header(PageHeader(
    title='ETF Flow Pressure', eyebrow='ADFM Flows + Sentiment',
    description='Weekly ETF net issuance by asset class, with daily trading pressure across tactical ETF exposures.',
))

issuance, issuance_delivery, issuance_error = load_issuance()
st.subheader('Reported ETF capital flows')
if not issuance.empty:
    displayed = issuance.copy()
    displayed.columns = [stamp.strftime('%b %d, %Y') for stamp in displayed.columns]
    displayed.index.name = 'Category'
    st.dataframe(displayed.style.format('{:+.2f}').map(tint), width='stretch', height=318)
    st.caption(f'ICI weekly net issuance estimates, $ billions. Latest week ended {issuance.columns.max():%B %d, %Y}. Industry categories cover more funds than the tactical ETF universe below.')
else:
    st.caption('The weekly ICI report is currently unavailable. Daily ETF trading pressure remains available below.')

st.subheader('ETF trading pressure')
now = datetime.now(ZoneInfo('America/New_York'))
bucket = now.strftime('%Y-%m-%d %H:') + str(now.minute // 15)
with st.spinner('Loading ETF trading history...'):
    frames = load_price_history(tuple(['SPY'] + list(etf_tickers)), bucket)
benchmark = frames.get('SPY', pd.DataFrame())
sessions = pd.DatetimeIndex(benchmark.loc[benchmark['Close'].notna()].index) if 'Close' in benchmark else pd.DatetimeIndex([])
if len(sessions) and (pd.Timestamp(now.date()) - sessions[-1]).days > 7:
    sessions = pd.DatetimeIndex([])
window = int((sessions.year == sessions[-1].year).sum()) if window_label == 'YTD' and len(sessions) else int(window_label.split()[0]) if window_label != 'YTD' else 1
rows = []
for ticker in etf_tickers:
    reading = pressure_reading(frames.get(ticker, pd.DataFrame()), sessions, window)
    rows.append({'Ticker': ticker, 'Exposure': etf_info[ticker][0], 'Asset class': infer_asset_class(ticker), **reading})
full = pd.DataFrame(rows)
filtered = full if asset_filter == 'All' else full.loc[full['Asset class'] == asset_filter]
ranked = filtered.loc[filtered['Pressure (%)'].notna()].sort_values('Pressure (%)', ascending=False)
if ranked.empty:
    st.caption('No completed ETF observations are available for this selection.')
else:
    st.caption(f'Completed session: {sessions[-1]:%b %d, %Y} · {len(ranked)} of {len(filtered)} ETFs · {window_label}. Positive pressure means more traded value closed near session highs; negative pressure means near lows.')
    st.plotly_chart(pressure_chart(ranked), width='stretch', config={'displaylogo': False})
    st.caption('Chart: up to ten strongest positive and ten strongest negative readings. Ticker pressure is a price-volume statistic; the reported capital-flow estimates are in the table above.')
    columns = ['Ticker', 'Exposure', 'Asset class', 'Pressure (%)', 'Pressure change (pp)', 'Return (%)', 'Up/down volume (%)', 'Activity (x)', 'Avg daily traded ($M)']
    st.dataframe(ranked[columns].style.format({
        'Pressure (%)': '{:+.1f}', 'Pressure change (pp)': '{:+.1f}', 'Return (%)': '{:+.2f}',
        'Up/down volume (%)': '{:+.1f}', 'Activity (x)': '{:.2f}', 'Avg daily traded ($M)': '{:,.1f}',
    }, na_rep='N/A').map(tint, subset=['Pressure (%)', 'Pressure change (pp)', 'Return (%)', 'Up/down volume (%)']),
        hide_index=True, width='stretch', height=650)
    st.caption('Pressure change compares with the preceding equal-length window. Up/down volume is the traded-value balance of positive versus negative adjusted-return sessions. Activity compares average share volume with its prior 63-session median. Click column headers to sort.')
    st.download_button('Download ETF readings', full.to_csv(index=False).encode(),
                       file_name=f'etf_trading_pressure_{sessions[-1]:%Y%m%d}.csv', mime='text/csv')

with st.expander('Method and source details', expanded=False):
    st.markdown('''**Trading pressure** is the close-location value `(2 × close − high − low) / (high − low)`, weighted by each session's close × share volume and normalized to a −100% to +100% scale. A flat high/low session contributes neutral pressure when its price and volume are valid. Large funds receive no automatic advantage from their dollar turnover.

**Returns** use adjusted closes, including distributions when provided. **Up/down volume** assigns each session's traded value the sign of its adjusted return. These are complementary descriptions of the same trading tape, not independent evidence of investor purchases. **Activity** uses a prior-volume median and excludes the measured window.

**ICI figures** are published weekly estimates of primary-market ETF net issuance. The table shows non-overlapping categories, with the total displayed separately. They are industry aggregates and are never allocated to individual tickers.

ETFs use one completed SPY session calendar. Missing or invalid price-volume windows remain unranked. The ETF universe can contain overlapping underlying exposures; no cross-ETF capital-flow total is constructed.''')
    st.markdown(f'[ICI published ETF issuance report]({ICI_URL})')
    st.caption(f'ICI delivery: {issuance_delivery}' + (f' · Refresh: {issuance_error}' if issuance_error else ''))
    omitted = full.loc[full['Pressure (%)'].isna(), ['Ticker', 'Exposure', 'As of', 'Status']]
    if not omitted.empty:
        st.dataframe(omitted, hide_index=True, width='stretch')
    if not issuance.empty:
        export = issuance.copy()
        export.columns = [stamp.date().isoformat() for stamp in export.columns]
        st.download_button('Download reported weekly issuance', export.to_csv().encode(),
                           file_name='ici_etf_net_issuance.csv', mime='text/csv')

render_footer(data_note='Reported weekly ETF net issuance: Investment Company Institute. Daily ETF trading pressure and adjusted returns: Yahoo Finance. Observation dates are shown with each output.')
