# streamlit_app.py
# ADFM | Unusual Options Flow Tracker
# Full-universe S&P 500 options scanner using public Yahoo chain data
# Hardened for Streamlit Cloud and repeat-safe cluster logic

import json
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Unusual Options Flow Tracker", layout="wide")

APP_TITLE = "Unusual Options Flow Tracker"
APP_SUBTITLE = (
    "Full S&P 500 scan every run. All symbols. All expiries that Yahoo returns. "
    "Built to approximate a tape-first unusual-flow workflow with public data."
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1650px;}
    h1, h2, h3 {font-weight: 700; letter-spacing: 0.1px;}
    .stMetric {background: #fafafa; border: 1px solid #ececec; border-radius: 14px; padding: 10px 14px;}
    .stDataFrame, .stPlotlyChart {border-radius: 14px;}
    div[data-testid="stSidebarContent"] {padding-top: 0.6rem;}
    .small-note {color: #6b7280; font-size: 0.88rem; line-height: 1.4;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CACHE_DIR = Path(".uw_cache")
CACHE_DIR.mkdir(exist_ok=True)

SP500_META_CACHE = CACHE_DIR / "sp500_meta.pkl"
LAST_GOOD_SCAN = CACHE_DIR / "last_good_flow.pkl"
LAST_GOOD_META = CACHE_DIR / "last_good_meta.json"

DEFAULT_MIN_PREMIUM = 25_000.0
DEFAULT_MIN_VOLUME = 50
DEFAULT_MIN_VOL_OI = 1.0
DEFAULT_MAX_SPREAD_PCT = 35.0
DEFAULT_TOP_N = 200
DEFAULT_MAX_WORKERS = 20
DEFAULT_DTE_MIN = 0
DEFAULT_DTE_MAX = 365

# -----------------------------------------------------------------------------
# Built-in fallback S&P 500 universe
# -----------------------------------------------------------------------------
SP500_FALLBACK_SYMBOLS = sorted([
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE",
    "LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
    "ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO",
    "AVB","AVY","AXON","BKR","BALL","BAC","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","BK","BA","BKNG",
    "BWA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH",
    "KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CRL","SCHW","CHTR","CVX",
    "CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP",
    "ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR",
    "DRI","DVA","DAY","DECK","DE","DELL","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW",
    "DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT",
    "EFX","EQIX","EQR","ERIE","ESS","EL","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO",
    "FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT",
    "GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA",
    "DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII",
    "IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV",
    "IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR",
    "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU",
    "LYB","MTB","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD",
    "MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI",
    "NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA",
    "NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH","PAYX",
    "PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
    "PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY",
    "ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA",
    "SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
    "TRGP","TGT","TEL","TDY","TER","TSLA","TXN","TPL","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX",
    "VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WSM",
    "WMB","WTW","WDAY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def human_money(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def human_num(x: float) -> str:
    x = 0.0 if pd.isna(x) else float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def midpoint(bid: float, ask: float) -> float:
    bid = 0.0 if pd.isna(bid) else float(bid)
    ask = 0.0 if pd.isna(ask) else float(ask)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ask > 0:
        return ask
    if bid > 0:
        return bid
    return 0.0

def estimate_fill_price(last_price: float, bid: float, ask: float) -> float:
    lp = 0.0 if pd.isna(last_price) else float(last_price)
    bid = 0.0 if pd.isna(bid) else float(bid)
    ask = 0.0 if pd.isna(ask) else float(ask)
    mid = midpoint(bid, ask)
    if lp > 0:
        return lp
    if mid > 0:
        return mid
    if ask > 0:
        return ask
    if bid > 0:
        return bid
    return 0.0

def classify_side(fill: float, bid: float, ask: float) -> str:
    bid = 0.0 if pd.isna(bid) else float(bid)
    ask = 0.0 if pd.isna(ask) else float(ask)
    fill = 0.0 if pd.isna(fill) else float(fill)

    if bid <= 0 and ask <= 0:
        return "UNKNOWN"
    if ask <= 0:
        return "BID"
    if bid <= 0:
        return "ASK"

    spread = ask - bid
    if spread <= 0:
        return "MID"

    near_bid = abs(fill - bid)
    near_ask = abs(fill - ask)
    tol = max(spread * 0.2, 0.02)

    if near_ask <= tol and near_ask <= near_bid:
        return "ASK"
    if near_bid <= tol and near_bid < near_ask:
        return "BID"
    return "MID"

def classify_direction(option_type: str, side: str) -> str:
    option_type = option_type.upper()
    side = side.upper()

    if option_type == "CALL":
        if side in ("ASK", "MID"):
            return "BULLISH"
        if side == "BID":
            return "BEARISH"
    elif option_type == "PUT":
        if side == "BID":
            return "BULLISH"
        if side in ("ASK", "MID"):
            return "BEARISH"
    return "NEUTRAL"

def compute_unusual_score(
    premium: float,
    volume: float,
    open_interest: float,
    spread_pct: float,
    pct_otm: float,
    dte: int,
    side: str,
) -> float:
    premium_score = min(math.log10(max(premium, 0) + 1.0) / 6.5, 1.0)
    abs_size_score = min(max(volume, 0) / 2000.0, 1.0)
    voi_score = min((max(volume, 0) / max(open_interest, 1.0)) / 5.0, 1.0)
    otm_score = min(max(pct_otm, 0) / 0.12, 1.0)
    dte_score = 1.0 if dte <= 45 else max(0.2, 1.0 - ((dte - 45) / 365.0))
    spread_penalty = min(max(spread_pct, 0) / 100.0, 1.0)
    side_bonus = 0.08 if side in ("ASK", "BID") else 0.03 if side == "MID" else 0.0

    raw = (
        0.38 * premium_score
        + 0.20 * abs_size_score
        + 0.22 * voi_score
        + 0.10 * otm_score
        + 0.10 * dte_score
        + side_bonus
        - 0.18 * spread_penalty
    )
    return float(np.clip(raw, 0.0, 1.0) * 100.0)

def load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def save_pickle(obj, path: Path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

def save_json(obj: dict, path: Path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Deterministic universe loader
# -----------------------------------------------------------------------------
def get_sp500_universe() -> pd.DataFrame:
    cached = load_pickle(SP500_META_CACHE)
    if isinstance(cached, pd.DataFrame) and not cached.empty and "symbol" in cached.columns:
        cached = cached.copy()
        cached["symbol"] = cached["symbol"].astype(str).str.upper()
        cached = cached.drop_duplicates("symbol").reset_index(drop=True)
        if len(cached) >= 500:
            return cached

    df = pd.DataFrame({"symbol": SP500_FALLBACK_SYMBOLS})
    df["security"] = df["symbol"]
    df["sector"] = "Unknown"
    df["sub_industry"] = "Unknown"
    df = df.drop_duplicates("symbol").reset_index(drop=True)
    save_pickle(df, SP500_META_CACHE)
    return df

# -----------------------------------------------------------------------------
# Batch spot snapshot
# -----------------------------------------------------------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_spot_snapshot(symbols: List[str]) -> pd.DataFrame:
    try:
        data = yf.download(
            tickers=symbols,
            period="15d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume"])

    rows = []
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["symbol", "spot", "prev_close", "avg_20d_vol", "day_volume"])

    multi = isinstance(data.columns, pd.MultiIndex)

    for sym in symbols:
        try:
            if multi:
                if sym not in data.columns.get_level_values(0):
                    continue
                sub = data[sym].copy()
            else:
                sub = data.copy()

            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue

            close = sub["Close"].dropna()
            vol = sub["Volume"].dropna() if "Volume" in sub.columns else pd.Series(dtype=float)

            if close.empty:
                continue

            spot = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else spot
            avg_20d_vol = float(vol.tail(20).mean()) if len(vol) > 0 else np.nan
            day_volume = float(vol.iloc[-1]) if len(vol) > 0 else np.nan

            rows.append({
                "symbol": sym,
                "spot": spot,
                "prev_close": prev_close,
                "avg_20d_vol": avg_20d_vol,
                "day_volume": day_volume,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Option scan
# -----------------------------------------------------------------------------
def scan_one_symbol(
    symbol: str,
    spot_map: Dict[str, float],
    min_premium: float,
    min_volume: int,
    min_vol_oi: float,
    max_spread_pct: float,
    dte_min: int,
    dte_max: int,
) -> Tuple[List[dict], dict]:
    diagnostics = {
        "symbol": symbol,
        "expiries_seen": 0,
        "contracts_kept": 0,
        "errors": 0,
    }
    rows: List[dict] = []

    try:
        tkr = yf.Ticker(symbol)
        expiries = list(tkr.options or [])
    except Exception:
        diagnostics["errors"] += 1
        return rows, diagnostics

    if not expiries:
        return rows, diagnostics

    spot = safe_float(spot_map.get(symbol, np.nan), np.nan)
    if pd.isna(spot) or spot <= 0:
        try:
            hist = tkr.history(period="5d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                spot = safe_float(hist["Close"].dropna().iloc[-1], np.nan)
        except Exception:
            pass

    if pd.isna(spot) or spot <= 0:
        return rows, diagnostics

    today = pd.Timestamp.utcnow().normalize()

    for exp in expiries:
        diagnostics["expiries_seen"] += 1
        try:
            chain = tkr.option_chain(exp)
            chain_parts = [("CALL", chain.calls), ("PUT", chain.puts)]
        except Exception:
            diagnostics["errors"] += 1
            continue

        try:
            expiry_ts = pd.Timestamp(exp)
            dte = int((expiry_ts.normalize() - today.tz_localize(None)).days)
        except Exception:
            continue

        if dte < dte_min or dte > dte_max:
            continue

        for option_type, df in chain_parts:
            if df is None or df.empty:
                continue

            local = df.copy()
            for col in ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
                if col not in local.columns:
                    local[col] = np.nan

            local = local[["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]]

            for _, r in local.iterrows():
                strike = safe_float(r["strike"], np.nan)
                if pd.isna(strike) or strike <= 0:
                    continue

                last_price = safe_float(r["lastPrice"], 0.0)
                bid = safe_float(r["bid"], 0.0)
                ask = safe_float(r["ask"], 0.0)
                volume = safe_float(r["volume"], 0.0)
                oi = safe_float(r["openInterest"], 0.0)
                iv = safe_float(r["impliedVolatility"], np.nan)

                if volume < min_volume:
                    continue

                fill = estimate_fill_price(last_price, bid, ask)
                if fill <= 0:
                    continue

                premium = fill * volume * 100.0
                if premium < min_premium:
                    continue

                vol_oi = volume / max(oi, 1.0)
                if vol_oi < min_vol_oi:
                    continue

                mid = midpoint(bid, ask)
                spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 and ask >= bid and bid > 0 else np.nan
                if not pd.isna(spread_pct) and spread_pct > max_spread_pct:
                    continue

                side = classify_side(fill, bid, ask)
                direction = classify_direction(option_type, side)

                if option_type == "CALL":
                    pct_otm = max((strike - spot) / spot, 0.0)
                    moneyness = "OTM" if strike > spot else "ITM" if strike < spot else "ATM"
                else:
                    pct_otm = max((spot - strike) / spot, 0.0)
                    moneyness = "OTM" if strike < spot else "ITM" if strike > spot else "ATM"

                score = compute_unusual_score(
                    premium=premium,
                    volume=volume,
                    open_interest=oi,
                    spread_pct=0.0 if pd.isna(spread_pct) else spread_pct,
                    pct_otm=pct_otm,
                    dte=dte,
                    side=side,
                )

                rows.append({
                    "symbol": symbol,
                    "contract_symbol": r["contractSymbol"],
                    "type": option_type,
                    "expiry": exp,
                    "dte": int(dte),
                    "spot": float(spot),
                    "strike": float(strike),
                    "moneyness": moneyness,
                    "pct_otm": float(pct_otm * 100.0),
                    "side": side,
                    "direction": direction,
                    "bid": float(bid),
                    "ask": float(ask),
                    "mid": float(mid),
                    "fill_est": float(fill),
                    "volume": float(volume),
                    "open_interest": float(oi),
                    "vol_oi": float(vol_oi),
                    "premium": float(premium),
                    "spread_pct": float(spread_pct) if not pd.isna(spread_pct) else np.nan,
                    "iv": float(iv) if not pd.isna(iv) else np.nan,
                    "unusual_score": float(score),
                })
                diagnostics["contracts_kept"] += 1

    return rows, diagnostics

def build_cluster_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotent and safe to call multiple times.
    Rebuilds cluster columns from scratch every time.
    """
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()

    stale_cols = [
        "cluster_key",
        "cluster_contracts",
        "cluster_premium",
        "cluster_volume",
        "cluster_score",
        "cluster_flag",
    ]
    existing_stale = [c for c in stale_cols if c in out.columns]
    if existing_stale:
        out = out.drop(columns=existing_stale, errors="ignore")

    required = ["symbol", "expiry", "type", "direction", "contract_symbol", "premium", "volume", "unusual_score"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        for c in ["cluster_contracts", "cluster_premium", "cluster_volume", "cluster_score", "cluster_flag"]:
            out[c] = np.nan if c != "cluster_flag" else ""
        return out

    out["cluster_key"] = (
        out["symbol"].astype(str) + "|" +
        out["expiry"].astype(str) + "|" +
        out["type"].astype(str) + "|" +
        out["direction"].astype(str)
    )

    grp = (
        out.groupby("cluster_key", dropna=False)
        .agg(
            cluster_contracts=("contract_symbol", "count"),
            cluster_premium=("premium", "sum"),
            cluster_volume=("volume", "sum"),
            cluster_score=("unusual_score", "mean"),
        )
        .reset_index()
    )

    if grp.empty:
        out["cluster_contracts"] = 1
        out["cluster_premium"] = out["premium"]
        out["cluster_volume"] = out["volume"]
        out["cluster_score"] = out["unusual_score"]
        out["cluster_flag"] = ""
        return out

    out = out.merge(grp, on="cluster_key", how="left", validate="many_to_one")

    for col, fill in {
        "cluster_contracts": 1,
        "cluster_premium": 0.0,
        "cluster_volume": 0.0,
        "cluster_score": 0.0,
    }.items():
        if col not in out.columns:
            out[col] = fill
        out[col] = out[col].fillna(fill)

    out["cluster_flag"] = np.where(
        (out["cluster_contracts"] >= 3) & (out["cluster_premium"] >= 250_000),
        "CLUSTER",
        "",
    )
    return out

def run_full_scan(
    symbols: List[str],
    spot_snapshot: pd.DataFrame,
    min_premium: float,
    min_volume: int,
    min_vol_oi: float,
    max_spread_pct: float,
    dte_min: int,
    dte_max: int,
    max_workers: int,
):
    spot_map = {}
    if spot_snapshot is not None and not spot_snapshot.empty and "symbol" in spot_snapshot.columns and "spot" in spot_snapshot.columns:
        spot_map = dict(zip(spot_snapshot["symbol"], spot_snapshot["spot"]))

    all_rows = []
    all_diags = []

    progress_bar = st.progress(0.0)
    status = st.empty()

    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                scan_one_symbol,
                sym,
                spot_map,
                min_premium,
                min_volume,
                min_vol_oi,
                max_spread_pct,
                dte_min,
                dte_max,
            ): sym
            for sym in symbols
        }

        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                rows, diag = fut.result()
                if rows:
                    all_rows.extend(rows)
                all_diags.append(diag)
            except Exception:
                all_diags.append({"symbol": sym, "expiries_seen": 0, "contracts_kept": 0, "errors": 1})

            completed += 1
            progress_bar.progress(completed / max(total, 1))
            status.caption(f"Scanning options chains... {completed:,}/{total:,} symbols completed")

    progress_bar.empty()
    status.empty()

    flow = pd.DataFrame(all_rows)
    diags = pd.DataFrame(all_diags)

    if not flow.empty:
        flow = build_cluster_flags(flow)
        sort_cols = [c for c in ["unusual_score", "premium", "vol_oi", "volume"] if c in flow.columns]
        if sort_cols:
            flow = flow.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    return flow, diags

def summarize_tape(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No qualifying flow made it through the current filters."

    bullish = df.loc[df["direction"] == "BULLISH", "premium"].sum() if "direction" in df.columns else 0.0
    bearish = df.loc[df["direction"] == "BEARISH", "premium"].sum() if "direction" in df.columns else 0.0
    top = df.iloc[0]

    leader = (
        f"The tape is led by {top['symbol']} {top['type'].lower()} flow into "
        f"{top['expiry']} {top['strike']:.0f}s, estimated premium {human_money(top['premium'])}, "
        f"volume/OI {top['vol_oi']:.2f}x, score {top['unusual_score']:.1f}."
    )

    if bullish > bearish * 1.2:
        tilt = f"Aggregate directional premium is net bullish, with {human_money(bullish)} bullish versus {human_money(bearish)} bearish."
    elif bearish > bullish * 1.2:
        tilt = f"Aggregate directional premium is net bearish, with {human_money(bearish)} bearish versus {human_money(bullish)} bullish."
    else:
        tilt = f"Directional premium is relatively balanced, with {human_money(bullish)} bullish versus {human_money(bearish)} bearish."

    clusters = int((df["cluster_flag"] == "CLUSTER").sum()) if "cluster_flag" in df.columns else 0
    cluster_line = (
        f"There are {clusters} contracts sitting inside same-expiry directional clusters."
        if clusters > 0 else
        "The scan is more single-print than cluster-driven right now."
    )
    return f"{leader} {tilt} {cluster_line}"

def prep_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "premium" in out.columns:
        out["Premium"] = out["premium"].map(human_money)
    if "volume" in out.columns:
        out["Vol"] = out["volume"].map(human_num)
    if "open_interest" in out.columns:
        out["OI"] = out["open_interest"].map(human_num)
    if "vol_oi" in out.columns:
        out["Vol/OI"] = out["vol_oi"].map(lambda x: f"{x:.2f}x")
    if "fill_est" in out.columns:
        out["Fill"] = out["fill_est"].map(lambda x: f"{x:.2f}")
    if "bid" in out.columns:
        out["Bid"] = out["bid"].map(lambda x: f"{x:.2f}")
    if "ask" in out.columns:
        out["Ask"] = out["ask"].map(lambda x: f"{x:.2f}")
    if "spread_pct" in out.columns:
        out["Spread %"] = out["spread_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    if "iv" in out.columns:
        out["IV"] = out["iv"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    if "pct_otm" in out.columns:
        out["OTM %"] = out["pct_otm"].map(lambda x: f"{x:.1f}%")
    if "unusual_score" in out.columns:
        out["Score"] = out["unusual_score"].map(lambda x: f"{x:.1f}")

    keep = [
        "symbol", "type", "expiry", "dte", "direction", "side", "strike", "spot",
        "moneyness", "OTM %", "Fill", "Bid", "Ask", "Spread %", "Vol", "OI",
        "Vol/OI", "Premium", "IV", "Score", "cluster_flag", "contract_symbol"
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This scanner is built to mimic how a flow trader triages large options activity using public data.

        What it does
        • Uses a deterministic local S&P 500 universe so the app does not die when public constituent pages fail
        • Checks every symbol every run
        • Walks every listed expiry Yahoo exposes for that symbol
        • Scores contracts by premium, volume/OI, side estimate, DTE, OTM distance, and spread quality
        • Flags directional clusters across the same symbol, expiry, and contract side

        What it cannot do
        • True exchange-level time-and-sales
        • Official sweep / split / multi-leg flags
        • Exact opening-versus-closing print classification
        """
    )

    st.subheader("Filters")
    min_premium = st.number_input("Minimum premium ($)", min_value=0.0, value=float(DEFAULT_MIN_PREMIUM), step=25_000.0, format="%.0f")
    min_volume = st.number_input("Minimum contract volume", min_value=1, value=int(DEFAULT_MIN_VOLUME), step=10)
    min_vol_oi = st.number_input("Minimum volume / OI", min_value=0.0, value=float(DEFAULT_MIN_VOL_OI), step=0.25, format="%.2f")
    max_spread_pct = st.number_input("Maximum spread %", min_value=0.0, value=float(DEFAULT_MAX_SPREAD_PCT), step=5.0, format="%.1f")
    dte_range = st.slider("DTE range", min_value=0, max_value=1095, value=(DEFAULT_DTE_MIN, DEFAULT_DTE_MAX), step=1)
    top_n = st.slider("Rows to display", min_value=25, max_value=500, value=DEFAULT_TOP_N, step=25)

    st.subheader("Engine")
    max_workers = st.slider("Concurrent workers", min_value=4, max_value=40, value=DEFAULT_MAX_WORKERS, step=2)
    allow_cached_fallback = st.toggle("Use last-good cached scan if live scan returns nothing", value=True)

    st.markdown(
        '<div class="small-note">Cluster logic is now rebuild-safe, so reruns and cached frames should not throw merge-related KeyErrors.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
scan_started = time.time()
live_scan_used = True

universe = get_sp500_universe()
symbols = universe["symbol"].dropna().astype(str).str.upper().unique().tolist()

with st.spinner(f"Loading spot snapshot for {len(symbols):,} symbols..."):
    spot_snapshot = fetch_spot_snapshot(symbols)

with st.spinner(f"Running full options scan across {len(symbols):,} S&P 500 symbols and all listed expiries..."):
    try:
        flow, diags = run_full_scan(
            symbols=symbols,
            spot_snapshot=spot_snapshot,
            min_premium=min_premium,
            min_volume=min_volume,
            min_vol_oi=min_vol_oi,
            max_spread_pct=max_spread_pct,
            dte_min=dte_range[0],
            dte_max=dte_range[1],
            max_workers=max_workers,
        )
    except Exception as e:
        st.warning(f"Live scan hit an error: {e}")
        flow = pd.DataFrame()
        diags = pd.DataFrame()

if (flow is None or flow.empty) and allow_cached_fallback:
    cached_flow = load_pickle(LAST_GOOD_SCAN)
    cached_meta = load_json(LAST_GOOD_META)
    if isinstance(cached_flow, pd.DataFrame) and not cached_flow.empty:
        flow = build_cluster_flags(cached_flow.copy())
        live_scan_used = False
        st.warning(
            f"Live scan returned no qualifying rows. Showing last-good cached snapshot from "
            f"{cached_meta.get('saved_at', 'unknown time') if isinstance(cached_meta, dict) else 'unknown time'}."
        )

if flow is not None and not flow.empty and live_scan_used:
    flow = build_cluster_flags(flow)
    save_pickle(flow, LAST_GOOD_SCAN)
    save_json(
        {
            "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": int(len(flow)),
            "symbols_requested": int(len(symbols)),
        },
        LAST_GOOD_META,
    )

scan_seconds = time.time() - scan_started

if flow is None:
    flow = pd.DataFrame()
if diags is None:
    diags = pd.DataFrame()

if not flow.empty:
    flow = build_cluster_flags(flow)

qualifying_rows = int(len(flow))
symbols_requested = int(len(symbols))
symbols_completed = int(diags["symbol"].nunique()) if not diags.empty and "symbol" in diags.columns else 0
expiries_seen = int(diags["expiries_seen"].sum()) if not diags.empty and "expiries_seen" in diags.columns else 0
scan_errors = int(diags["errors"].sum()) if not diags.empty and "errors" in diags.columns else 0

bullish_premium = float(flow.loc[flow["direction"] == "BULLISH", "premium"].sum()) if not flow.empty and "direction" in flow.columns else 0.0
bearish_premium = float(flow.loc[flow["direction"] == "BEARISH", "premium"].sum()) if not flow.empty and "direction" in flow.columns else 0.0
total_premium = float(flow["premium"].sum()) if not flow.empty and "premium" in flow.columns else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Symbols requested", f"{symbols_requested:,}")
c2.metric("Symbols completed", f"{symbols_completed:,}")
c3.metric("Expiries checked", f"{expiries_seen:,}")
c4.metric("Qualifying rows", f"{qualifying_rows:,}")
c5.metric("Bullish premium", human_money(bullish_premium))
c6.metric("Bearish premium", human_money(bearish_premium))

st.caption(
    f"Scan mode: {'Live' if live_scan_used else 'Cached fallback'} | "
    f"Total qualifying premium: {human_money(total_premium)} | "
    f"Errors: {scan_errors:,} | "
    f"Elapsed: {scan_seconds:.1f}s"
)

if flow.empty:
    st.info("No rows passed the current filters. Lower premium, volume, volume/OI, or spread thresholds and rerun.")
    st.stop()

flow["rank"] = np.arange(1, len(flow) + 1)

overall = flow.nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
bullish = flow[flow["direction"] == "BULLISH"].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
bearish = flow[flow["direction"] == "BEARISH"].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])
zero_dte = flow[flow["dte"] == 0].nlargest(top_n, ["unusual_score", "premium", "vol_oi"])

def sum_where(group_df: pd.DataFrame, direction: str) -> float:
    if group_df.empty:
        return 0.0
    return float(group_df.loc[group_df["direction"] == direction, "premium"].sum())

ticker_summary = (
    flow.groupby("symbol", dropna=False, as_index=False)
    .apply(lambda g: pd.Series({
        "rows": int(len(g)),
        "total_premium": float(g["premium"].sum()),
        "bullish_premium": sum_where(g, "BULLISH"),
        "bearish_premium": sum_where(g, "BEARISH"),
        "avg_score": float(g["unusual_score"].mean()),
        "max_score": float(g["unusual_score"].max()),
    }))
    .reset_index(drop=True)
)

ticker_summary["net_premium"] = ticker_summary["bullish_premium"] - ticker_summary["bearish_premium"]
ticker_summary["flow_bias"] = np.where(
    ticker_summary["net_premium"] > 0, "BULLISH",
    np.where(ticker_summary["net_premium"] < 0, "BEARISH", "BALANCED")
)
ticker_summary = ticker_summary.sort_values(
    ["total_premium", "max_score", "rows"], ascending=[False, False, False]
).reset_index(drop=True)

st.subheader("Tape Commentary")
st.write(summarize_tape(overall))

leaderboard_cols = st.columns([1.2, 1.2, 1.1, 1.1])
top_contract = f"{overall.iloc[0]['symbol']} {overall.iloc[0]['expiry']} {overall.iloc[0]['type']} {overall.iloc[0]['strike']:.0f}"
leaderboard_cols[0].metric("Top ticker by premium", str(ticker_summary.iloc[0]["symbol"]))
leaderboard_cols[1].metric("Top contract", top_contract)
leaderboard_cols[2].metric("0DTE rows", f"{len(zero_dte):,}")
leaderboard_cols[3].metric("Cluster flags", f"{int((flow['cluster_flag'] == 'CLUSTER').sum()) if 'cluster_flag' in flow.columns else 0:,}")

left, right = st.columns(2)

with left:
    call_put = flow.groupby("type", as_index=False)["premium"].sum()
    fig_cp = px.bar(call_put, x="type", y="premium", title="Premium by Contract Type")
    fig_cp.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_cp, use_container_width=True)

with right:
    bubble = overall.copy()
    bubble["label"] = bubble["symbol"] + " " + bubble["type"] + " " + bubble["expiry"]
    fig_bubble = px.scatter(
        bubble.head(300),
        x="vol_oi",
        y="premium",
        size="unusual_score",
        color="direction",
        hover_name="label",
        hover_data={
            "strike": True,
            "dte": True,
            "side": True,
            "pct_otm": ":.1f",
            "spread_pct": ":.1f",
            "unusual_score": ":.1f",
            "vol_oi": ":.2f",
            "premium": ":,.0f",
        },
        title="Flow Map: Premium vs Volume/OI",
    )
    fig_bubble.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_bubble, use_container_width=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Top Flow", "Bullish", "Bearish", "0DTE", "By Ticker", "Download"]
)

with tab1:
    st.dataframe(prep_display(overall), use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(prep_display(bullish), use_container_width=True, hide_index=True)

with tab3:
    st.dataframe(prep_display(bearish), use_container_width=True, hide_index=True)

with tab4:
    if zero_dte.empty:
        st.info("No 0DTE rows passed the current filters.")
    else:
        st.dataframe(prep_display(zero_dte), use_container_width=True, hide_index=True)

with tab5:
    display_ticker = ticker_summary.copy()
    display_ticker["Total Premium"] = display_ticker["total_premium"].map(human_money)
    display_ticker["Bullish Premium"] = display_ticker["bullish_premium"].map(human_money)
    display_ticker["Bearish Premium"] = display_ticker["bearish_premium"].map(human_money)
    display_ticker["Net Premium"] = display_ticker["net_premium"].map(human_money)
    display_ticker["Avg Score"] = display_ticker["avg_score"].map(lambda x: f"{x:.1f}")
    display_ticker["Max Score"] = display_ticker["max_score"].map(lambda x: f"{x:.1f}")
    display_ticker = display_ticker[
        ["symbol", "rows", "Total Premium", "Bullish Premium", "Bearish Premium", "Net Premium", "flow_bias", "Avg Score", "Max Score"]
    ]
    st.dataframe(display_ticker.head(300), use_container_width=True, hide_index=True)

    ticker_pick = st.selectbox("Inspect ticker", options=ticker_summary["symbol"].tolist(), index=0)
    ticker_rows = flow[flow["symbol"] == ticker_pick].nlargest(200, ["unusual_score", "premium", "vol_oi"])
    st.dataframe(prep_display(ticker_rows), use_container_width=True, hide_index=True)

with tab6:
    csv_bytes = flow.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full scan CSV",
        data=csv_bytes,
        file_name=f"unusual_options_flow_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    if not diags.empty:
        st.dataframe(diags.sort_values(["errors", "expiries_seen"], ascending=[False, False]), use_container_width=True, hide_index=True)

st.caption("Cluster logic is now idempotent and safe on reruns, cached frames, and repeated post-processing.")
