"""Core calculations and catalog for the Sector Breadth and Rotation page."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from adfm_sector_rotation_config import MAJOR_SECTORS, SUBSECTOR_ROWS

NEUTRAL_BAND = 0.005
CONFIRMATION_SESSIONS = 4
MOVEMENT_SESSIONS = 5
PARENT_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

SUPPLEMENTAL_ETFS = [
    {"Ticker": "XSW", "Name": "Software equal-weight", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "IHAK", "Name": "Cybersecurity diversified", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "SOXQ", "Name": "Semiconductors broad alternative", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "IPAY", "Name": "Digital payments", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "CRUZ", "Name": "Hotels, airlines and cruise lines", "Sector Group": "Consumer Discretionary", "Tier": "Thematic"},
    {"Ticker": "SRVR", "Name": "Data-center and digital-infrastructure REITs", "Sector Group": "Real Estate", "Tier": "Thematic"},
    {"Ticker": "URNM", "Name": "Uranium miners", "Sector Group": "Materials", "Tier": "Thematic"},
]

# Equal-weight, transparent baskets are used where a broad ETF obscures the
# industry exposure. Constituents are intentionally compact and auditable.
STOCK_BASKETS: List[Dict[str, object]] = [
    {"Id": "BASKET_CUSTODY", "Name": "Custody & Trust Banks", "Sector Group": "Financials", "Members": ["BK", "STT", "NTRS"]},
    {"Id": "BASKET_EXCHANGES", "Name": "Exchanges & Market Infrastructure", "Sector Group": "Financials", "Members": ["CME", "ICE", "NDAQ", "CBOE", "MKTX"]},
    {"Id": "BASKET_PAYMENTS", "Name": "Payments Networks", "Sector Group": "Financials", "Members": ["V", "MA", "AXP", "FI", "GPN"]},
    {"Id": "BASKET_CONSUMER_FIN", "Name": "Consumer Finance", "Sector Group": "Financials", "Members": ["COF", "SYF", "DFS", "ALLY", "OMF"]},
    {"Id": "BASKET_MORTGAGE_FIN", "Name": "Mortgage Finance", "Sector Group": "Financials", "Members": ["RKT", "UWMC", "PFSI", "COOP"]},
    {"Id": "BASKET_ALT_MGRS", "Name": "Alternative Asset Managers", "Sector Group": "Financials", "Members": ["BX", "KKR", "APO", "ARES", "CG", "OWL"]},
    {"Id": "BASKET_SEMICAP", "Name": "Semiconductor Equipment", "Sector Group": "Technology", "Members": ["AMAT", "LRCX", "KLAC", "ASML", "TER", "ONTO"]},
    {"Id": "BASKET_NETWORKING", "Name": "Data-Center Networking", "Sector Group": "Technology", "Members": ["ANET", "CSCO", "CIEN", "JNPR", "LITE", "COHR"]},
    {"Id": "BASKET_DATACENTER_HW", "Name": "Data-Center Hardware", "Sector Group": "Technology", "Members": ["DELL", "HPE", "SMCI", "VRT", "NTAP", "PSTG"]},
    {"Id": "BASKET_HARDWARE", "Name": "IT Hardware", "Sector Group": "Technology", "Members": ["AAPL", "DELL", "HPQ", "HPE", "LOGI"]},
    {"Id": "BASKET_CYBER", "Name": "Cybersecurity Leaders", "Sector Group": "Technology", "Members": ["PANW", "CRWD", "FTNT", "ZS", "OKTA", "CYBR"]},
    {"Id": "BASKET_SOFTWARE", "Name": "Enterprise Software", "Sector Group": "Technology", "Members": ["MSFT", "ORCL", "NOW", "CRM", "ADBE", "INTU"]},
    {"Id": "BASKET_ELECTRICAL", "Name": "Electrical Equipment", "Sector Group": "Industrials", "Members": ["ETN", "HUBB", "EMR", "ROK", "AME", "VRT"]},
    {"Id": "BASKET_MACHINERY", "Name": "Industrial Machinery", "Sector Group": "Industrials", "Members": ["CAT", "DE", "PCAR", "CMI", "PH", "IR"]},
    {"Id": "BASKET_EC", "Name": "Engineering & Construction", "Sector Group": "Industrials", "Members": ["PWR", "EME", "FIX", "MTZ", "J", "ACM"]},
    {"Id": "BASKET_LOGISTICS", "Name": "Logistics & Parcel", "Sector Group": "Industrials", "Members": ["UPS", "FDX", "EXPD", "CHRW", "XPO", "ODFL"]},
    {"Id": "BASKET_DEFENSE", "Name": "Defense Primes", "Sector Group": "Industrials", "Members": ["LMT", "NOC", "RTX", "GD", "LHX", "HII"]},
    {"Id": "BASKET_RESTAURANTS", "Name": "Restaurants", "Sector Group": "Consumer Discretionary", "Members": ["MCD", "SBUX", "CMG", "YUM", "DRI", "CAVA"]},
    {"Id": "BASKET_HOTELS", "Name": "Hotels", "Sector Group": "Consumer Discretionary", "Members": ["MAR", "HLT", "H", "IHG", "WH"]},
    {"Id": "BASKET_CRUISE", "Name": "Cruise Lines", "Sector Group": "Consumer Discretionary", "Members": ["CCL", "RCL", "NCLH"]},
    {"Id": "BASKET_LUXURY", "Name": "Luxury Goods", "Sector Group": "Consumer Discretionary", "Members": ["CPRI", "TPR", "RACE", "LVMUY", "PPRUY"]},
    {"Id": "BASKET_APPAREL", "Name": "Apparel & Footwear", "Sector Group": "Consumer Discretionary", "Members": ["NKE", "LULU", "DECK", "RL", "PVH", "VFC"]},
    {"Id": "BASKET_APARTMENTS", "Name": "Apartment REITs", "Sector Group": "Real Estate", "Members": ["AVB", "EQR", "ESS", "MAA", "UDR", "CPT"]},
    {"Id": "BASKET_INDUSTRIAL_RE", "Name": "Industrial REITs", "Sector Group": "Real Estate", "Members": ["PLD", "REXR", "FR", "EGP", "STAG"]},
    {"Id": "BASKET_OFFICE_RE", "Name": "Office REITs", "Sector Group": "Real Estate", "Members": ["BXP", "VNO", "KRC", "HIW", "SLG"]},
    {"Id": "BASKET_RETAIL_RE", "Name": "Retail REITs", "Sector Group": "Real Estate", "Members": ["SPG", "REG", "FRT", "KIM", "O"]},
    {"Id": "BASKET_DIGITAL_RE", "Name": "Towers & Data-Center REITs", "Sector Group": "Real Estate", "Members": ["AMT", "CCI", "EQIX", "DLR", "SBAC"]},
    {"Id": "BASKET_REFINERS", "Name": "Refiners", "Sector Group": "Energy", "Members": ["VLO", "MPC", "PSX", "PBF", "DK"]},
    {"Id": "BASKET_INTEGRATED_LNG", "Name": "Integrated Oil & LNG", "Sector Group": "Energy", "Members": ["XOM", "CVX", "SHEL", "BP", "LNG", "CTRA"]},
    {"Id": "BASKET_COAL", "Name": "Coal Producers", "Sector Group": "Energy", "Members": ["BTU", "ARCH", "CEIX", "HCC", "AMR"]},
    {"Id": "BASKET_CHEM_FERT", "Name": "Chemicals & Fertilizers", "Sector Group": "Materials", "Members": ["LIN", "APD", "SHW", "CF", "MOS", "NTR"]},
    {"Id": "BASKET_METALS", "Name": "Copper & Precious-Metal Miners", "Sector Group": "Materials", "Members": ["FCX", "SCCO", "NEM", "AEM", "GOLD"]},
    {"Id": "BASKET_MANAGED_CARE", "Name": "Managed Care", "Sector Group": "Health Care", "Members": ["UNH", "ELV", "CI", "HUM", "CNC", "MOH"]},
    {"Id": "BASKET_HOSPITALS", "Name": "Hospitals", "Sector Group": "Health Care", "Members": ["HCA", "THC", "UHS", "CYH"]},
    {"Id": "BASKET_LIFE_SCI", "Name": "Life-Science Tools", "Sector Group": "Health Care", "Members": ["TMO", "DHR", "A", "IQV", "RVTY", "WAT"]},
    {"Id": "BASKET_MEDTECH", "Name": "Medical Technology", "Sector Group": "Health Care", "Members": ["ABT", "MDT", "SYK", "BSX", "EW", "ISRG"]},
]

COUNTRY_ETFS = [
    ("EFA", "Developed ex-US"), ("VEA", "Developed Markets"), ("EEM", "Emerging Markets"), ("VWO", "Emerging Markets Broad"),
    ("EZU", "Eurozone"), ("VGK", "Europe"), ("EWJ", "Japan"), ("EWG", "Germany"), ("EWQ", "France"), ("EWU", "United Kingdom"),
    ("EWI", "Italy"), ("EWP", "Spain"), ("EWL", "Switzerland"), ("EWN", "Netherlands"), ("EWD", "Sweden"), ("EWO", "Austria"),
    ("EWK", "Belgium"), ("EWY", "South Korea"), ("ASHR", "China A-Shares"), ("FXI", "China Large Caps"), ("MCHI", "China Broad"),
    ("EWT", "Taiwan"), ("INDA", "India"), ("EWS", "Singapore"), ("EWA", "Australia"), ("EWH", "Hong Kong"), ("EPHE", "Philippines"),
    ("EWM", "Malaysia"), ("IDX", "Indonesia"), ("THD", "Thailand"), ("VNM", "Vietnam"), ("EWZ", "Brazil"), ("EWW", "Mexico"),
    ("EWC", "Canada"), ("EPU", "Peru"), ("ECH", "Chile"), ("ARGT", "Argentina"), ("GXG", "Colombia"), ("TUR", "Turkey"), ("EZA", "South Africa"),
]


def build_catalog() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ticker, name in MAJOR_SECTORS.items():
        rows.append({
            "Id": ticker, "Ticker": ticker, "Name": name, "Sector Group": name,
            "Universe": "Sectors", "Kind": "ETF", "Parent": ticker,
            "Benchmark": "SPY", "Members": None,
        })
    for row in list(SUBSECTOR_ROWS) + SUPPLEMENTAL_ETFS:
        universe = "Industries" if row["Tier"] == "Core" else "Themes"
        rows.append({
            "Id": row["Ticker"], "Ticker": row["Ticker"], "Name": row["Name"],
            "Sector Group": row["Sector Group"], "Universe": universe, "Kind": "ETF",
            "Parent": PARENT_ETF[row["Sector Group"]], "Benchmark": "SPY", "Members": None,
        })
    for row in STOCK_BASKETS:
        rows.append({
            "Id": row["Id"], "Ticker": row["Id"], "Name": row["Name"],
            "Sector Group": row["Sector Group"], "Universe": "Industries", "Kind": "Stock Basket",
            "Parent": PARENT_ETF[row["Sector Group"]], "Benchmark": "SPY", "Members": tuple(row["Members"]),
        })
    for ticker, name in COUNTRY_ETFS:
        rows.append({
            "Id": ticker, "Ticker": ticker, "Name": name, "Sector Group": "Global Equities",
            "Universe": "Countries", "Kind": "ETF", "Parent": "ACWI", "Benchmark": "ACWI", "Members": None,
        })
    out = pd.DataFrame(rows)
    if out["Id"].duplicated().any():
        dups = out.loc[out["Id"].duplicated(keep=False), "Id"].tolist()
        raise ValueError(f"Duplicate sector-rotation identifiers: {dups}")
    return out


def classify_coordinates(long_value: float, short_value: float, neutral_band: float = NEUTRAL_BAND) -> str:
    if pd.isna(long_value) or pd.isna(short_value):
        return "Neutral"
    if abs(float(long_value)) <= neutral_band or abs(float(short_value)) <= neutral_band:
        return "Neutral"
    if long_value > 0 and short_value > 0:
        return "Leading"
    if long_value < 0 and short_value > 0:
        return "Improving"
    if long_value < 0 and short_value < 0:
        return "Lagging"
    return "Weakening"


def movement_from_coordinates(x: pd.Series, y: pd.Series, periods: int = MOVEMENT_SESSIONS) -> Dict[str, float]:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(frame) <= periods:
        return {"dx": np.nan, "dy": np.nan, "speed": np.nan, "angle": np.nan}
    dx = float(frame.iloc[-1, 0] - frame.iloc[-1 - periods, 0])
    dy = float(frame.iloc[-1, 1] - frame.iloc[-1 - periods, 1])
    return {
        "dx": dx, "dy": dy, "speed": float(np.hypot(dx, dy)),
        "angle": float(np.degrees(np.arctan2(dy, dx))),
    }


def confirm_state_series(raw_states: pd.Series, confirmation_sessions: int = CONFIRMATION_SESSIONS) -> tuple[pd.Series, pd.Series]:
    raw = raw_states.astype("object")
    if raw.empty:
        return raw.copy(), pd.Series(dtype="int64")
    confirmed: List[object] = []
    days: List[int] = []
    current = raw.iloc[0]
    candidate = None
    candidate_count = 0
    current_days = 0
    for value in raw:
        if pd.isna(value):
            confirmed.append(current)
            current_days += 1
            days.append(current_days)
            continue
        if value == current:
            candidate = None
            candidate_count = 0
        else:
            if value == candidate:
                candidate_count += 1
            else:
                candidate = value
                candidate_count = 1
            if candidate_count >= confirmation_sessions:
                current = candidate
                candidate = None
                candidate_count = 0
                current_days = 0
        current_days += 1
        confirmed.append(current)
        days.append(current_days)
    return pd.Series(confirmed, index=raw.index), pd.Series(days, index=raw.index, dtype="int64")


def _trailing_return(series: pd.Series, periods: int, offset: int = 0) -> float:
    s = pd.to_numeric(series, errors="coerce")
    end = len(s) - 1 - offset
    start = end - periods
    if start < 0 or end < 0:
        return np.nan
    a, b = s.iloc[start], s.iloc[end]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return float(b / a - 1.0)


def compute_relative_metrics(asset: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {k: np.nan for k in ("rel_1w", "rel_1m", "rel_3m", "rel_1w_change")}
    ratio = aligned["asset"] / aligned["benchmark"]
    one_week = _trailing_return(ratio, 5)
    prior_week = _trailing_return(ratio, 5, offset=5)
    return {
        "rel_1w": one_week,
        "rel_1m": _trailing_return(ratio, 21),
        "rel_3m": _trailing_return(ratio, 63),
        "rel_1w_change": one_week - prior_week if np.isfinite(one_week) and np.isfinite(prior_week) else np.nan,
    }


def extension_metrics(asset: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(asset, errors="coerce")
    if s.empty or pd.isna(s.iloc[-1]):
        return {"abs_1m": np.nan, "dist_50d": np.nan, "drawdown_52w": np.nan}
    last = float(s.iloc[-1])
    ma50 = s.iloc[-50:].mean() if len(s) >= 50 else np.nan
    high252 = s.iloc[-252:].max() if len(s) >= 2 else np.nan
    return {
        "abs_1m": _trailing_return(s, 21),
        "dist_50d": float(last / ma50 - 1.0) if np.isfinite(ma50) and ma50 != 0 else np.nan,
        "drawdown_52w": float(last / high252 - 1.0) if np.isfinite(high252) and high252 != 0 else np.nan,
    }


def _breadth_at(prices: pd.DataFrame, end_pos: int) -> tuple[float, float, int]:
    above50: List[bool] = []
    above200: List[bool] = []
    for col in prices.columns:
        s = pd.to_numeric(prices[col], errors="coerce")
        if end_pos < 0 or end_pos >= len(s) or pd.isna(s.iloc[end_pos]):
            continue
        hist = s.iloc[: end_pos + 1]
        if len(hist.dropna()) < 50:
            continue
        last = float(s.iloc[end_pos])
        ma50 = hist.iloc[-50:].mean()
        if np.isfinite(ma50):
            above50.append(last > ma50)
        if len(hist.dropna()) >= 200:
            ma200 = hist.iloc[-200:].mean()
            if np.isfinite(ma200):
                above200.append(last > ma200)
    p50 = float(np.mean(above50) * 100.0) if above50 else np.nan
    p200 = float(np.mean(above200) * 100.0) if above200 else np.nan
    return p50, p200, len(above50)


def compute_breadth(prices: pd.DataFrame) -> Dict[str, float]:
    if prices.empty:
        return {"above_50d": np.nan, "above_200d": np.nan, "breadth_1m_change": np.nan, "coverage": 0}
    p50, p200, coverage = _breadth_at(prices, len(prices) - 1)
    prior50, _, _ = _breadth_at(prices, max(0, len(prices) - 22))
    change = p50 - prior50 if np.isfinite(p50) and np.isfinite(prior50) else np.nan
    return {"above_50d": p50, "above_200d": p200, "breadth_1m_change": change, "coverage": coverage}


def synthetic_equal_weight_level(prices: pd.DataFrame) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    daily = prices.apply(pd.to_numeric, errors="coerce").pct_change(fill_method=None)
    basket_ret = daily.mean(axis=1, skipna=True).where(daily.notna().sum(axis=1) > 0)
    first_valid = basket_ret.first_valid_index()
    if first_valid is None:
        return pd.Series(index=prices.index, dtype=float)
    out = pd.Series(index=prices.index, dtype=float)
    out.loc[first_valid:] = (1.0 + basket_ret.loc[first_valid:].fillna(0.0)).cumprod() * 100.0
    return out


def rotation_history(asset: pd.Series, benchmark: pd.Series, short_window: int = 21, long_window: int = 63) -> pd.DataFrame:
    aligned = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["x", "y", "raw_state", "state", "days_in_state"])
    ratio = aligned["asset"] / aligned["benchmark"]
    x = ratio.pct_change(long_window, fill_method=None)
    y = ratio.pct_change(short_window, fill_method=None)
    raw = pd.Series([classify_coordinates(a, b) for a, b in zip(x, y, strict=True)], index=ratio.index)
    confirmed, days = confirm_state_series(raw)
    return pd.DataFrame({"x": x, "y": y, "raw_state": raw, "state": confirmed, "days_in_state": days})


def rank_change(metric_history: pd.DataFrame, periods: int = 5) -> pd.Series:
    if metric_history.empty or len(metric_history) <= periods:
        return pd.Series(index=metric_history.columns, dtype=float)
    current_rank = metric_history.iloc[-1].rank(ascending=False, method="min")
    prior_rank = metric_history.iloc[-1 - periods].rank(ascending=False, method="min")
    return prior_rank - current_rank


def adaptive_axis_range(values: Iterable[float], min_padding: float = 0.0025) -> tuple[float, float]:
    clean = pd.Series(list(values), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (-0.05, 0.05)
    lo, hi = float(clean.min()), float(clean.max())
    span = max(hi - lo, min_padding * 2)
    pad = max(min_padding, span * 0.10)
    lo, hi = lo - pad, hi + pad
    lo = min(lo, -min_padding)
    hi = max(hi, min_padding)
    return lo, hi


__all__ = [
    "CONFIRMATION_SESSIONS", "NEUTRAL_BAND", "PARENT_ETF", "STOCK_BASKETS",
    "adaptive_axis_range", "build_catalog", "classify_coordinates", "compute_breadth",
    "compute_relative_metrics", "confirm_state_series", "extension_metrics",
    "movement_from_coordinates", "rank_change", "rotation_history", "synthetic_equal_weight_level",
]
