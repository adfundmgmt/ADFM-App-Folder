"""Original macro voting, classifications, and evidence, preserved verbatim."""
from __future__ import annotations
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
TITLE = "Global Macro Regime"


SUBTITLE = "A linear cross-asset read of growth, inflation, rates, liquidity and risk confirmation."


TICKERS: Dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "RSP": "Equal Weight S&P 500",
    "IWM": "Russell 2000",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "FXI": "China Large Caps",
    "SMH": "Semiconductors",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "HYG": "High Yield Credit",
    "LQD": "Investment Grade Credit",
    "BKLN": "Senior Loans",
    "SHY": "1-3Y Treasuries",
    "IEF": "7-10Y Treasuries",
    "TLT": "20Y+ Treasuries",
    "TIP": "TIPS",
    "DX-Y.NYB": "U.S. Dollar Index",
    "UUP": "Dollar ETF",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "GC=F": "Gold",
    "HG=F": "Copper",
    "^VIX": "VIX",
}


PERFORMANCE_ROWS = [
    ("Equities", "SPY"),
    ("Equities", "QQQ"),
    ("Equities", "RSP"),
    ("Equities", "IWM"),
    ("Equities", "EFA"),
    ("Equities", "EEM"),
    ("Equities", "FXI"),
    ("Equities", "SMH"),
    ("Sectors", "XLF"),
    ("Sectors", "XLI"),
    ("Sectors", "XLE"),
    ("Credit", "HYG"),
    ("Credit", "LQD"),
    ("Rates", "IEF"),
    ("Rates", "TLT"),
    ("FX", "UUP"),
    ("Commodities", "GC=F"),
    ("Commodities", "CL=F"),
    ("Commodities", "HG=F"),
]


POSITIVE_RGB = ",".join(
    str(int(PASTEL["sage"][offset : offset + 2], 16)) for offset in (1, 3, 5)
)


NEGATIVE_RGB = ",".join(
    str(int(PASTEL["rose"][offset : offset + 2], 16)) for offset in (1, 3, 5)
)


def is_valid(value: object) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.loc[out.index.notna()].sort_index()


def latest_date(series: pd.Series | None) -> pd.Timestamp | None:
    clean = clean_series(series)
    return pd.Timestamp(clean.index[-1]) if not clean.empty else None


def value_asof(series: pd.Series | None, asof: pd.Timestamp) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index <= pd.Timestamp(asof)]
    return float(subset.iloc[-1]) if not subset.empty else np.nan


def pct_change_days(series: pd.Series | None, asof: pd.Timestamp, days: int) -> float:
    current = value_asof(series, asof)
    previous = value_asof(series, pd.Timestamp(asof) - timedelta(days=days))
    if not is_valid(current) or not is_valid(previous) or previous == 0:
        return np.nan
    return float((current / previous - 1.0) * 100.0)


def abs_change_days(series: pd.Series | None, asof: pd.Timestamp, days: int) -> float:
    current = value_asof(series, asof)
    previous = value_asof(series, pd.Timestamp(asof) - timedelta(days=days))
    if not is_valid(current) or not is_valid(previous):
        return np.nan
    return float(current - previous)


def ytd_change(series: pd.Series | None, asof: pd.Timestamp) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index <= pd.Timestamp(asof)]
    if subset.empty:
        return np.nan
    year_start = pd.Timestamp(pd.Timestamp(asof).year, 1, 1)
    ytd = subset.loc[subset.index >= year_start]
    if len(ytd) < 2 or ytd.iloc[0] == 0:
        return np.nan
    return float((ytd.iloc[-1] / ytd.iloc[0] - 1.0) * 100.0)


def safe_ratio(prices: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    if numerator not in prices.columns or denominator not in prices.columns:
        return pd.Series(dtype=float)
    return clean_series(prices[numerator] / prices[denominator].replace(0, np.nan))


def fmt_pct(value: float) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):+.2f}%"


def fmt_pct_level(value: float) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):.2f}%"


def fmt_bp(value: float) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):+.0f} bp"


def fmt_bn(value: float) -> str:
    return "N/A" if not is_valid(value) else f"${float(value):+.0f}bn"


def fmt_number(value: float, digits: int = 2) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):.{digits}f}"


def vote(value: float, threshold: float, orientation: int = 1) -> int:
    if not is_valid(value):
        return 0
    adjusted = float(value) * int(orientation)
    if adjusted >= threshold:
        return 1
    if adjusted <= -threshold:
        return -1
    return 0


def classify(indicators: List[dict], positive: str, negative: str, mixed: str = "Mixed") -> str:
    votes = [int(item["vote"]) for item in indicators if item.get("available", False)]
    positives = sum(v > 0 for v in votes)
    negatives = sum(v < 0 for v in votes)
    required = max(2, int(np.ceil(len(votes) / 2.0))) if votes else 2
    if positives >= required and positives > negatives:
        return positive
    if negatives >= required and negatives > positives:
        return negative
    return mixed


def indicator(
    name: str,
    series: pd.Series,
    asof: pd.Timestamp,
    threshold: float,
    *,
    kind: str = "pct",
    orientation: int = 1,
    days: int = 30,
) -> dict:
    if kind == "pct":
        move = pct_change_days(series, asof, days)
        text = fmt_pct(move)
    elif kind == "bp":
        move = abs_change_days(series, asof, days) * 100.0
        text = fmt_bp(move)
    elif kind == "bn":
        move = abs_change_days(series, asof, days) / 1_000.0
        text = fmt_bn(move)
    else:
        raise ValueError(kind)
    return {
        "name": name,
        "move": move,
        "move_text": text,
        "vote": vote(move, threshold, orientation),
        "available": is_valid(move),
    }


def macro_series(panel: pd.DataFrame, key: str) -> pd.Series:
    if key not in panel.columns:
        return pd.Series(dtype=float)
    return clean_series(panel[key])


def build_net_liquidity(macro: pd.DataFrame) -> pd.Series:
    if not {"walcl", "tga", "rrp"}.issubset(macro.columns):
        return pd.Series(dtype=float)
    frame = macro[["walcl", "tga", "rrp"]].copy().sort_index().ffill()
    return clean_series(frame["walcl"] - frame["tga"] - frame["rrp"] * 1_000.0)


def build_snapshot(prices: pd.DataFrame, macro: pd.DataFrame, asof: pd.Timestamp) -> dict:
    rsp_spy = safe_ratio(prices, "RSP", "SPY")
    iwm_spy = safe_ratio(prices, "IWM", "SPY")
    eem_spy = safe_ratio(prices, "EEM", "SPY")
    hyg_lqd = safe_ratio(prices, "HYG", "LQD")
    copper_gold = safe_ratio(prices, "HG=F", "GC=F")
    tip_ief = safe_ratio(prices, "TIP", "IEF")

    dollar = (
        clean_series(prices["DX-Y.NYB"])
        if "DX-Y.NYB" in prices and not clean_series(prices["DX-Y.NYB"]).empty
        else clean_series(prices["UUP"]) if "UUP" in prices else pd.Series(dtype=float)
    )
    net_liquidity = build_net_liquidity(macro)

    growth_indicators = [
        indicator("Breadth · RSP/SPY", rsp_spy, asof, 0.50),
        indicator("Small caps · IWM/SPY", iwm_spy, asof, 1.00),
        indicator("Copper/Gold", copper_gold, asof, 2.00),
        indicator("EM vs U.S. · EEM/SPY", eem_spy, asof, 1.00),
        indicator("High Yield OAS", macro_series(macro, "hy_oas"), asof, 10.0, kind="bp", orientation=-1),
    ]
    growth = classify(growth_indicators, "Improving", "Weakening")

    inflation_indicators = [
        indicator("10Y Breakeven", macro_series(macro, "t10yie"), asof, 5.0, kind="bp"),
        indicator("WTI Crude", clean_series(prices["CL=F"]) if "CL=F" in prices else pd.Series(dtype=float), asof, 5.0),
        indicator("Copper/Gold", copper_gold, asof, 2.00),
        indicator("TIPS vs Treasuries · TIP/IEF", tip_ief, asof, 0.75),
    ]
    inflation = classify(inflation_indicators, "Rising", "Falling")

    rates_indicators = [
        indicator("U.S. 2Y", macro_series(macro, "dgs2"), asof, 10.0, kind="bp"),
        indicator("U.S. 10Y", macro_series(macro, "dgs10"), asof, 10.0, kind="bp"),
        indicator("U.S. 30Y", macro_series(macro, "dgs30"), asof, 10.0, kind="bp"),
        indicator("10Y Real Yield", macro_series(macro, "dfii10"), asof, 10.0, kind="bp"),
    ]
    rates = classify(rates_indicators, "Rising", "Falling")

    liquidity_indicators = [
        indicator("U.S. Dollar", dollar, asof, 1.00, orientation=-1),
        indicator("10Y Real Yield", macro_series(macro, "dfii10"), asof, 10.0, kind="bp", orientation=-1),
        indicator("High Yield OAS", macro_series(macro, "hy_oas"), asof, 10.0, kind="bp", orientation=-1),
        indicator("Fed - TGA - RRP", net_liquidity, asof, 100.0, kind="bn"),
        indicator(
            "EUR/USD",
            clean_series(prices["EURUSD=X"]) if "EURUSD=X" in prices else pd.Series(dtype=float),
            asof,
            1.00,
        ),
    ]
    liquidity = classify(liquidity_indicators, "Easing", "Tightening")

    risk_indicators = [
        indicator("S&P 500", clean_series(prices["SPY"]) if "SPY" in prices else pd.Series(dtype=float), asof, 2.00),
        indicator("Breadth · RSP/SPY", rsp_spy, asof, 0.50),
        indicator("Credit · HYG/LQD", hyg_lqd, asof, 0.50),
        indicator("VIX", clean_series(prices["^VIX"]) if "^VIX" in prices else pd.Series(dtype=float), asof, 10.0, orientation=-1),
        indicator("EM vs U.S. · EEM/SPY", eem_spy, asof, 1.00),
    ]
    base_risk = classify(risk_indicators, "Constructive", "Defensive")

    spy_1m = pct_change_days(prices["SPY"] if "SPY" in prices else None, asof, 30)
    breadth_1m = pct_change_days(rsp_spy, asof, 30)
    credit_1m = pct_change_days(hyg_lqd, asof, 30)
    vix_1m = pct_change_days(prices["^VIX"] if "^VIX" in prices else None, asof, 30)

    if is_valid(spy_1m) and spy_1m > 2.0 and is_valid(breadth_1m) and breadth_1m > 0.50 and is_valid(credit_1m) and credit_1m > 0.50:
        risk = "Broad risk-on"
    elif is_valid(spy_1m) and spy_1m > 1.0 and (
        (is_valid(breadth_1m) and breadth_1m < -0.50)
        or (is_valid(credit_1m) and credit_1m < -0.50)
    ):
        risk = "Narrow risk-on"
    elif is_valid(spy_1m) and spy_1m < -2.0 and (
        (is_valid(credit_1m) and credit_1m < -0.50)
        or (is_valid(vix_1m) and vix_1m > 10.0)
    ):
        risk = "Risk-off"
    else:
        risk = base_risk

    if growth == "Improving" and inflation == "Falling":
        regime = "Goldilocks / disinflationary growth"
    elif growth == "Improving" and inflation == "Rising":
        regime = "Reflation"
    elif growth == "Weakening" and inflation == "Rising":
        regime = "Stagflation pressure"
    elif growth == "Weakening" and inflation == "Falling":
        regime = "Growth scare / disinflation"
    else:
        regime = "Transition / mixed"

    return {
        "asof": pd.Timestamp(asof),
        "regime": regime,
        "growth": growth,
        "inflation": inflation,
        "rates": rates,
        "liquidity": liquidity,
        "risk": risk,
        "growth_indicators": growth_indicators,
        "inflation_indicators": inflation_indicators,
        "rates_indicators": rates_indicators,
        "liquidity_indicators": liquidity_indicators,
        "risk_indicators": risk_indicators,
        "ratios": {
            "rsp_spy": rsp_spy,
            "iwm_spy": iwm_spy,
            "eem_spy": eem_spy,
            "hyg_lqd": hyg_lqd,
            "copper_gold": copper_gold,
            "tip_ief": tip_ief,
        },
        "dollar": dollar,
        "net_liquidity": net_liquidity,
    }


def narrative(snapshot: dict) -> str:
    base = {
        "Goldilocks / disinflationary growth":
            "Growth-sensitive markets are improving while inflation pressure is fading. That is the cleanest mix for duration-sensitive risk when breadth and credit confirm.",
        "Reflation":
            "Growth and inflation proxies are rising together. Cyclical risk can work, but real yields and the long end become the constraint on valuation.",
        "Stagflation pressure":
            "Growth-sensitive markets are weakening while inflation pressure remains firm. This is the most hostile mix for duration and weaker balance sheets.",
        "Growth scare / disinflation":
            "Growth and inflation pressure are falling together. Duration should normally confirm; if long yields rise anyway, term premium or fiscal pressure is the governing signal.",
        "Transition / mixed":
            "Growth and inflation do not define a clean quadrant. The useful information is in rates, liquidity and cross-asset confirmation rather than a forced regime label.",
    }[snapshot["regime"]]
    return (
        f"{base} Rates are {snapshot['rates'].lower()}, liquidity is {snapshot['liquidity'].lower()}, "
        f"and risk confirmation is {snapshot['risk'].lower()}."
    )


def evidence(indicators: List[dict], limit: int = 5) -> str:
    usable = [item for item in indicators if item.get("available", False)]
    if not usable:
        return "No current observations."
    return " · ".join(f"{item['name']} {item['move_text']}" for item in usable[:limit])


def build_tensions(snapshot: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> List[Tuple[str, str]]:
    asof = snapshot["asof"]
    tensions: List[Tuple[str, str]] = []

    spy = pct_change_days(prices["SPY"] if "SPY" in prices else None, asof, 30)
    breadth = pct_change_days(snapshot["ratios"]["rsp_spy"], asof, 30)
    credit = pct_change_days(snapshot["ratios"]["hyg_lqd"], asof, 30)
    copper_gold = pct_change_days(snapshot["ratios"]["copper_gold"], asof, 30)
    ten_year_bp = abs_change_days(macro_series(macro, "dgs10"), asof, 30) * 100.0
    real_yield_bp = abs_change_days(macro_series(macro, "dfii10"), asof, 30) * 100.0
    hy_oas_bp = abs_change_days(macro_series(macro, "hy_oas"), asof, 30) * 100.0
    dollar = pct_change_days(snapshot["dollar"], asof, 30)

    if snapshot["growth"] == "Weakening" and is_valid(ten_year_bp) and ten_year_bp > 10:
        tensions.append((
            "Growth vs long rates",
            f"Growth proxies are weakening while the 10Y yield is up {ten_year_bp:.0f} bp over one month. "
            "That points toward term-premium, fiscal or supply pressure rather than a clean cyclical selloff in bonds.",
        ))

    if snapshot["inflation"] == "Falling" and is_valid(ten_year_bp) and ten_year_bp > 10:
        tensions.append((
            "Disinflation vs duration",
            f"Inflation pressure is falling but the 10Y is up {ten_year_bp:.0f} bp. "
            "The bond market is refusing the normal disinflationary duration bid.",
        ))

    if is_valid(spy) and spy > 2 and is_valid(breadth) and breadth < -0.50:
        tensions.append((
            "Index strength vs breadth",
            f"SPY is up {spy:.2f}% over one month while RSP/SPY is {breadth:.2f}%. "
            "The headline index is being carried by a narrower leadership set.",
        ))

    if is_valid(spy) and spy > 2 and (
        (is_valid(credit) and credit < -0.50)
        or (is_valid(hy_oas_bp) and hy_oas_bp > 10)
    ):
        tensions.append((
            "Equities vs credit",
            f"Equities are advancing without clean credit confirmation. HYG/LQD is {fmt_pct(credit)} over one month "
            f"and HY OAS has moved {fmt_bp(hy_oas_bp)}.",
        ))

    if is_valid(dollar) and dollar < -1 and is_valid(real_yield_bp) and real_yield_bp > 10:
        tensions.append((
            "Dollar easing vs real-rate tightening",
            f"The dollar is down {abs(dollar):.2f}% over one month, but the 10Y real yield is up {real_yield_bp:.0f} bp. "
            "FX is easing global conditions while the discount rate is tightening them.",
        ))

    if snapshot["liquidity"] == "Easing" and snapshot["risk"] not in {"Broad risk-on", "Constructive"}:
        tensions.append((
            "Liquidity vs transmission",
            f"Liquidity is easing, but risk confirmation is still {snapshot['risk'].lower()}. "
            "The easier financial-conditions impulse has not yet transmitted broadly through equities and credit.",
        ))

    if is_valid(spy) and spy > 2 and is_valid(copper_gold) and copper_gold < -2:
        tensions.append((
            "Equities vs cyclicality",
            f"SPY is up {spy:.2f}% while Copper/Gold is down {abs(copper_gold):.2f}% over one month. "
            "Equity strength is running ahead of the cyclical growth signal.",
        ))

    if not tensions:
        tensions.append((
            "No major cross-asset contradiction",
            "Rates and liquidity are broadly consistent with the current macro mix. The main question is whether breadth and credit strengthen enough to make risk confirmation more durable.",
        ))
    return tensions[:5]


def state_table(current: dict, one_month: dict, three_month: dict) -> pd.DataFrame:
    rows = [
        ("Macro regime", three_month["regime"], one_month["regime"], current["regime"]),
        ("Growth", three_month["growth"], one_month["growth"], current["growth"]),
        ("Inflation", three_month["inflation"], one_month["inflation"], current["inflation"]),
        ("Rates", three_month["rates"], one_month["rates"], current["rates"]),
        ("Liquidity", three_month["liquidity"], one_month["liquidity"], current["liquidity"]),
        ("Risk confirmation", three_month["risk"], one_month["risk"], current["risk"]),
    ]
    return pd.DataFrame(rows, columns=["Regime component", "3M ago", "1M ago", "Current"])


def rates_fci_table(snapshot: dict, macro: pd.DataFrame) -> pd.DataFrame:
    asof = snapshot["asof"]
    rows: List[dict] = []
    specs = [
        ("U.S. 2Y Treasury", macro_series(macro, "dgs2")),
        ("U.S. 10Y Treasury", macro_series(macro, "dgs10")),
        ("U.S. 30Y Treasury", macro_series(macro, "dgs30")),
        ("10Y Real Yield", macro_series(macro, "dfii10")),
        ("10Y Breakeven", macro_series(macro, "t10yie")),
        ("High Yield OAS", macro_series(macro, "hy_oas")),
    ]
    for label, series in specs:
        level = value_asof(series, asof)
        rows.append({
            "Series": label,
            "Current": fmt_pct_level(level),
            "1W": fmt_bp(abs_change_days(series, asof, 7) * 100.0),
            "1M": fmt_bp(abs_change_days(series, asof, 30) * 100.0),
            "3M": fmt_bp(abs_change_days(series, asof, 90) * 100.0),
            "Interpretation": (
                "Higher = tighter discount rate"
                if "Yield" in label or "Treasury" in label
                else "Higher = tighter credit"
            ),
        })

    dollar = snapshot["dollar"]
    rows.append({
        "Series": "U.S. Dollar",
        "Current": fmt_number(value_asof(dollar, asof), 2),
        "1W": fmt_pct(pct_change_days(dollar, asof, 7)),
        "1M": fmt_pct(pct_change_days(dollar, asof, 30)),
        "3M": fmt_pct(pct_change_days(dollar, asof, 90)),
        "Interpretation": "Higher = tighter global dollar liquidity",
    })

    net_liquidity = snapshot["net_liquidity"]
    rows.append({
        "Series": "Fed - TGA - RRP",
        "Current": fmt_bn(value_asof(net_liquidity, asof) / 1_000.0),
        "1W": fmt_bn(abs_change_days(net_liquidity, asof, 7) / 1_000.0),
        "1M": fmt_bn(abs_change_days(net_liquidity, asof, 30) / 1_000.0),
        "3M": fmt_bn(abs_change_days(net_liquidity, asof, 90) / 1_000.0),
        "Interpretation": "Higher = more system liquidity",
    })
    return pd.DataFrame(rows)


def cross_asset_table(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    rows: List[dict] = []
    for group, ticker in PERFORMANCE_ROWS:
        if ticker not in prices.columns:
            continue
        series = clean_series(prices[ticker])
        if series.empty:
            continue
        rows.append({
            "Group": group,
            "Ticker": ticker,
            "Asset": TICKERS[ticker],
            "1W": pct_change_days(series, asof, 7),
            "1M": pct_change_days(series, asof, 30),
            "3M": pct_change_days(series, asof, 90),
            "YTD": ytd_change(series, asof),
        })
    return pd.DataFrame(rows)


