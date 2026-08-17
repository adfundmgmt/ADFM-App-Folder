"""CFTC Commitments of Traders access and positioning analytics."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final, Mapping

import numpy as np
import pandas as pd
import requests

CFTC_HOST: Final = "https://publicreporting.cftc.gov"
DATASETS: Final = {"TFF": "gpe5-46if", "Disaggregated": "72hh-3qpy"}
REPORT_LABELS: Final = {
    "TFF": "Traders in Financial Futures · futures only",
    "Disaggregated": "Disaggregated COT · futures only",
}
COHORTS: Final = {
    "TFF": {
        "Asset Managers": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
        "Leveraged Funds": ("lev_money_positions_long", "lev_money_positions_short"),
        "Asset Managers + Leveraged Funds": (
            ("asset_mgr_positions_long", "lev_money_positions_long"),
            ("asset_mgr_positions_short", "lev_money_positions_short"),
        ),
        "Dealers": ("dealer_positions_long_all", "dealer_positions_short_all"),
        "Other Reportables": ("other_rept_positions_long", "other_rept_positions_short"),
    },
    "Disaggregated": {
        "Managed Money": ("m_money_positions_long_all", "m_money_positions_short_all"),
        "Producer / Merchant": ("prod_merc_positions_long", "prod_merc_positions_short"),
        "Swap Dealers": ("swap_positions_long_all", "swap__positions_short_all"),
        "Other Reportables": ("other_rept_positions_long", "other_rept_positions_short"),
    },
}
DEFAULT_COHORT: Final = {
    "TFF": "Asset Managers + Leveraged Funds",
    "Disaggregated": "Managed Money",
}
BASE_FIELDS: Final = (
    "market_and_exchange_names",
    "contract_market_name",
    "commodity_name",
    "report_date_as_yyyy_mm_dd",
    "cftc_contract_market_code",
    "open_interest_all",
)

PRICE_PROXIES: Final = {
    "209742": ("NQ=F", "Nasdaq-100 E-mini", 20.0),
    "13874A": ("ES=F", "S&P 500 E-mini", 50.0),
    "239742": ("RTY=F", "Russell 2000 E-mini", 50.0),
    "124603": ("YM=F", "Dow Jones $5", 5.0),
    "088691": ("GC=F", "Gold", 100.0),
    "084691": ("SI=F", "Silver", 5000.0),
    "085692": ("HG=F", "Copper", 25000.0),
    "067651": ("CL=F", "WTI crude", 1000.0),
    "023651": ("NG=F", "Henry Hub natural gas", 10000.0),
    "043602": ("ZN=F", "10Y Treasury note", None),
    "044601": ("ZF=F", "5Y Treasury note", None),
    "042601": ("ZT=F", "2Y Treasury note", None),
    "099741": ("6E=F", "Euro FX", None),
    "097741": ("6J=F", "Japanese yen", None),
    "096742": ("6B=F", "British pound", None),
    "092741": ("6C=F", "Canadian dollar", None),
    "232741": ("6A=F", "Australian dollar", None),
}


def _cohort_fields(report_type: str, cohort: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = COHORTS[report_type][cohort]
    if isinstance(raw[0], tuple):
        return raw  # type: ignore[return-value]
    return (raw[0],), (raw[1],)  # type: ignore[index]


def _all_position_fields(report_type: str) -> tuple[str, ...]:
    fields: list[str] = []
    for cohort in COHORTS[report_type]:
        longs, shorts = _cohort_fields(report_type, cohort)
        fields.extend(longs)
        fields.extend(shorts)
    return tuple(dict.fromkeys(fields))


def _request(report_type: str, params: Mapping[str, object], timeout: int = 30) -> pd.DataFrame:
    url = f"{CFTC_HOST}/resource/{DATASETS[report_type]}.json"
    headers = {"User-Agent": "ADFM-Analytics/1.0"}
    response = requests.get(url, params=dict(params), headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected CFTC API payload")
    return pd.DataFrame(payload)


def normalize(raw: pd.DataFrame, report_type: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    frame["report_date"] = pd.to_datetime(
        frame["report_date_as_yyyy_mm_dd"], errors="coerce"
    ).dt.normalize()
    frame["contract_code"] = (
        frame["cftc_contract_market_code"].astype("string").str.strip()
    )
    frame["market_name"] = (
        frame["market_and_exchange_names"]
        .fillna(frame["contract_market_name"])
        .astype("string")
        .str.strip()
    )
    if "commodity_name" not in frame:
        frame["commodity_name"] = ""
    frame["commodity_name"] = frame["commodity_name"].astype("string").str.strip()
    frame["open_interest"] = pd.to_numeric(frame["open_interest_all"], errors="coerce")
    for field in _all_position_fields(report_type):
        if field not in frame:
            frame[field] = np.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    keep = [
        "report_date",
        "contract_code",
        "market_name",
        "commodity_name",
        "open_interest",
        *_all_position_fields(report_type),
    ]
    return (
        frame[keep]
        .dropna(subset=["report_date", "contract_code"])
        .sort_values(["contract_code", "report_date"])
    )


def add_metrics(frame: pd.DataFrame, report_type: str, cohort: str) -> pd.DataFrame:
    out = frame.copy()
    longs, shorts = _cohort_fields(report_type, cohort)
    out["cohort_long"] = out[list(longs)].sum(axis=1, min_count=1)
    out["cohort_short"] = out[list(shorts)].sum(axis=1, min_count=1)
    out["net_contracts"] = out["cohort_long"] - out["cohort_short"]
    out["net_pct_oi"] = out["net_contracts"] / out["open_interest"].replace(0, np.nan)
    return out


def percentile_rank(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    latest = clean.iloc[-1]
    return float(
        (clean.lt(latest).sum() + 0.5 * clean.eq(latest).sum())
        / len(clean)
        * 100.0
    )


def positioning_signal(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "N/A"
    if percentile <= 2.5:
        return "Extreme Short"
    if percentile <= 15:
        return "Crowded Short"
    if percentile >= 97.5:
        return "Extreme Long"
    if percentile >= 85:
        return "Crowded Long"
    return "Neutral"


def zscore_latest(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 8:
        return np.nan
    std = float(clean.std(ddof=0))
    return float((clean.iloc[-1] - clean.mean()) / std) if std else np.nan


def infer_asset_class(report_type: str, market: str, commodity: str = "") -> str:
    text = f"{market} {commodity}".upper()
    if report_type == "TFF":
        if any(x in text for x in ("BITCOIN", "ETHER", "SOLANA")):
            return "Crypto"
        if any(
            x in text
            for x in ("S&P", "NASDAQ", "RUSSELL", "DOW JONES", "VIX", "MSCI", "NIKKEI")
        ):
            return "Equity / Vol"
        if any(x in text for x in ("TREASURY", "UST ", "SOFR", "FED FUNDS", "EURODOLLAR")):
            return "Rates"
        if any(
            x in text
            for x in ("EURO FX", "YEN", "POUND", "FRANC", "DOLLAR", "PESO", "REAL")
        ):
            return "FX"
        return "Financial"
    if any(
        x in text for x in ("CRUDE", "GASOLINE", "HEATING OIL", "NATURAL GAS", "PROPANE")
    ):
        return "Energy"
    if any(
        x in text for x in ("GOLD", "SILVER", "COPPER", "PLATINUM", "PALLADIUM", "ALUMINUM")
    ):
        return "Metals"
    if any(x in text for x in ("CORN", "WHEAT", "SOY", "OATS", "RICE", "CANOLA")):
        return "Grains / Oilseeds"
    if any(
        x in text for x in ("COCOA", "COFFEE", "SUGAR", "COTTON", "ORANGE JUICE", "LUMBER")
    ):
        return "Softs"
    if any(x in text for x in ("CATTLE", "HOG", "MILK", "BUTTER", "CHEESE")):
        return "Livestock / Dairy"
    return "Physical Commodity"


def fetch_recent(report_type: str, years: int = 5, timeout: int = 30) -> pd.DataFrame:
    start = date.today() - timedelta(days=max(years, 1) * 366)
    select = ",".join((*BASE_FIELDS, *_all_position_fields(report_type)))
    params = {
        "$select": select,
        "$where": f"report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 50000,
    }
    return normalize(_request(report_type, params, timeout), report_type)


def fetch_contract_history(
    report_type: str, contract_code: str, timeout: int = 30
) -> pd.DataFrame:
    code = str(contract_code).replace("'", "''").strip()
    select = ",".join((*BASE_FIELDS, *_all_position_fields(report_type)))
    params = {
        "$select": select,
        "$where": f"cftc_contract_market_code = '{code}'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 10000,
    }
    return normalize(_request(report_type, params, timeout), report_type)


def build_scanner(
    frame: pd.DataFrame,
    report_type: str,
    cohort: str,
    lookback_weeks: int = 156,
    max_stale_days: int = 21,
) -> pd.DataFrame:
    metrics = add_metrics(frame, report_type, cohort)
    if metrics.empty:
        return pd.DataFrame()
    report_latest = pd.Timestamp(metrics["report_date"].max())
    rows = []
    for code, group in metrics.groupby("contract_code", sort=False):
        history = group.sort_values("report_date").tail(lookback_weeks)
        latest = history.iloc[-1]
        stale_days = int((report_latest - pd.Timestamp(latest["report_date"])).days)
        if stale_days > max_stale_days:
            continue
        pct_history = history["net_pct_oi"].dropna()
        enough = len(pct_history) >= 26
        percentile = percentile_rank(pct_history) if enough else np.nan
        zscore = zscore_latest(pct_history) if enough else np.nan
        net = history["net_contracts"].dropna()
        one_week = float(net.iloc[-1] - net.iloc[-2]) if len(net) >= 2 else np.nan
        four_week = float(net.iloc[-1] - net.iloc[-5]) if len(net) >= 5 else np.nan
        record = ""
        if enough and latest["net_pct_oi"] == pct_history.min():
            record = f"{len(pct_history)}W Low"
        elif enough and latest["net_pct_oi"] == pct_history.max():
            record = f"{len(pct_history)}W High"
        rows.append(
            {
                "report_type": report_type,
                "report_date": latest["report_date"],
                "contract_code": str(code),
                "market": str(latest["market_name"]),
                "commodity": str(latest["commodity_name"]),
                "asset_class": infer_asset_class(
                    report_type,
                    str(latest["market_name"]),
                    str(latest["commodity_name"]),
                ),
                "open_interest": float(latest["open_interest"]),
                "net_contracts": float(latest["net_contracts"]),
                "net_pct_oi": float(latest["net_pct_oi"]),
                "one_week_change": one_week,
                "four_week_change": four_week,
                "percentile": percentile,
                "zscore": zscore,
                "history_weeks": len(pct_history),
                "stale_days": stale_days,
                "signal": positioning_signal(percentile),
                "record": record,
            }
        )
    return pd.DataFrame(rows)


def rolling_metrics(
    frame: pd.DataFrame,
    report_type: str,
    cohort: str,
    window_weeks: int = 156,
) -> pd.DataFrame:
    out = add_metrics(frame, report_type, cohort).sort_values("report_date").copy()
    window = max(window_weeks, 26)
    out["rolling_zscore"] = out["net_pct_oi"].rolling(
        window, min_periods=26
    ).apply(lambda x: zscore_latest(pd.Series(x)), raw=False)
    out["rolling_percentile"] = out["net_pct_oi"].rolling(
        window, min_periods=26
    ).apply(lambda x: percentile_rank(pd.Series(x)), raw=False)
    return out


def price_proxy(contract_code: str) -> tuple[str, str, float | None] | None:
    return PRICE_PROXIES.get(str(contract_code).strip())


def estimate_notional(
    net_contracts: pd.Series, price: pd.Series, multiplier: float | None
) -> pd.Series:
    if multiplier is None:
        return pd.Series(np.nan, index=net_contracts.index, dtype=float)
    return (
        pd.to_numeric(net_contracts, errors="coerce")
        * pd.to_numeric(price, errors="coerce")
        * multiplier
    )
