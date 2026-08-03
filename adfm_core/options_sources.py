"""Secondary option-chain sources for the Options Positioning Compass."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Mapping
from urllib.request import Request, urlopen

import pandas as pd


CBOE_DELAYED_OPTIONS_URL = (
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
)


def parse_occ_option_symbol(contract_symbol: str) -> tuple[str, str, float]:
    """Return expiration, option type, and strike from an OCC option symbol."""
    value = str(contract_symbol or "").strip().upper()
    if len(value) < 16:
        raise ValueError(f"Invalid OCC option symbol: {contract_symbol!r}")
    suffix = value[-15:]
    try:
        expiry = datetime.strptime(suffix[:6], "%y%m%d").date().isoformat()
        option_code = suffix[6]
        strike = int(suffix[7:]) / 1_000.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid OCC option symbol: {contract_symbol!r}") from exc
    if option_code not in {"C", "P"}:
        raise ValueError(f"Invalid OCC option type in {contract_symbol!r}")
    return expiry, "call" if option_code == "C" else "put", strike


def cboe_payload_to_chains(
    payload: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """Normalize Cboe's delayed quote payload into Yahoo-style chain columns."""
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Cboe response did not contain option data")
    raw_options = data.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("Cboe response contained no option contracts")

    rows: list[dict[str, object]] = []
    for raw in raw_options:
        if not isinstance(raw, Mapping):
            continue
        contract_symbol = str(raw.get("option", ""))
        try:
            expiry, option_type, strike = parse_occ_option_symbol(contract_symbol)
        except ValueError:
            continue
        rows.append(
            {
                "contractSymbol": contract_symbol,
                "lastTradeDate": raw.get("last_trade_time"),
                "strike": strike,
                "lastPrice": raw.get("last_trade_price"),
                "bid": raw.get("bid"),
                "ask": raw.get("ask"),
                "volume": raw.get("volume"),
                "openInterest": raw.get("open_interest"),
                "impliedVolatility": raw.get("iv"),
                "iv_source": "Cboe",
                "expiry": expiry,
                "option_type": option_type,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cboe response contained no parseable option contracts")

    timestamp = str(payload.get("timestamp", "")).strip()
    underlying = {
        "regularMarketPrice": data.get("current_price"),
        "symbol": data.get("symbol"),
    }
    return frame, underlying, timestamp


def fetch_cboe_delayed_options(
    symbol: str,
    *,
    timeout: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """Fetch and normalize the official Cboe delayed option chain."""
    request = Request(
        CBOE_DELAYED_OPTIONS_URL.format(symbol=str(symbol).upper()),
        headers={
            "Accept": "application/json",
            "User-Agent": "ADFM-Options-Compass/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return cboe_payload_to_chains(payload)


def expirations_from_cboe(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return sorted unique expirations from a normalized Cboe chain."""
    if frame is None or frame.empty or "expiry" not in frame:
        return ()
    values = pd.to_datetime(frame["expiry"], errors="coerce").dropna().dt.date
    return tuple(sorted({value.isoformat() for value in values}))


def select_cboe_expiry(
    frame: pd.DataFrame, expiry: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one Cboe expiration into Yahoo-style call and put frames."""
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = frame.loc[frame["expiry"].eq(str(expiry))]
    calls = selected.loc[selected["option_type"].eq("call")].copy()
    puts = selected.loc[selected["option_type"].eq("put")].copy()
    return calls, puts
