"""
Yahoo / yfinance adapter — FX spot  (spec §3: "better than FRED
for spot").

FX: pulls the 7 non-USD legs vs USD plus DXY for the dollar's own price leg,
then normalizes every non-USD leg to a common basis — USD value of 1 unit of
the currency — by inverting the legs Yahoo quotes as (foreign per USD). USD's
own leg is the DXY index level (spec §1). Each leg is later z-scored against its
own history, so the unit mismatch between DXY (index) and the others
(USD-per-unit) is harmless.

scoring; they feed correlation/technical context and commentary.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from cte.adapters.base import utcnow
from cte.config import (
    YF_DXY,
    YF_PAIR_INVERTED,
    YF_USD_PAIRS,
)


def _download(tickers: list[str], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return raw


def _close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract the Close column for one ticker from a (possibly multiindexed) frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        if (ticker, "Close") in raw.columns:
            s = raw[(ticker, "Close")]
        else:
            return pd.Series(dtype=float)
    else:  # single ticker -> flat columns
        s = raw["Close"]
    return s.dropna()


def fetch_fx_spot(period: str = "10y") -> pd.DataFrame:
    """Return contract: [date, ccy, value, source, fetched_at]; value = USD/unit."""
    tickers = list(YF_USD_PAIRS.values()) + [YF_DXY]
    raw = _download(tickers, period)
    rows = []

    for ccy, tk in YF_USD_PAIRS.items():
        s = _close_series(raw, tk)
        if s.empty:
            continue
        if YF_PAIR_INVERTED[ccy]:
            s = 1.0 / s  # foreign-per-USD -> USD-per-foreign
        for d, v in s.items():
            rows.append({"date": d, "ccy": ccy, "value": float(v)})

    dxy = _close_series(raw, YF_DXY)
    for d, v in dxy.items():
        rows.append({"date": d, "ccy": "USD", "value": float(v)})  # index level

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["source"] = "yahoo"
    df["fetched_at"] = utcnow()
    df = df[["date", "ccy", "value", "source", "fetched_at"]].sort_values(
        ["ccy", "date"]
    ).reset_index(drop=True)
    return df


