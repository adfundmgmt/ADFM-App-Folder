from __future__ import annotations

import os
from math import isfinite
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from adfm_core.data_integrity import DataIntegrityPolicy, build_data_quality_report
from adfm_core.market_data import fetch_daily_ohlcv
from adfm_core.rate_of_change import compute_features

app = FastAPI(
    title="ADFM Analytics API",
    version="0.1.0",
    description="Python API for the ADFM analytics web application.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("ADFM_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

TIMEFRAMES = {
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
    "25Y": "25y",
    "Max": "max",
}
ROC_PERIODS = {"10D": 10, "20D": 20, "63D": 63, "126D": 126, "252D": 252}


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tools/rate-of-change")
def rate_of_change(
    ticker: str = Query("^SPX", min_length=1, max_length=24),
    timeframe: Literal["3M", "6M", "1Y", "3Y", "5Y", "10Y", "25Y", "Max"] = "3Y",
    roc_period: Literal["10D", "20D", "63D", "126D", "252D"] = "63D",
) -> dict:
    symbol = ticker.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    frames, dropped = fetch_daily_ohlcv((symbol,), TIMEFRAMES[timeframe])
    if symbol not in frames:
        reason = "No valid OHLCV data returned"
        if not dropped.empty:
            reason = str(dropped.iloc[0].get("Reason", reason))
        raise HTTPException(status_code=404, detail=f"Failed to fetch {symbol}: {reason}")

    frame = frames[symbol]
    quality = build_data_quality_report(
        {symbol: frame},
        symbol,
        policy=DataIntegrityPolicy(min_valid_sessions=1, max_stale_sessions=0),
    )

    features = compute_features(frame, ROC_PERIODS[roc_period]).dropna(
        subset=["ROC", "Second_Derivative"], how="any"
    )
    if features.empty:
        raise HTTPException(
            status_code=422,
            detail="Data became empty after indicator calculations. Try a longer analysis window.",
        )

    rows: list[dict] = []
    for date, row in features.iterrows():
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "open": _number(row.get("Open")),
                "high": _number(row.get("High")),
                "low": _number(row.get("Low")),
                "close": _number(row.get("Close")),
                "sma21": _number(row.get("SMA_21")),
                "sma50": _number(row.get("SMA_50")),
                "sma100": _number(row.get("SMA_100")),
                "sma200": _number(row.get("SMA_200")),
                "roc": _number(row.get("ROC")),
                "rocSlope": _number(row.get("ROC_Slope")),
                "acceleration": _number(row.get("Second_Derivative")),
                "accelerationSlope": _number(row.get("Second_Derivative_Slope")),
                "positiveInflection": bool(row.get("Pos_Inflect", False)),
                "negativeInflection": bool(row.get("Neg_Inflect", False)),
            }
        )

    latest = rows[-1]
    return {
        "tool": "Rate of Change Regime Explorer",
        "ticker": symbol,
        "timeframe": timeframe,
        "rocPeriod": roc_period,
        "source": "Yahoo Finance",
        "dataThrough": (
            quality.data_through.date().isoformat()
            if quality.data_through is not None
            else latest["date"]
        ),
        "dataQuality": (
            "complete daily sessions" if quality.benchmark_ready else quality.reason_for(symbol)
        ),
        "observationCount": len(rows),
        "warning": (
            "The selected window has fewer than 60 observations, so derivative readings may be unstable."
            if len(frame) < 60
            else None
        ),
        "latest": {
            "close": latest["close"],
            "roc": latest["roc"],
            "acceleration": latest["acceleration"],
        },
        "series": rows,
    }
