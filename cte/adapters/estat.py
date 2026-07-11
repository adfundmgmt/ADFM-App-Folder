"""Japan e-Stat adapter — Japan CPI YoY (closes the last macro gap: JPY inflation).

Japan CPI is absent from FRED (frozen), OECD prices (Japan not reported there), and
Eurostat (EU-only). Japan's official Statistics Bureau portal (e-Stat) serves it via
a free appId, read from env ESTAT_APP_ID (never committed).

We pull the pre-computed year-on-year series (tab=3) for the all-items headline
(cat01=0001, 総合) nationwide (area=00000), matching the headline basis used for the
other OECD-sourced legs. e-Stat's YoY runs back to 1971.

Returns the tidy contract: [date, ccy, metric, value, source, fetched_at]
"""
from __future__ import annotations

import os

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import (
    ESTAT_APP_ID_ENV, ESTAT_BASE, ESTAT_CPI_AREA_NATIONWIDE,
    ESTAT_CPI_ITEM_HEADLINE, ESTAT_CPI_TAB_YOY, ESTAT_CPI_TABLE,
    HTTP_TIMEOUT, HTTP_UA,
)


def get_app_id() -> str:
    app = os.environ.get(ESTAT_APP_ID_ENV)
    if not app:
        raise RuntimeError(f"e-Stat appId not set: export {ESTAT_APP_ID_ENV}=...")
    return app


def _parse_estat_time(code: str) -> pd.Timestamp | None:
    """e-Stat monthly time code 'YYYY00MMMM' -> period-end timestamp.
    Year is the first 4 chars; month is the last 2 (e.g. '2026000505' -> May 2026).
    Returns None for non-monthly codes (annual/other aggregates end in 00)."""
    year, month = int(code[:4]), int(code[-2:])
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def fetch_estat_cpi(session=None) -> pd.DataFrame:
    sess = session or make_session()
    params = {
        "appId": get_app_id(),
        "statsDataId": ESTAT_CPI_TABLE,
        "cdTab": ESTAT_CPI_TAB_YOY,
        "cdCat01": ESTAT_CPI_ITEM_HEADLINE,
        "cdArea": ESTAT_CPI_AREA_NATIONWIDE,
        "limit": "5000",
    }
    r = sess.get(ESTAT_BASE, params=params, headers={"User-Agent": HTTP_UA},
                 timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    result = j.get("GET_STATS_DATA", {}).get("RESULT", {})
    if str(result.get("STATUS")) != "0":
        raise RuntimeError(f"e-Stat error: {result.get('ERROR_MSG')}")
    vals = (j["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"])
    rows = []
    for v in vals:
        raw = v.get("$")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue  # skip suppressed / non-numeric marks
        ts = _parse_estat_time(v["@time"])
        if ts is not None:
            rows.append((ts, value))
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["ccy"] = "JPY"
    df["metric"] = "cpi_yoy"
    df["source"] = "estat_cpi_headline"
    df["fetched_at"] = utcnow()
    return (df.dropna(subset=["value"]).sort_values("date")
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


if __name__ == "__main__":
    df = fetch_estat_cpi()
    last = df.iloc[-1]
    print(f"rows: {len(df)} | range {df.date.min().date()} -> {df.date.max().date()}")
    print(f"JPY headline CPI YoY latest: {last.value}% ({last.date.date()})")
