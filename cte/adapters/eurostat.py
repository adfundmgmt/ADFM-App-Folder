"""Eurostat adapter — euro-area harmonised unemployment rate (FRED gap: EUR/unemp).

The euro-area aggregate isn't in the OECD labour dataflow and froze on FRED, but
Eurostat publishes it monthly (dataset une_rt_m). Parameter-based filtering keeps
the pull to the single series we need (SA, total sex, all ages, % of active pop).

Returns the FRED/OECD tidy contract: [date, ccy, metric, value, source, fetched_at]
"""
from __future__ import annotations

import io

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import HTTP_TIMEOUT, HTTP_UA

_URL = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_m/"
        "M.SA.TOTAL.PC_ACT.T.EA21?format=SDMX-CSV&startPeriod=1990-01")


def fetch_ea_unemployment(session=None) -> pd.DataFrame:
    sess = session or make_session()
    r = sess.get(_URL, headers={"User-Agent": HTTP_UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content))
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m") + pd.offsets.MonthEnd(0),
        "ccy": "EUR",
        "metric": "unemp",
        "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
        "source": "eurostat_une_rt_m",
    })
    df["fetched_at"] = utcnow()
    return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_ea_unemployment()
    last = df.iloc[-1]
    print(f"rows: {len(df)} | range {df.date.min().date()} -> {df.date.max().date()}")
    print(f"EA unemployment latest: {last.value}% ({last.date.date()})")
