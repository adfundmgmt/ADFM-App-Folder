"""ONS adapter — UK unemployment rate, sourced from the UK's own statistics office.

The OECD harmonised UK rate is *derived from* the ONS Labour Force Survey; empirically
the two are the same series (mean abs diff 0.016pp over 434 months, identical in recent
months). So ONS gives the identical number the other legs' OECD harmonised rates use,
but published 1-2 months fresher by the primary source. Series MGSX = unemployment rate,
aged 16+, seasonally adjusted; history back to 1971.

Tidy contract: [date, ccy, metric, value, source, fetched_at]
"""
from __future__ import annotations

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import HTTP_TIMEOUT, HTTP_UA, ONS_UNEMP_URL


def fetch_uk_unemployment(session=None) -> pd.DataFrame:
    sess = session or make_session()
    r = sess.get(ONS_UNEMP_URL, headers={"User-Agent": HTTP_UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    months = r.json().get("months", [])
    rows = []
    for m in months:
        try:
            val = float(m["value"])
        except (TypeError, ValueError, KeyError):
            continue
        # ONS date format e.g. "2026 MAR" -> month-end timestamp
        dt = pd.to_datetime(m["date"].title(), format="%Y %b") + pd.offsets.MonthEnd(0)
        rows.append((dt, val))
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["ccy"] = "GBP"
    df["metric"] = "unemp"
    df["source"] = "ons_lfs"
    df["fetched_at"] = utcnow()
    return (df.dropna(subset=["value"]).sort_values("date")
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


if __name__ == "__main__":
    df = fetch_uk_unemployment()
    last = df.iloc[-1]
    print(f"rows: {len(df)} | range {df.date.min().date()} -> {df.date.max().date()}")
    print(f"UK unemployment latest: {last.value}% ({last.date.date()})")
