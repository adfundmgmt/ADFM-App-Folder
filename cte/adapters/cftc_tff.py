"""CFTC Traders-in-Financial-Futures (TFF) positioning adapter — spec flag layer.

Weekly leveraged-money and asset-manager positioning for the eight currency
legs, pulled from the CFTC public-reporting Socrata dataset (Futures-Only).
This feeds the positioning *flag* overlay only; per spec §11 it is not part of
the axis composites.

Leveraged money  = fast / spec money (hedge funds, CTAs) — the crowding signal.
Asset manager    = real money (institutional) — the slower structural leg.

Tidy contract returned:
  [date, ccy, lev_long, lev_short, lev_net, lev_net_pct_oi,
   am_long, am_short, am_net, open_interest, source, fetched_at]
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from cte.adapters.base import http_get, utcnow
from cte.config import CFTC_TFF_CODES, CFTC_TFF_DATASET

_BASE = "https://publicreporting.cftc.gov/resource/{ds}.json"


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def fetch_tff(weeks_back: int = 260) -> pd.DataFrame:
    """Pull ~5y (260 weeks) of TFF positioning for all eight currency legs.

    One Socrata query, filtered to our contract codes, ordered newest-first.
    """
    codes = CFTC_TFF_CODES
    rev = {v: k for k, v in codes.items()}
    inlist = ",".join(f"'{c}'" for c in codes.values())
    # rows per week (8 contracts) * weeks_back, plus headroom
    limit = max(8 * weeks_back, 2000)
    select = ",".join([
        "cftc_contract_market_code", "report_date_as_yyyy_mm_dd",
        "lev_money_positions_long", "lev_money_positions_short",
        "asset_mgr_positions_long", "asset_mgr_positions_short",
        "open_interest_all",
    ])
    where = f"cftc_contract_market_code in({inlist})"
    url = (f"{_BASE.format(ds=CFTC_TFF_DATASET)}"
           f"?$where={where}&$select={select}"
           f"&$order=report_date_as_yyyy_mm_dd DESC&$limit={limit}")
    rows = http_get(url).json()

    fetched = utcnow()
    out = []
    for r in rows:
        ccy = rev.get(r.get("cftc_contract_market_code"))
        if ccy is None:
            continue
        date = pd.to_datetime(r.get("report_date_as_yyyy_mm_dd"), errors="coerce")
        if pd.isna(date):
            continue
        lev_l = _to_int(r.get("lev_money_positions_long"))
        lev_s = _to_int(r.get("lev_money_positions_short"))
        am_l = _to_int(r.get("asset_mgr_positions_long"))
        am_s = _to_int(r.get("asset_mgr_positions_short"))
        oi = _to_int(r.get("open_interest_all"))
        lev_net = (lev_l - lev_s) if (lev_l is not None and lev_s is not None) else None
        am_net = (am_l - am_s) if (am_l is not None and am_s is not None) else None
        lev_net_pct_oi = (round(100 * lev_net / oi, 2)
                          if (lev_net is not None and oi) else None)
        out.append({
            "date": date.normalize(), "ccy": ccy,
            "lev_long": lev_l, "lev_short": lev_s, "lev_net": lev_net,
            "lev_net_pct_oi": lev_net_pct_oi,
            "am_long": am_l, "am_short": am_s, "am_net": am_net,
            "open_interest": oi,
            "source": "cftc_tff_futonly", "fetched_at": fetched,
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["ccy", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = fetch_tff(weeks_back=52)
    print(f"rows: {len(df)} | currencies: {sorted(df.ccy.unique())}")
    print(f"date range: {df.date.min().date()} -> {df.date.max().date()}")
    latest = df.sort_values("date").groupby("ccy").tail(1)
    print(latest[["ccy", "date", "lev_net", "lev_net_pct_oi", "am_net",
                  "open_interest"]].to_string(index=False))
