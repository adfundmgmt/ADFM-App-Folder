"""Holdings helpers for sector breadth."""

from __future__ import annotations

from typing import List

import pandas as pd


def parse_spdr_holdings_table(raw: pd.DataFrame) -> List[str]:
    """Extract equity tickers from a raw SPDR holdings worksheet.

    The parser keys off the actual Ticker/Name header and excludes only explicit
    cash rows, so company names containing the word ``Cash`` remain valid.
    """
    if raw is None or raw.empty:
        return []

    header_idx = None
    ticker_col = None
    name_col = None
    for i, row in raw.iterrows():
        values = [str(v).strip() if pd.notna(v) else "" for v in row.tolist()]
        lowered = [v.lower() for v in values]
        if "ticker" in lowered:
            header_idx = i
            ticker_col = lowered.index("ticker")
            name_col = lowered.index("name") if "name" in lowered else None
            break

    if header_idx is None or ticker_col is None:
        return []

    out: List[str] = []
    for _, row in raw.loc[header_idx + 1 :].iterrows():
        ticker = (
            str(row.iloc[ticker_col]).strip().upper()
            if pd.notna(row.iloc[ticker_col])
            else ""
        )
        name = ""
        if name_col is not None and name_col < len(row) and pd.notna(row.iloc[name_col]):
            name = str(row.iloc[name_col]).strip().upper()

        if not ticker or ticker in {"NAN", "NONE", "-"}:
            continue
        if (
            ticker.startswith("CASH_")
            or ticker in {"USD", "CASH"}
            or name in {"US DOLLAR", "U.S. DOLLAR", "CASH"}
        ):
            continue
        if ticker.replace(".", "").replace("-", "").isalnum():
            out.append(ticker)

    return list(dict.fromkeys(out))
