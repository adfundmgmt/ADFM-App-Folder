"""Preserved from the original Cross-Asset Ratio Chartbook; no UI runtime."""
from datetime import date
from typing import Tuple, List, Sequence, Iterable, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.analytics.ratio_universe import unique_keep_order

def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


@ttl_cache(seconds=3600)
def fetch_closes(tickers: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    ticker_list = unique_keep_order(tickers)

    if not ticker_list:
        return pd.DataFrame()

    def _extract_close(block: pd.DataFrame) -> Optional[pd.Series]:
        if block is None or block.empty:
            return None

        for field in ("Close", "Adj Close"):
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce")

        numeric = block.select_dtypes(include=[np.number])

        if numeric.empty:
            return None

        return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    def _normalize(raw: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).astype(str)
            level1 = raw.columns.get_level_values(1).astype(str)

            tickers_in_level0 = set(level0).intersection(requested)
            tickers_in_level1 = set(level1).intersection(requested)

            if tickers_in_level0:
                for ticker in requested:
                    if ticker not in tickers_in_level0:
                        continue

                    try:
                        close = _extract_close(raw[ticker])

                        if close is not None:
                            out[ticker] = close
                    except Exception:
                        continue

            elif tickers_in_level1:
                for ticker in requested:
                    if ticker not in tickers_in_level1:
                        continue

                    for field in ("Close", "Adj Close"):
                        try:
                            if (field, ticker) in raw.columns:
                                out[ticker] = pd.to_numeric(
                                    raw[(field, ticker)],
                                    errors="coerce",
                                )
                                break
                        except Exception:
                            continue

        else:
            if len(requested) == 1:
                close = _extract_close(raw)

                if close is not None:
                    out[requested[0]] = close

        if out.empty:
            return pd.DataFrame()

        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.ffill()
        out = out.dropna(how="all")

        return out

    frames = []

    for batch in chunked(ticker_list, 30):
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = _normalize(raw, batch)

            if not normalized.empty:
                frames.append(normalized)
                continue
        except Exception:
            pass

        try:
            raw = yf.download(
                tickers=batch,
                period="max",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = _normalize(raw, batch)

            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.sort_index().ffill().dropna(how="all")

    return out


