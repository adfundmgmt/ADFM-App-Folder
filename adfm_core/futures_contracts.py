"""Verified USD futures quote units; no inference for unreviewed contracts.

Multipliers apply to exchange quote units, not unverified provider rescaling.
Specifications reviewed 2026-09-29. These are contract P&L units, not a claim
that a provider's continuous price history is a realizable trading return.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    name: str
    contract_size: float
    contract_unit: str
    quote_unit: str
    multiplier: float
    spec_url: str


CME = "https://www.cmegroup.com/"
_SPECS = [
    ContractSpec("CL=F", "WTI crude oil", 1000, "barrels", "USD/barrel", 1000, CME + "education/courses/introduction-to-energy/introduction-to-crude-oil/wti-overview"),
    ContractSpec("NG=F", "Henry Hub natural gas", 10000, "MMBtu", "USD/MMBtu", 10000, CME + "markets/energy/natural-gas-futures.html"),
    ContractSpec("HO=F", "NY Harbor ULSD", 42000, "US gallons", "USD/gallon", 42000, CME + "rulebook/NYMEX/1a/150.pdf"),
    ContractSpec("RB=F", "RBOB gasoline", 42000, "US gallons", "USD/gallon", 42000, CME + "rulebook/NYMEX/1a/191.pdf"),
    ContractSpec("GC=F", "COMEX gold", 100, "troy ounces", "USD/troy ounce", 100, CME + "rulebook/COMEX/1a/113.pdf"),
    ContractSpec("MGC=F", "Micro gold", 10, "troy ounces", "USD/troy ounce", 10, CME + "education/lessons/micro-gold-and-micro-silver-futures-product-overview"),
    ContractSpec("SI=F", "COMEX silver", 5000, "troy ounces", "USD/troy ounce", 5000, CME + "rulebook/COMEX/1a/112.pdf"),
    ContractSpec("SIL=F", "Micro silver", 1000, "troy ounces", "USD/troy ounce", 1000, CME + "education/lessons/micro-gold-and-micro-silver-futures-product-overview"),
    ContractSpec("HG=F", "COMEX copper", 25000, "pounds", "USD/pound", 25000, CME + "education/lessons/copper-product-overview"),
    ContractSpec("ZC=F", "Corn", 5000, "bushels", "US cents/bushel", 50, CME + "trading/agricultural/files/grain-and-oilseed-futures-options-fact-card.pdf"),
    ContractSpec("ZS=F", "Soybeans", 5000, "bushels", "US cents/bushel", 50, CME + "trading/agricultural/files/grain-and-oilseed-futures-options-fact-card.pdf"),
]
CONTRACT_SPECS = {spec.symbol: spec for spec in _SPECS}


def get_contract_spec(symbol: str) -> ContractSpec | None:
    """Unknown/discontinued symbols remain unavailable; do not guess a multiplier."""
    return CONTRACT_SPECS.get(str(symbol).strip().upper())


def futures_pnl(symbol: str, quantity: float, price_change: float) -> float:
    """Signed USD P&L for an exact quoted-price change on one specified contract."""
    spec = get_contract_spec(symbol)
    if spec is None:
        raise ValueError(f"Unverified futures contract: {symbol}")
    if not isfinite(quantity) or not isfinite(price_change):
        raise ValueError("Quantity and price change must be finite")
    if quantity != int(quantity):
        raise ValueError("Futures contract quantity must be an integer")
    return float(quantity * price_change * spec.multiplier)


def discontinuity_candidates(close: pd.Series, *, threshold: float = 0.15) -> pd.DataFrame:
    """Flag large moves/sign changes for inspection; never identify a roll as fact.

    Robust trailing change scale is estimated without including the candidate.
    Real market gaps and negative-price episodes may also trigger this diagnostic.
    """
    prices = pd.to_numeric(close, errors="coerce")
    change = prices.diff()
    relative = change / prices.shift(1).abs().replace(0, np.nan)
    trailing_scale = relative.abs().shift(1).rolling(60, min_periods=20).median()
    gap = relative.abs().gt(threshold) & relative.abs().gt(8 * trailing_scale.fillna(0))
    nonpositive = prices.le(0) | prices.shift(1).le(0)
    mask = (gap | nonpositive) & change.notna()
    result = pd.DataFrame({"Close": prices, "Prior Close": prices.shift(1), "Price Change": change,
                           "Reason": np.where(nonpositive, "non-positive price: percentage return unavailable", "possible roll or market move")})
    return result.loc[mask]
