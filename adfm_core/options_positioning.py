"""Deterministic option-chain analytics for the Options Positioning Compass."""

from __future__ import annotations

import math
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

OPTION_COLUMNS = (
    "contractSymbol",
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_delta(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    *,
    option_type: str,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> float:
    """Return a European Black-Scholes delta for a call or put."""
    inputs = (spot, strike, time_years, volatility)
    if not all(np.isfinite(value) and value > 0 for value in inputs):
        return np.nan
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * math.sqrt(time_years))
    call_delta = math.exp(-dividend_yield * time_years) * _normal_cdf(d1)
    if option_type.lower() == "call":
        return call_delta
    if option_type.lower() == "put":
        return call_delta - math.exp(-dividend_yield * time_years)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    *,
    option_type: str,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> float:
    """Return a European Black-Scholes option value."""
    inputs = (spot, strike, time_years, volatility)
    if not all(np.isfinite(value) and value > 0 for value in inputs):
        return np.nan
    root_t = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discounted_spot = spot * math.exp(-dividend_yield * time_years)
    discounted_strike = strike * math.exp(-risk_free_rate * time_years)
    if option_type.lower() == "call":
        return discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    if option_type.lower() == "put":
        return discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility_from_price(
    price: float,
    spot: float,
    strike: float,
    time_years: float,
    *,
    option_type: str,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> float:
    """Solve Black-Scholes IV by bisection, returning NaN outside price bounds."""
    if not all(np.isfinite(value) and value > 0 for value in (price, spot, strike, time_years)):
        return np.nan
    discounted_spot = spot * math.exp(-dividend_yield * time_years)
    discounted_strike = strike * math.exp(-risk_free_rate * time_years)
    if option_type.lower() == "call":
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        upper_bound = discounted_spot
    elif option_type.lower() == "put":
        lower_bound = max(discounted_strike - discounted_spot, 0.0)
        upper_bound = discounted_strike
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    if price < lower_bound - 1e-8 or price > upper_bound + 1e-8:
        return np.nan
    low, high = 0.001, 5.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        model = black_scholes_price(
            spot,
            strike,
            time_years,
            midpoint,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        if model > price:
            high = midpoint
        else:
            low = midpoint
    return (low + high) / 2.0


def prepare_chain(
    frame: pd.DataFrame,
    option_type: str,
    *,
    spot: float | None = None,
    time_years: float | None = None,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Normalize a Yahoo-style chain and derive a usable premium estimate."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[*OPTION_COLUMNS, "type", "mid", "premium_activity"])
    out = frame.copy()
    for column in OPTION_COLUMNS:
        if column not in out:
            out[column] = np.nan
    numeric = (
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
    )
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    quoted = out["bid"].gt(0) & out["ask"].ge(out["bid"])
    out["mid"] = out["lastPrice"]
    out.loc[quoted, "mid"] = (out.loc[quoted, "bid"] + out.loc[quoted, "ask"]) / 2.0
    out["volume"] = out["volume"].fillna(0).clip(lower=0)
    out["openInterest"] = out["openInterest"].fillna(0).clip(lower=0)
    out["premium_activity"] = out["mid"].clip(lower=0) * out["volume"] * 100.0
    out["type"] = option_type.lower()
    out["iv_source"] = "Yahoo"
    if (
        spot is not None
        and time_years is not None
        and np.isfinite(spot)
        and spot > 0
        and np.isfinite(time_years)
        and time_years > 0
    ):
        invalid_iv = ~out["impliedVolatility"].between(0.02, 5.0)
        for index in out.index[invalid_iv & out["mid"].gt(0)]:
            solved = implied_volatility_from_price(
                float(out.at[index, "mid"]),
                float(spot),
                float(out.at[index, "strike"]),
                float(time_years),
                option_type=option_type,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            if np.isfinite(solved) and 0.02 <= solved <= 5.0:
                out.at[index, "impliedVolatility"] = solved
                out.at[index, "iv_source"] = "Solved from price"
    return out.replace([np.inf, -np.inf], np.nan)


def _valid_iv(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["impliedVolatility"].between(0.02, 5.0)].copy()


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    readings: list[float] = []
    for frame in (calls, puts):
        valid = _valid_iv(frame)
        if not valid.empty:
            row = valid.loc[(valid["strike"] - spot).abs().idxmin()]
            readings.append(float(row["impliedVolatility"]))
    return float(np.mean(readings)) if readings else np.nan


def _target_delta_contract(
    frame: pd.DataFrame,
    *,
    spot: float,
    time_years: float,
    option_type: str,
    target_abs_delta: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.Series | None:
    valid = _valid_iv(frame)
    liquid = valid.loc[
        valid["volume"].gt(0)
        | valid["openInterest"].gt(0)
        | (valid["bid"].gt(0) & valid["ask"].gt(0))
    ]
    if not liquid.empty:
        valid = liquid
    otm = valid.loc[
        valid["strike"].ge(spot)
        if option_type == "call"
        else valid["strike"].le(spot)
    ]
    if not otm.empty:
        valid = otm
    if valid.empty:
        return None
    valid["delta"] = [
        black_scholes_delta(
            spot,
            strike,
            time_years,
            volatility,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        for strike, volatility in zip(
            valid["strike"], valid["impliedVolatility"], strict=False
        )
    ]
    valid["delta_distance"] = valid["delta"].abs().sub(target_abs_delta).abs()
    valid = valid.dropna(subset=["delta_distance"])
    return None if valid.empty else valid.loc[valid["delta_distance"].idxmin()]


def option_snapshot(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    spot: float,
    expiry: str | date,
    as_of: date,
    target_abs_delta: float = 0.25,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> dict[str, float | str]:
    """Summarize one expiration with ATM IV, skew, activity, and positioning."""
    expiry_date = pd.Timestamp(expiry).date()
    dte = max((expiry_date - as_of).days, 0)
    time_years = max(dte, 1) / 365.0
    call_chain = prepare_chain(
        calls,
        "call",
        spot=spot,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    put_chain = prepare_chain(
        puts,
        "put",
        spot=spot,
        time_years=time_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    call_25 = _target_delta_contract(
        call_chain,
        spot=spot,
        time_years=time_years,
        option_type="call",
        target_abs_delta=target_abs_delta,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    put_25 = _target_delta_contract(
        put_chain,
        spot=spot,
        time_years=time_years,
        option_type="put",
        target_abs_delta=target_abs_delta,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    call_iv = float(call_25["impliedVolatility"]) if call_25 is not None else np.nan
    put_iv = float(put_25["impliedVolatility"]) if put_25 is not None else np.nan
    call_volume = float(call_chain["volume"].sum())
    put_volume = float(put_chain["volume"].sum())
    call_oi = float(call_chain["openInterest"].sum())
    put_oi = float(put_chain["openInterest"].sum())
    combined = pd.concat([call_chain, put_chain], ignore_index=True)
    return {
        "expiry": expiry_date.isoformat(),
        "dte": float(dte),
        "spot": float(spot),
        "atm_iv": _atm_iv(call_chain, put_chain, spot),
        "put_25d_iv": put_iv,
        "call_25d_iv": call_iv,
        "put_skew": put_iv - call_iv,
        "risk_reversal": call_iv - put_iv,
        "put_call_volume": put_volume / call_volume if call_volume > 0 else np.nan,
        "put_call_oi": put_oi / call_oi if call_oi > 0 else np.nan,
        "option_volume": call_volume + put_volume,
        "option_open_interest": call_oi + put_oi,
        "premium_activity": float(combined["premium_activity"].sum()),
        "put_25d_strike": float(put_25["strike"]) if put_25 is not None else np.nan,
        "call_25d_strike": float(call_25["strike"]) if call_25 is not None else np.nan,
    }


def prior_percentile(value: float, peers: pd.Series) -> float:
    """Return a mid-rank percentile for a finite value within a peer series."""
    clean = pd.to_numeric(peers, errors="coerce").dropna().to_numpy(dtype=float)
    if not np.isfinite(value) or len(clean) < 2:
        return np.nan
    below = float(np.sum(clean < value))
    tied = float(np.sum(clean == value))
    return (below + 0.5 * tied) / len(clean) * 100.0


def ordinal(value: float) -> str:
    """Format a rounded numeric rank with its English ordinal suffix."""
    if not np.isfinite(value):
        return "N/A"
    number = int(round(value))
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def add_cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Add comparable current-snapshot ranks without implying historical ranks."""
    out = frame.copy()
    out["iv_richness"] = out["atm_iv"] - out["realized_vol_21d"]
    for source, target in (
        ("iv_richness", "iv_richness_percentile"),
        ("put_skew", "put_skew_percentile"),
    ):
        out[target] = [prior_percentile(value, out[source]) for value in out[source]]
    return out


def build_positioning_commentary(row: Mapping[str, object]) -> str:
    """Create restrained commentary from one ranked current snapshot."""
    ticker = str(row.get("ticker", "The selected ticker"))
    atm_iv = float(row.get("atm_iv", np.nan))
    richness_rank = float(row.get("iv_richness_percentile", np.nan))
    skew = float(row.get("put_skew", np.nan))
    skew_rank = float(row.get("put_skew_percentile", np.nan))
    volume_ratio = float(row.get("put_call_volume", np.nan))
    oi_ratio = float(row.get("put_call_oi", np.nan))
    parts: list[str] = []
    if np.isfinite(atm_iv) and np.isfinite(richness_rank):
        parts.append(
            f"{ticker} ATM implied volatility is {atm_iv * 100:.1f}%, with its IV-minus-21D-realized spread "
            f"at the {ordinal(richness_rank)} percentile of the selected universe."
        )
    if np.isfinite(skew) and np.isfinite(skew_rank):
        if skew >= 0:
            signal = "downside puts carry richer implied volatility than comparable upside calls"
        else:
            signal = "upside calls carry richer implied volatility than comparable downside puts"
        parts.append(
            f"The 25-delta put-minus-call skew is {skew * 100:+.1f} vol points "
            f"({ordinal(skew_rank)} cross-sectional percentile), so {signal}."
        )
    if np.isfinite(volume_ratio) or np.isfinite(oi_ratio):
        volume_text = f"{volume_ratio:.2f}x" if np.isfinite(volume_ratio) else "unavailable"
        oi_text = f"{oi_ratio:.2f}x" if np.isfinite(oi_ratio) else "unavailable"
        parts.append(
            f"Aggregate put/call volume is {volume_text}, versus {oi_text} on open interest; "
            "these are positioning proxies and do not identify whether trades opened or closed."
        )
    return " ".join(parts) or "The current chain does not contain enough valid observations for commentary."
