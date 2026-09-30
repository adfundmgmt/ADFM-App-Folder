"""Pure, session-only portfolio stress math. All monetary outputs are USD.

Scenarios are instantaneous: no theta/time passage. Option P&L uses changes in
European model value anchored to the supplied mark, not an invented market quote.
Convexity is dollars per bp squared per position unit (not conventional duration).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from math import erf, exp, isfinite, log, pi, sqrt

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    equity: float = 0.0
    yield_bp: float = 0.0
    fx: float = 0.0
    commodity: float = 0.0
    targets: Mapping[str, float] | None = None  # explicit local-quote underlying marks
    iv: float = 0.0  # absolute annualized volatility change: .05 = five vol points


def _number(value, label, *, minimum=None, strict=False):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(result) or (
        minimum is not None and (result <= minimum if strict else result < minimum)
    ):
        raise ValueError(f"{label} is outside its valid range")
    return result


def _date(value, label):
    # Strict calendar dates avoid silently normalizing times or ambiguous formats.
    try:
        result = date.fromisoformat(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD") from exc
    return result


def _cdf(value):
    return (1 + erf(value / sqrt(2))) / 2


def _option_terms(spot, strike, years, volatility, rate, dividend):
    d1 = (log(spot / strike) + (rate - dividend + volatility**2 / 2) * years) / (
        volatility * sqrt(years)
    )
    return d1, d1 - volatility * sqrt(years)


def european_option(spot, strike, years, volatility, rate, dividend, kind):
    """Black–Scholes–Merton European value in the underlying's quote currency."""
    for label, value in [
        ("spot", spot),
        ("strike", strike),
        ("years", years),
        ("volatility", volatility),
    ]:
        _number(value, label, minimum=0, strict=True)
    _number(rate, "rate")
    _number(dividend, "dividend")
    if kind not in {"call", "put"}:
        raise ValueError("Option kind must be call or put")
    d1, d2 = _option_terms(spot, strike, years, volatility, rate, dividend)
    if kind == "call":
        return spot * exp(-dividend * years) * _cdf(d1) - strike * exp(
            -rate * years
        ) * _cdf(d2)
    return strike * exp(-rate * years) * _cdf(-d2) - spot * exp(
        -dividend * years
    ) * _cdf(-d1)


def validate_holdings(frame: pd.DataFrame, valuation_date: str) -> pd.DataFrame:
    """Normalize every row or reject the entire input; never omit invalid holdings.

    All marks are supplied as of valuation_date, with no automatic quote fetching.
    Signed quantity encodes direction; an optional direction must agree. Shares
    use multiplier=1, FX is USD per foreign unit, derivatives have explicit units.
    """
    asof = _date(valuation_date, "Valuation date")
    if asof > date.today():
        raise ValueError("Valuation date cannot be in the future")
    required = {
        "symbol",
        "kind",
        "quantity",
        "multiplier",
        "mark",
        "mark_date",
        "currency",
        "fx_rate",
        "factor",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Supply nonempty holdings with all required schema columns")
    allowed = required | {
        "underlying",
        "strike",
        "expiry",
        "volatility",
        "rate",
        "dividend_yield",
        "dv01",
        "convexity",
        "margin_rate",
        "margin_per_contract",
        "direction",
        "underlying_symbol",
        "underlying_dv01",
        "underlying_convexity",
    }
    if set(frame.columns) - allowed:
        raise ValueError("Unrecognized holdings columns; use the documented schema")
    rows = []
    for index, raw in enumerate(frame.to_dict("records"), start=1):
        label = f"Row {index}"
        row = dict(raw)
        kind = str(row["kind"]).strip().lower()
        if kind not in {"shares", "fx", "futures", "dv01", "call", "put"}:
            raise ValueError(f"{label}: unsupported kind")
        row["kind"] = kind
        row["symbol"] = str(row["symbol"]).strip().upper()
        if pd.isna(raw["symbol"]) or not row["symbol"]:
            raise ValueError(f"{label}: symbol is required")
        if _date(row["mark_date"], f"{label} mark_date") != asof:
            raise ValueError(
                f"{label}: mark_date must equal valuation date; reconcile stale marks explicitly"
            )
        for field in ("quantity", "multiplier", "mark", "fx_rate"):
            row[field] = _number(row[field], f"{label} {field}")
        if row["quantity"] == 0 or row["multiplier"] <= 0 or row["fx_rate"] <= 0:
            raise ValueError(
                f"{label}: quantity must be signed/nonzero; multiplier and fx_rate positive"
            )
        if row["mark"] < 0 or (
            row["mark"] == 0 and kind not in {"dv01", "call", "put"}
        ):
            raise ValueError(f"{label}: invalid mark")
        if pd.isna(row["currency"]):
            raise ValueError(f"{label}: currency must be a three-letter code")
        row["currency"] = str(row["currency"]).strip().upper()
        if len(row["currency"]) != 3 or not row["currency"].isalpha():
            raise ValueError(f"{label}: currency must be a three-letter code")
        if row["currency"] == "USD" and row["fx_rate"] != 1:
            raise ValueError(f"{label}: USD fx_rate must equal 1")
        if pd.notna(row.get("direction")):
            direction = str(row["direction"]).strip().lower()
            if direction != ("long" if row["quantity"] > 0 else "short"):
                raise ValueError(f"{label}: direction conflicts with signed quantity")
        factor = str(row["factor"]).strip().lower()
        permitted = {
            "shares": {"equity", "rates"},
            "fx": {"fx"},
            "futures": {"equity", "commodity"},
            "dv01": {"yield"},
            "call": {"equity", "rates"},
            "put": {"equity", "rates"},
        }
        if factor not in permitted[kind]:
            raise ValueError(f"{label}: factor does not match kind")
        row["factor"] = factor
        target_symbol = (
            row.get("underlying_symbol") if kind in {"call", "put"} else None
        )
        row["target_symbol"] = (
            str(target_symbol).strip().upper()
            if pd.notna(target_symbol)
            else row["symbol"]
        )
        if not row["target_symbol"]:
            raise ValueError(f"{label}: underlying_symbol cannot be blank")
        if factor == "rates":
            sensitivity = row.get("underlying_dv01")
            row["underlying_dv01"] = (
                _number(sensitivity, f"{label} underlying_dv01", minimum=0, strict=True)
                if pd.notna(sensitivity)
                else None
            )
            convexity = row.get("underlying_convexity")
            row["underlying_convexity"] = (
                _number(convexity, f"{label} underlying_convexity", minimum=0)
                if pd.notna(convexity)
                else 0.0
            )
        if kind in {"shares", "fx", "dv01"} and row["multiplier"] != 1:
            raise ValueError(f"{label}: shares, fx and dv01 require multiplier=1")
        if kind == "fx" and (row["currency"] != "USD" or row["fx_rate"] != 1):
            raise ValueError(
                f"{label}: FX spot mark must be USD per foreign unit, currency=USD, fx_rate=1"
            )
        if kind in {"futures", "call", "put"} and not row["quantity"].is_integer():
            raise ValueError(f"{label}: contract quantity must be an integer")
        if kind == "futures":
            from adfm_core.futures_contracts import get_contract_spec

            spec = get_contract_spec(row["symbol"])
            if spec and row["currency"] != "USD":
                raise ValueError(
                    f"{label}: verified contract quote currency must be USD"
                )
            if spec and row["multiplier"] != spec.multiplier:
                raise ValueError(
                    f"{label}: multiplier disagrees with verified contract specification"
                )
        if kind == "dv01":
            row["dv01"] = _number(
                row.get("dv01"), f"{label} dv01", minimum=0, strict=True
            )
            row["convexity"] = _number(
                row.get("convexity", 0) if pd.notna(row.get("convexity")) else 0,
                f"{label} convexity",
                minimum=0,
            )
        if kind in {"call", "put"}:
            for field in ("underlying", "strike", "volatility"):
                row[field] = _number(
                    row.get(field), f"{label} {field}", minimum=0, strict=True
                )
            for field in ("rate", "dividend_yield"):
                row[field] = _number(row.get(field), f"{label} {field}")
            expiry = _date(row.get("expiry"), f"{label} expiry")
            if expiry <= asof:
                raise ValueError(f"{label}: expiry must follow valuation date")
            row["years"] = (expiry - asof).days / 365
            # Exercise-style-independent practical upper bound on supplied marks.
            if row["mark"] > (row["underlying"] if kind == "call" else row["strike"]):
                raise ValueError(
                    f"{label}: option mark exceeds underlying/strike bound"
                )
        for field in ("margin_rate", "margin_per_contract"):
            value = row.get(field)
            row[field] = (
                _number(value, f"{label} {field}", minimum=0)
                if pd.notna(value)
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _risk(rows):
    risk = {
        "Gross equity USD": 0.0,
        "Gross rates USD": 0.0,
        "Gross commodity USD": 0.0,
        "Gross FX USD": 0.0,
        "Gross DV01 USD/bp": 0.0,
        "Gross vega USD/vol point": 0.0,
    }
    for row in rows:
        scale = abs(row["quantity"] * row["multiplier"]) * row["fx_rate"]
        value = scale * row["mark"]
        if row["kind"] in {"call", "put"}:
            value = (
                scale * row["underlying"]
            )  # gross underlying notional, not premium or delta
            d1, _ = _option_terms(
                row["underlying"],
                row["strike"],
                row["years"],
                row["volatility"],
                row["rate"],
                row["dividend_yield"],
            )
            risk["Gross vega USD/vol point"] += (
                scale
                * row["underlying"]
                * exp(-row["dividend_yield"] * row["years"])
                * exp(-d1 * d1 / 2)
                / sqrt(2 * pi)
                * sqrt(row["years"])
                / 100
            )
        if row["factor"] == "rates":
            if pd.isna(row["underlying_dv01"]):
                risk["Gross DV01 USD/bp"] = float("nan")
            else:
                delta = 1.0
                if row["kind"] in {"call", "put"}:
                    delta = exp(-row["dividend_yield"] * row["years"]) * (
                        _cdf(d1) if row["kind"] == "call" else _cdf(d1) - 1
                    )
                risk["Gross DV01 USD/bp"] += scale * abs(delta) * row["underlying_dv01"]
        if row["factor"] in {"equity", "commodity", "rates"}:
            risk[f"Gross {row['factor']} USD"] += value
        if row["kind"] == "fx" or row["currency"] != "USD":
            risk["Gross FX USD"] += value
        if row["kind"] == "dv01":
            risk["Gross DV01 USD/bp"] += (
                abs(row["quantity"]) * row["dv01"] * row["fx_rate"]
            )
    return risk


def _shocked_price(row, shock):
    old = row["underlying"] if row["kind"] in {"call", "put"} else row["mark"]
    if shock.targets and row["target_symbol"] in shock.targets:
        return shock.targets[row["target_symbol"]]
    if row["factor"] == "rates":
        if shock.yield_bp == 0:
            return old
        if pd.isna(row["underlying_dv01"]):
            raise ValueError(
                "Yield shock requires rates underlying_dv01 or an explicit symbol price target"
            )
        value = (
            old
            - row["underlying_dv01"] * shock.yield_bp
            + 0.5 * row["underlying_convexity"] * shock.yield_bp**2
        )
        return _number(value, "Shocked rates underlying", minimum=0, strict=True)
    return old * (1 + getattr(shock, row["factor"]))


def _position_pnl(row, shock):
    scale = row["quantity"] * row["multiplier"]
    fx = row["fx_rate"] * (1 + shock.fx if row["currency"] != "USD" else 1)
    old = row["mark"]
    if row["kind"] in {"call", "put"}:
        args = (row["strike"], row["years"])
        base_model = european_option(
            row["underlying"],
            *args,
            row["volatility"],
            row["rate"],
            row["dividend_yield"],
            row["kind"],
        )
        new_model = european_option(
            _shocked_price(row, shock),
            *args,
            row["volatility"] + shock.iv,
            row["rate"] + shock.yield_bp / 10000,
            row["dividend_yield"],
            row["kind"],
        )
        new = max(0.0, old + new_model - base_model)
    elif row["kind"] == "dv01":
        new = (
            old
            - row["dv01"] * shock.yield_bp
            + 0.5 * row["convexity"] * shock.yield_bp**2
        )
    else:
        new = _shocked_price(row, shock)
    if row["kind"] == "futures":
        # Futures have zero funded market value. Convert the payoff only; foreign
        # FX movement alone does not turn futures notional into currency P&L.
        pnl = scale * (new - old) * fx
    else:
        pnl = scale * (new * fx - old * row["fx_rate"])
    return pnl, abs(scale * new * fx)


def portfolio_scenarios(holdings, nav, valuation_date, scenarios):
    """Return a complete scenario table or raise; no I/O, cache or telemetry."""
    nav = _number(nav, "Current supplied NAV", minimum=0, strict=True)
    rows = validate_holdings(holdings, valuation_date).to_dict("records")
    risk = _risk(rows)
    if not all(
        isfinite(value) or (key == "Gross DV01 USD/bp" and pd.isna(value))
        for key, value in risk.items()
    ):
        raise ValueError("Exposure exceeds supported numeric range")
    if not scenarios:
        raise ValueError("Supply at least one scenario")
    results = []
    for shock in scenarios:
        target_symbols = {row["target_symbol"] for row in rows if row["kind"] != "dv01"}
        targets = {}
        if shock.targets is not None:
            if not isinstance(shock.targets, Mapping):
                raise ValueError(
                    "Scenario targets must map underlying symbols to local-quote prices"
                )
            for key, value in shock.targets.items():
                symbol = str(key).strip().upper()
                if symbol not in target_symbols or symbol in targets:
                    raise ValueError(
                        "Scenario price target is unused or duplicated; reconcile holdings symbols"
                    )
                targets[symbol] = _number(
                    value, "Scenario target price", minimum=0, strict=True
                )
        shock = replace(shock, targets=targets)
        for field in ("equity", "yield_bp", "fx", "commodity", "iv"):
            value = _number(getattr(shock, field), f"Scenario {field}")
            if field in {"equity", "fx", "commodity"} and value <= -1:
                raise ValueError("Price/FX shocks must be greater than -100%")
        pnl = margin = base_margin = 0.0
        available = True
        for row in rows:
            contribution, value = _position_pnl(row, shock)
            pnl += contribution
            if row["kind"] in {"shares", "fx"}:
                rate = row["margin_rate"]
                if rate is None or pd.isna(rate):
                    available = False
                else:
                    margin += value * rate
                    base_margin += (
                        abs(row["quantity"] * row["mark"] * row["fx_rate"]) * rate
                    )
            else:
                per_contract = row["margin_per_contract"]
                if per_contract is None or pd.isna(per_contract):
                    available = False
                else:
                    amount = abs(row["quantity"]) * per_contract
                    margin += amount
                    base_margin += amount
        if not all(
            isfinite(value)
            for value in (pnl, margin, base_margin, nav + pnl, pnl / nav * 100)
        ):
            raise ValueError("Scenario exceeds supported numeric range")
        loss = max(0.0, -pnl)
        results.append(
            {
                "Scenario": shock.name,
                "Equity %": shock.equity * 100,
                "Yield bp": shock.yield_bp,
                "FX %": shock.fx * 100,
                "Commodity %": shock.commodity * 100,
                "IV points": shock.iv * 100,
                "P&L USD": pnl,
                "NAV impact %": pnl / nav * 100,
                "Scenario NAV USD": nav + pnl,
                **risk,
                "Risk basis": "base gross exposures; rates DV01 unavailable"
                if pd.isna(risk["Gross DV01 USD/bp"])
                else "base gross notionals and sensitivity approximations",
                "Price target count": len(targets),
                "Loss funding USD": loss,
                "Estimated margin USD": margin if available else float("nan"),
                "Estimated cash buffer USD": loss + max(margin - base_margin, 0)
                if available
                else float("nan"),
                "Margin basis": "supplied broker inputs; estimate"
                if available
                else "unavailable: missing broker inputs",
            }
        )
    return pd.DataFrame(results)
