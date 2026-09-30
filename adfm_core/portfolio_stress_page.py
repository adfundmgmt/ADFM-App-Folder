"""Private, in-memory portfolio upload and one consolidated stress table."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from adfm_core.portfolio_stress import Scenario, portfolio_scenarios

SCHEMA = "symbol,kind,quantity,multiplier,mark,mark_date,currency,fx_rate,factor,underlying,strike,expiry,volatility,rate,dividend_yield,dv01,convexity,margin_rate,margin_per_contract,underlying_symbol,underlying_dv01,underlying_convexity"
TEMPLATE = (
    SCHEMA
    + "\n"
    + ",".join(
        (
            "EXAMPLE",
            "shares",
            "10",
            "1",
            "100",
            "2026-09-29",
            "USD",
            "1",
            "equity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "0.5",
            "",
            "",
            "",
            "",
        )
    )
    + "\n"
)


def render_portfolio_stress():
    st.subheader("Portfolio stress")
    st.caption(
        "Private session input · explicitly supplied marks and current USD NAV · instantaneous combined shocks"
    )
    with st.expander("CSV schema, template and model limits"):
        st.download_button(
            "Download holdings template", TEMPLATE, "holdings_template.csv", "text/csv"
        )
        st.markdown(
            "The template is an illustrative row, not a live holding or quote. Replace every value. "
            "Required columns: `symbol, kind, quantity, multiplier, mark, mark_date, currency, fx_rate, factor`. "
            "Use signed quantities (positive long, negative short); contract quantities must be integers. "
            "Kinds: `shares`, `fx`, `futures`, `dv01`, `call`, `put`. Factors: equity or rates for shares/options, "
            "equity or commodity for futures, fx for FX spot, yield for DV01. "
            "All mark dates must equal the entered valuation date. USD marks require fx_rate=1; "
            "other marks are local currency with supplied USD per local currency fx_rate. "
            "FX spot uses foreign-unit quantity and USD per unit mark, currency=USD and fx_rate=1. "
            "Shares/FX/DV01 use multiplier=1; futures/options require explicit quote-unit multipliers. "
            "Known futures multipliers are checked against contract specifications; unknown contracts rely on your supplied specification. "
            "A ticker ending =F is a provider series identifier: use your actual contract mark, not an inferred continuous-series execution price."
        )
        st.markdown(
            "Bond/rate ETFs such as TLT use factor=`rates`, so equity shocks do not alter their underlying. "
            "For yield repricing, supply `underlying_dv01` (local currency per bp per underlying share) and optional "
            "`underlying_convexity` (local currency per bp² per share, default zero). Underlying change is "
            "−underlying_dv01 × yield bp + ½ underlying_convexity × yield bp² before repricing options. "
            "No duration is guessed: when sensitivities are missing, a nonzero yield shock requires an explicit "
            "symbol price target and gross rates DV01 is labeled unavailable. Options can provide `underlying_symbol` "
            "to connect their ticker/contract label to a target, e.g. TLT."
        )
        st.markdown(
            "DV01 rows require positive `dv01` in local currency per bp per position unit; optional `convexity` "
            "is local currency per bp² per unit (default zero). P&L = quantity × (−DV01 × yield bp "
            "+ ½ convexity × yield bp²). Supply the funded market value as mark when FX translation applies; "
            "mark=0 means a standalone sensitivity exposure. This is a local rate approximation."
        )
        st.markdown(
            "Calls/puts require `underlying, strike, expiry` (YYYY-MM-DD), `volatility, rate, dividend_yield` "
            "as decimal annual inputs and an observed premium mark in the same quote currency. "
            "European Black–Scholes–Merton changes are anchored to the supplied premium with a zero premium floor; "
            "the base scenario has zero P&L. Equity shocks reprice the underlying, yield shocks change the risk-free rate, "
            "IV shocks add absolute volatility points; FX changes translate foreign values. "
            "No time passes (ACT/365 to expiry), dividend cash flows are approximated by a continuous yield. "
            "American early exercise, assignment, delivery, borrow, liquidity, transaction costs and settlement cash "
            "are not modeled; stress results and cash estimates are approximations."
        )
        st.markdown(
            "Optional broker inputs: `margin_rate` (decimal fraction of shocked gross value for shares/FX) and "
            "`margin_per_contract` (USD per contract/unit for futures/options/DV01). Every row must supply the "
            "applicable input for a portfolio margin estimate; otherwise margin and cash buffer are unavailable. "
            "Contract margin inputs stay fixed across scenarios; actual broker stress/offset rules are not inferred. "
            "Loss funding = max(−P&L, 0); cash buffer = loss funding + positive increase in estimated margin. "
            "This conservative funding proxy is not a precise margin call or actual cash settlement schedule. "
            "Gross factor exposures are base notionals (options: full underlying notional), overlapping foreign-currency "
            "notionals, gross DV01 and model vega per 1 vol point. Gross FX notional includes foreign derivatives, "
            "whose FX payoff sensitivity depends on the simultaneous price shock. No correlation, netting or diversification is assumed."
        )
        st.caption(
            "The uploaded contents are read only in this session: no disk writes, shared cache, provider calls or portfolio telemetry."
        )
    first, second = st.columns(2)
    nav = first.number_input(
        "Current portfolio NAV (USD, required)",
        min_value=0.01,
        value=None,
        key="stress_nav",
        placeholder="Enter actual NAV",
        format="%.2f",
    )
    valuation = second.date_input(
        "Valuation date (required)", value=None, key="stress_date"
    )
    upload = st.file_uploader(
        "Dated holdings CSV (session only)", type=["csv"], key="stress_upload"
    )
    with st.expander("Combined scenario inputs", expanded=False):
        left, right = st.columns(2)
        equity = left.number_input(
            "Equity shock %", min_value=-99.9, value=-10.0, key="stress_equity"
        )
        yields = right.number_input("Yield shock bp", value=100.0, key="stress_yield")
        fx = left.number_input(
            "Foreign currency vs USD shock %",
            min_value=-99.9,
            value=-5.0,
            key="stress_fx",
        )
        commodity = right.number_input(
            "Commodity shock %", min_value=-99.9, value=-15.0, key="stress_commodity"
        )
        iv = left.number_input(
            "IV shock (absolute volatility points)", value=5.0, key="stress_iv"
        )
        target_text = st.text_area(
            "Optional symbol price targets (local quote units)",
            value="",
            key="stress_targets",
            placeholder="TLT=95",
        )
        st.caption(
            "Comma or newline separated SYMBOL=PRICE; targets replace that symbol's factor price shock, while FX, IV and rate changes still apply. Unmatched targets reject the scenario."
        )
    if upload is None or nav is None or valuation is None:
        st.info(
            "Supply a dated CSV, actual current NAV and valuation date to calculate the complete portfolio."
        )
        return
    try:
        upload.seek(0)
        contents = upload.read(2 * 1024 * 1024 + 1)
        if len(contents) > 2 * 1024 * 1024:
            st.error("CSV exceeds the 2 MB session input limit.")
            return
        frame = pd.read_csv(io.BytesIO(contents))
        if len(frame) > 10000:
            st.error("CSV exceeds the 10,000 holding row limit.")
            return
    except (ValueError, UnicodeError, OSError):
        st.error(
            "CSV could not be parsed. Use the holdings template; no portfolio total calculated."
        )
        return
    try:
        targets = {}
        for line in target_text.replace(",", "\n").splitlines():
            if not line.strip():
                continue
            pair = line.rsplit("=", 1)
            if len(pair) != 2 or not pair[0].strip():
                raise ValueError("Price targets must use SYMBOL=PRICE")
            symbol = pair[0].strip().upper()
            if symbol in targets:
                raise ValueError("Duplicate price targets are not allowed")
            try:
                targets[symbol] = float(pair[1])
            except ValueError as exc:
                raise ValueError("Price targets must contain numeric prices") from exc
        table = portfolio_scenarios(
            frame,
            nav,
            valuation.isoformat(),
            [
                Scenario("Base supplied marks"),
                Scenario(
                    "Combined custom stress",
                    equity=equity / 100,
                    yield_bp=yields,
                    fx=fx / 100,
                    commodity=commodity / 100,
                    iv=iv / 100,
                    targets=targets,
                ),
            ],
        )
    except ValueError as exc:
        # Validation messages contain only row/field labels, never holdings.
        st.error(f"{exc}. No portfolio total calculated.")
        return
    except OverflowError:
        st.error("Inputs exceed supported model range. No portfolio total calculated.")
        return
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(format="%.2f")
            for name in table.columns
            if name not in {"Scenario", "Margin basis", "Risk basis"}
        },
    )
    st.caption(
        f"{len(frame):,} validated holdings · valuation/NAV date {valuation:%Y-%m-%d} · USD reporting · header sorting enabled"
    )
