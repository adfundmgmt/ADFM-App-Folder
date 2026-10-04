"""Reproduce SPX-only callout refinement; evaluate NDX after selection."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adfm_core.hedge_timer_data import callout_session_inputs  # noqa: E402
from adfm_core.hedge_timer_model import (  # noqa: E402
    CALLOUT_LEAD_LOOKBACK,
    FROZEN_CALLOUT_RULES,
    MODEL_FIT_END,
    NDX_TICKER,
    SPX_TICKER,
    compute_callouts,
    episode_audit,
    warning_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/hedge_timer/callout_calibration.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/hedge_timer/callout_drawdown_audit.csv")
    args = parser.parse_args()
    close = pd.read_csv(ROOT / "data/hedge_timer/research_inputs.csv", index_col="Date", parse_dates=True).loc[:MODEL_FIT_END]
    close = callout_session_inputs(close)
    bars = pd.read_csv(ROOT / "data/hedge_timer/research_indices_ohlc.csv", header=[0, 1], index_col=0, parse_dates=True).loc[:MODEL_FIT_END]
    spx_prices = bars.xs(SPX_TICKER, level=1, axis=1)
    candidates = []
    # Earlier SPX probes established a 1.25% retreat and a five-session
    # pre-peak window. This bounded refinement reduces repeat/false dots.
    for credit, shock_loss, shock_vol, spacing in product(
        [.005, .0075, .01, .015], [.01, .0125, .015], [.10, .15, .20], [5, 8, 10]
    ):
        rules = replace(FROZEN_CALLOUT_RULES, credit_retreat=credit, shock_loss=shock_loss,
                        shock_volatility=shock_vol, minimum_spacing=spacing)
        dots = compute_callouts(close, SPX_TICKER, rules)["Callout"]
        summary = warning_summary(spx_prices, dots, lookback=CALLOUT_LEAD_LOOKBACK)
        candidates.append(((-summary["captured"], summary["false_warnings"],
                            summary["late_warnings"], summary["warnings"]), rules, summary))
    candidates.sort(key=lambda row: row[0])
    _, selected, fit_summary = candidates[0]
    if selected != FROZEN_CALLOUT_RULES:
        raise RuntimeError("Provider history no longer reproduces frozen rules; review without refitting the live page")
    audits, transfer_summary = [], None
    for ticker in (SPX_TICKER, NDX_TICKER):
        result = compute_callouts(close, ticker, selected)
        px = bars.xs(ticker, level=1, axis=1)
        audit = episode_audit(ticker, px, result["Callout"], lookback=CALLOUT_LEAD_LOOKBACK)
        audit["Trigger"] = audit["First warning"].map(result["Trigger"]).fillna("")
        audits.append(audit.rename(columns={"First warning": "First callout", "Loss at warning": "Loss at callout"}))
        if ticker == NDX_TICKER:
            transfer_summary = warning_summary(px, result["Callout"], lookback=CALLOUT_LEAD_LOOKBACK)
    payload = {
        "fit_index": SPX_TICKER, "fit_start": "2020-01-01", "fit_end": MODEL_FIT_END,
        "rules": asdict(selected), "refinement_candidates": len(candidates),
        "pre_peak_sessions": CALLOUT_LEAD_LOOKBACK, "carried_state_counts": False,
        "spx": fit_summary, "ndx_unchanged_transfer": transfer_summary,
        "source": json.loads((ROOT / "data/hedge_timer/research_source.json").read_text()),
        "limitations": ["SPX results are fitted in-sample", "NDX was evaluated after selection and has missed episodes",
                        "Retrospective provider prices may be revised", "Credit uses HYG/LQD; full publication-dated OAS history was unavailable"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    pd.concat(audits, ignore_index=True).to_csv(args.audit, index=False)
    print(json.dumps({"rules": asdict(selected), "spx": fit_summary, "ndx": transfer_summary}, indent=2))


if __name__ == "__main__":
    main()
