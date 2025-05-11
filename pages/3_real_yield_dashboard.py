#!/usr/bin/env python3
"""
10-Year Nominal & Real Yield Dashboard — *Inverted Momentum*
===========================================================
Momentum is defined so that **falling real yields = positive** (easing) and
**rising yields = negative** (tightening).

Update:
-------
* Banner spacing corrected — now rendered cleanly with clear line spacing between date and metrics.
* Metrics banner is now right-aligned for visual clarity.

Run
---
python real_yield_dashboard.py --window 5y
"""

import argparse
import sys, types
from typing import List, Tuple

import packaging.version
try:
    import distutils.version
except ImportError:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

import pandas as pd
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

REAL_SERIES = "DFII10"
NOM_SERIES = "DGS10"
WIN = 63
EASE_BP = 40
TIGHT_BP = -40
NBSP = "\u00A0"

# ── Helpers ───────────────────────────────────────────────────────────────

def fred(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = pdr.DataReader(series, "fred", start, end)[series]
    return s.ffill().dropna()

def contiguous(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segs, st = [], None
    for ts, flag in mask.items():
        if flag and st is None:
            st = ts
        elif not flag and st is not None:
            segs.append((st, ts)); st = None
    if st is not None:
        segs.append((st, mask.index[-1]))
    return segs

# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    cli = argparse.ArgumentParser(description="10-Year Yield Dashboard (Inverted Momentum)")
    cli.add_argument("--window", default="5y", help="Look-back horizon, e.g. 5y, 730d")
    args = cli.parse_args()

    end = pd.Timestamp.today().normalize()
    w = args.window.lower()
    if w.endswith("d"):
        start = end - pd.Timedelta(days=int(w[:-1]))
    elif w.endswith("y"):
        start = end - pd.DateOffset(years=int(w[:-1]))
    else:
        raise ValueError("--window must end with 'd' or 'y'")

    ext = start - pd.Timedelta(days=WIN*2)
    real = fred(REAL_SERIES, ext, end)
    nom = fred(NOM_SERIES, ext, end)

    real, nom = real.loc[start:], nom.loc[start:]
    mom = -(real - real.shift(WIN)) * 100

    latest = dict(date=real.index[-1], real=real.iloc[-1], nom=nom.iloc[-1], mom=mom.iloc[-1])
    regime = "Easing" if latest["mom"] > EASE_BP else "Tightening" if latest["mom"] < TIGHT_BP else "Neutral"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.05,
        subplot_titles=("Nominal vs Real 10-Yr Yield", "63-Day Inverted Momentum (bp)")
    )

    fig.add_trace(go.Scatter(x=nom.index, y=nom, name="Nominal", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=real.index, y=real, name="Real", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
    fig.update_yaxes(title="Yield (%)", tickformat=".2f", row=1, col=1)

    fig.add_trace(go.Scatter(x=mom.index, y=mom, name="Momentum (bp)", line=dict(color="#d62728", dash="dash", width=1.8)), row=2, col=1)
    for th, lab in [(EASE_BP, "+40 bp"), (0, "zero"), (TIGHT_BP, "-40 bp")]:
        fig.add_hline(y=th, line=dict(color="grey", dash="dot", width=1), row=2, col=1,
                      annotation_text=lab, annotation_position="right")

    for s, e in contiguous(mom > EASE_BP):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(0,128,0,0.12)", line_width=0, row=2, col=1)
    for s, e in contiguous(mom < TIGHT_BP):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=2, col=1)

    fig.update_yaxes(title="bp", tickformat=".0f", row=2, col=1)

    metrics = f"Nom {latest['nom']:.2f}%{NBSP*3}|{NBSP*3}Real {latest['real']:.2f}%{NBSP*3}|{NBSP*3}Inv Mom {latest['mom']:+.0f} bp{NBSP*3}|{NBSP*3}<b>{regime}</b>"
    fig.add_annotation(
        x=1.0, y=1.07, xref="paper", yref="paper",
        text=f"<span style='font-size:14px;'>{latest['date']:%b %d %Y}</span><br><span style='font-size:13px;'>{metrics}</span>",
        align="right", showarrow=False
    )

    fig.update_layout(
        template="plotly_white",
        height=720,
        title=dict(text="10-Year Yield Dashboard", x=0.5, xanchor="center"),
        legend=dict(orientation="h", x=0, y=1.11, xanchor="left"),
        margin=dict(l=60, r=40, t=90, b=60),
    )
    fig.update_xaxes(tickformat="%b-%y", row=2, col=1, title="Date")

    pyo.plot(fig, auto_open=True)

if __name__ == "__main__":
    main()
