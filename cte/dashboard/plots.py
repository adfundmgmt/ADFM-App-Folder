"""Dashboard chart functions (matplotlib) — kept separate from the Streamlit layout
so they can be rendered and tested without a running Streamlit server.

Each returns a matplotlib Figure built from the persisted engine snapshot.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

_BG = "#0e1117"
_PANEL = "#141821"
_FG = "#d7dce5"
_MUTE = "#6b7280"
_GRID = "#252a35"
_BLUE = "#5b9bd5"
_WARN = "#e07a5f"

# a calmer diverging map than RdYlGn (red -> slate -> teal-green)
_DIV = LinearSegmentedColormap.from_list(
    "cte_div", ["#c1543b", "#8a4a45", "#2b2f3a", "#3f7d6e", "#4fae8b"])

_QUAD = {("+", "+"): "Strong · expensive", ("-", "+"): "Vulnerable",
         ("+", "-"): "Attractive", ("-", "-"): "Washed out"}
_QUAD_TINT = {("+", "-"): "#193a2e", ("-", "+"): "#3a1f19"}  # attractive / vulnerable

PILLAR_LABEL = {
    "A_growth": "Growth (A)", "B_inflation": "Inflation (B)",
    "C_external": "External (C)", "D_fiscal": "Real 10Y (D)",
    "E_policy": "Policy (E)", "F_carry": "Carry (F)", "G_valuation": "Valuation (G)",
}
_AXIS1 = ["A_growth", "B_inflation", "C_external", "D_fiscal"]
_AXIS2 = ["E_policy", "F_carry", "G_valuation"]


def _style(ax):
    ax.set_facecolor(_PANEL)
    for s in ax.spines.values():
        s.set_color(_GRID)
    ax.tick_params(colors=_FG, length=0)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
        lab.set_color(_FG)


def tension_map_fig(tm, horizon="struct", flagged=None):
    xcol, ycol = f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"
    d = tm.dropna(subset=[xcol, ycol]).set_index("ccy")
    xs, ys = d[xcol], d[ycol]

    # fit each axis tightly to its own range (keeping 0 in view) so the currencies
    # spread across the panel instead of clustering on a wide symmetric scale
    def _bounds(v):
        lo, hi = min(v.min(), -0.1), max(v.max(), 0.1)
        pad = max((hi - lo) * 0.18, 0.15)
        return lo - pad, hi + pad
    xlo, xhi = _bounds(xs); ylo, yhi = _bounds(ys)

    fig, ax = plt.subplots(figsize=(8.4, 7.6), facecolor=_BG)
    _style(ax)
    # tints: attractive (improving+cheap) and vulnerable (deteriorating+stretched)
    ax.add_patch(plt.Rectangle((0, ylo), xhi, -ylo, color=_QUAD_TINT[("+", "-")],
                               alpha=0.5, zorder=0))
    ax.add_patch(plt.Rectangle((xlo, 0), -xlo, yhi, color=_QUAD_TINT[("-", "+")],
                               alpha=0.5, zorder=0))
    ax.axhline(0, color=_GRID, lw=1.2, zorder=1)
    ax.axvline(0, color=_GRID, lw=1.2, zorder=1)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    for (sx, sy), label in _QUAD.items():
        ax.text((xhi if sx == "+" else xlo) * 1.0 - (0.02 * (xhi - xlo) * (1 if sx == "+" else -1)),
                (yhi if sy == "+" else ylo) - (0.03 * (yhi - ylo) * (1 if sy == "+" else -1)),
                label, color=_MUTE, fontsize=9.5,
                ha=("right" if sx == "+" else "left"),
                va=("top" if sy == "+" else "bottom"))

    for ccy, r in d.iterrows():
        ax.scatter(r[xcol], r[ycol], s=1500, zorder=3, color=_BLUE, alpha=0.16,
                   edgecolors="none")
        ax.scatter(r[xcol], r[ycol], s=900, zorder=4, color=_BLUE, alpha=0.95,
                   edgecolors=_BG, linewidths=1.6)
        ax.text(r[xcol], r[ycol], ccy, color="white", fontsize=9.5,
                fontweight="bold", ha="center", va="center", zorder=5,
                path_effects=[pe.withStroke(linewidth=1.4, foreground=_BLUE)])

    hlabel = "Structural · ~10y" if horizon == "struct" else "Regime · ~2y"
    ax.set_xlabel("←  Deteriorating        Fundamental trajectory        Improving  →",
                  fontsize=9.5, labelpad=8)
    ax.set_ylabel("←  Cheap        Valuation & policy stretch        Maxed-out  →",
                  fontsize=9.5, labelpad=8)
    ax.set_title(f"Currency Tension Map   ·   {hlabel}", fontsize=13.5, pad=14,
                 fontweight="bold")
    fig.tight_layout()
    return fig


def carry_heatmap_fig(grid, title="2Y Real Carry  ·  Base − Quote (%)"):
    ccys = list(grid.index)
    n = len(ccys)
    vals = grid.values.astype(float)
    masked = np.ma.array(vals, mask=np.eye(n, dtype=bool))
    m = np.nanmax(np.abs(vals)) or 1
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor=_BG)
    _style(ax)
    im = ax.imshow(masked, cmap=_DIV, vmin=-m, vmax=m)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ccys, fontsize=10); ax.set_yticklabels(ccys, fontsize=10)
    ax.set_xlabel("Quote  (short leg)", color=_FG, fontsize=10, labelpad=6)
    ax.set_ylabel("Base  (long leg)", color=_FG, fontsize=10, labelpad=6)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, color=_PANEL, zorder=2))
        for j in range(n):
            if i != j:
                v = vals[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                        color=("white" if abs(v) > m * 0.45 else _FG), zorder=3)
    ax.set_title(title, color=_FG, fontsize=12.5, pad=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=_FG, length=0)
    cb.outline.set_edgecolor(_GRID)
    fig.tight_layout()
    return fig


def pillar_heatmap_fig(pillars, tm=None, horizon="struct"):
    """Signed pillar z-scores, grouped by axis, plus the Axis-1/Axis-2 composite each
    row rolls up into (the map coordinates) — so the grid reads pillar → axis → map."""
    p = pillars.set_index("ccy") if "ccy" in pillars.columns else pillars.copy()
    cols = [c for c in _AXIS1 + _AXIS2 if c in p.columns]
    p = p[cols]
    a1 = a2 = None
    if tm is not None:
        t = tm.set_index("ccy")
        order = t[f"axis1_fundamental_{horizon}"].reindex(p.index).sort_values(
            ascending=False).index
        p = p.loc[order]
        a1 = t[f"axis1_fundamental_{horizon}"].reindex(p.index)
        a2 = t[f"axis2_stretch_{horizon}"].reindex(p.index)

    n1 = len([c for c in _AXIS1 if c in cols])
    n2 = len(cols) - n1
    # assemble the display matrix: pillars, then a gap, then the two composites
    blocks = [p.values.astype(float)]
    labels = [PILLAR_LABEL[c] for c in cols]
    if a1 is not None:
        gap = np.full((len(p), 1), np.nan)
        blocks += [gap, a1.values.reshape(-1, 1), a2.values.reshape(-1, 1)]
        labels += ["", "Axis 1\n(fund.)", "Axis 2\n(stretch)"]
    M = np.hstack(blocks)
    ncol = M.shape[1]

    m = max(1.5, np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(11.2, 5.9), facecolor=_BG)
    _style(ax)
    ax.imshow(np.ma.masked_invalid(M), cmap=_DIV, vmin=-m, vmax=m, aspect="auto")
    ax.set_xticks(range(ncol)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index, fontsize=10)
    for i in range(len(p.index)):
        for j in range(ncol):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                comp = a1 is not None and j >= ncol - 2
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=(9.5 if comp else 9),
                        fontweight=("bold" if comp else "normal"),
                        color=("white" if abs(v) > m * 0.5 else _FG))
    ax.axvline(n1 - 0.5, color=_BG, lw=3)
    if a1 is not None:
        ax.axvline(len(cols) - 0.5 + 0.5, color=_BG, lw=6)  # gap before composites
    ax.text((n1 - 1) / 2, -0.72, "Axis 1 · Fundamental trajectory", color=_MUTE,
            fontsize=9, ha="center", va="center")
    ax.text(n1 + (n2 - 1) / 2, -0.72, "Axis 2 · Valuation & stretch", color=_MUTE,
            fontsize=9, ha="center", va="center")
    if a1 is not None:
        ax.text(ncol - 1.5, -0.72, "Composite", color=_MUTE, fontsize=9,
                ha="center", va="center")
    ax.set_title("Pillar Scores", color=_FG, fontsize=13, pad=26,
                 fontweight="bold")
    # legend as a discrete footnote below the grid, not competing with the title
    fig.text(0.5, 0.005,
             "Green = supportive / more stretched   ·   Red = drag / cheap   ·   "
             "Rows sorted by fundamental score",
             color=_MUTE, fontsize=8.5, ha="center", va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig
