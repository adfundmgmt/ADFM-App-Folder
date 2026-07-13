"""Currency Tension Engine — Streamlit dashboard (spec §14).

Reads the persisted engine snapshot from cache and renders the two-axis tension map,
the inflection warnings, pillar breakdown, carry grid, and overlays. A refresh button
rebuilds the snapshot from the latest cached data.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cte.adapters.base import read_cache
from cte.config import CACHE_DIR
from cte.dashboard.plots import (carry_heatmap_fig, pillar_heatmap_fig,
                                 positioning_fig, tension_map_fig)

st.set_page_config(page_title="Currency Tension Engine", layout="wide")


def _grid(name):
    g = read_cache(name)
    return g.set_index(g.columns[0]) if g is not None else None


@st.cache_data(show_spinner=False, ttl=600)
def _load():
    """ttl: the daily Action pushes data-only commits that can rerun this script
    inside the SAME container without clearing st.cache_data — a 10-minute ttl
    means fresh snapshots surface on their own instead of demanding a reboot."""
    tm = read_cache("tension_map")
    pillars = read_cache("pillar_scores")
    overlays = read_cache("overlays")
    wpath = CACHE_DIR / "warnings.json"
    warns = json.loads(wpath.read_text()) if wpath.exists() else {}
    return tm, pillars, overlays, warns


def _rebuild():
    from cte.scoring.engine import build_snapshot
    build_snapshot(persist=True)
    _load.clear()


st.title("Currency Tension Engine")
st.caption("Two axes: **fundamental trajectory** (deteriorating ↔ improving) × "
           "**valuation & policy stretch** (cheap ↔ maxed-out). Each currency scored "
           "against its own history on dual horizons; carry is pairwise.")

with st.sidebar:
    st.header("Controls")
    horizon = st.radio("Horizon",
                       ["Regime (~2y)", "Structural (~10y)", "Secular (~15y)"],
                       index=1,
                       help="Secular scores each input against ~15 years of its "
                            "own history — a full generation of cycles. Rate "
                            "pillars have shorter source series (EUR yields 2004, "
                            "US TIPS 2003), so the secular dial starts later than "
                            "the others; today's snapshot is fully populated.")
    hz = {"St": "struct", "Re": "regime", "Se": "secular"}[horizon[:2]]

    st.markdown("---")
    st.subheader("History")
    _hist = read_cache("snapshot_history")
    trail_n = st.slider("Trail (month-ends)", 0, 12, 6,
                        help="Fading path of each currency's last N month-end "
                             "positions — which way it's moving, and how fast.")
    asof_sel = "Live"
    CUSTOM_W, user_w = False, {}
    if _hist is not None and len(_hist):
        from cte.scoring.history import dial_options
        _avail = dial_options(_hist, hz)
        if _avail:
            _mode = st.radio("View", ["Live", "Historical"], horizontal=True,
                             help="Historical redraws the map as it stood at a past "
                                  "month-end. Notes and overlays always describe the "
                                  "live snapshot.")
            if _mode == "Historical":
                _years = sorted({d.year for d in _avail}, reverse=True)
                _yr = st.selectbox("Year", _years, index=0)
                _mopts = [d for d in _avail if d.year == _yr]
                _mo = st.selectbox(
                    "Month", _mopts, index=len(_mopts) - 1,
                    format_func=lambda d: d.strftime("%B"))
                asof_sel = _mo.strftime("%Y-%m-%d")
                st.caption(f"Dial range: {_avail[0].strftime('%b %Y')} — "
                           f"{_avail[-1].strftime('%b %Y')} on this horizon.")

    st.markdown("---")
    with st.expander("Pillar weights"):
        from cte.config import PILLAR_AXIS, PILLAR_DISPLAY, PILLAR_WEIGHT
        st.caption("Re-weight how the pillars aggregate into the two axes — the "
                   "map, trails, and time dial recompute instantly from the same "
                   "pillar scores. 0 excludes a pillar. Warnings, notes, "
                   "commentary, and carry grids keep the default weighting.")
        _wp = [p for p in PILLAR_WEIGHT if p != "F_carry"]
        if st.button("Reset to defaults"):
            for _p in _wp:
                st.session_state[f"pw_{_p}"] = float(PILLAR_WEIGHT[_p])
        user_w = {}
        for _axis, _lbl in (("axis1_fundamental", "Axis 1 · fundamental"),
                            ("axis2_stretch", "Axis 2 · stretch")):
            st.markdown(f"**{_lbl}**")
            for _p in [p for p in _wp if PILLAR_AXIS.get(p) == _axis]:
                user_w[_p] = st.slider(PILLAR_DISPLAY[_p], 0.0, 3.0,
                                       float(PILLAR_WEIGHT[_p]), 0.25,
                                       key=f"pw_{_p}")
        CUSTOM_W = any(abs(user_w[p] - PILLAR_WEIGHT[p]) > 1e-9 for p in _wp)
    # rebuild is admin-only: it needs the raw cache + API keys, absent on the public
    # deploy (there the scheduled Action refreshes the snapshot instead)
    import os
    if read_cache("macro_backbone") is not None and os.environ.get("FRED_API_KEY"):
        if st.button("↻ Rebuild Snapshot"):
            with st.spinner("Recomputing…"):
                _rebuild()
            st.success("Snapshot refreshed.")
    st.markdown("---")
    st.caption("Data: FRED · OECD · BIS · Eurostat · e-Stat · ONS · CFTC · Yahoo. "
               "See docs/DATA_SOURCES.md.")

tm, pillars, overlays, warns = _load()

# schema guard: if the committed snapshot predates the selected horizon (e.g. the
# Secular columns before the reseeding workflow run), degrade to Structural with a
# notice instead of crashing — code schema may briefly run ahead of committed data
if f"axis1_fundamental_{hz}" not in tm.columns:
    st.warning(f"The committed snapshot doesn't include **{horizon}** scores yet — "
               "run the daily workflow once (it reseeds the histories and the "
               "snapshot under the new schema), then reload. Showing Structural.")
    hz, horizon = "struct", "Structural (~10y)"
if tm is None:
    st.warning("No snapshot in cache. Run `python -m scripts.backfill` then "
               "`python -m cte.scoring.engine`, or click Rebuild.")
    st.stop()

# daily note — read-only from cache; generated by the scheduled Action, never here
from cte.commentary.narrator import load_commentary
_note, _meta = load_commentary()
with st.container(border=True):
    st.markdown("#### Daily Read")
    st.markdown(_note)
    if _meta.get("generated_at"):
        st.caption(f"Generated {_meta['generated_at'][:16]}Z · {_meta.get('model','')} "
                   "· refreshed once daily by the scheduled job (not per-user).")

flagged = set(warns)

# ---- custom pillar weights: recompose the map + entire history client-side from
# the persisted pillar scores (exact — same aggregation the engine applies)
if CUSTOM_W:
    _ph_all = read_cache("pillar_history")
    if _ph_all is not None and len(_ph_all):
        from cte.scoring.compositor import axes_from_pillars
        _ks = ("date", "ccy", "kind")
        _re = axes_from_pillars(_ph_all, "struct", user_w, keys=_ks).merge(
            axes_from_pillars(_ph_all, "regime", user_w, keys=_ks),
            on=list(_ks), how="outer")
        if len(_re):
            _hist = _re
            _last = _re[_re.date == _re.date.max()]
            tm = _last.drop(columns=["date", "kind"]).reset_index(drop=True)
    else:
        st.warning("Custom weights need the pillar history — run the daily "
                   "workflow once to seed it. Showing default weighting.")
        CUSTOM_W = False

# ---- historical view frames (dial set) — every tab except Overlays renders as-of
HIST_MODE = asof_sel != "Live" and _hist is not None
tm_v, pillars_v, carry_v, pos_v = tm, pillars, None, overlays
if HIST_MODE:
    _d = pd.Timestamp(asof_sel)
    tm_v = (_hist[(_hist.date + pd.offsets.MonthEnd(0)) == _d]
            .sort_values("date").groupby("ccy").tail(1))
    _ph = read_cache("pillar_history")
    if _ph is not None:
        _pr = (_ph[(_ph.date + pd.offsets.MonthEnd(0)) == _d]
               .sort_values("date").groupby(["ccy", "pillar"]).tail(1))
        if len(_pr):
            _vcol = hz if hz in _pr.columns else "struct"
            pillars_v = (_pr.pivot_table(index="ccy", columns="pillar",
                                         values=_vcol).round(2).reset_index())
    _ch = read_cache("carry_history")
    if _ch is not None:
        carry_v = (_ch[(_ch.date + pd.offsets.MonthEnd(0)) == _d]
                   .sort_values("date").groupby("ccy").tail(1).set_index("ccy"))
    from cte.flags.positioning import positioning_asof
    pos_v = positioning_asof(_d)
elif hz != "struct":
    _ph2 = read_cache("pillar_history")
    if _ph2 is not None and hz in getattr(_ph2, "columns", []):
        _lastp = _ph2[_ph2.date == _ph2.date.max()]
        if len(_lastp) and _lastp[hz].notna().any():
            pillars_v = (_lastp.pivot_table(index="ccy", columns="pillar",
                                            values=hz).round(2).reset_index())

crowded = set()
if overlays is not None and "pos_label" in overlays.columns:
    crowded = set(overlays.loc[overlays.pos_label.astype(str)
                  .str.startswith("CROWDED"), "ccy"])

left, right = st.columns([3, 2])
with left:
    if asof_sel != "Live" and _hist is not None:
        _d = pd.Timestamp(asof_sel)
        _rows = _hist[(_hist.date + pd.offsets.MonthEnd(0)) == _d]             .sort_values("date").groupby("ccy").tail(1)
        st.pyplot(tension_map_fig(_rows, hz, None, history=_hist,
                                  trail_months=trail_n, asof_label=asof_sel),
                  use_container_width=True)
        st.caption(f"Historical view — the map as of {asof_sel}, computed from "
                   "today's data vintage (revised macro; see Methodology). Notes "
                   "and rings apply to the live snapshot only.")
    else:
        st.pyplot(tension_map_fig(tm, hz, flagged, history=_hist,
                                  trail_months=trail_n, crowded=crowded),
                  use_container_width=True)
        if CUSTOM_W:
            _cw = ", ".join(f"{k.split('_', 1)[1].title()} {v:g}"
                            for k, v in user_w.items())
            st.caption(f"**Custom pillar weights active** ({_cw}) — map, trails, "
                       "and dial recomposed; scored as of "
                       f"{pd.Timestamp(tm['date'].iloc[0]).date() if 'date' in tm.columns else 'latest snapshot'}. "
                       "Warnings, notes, and carry reflect default weights.")
with right:
    st.subheader("Currency Notes")
    st.caption("Per-currency context and objective flags in plain language — the "
               "twin-signal read (cheap/improving vs expensive/deteriorating), regime "
               "divergence, crowded or fragile carry, one-legged positions, and where a "
               "conditional signal was bent by an overlay.")
    if not warns:
        st.info("No notable reads this snapshot.")
    for ccy, notes in warns.items():
        with st.expander(f"{ccy}  ({len(notes)})", expanded=(ccy in ("GBP",))):
            for n in notes:
                st.markdown(f"- {n}")

st.markdown("---")
t1, t2, t_pos, t3, t4 = st.tabs(["Pillar Scores", "Carry Grid", "Positioning", "Overlays", "Currency Detail"])

with t1:
    if HIST_MODE:
        st.caption(f"As of {asof_sel} ({hz} pillar scores).")
    st.pyplot(pillar_heatmap_fig(pillars_v, tm_v, hz), use_container_width=True)
    st.caption("Signed z-scores, ordered by fundamental score. Green = the axis's "
               "positive pole (Axis 1: supportive to the currency; Axis 2: more "
               "stretched). Inflation and Fiscal are overlay-adjusted for the trap "
               "and reward-vs-stress regimes — see the warnings panel.")

with t2:
    basis = st.radio("Basis", ["2Y real", "2Y nominal"], horizontal=True)
    _feat = "real_2y" if basis == "2Y real" else "nominal_2y"
    label = f"{basis.title()} Carry  ·  Base − Quote (%)"
    if HIST_MODE:
        grid = None
        if carry_v is not None and _feat in carry_v.columns:
            from cte.transform.pairwise import grid_from_values
            grid = grid_from_values(carry_v[_feat])
        st.caption(f"As of {asof_sel}." if grid is not None else
                   "No carry history at this date.")
    else:
        gname = "carry_grid_real" if basis == "2Y real" else "carry_grid_nominal"
        grid = _grid(gname)
    if grid is not None and len(grid):
        st.pyplot(carry_heatmap_fig(grid, label), use_container_width=True)
    st.caption("Pairwise — the dollar is one leg of eight, not a hub. Nominal shows "
               "raw rate carry; real strips the inflation tax (fragile carry).")

with t3:
    if HIST_MODE:
        _oh = read_cache("overlay_history")
        _ov = None
        if _oh is not None:
            _ov = (_oh[(_oh.date + pd.offsets.MonthEnd(0)) == _d]
                   .sort_values("date").groupby("ccy").tail(1)
                   .set_index("ccy").drop(columns=["date", "kind"],
                                          errors="ignore"))
        if _ov is not None and len(_ov):
            st.caption(f"As of {asof_sel} — the conditioning state that bent that "
                       "month's scores: FX-yield reward/stress regime, "
                       "hike-feasibility, carry-to-vol crowding (percentile within "
                       "history *up to that date*), and the positioning read from "
                       "the last weekly report on or before it.")
            _pcols_h = [c for c in ("lev_pct_oi", "am_pct_oi", "lev_z", "am_z",
                                    "pos_label", "pos_date")
                        if pos_v is not None and c in pos_v.columns]
            if _pcols_h:
                _ov = _ov.join(pos_v.set_index("ccy")[_pcols_h], how="left")
            st.dataframe(_ov, use_container_width=True)
        else:
            st.caption("No overlay history at this date — run the workflow once "
                       "to seed it.")
    else:
        st.dataframe(overlays.set_index("ccy"), use_container_width=True)
    st.caption("yld_fx_corr / yld_regime: is a currency rewarded or punished for its "
               "yield. feasibility / infl_mult: hike room vs. the inflation trap. "
               "ctv_pctile: carry-to-vol crowding.")

with t4:
    ccy = st.selectbox("Currency", list(tm_v["ccy"]))
    if HIST_MODE:
        st.caption(f"As of {asof_sel}. Warnings are live-only.")
    row = tm_v.set_index("ccy").loc[ccy]
    c1, c2 = st.columns(2)
    _f, _v = row.get(f"axis1_fundamental_{hz}"), row.get(f"axis2_stretch_{hz}")
    c1.metric(f"{ccy} fundamental ({hz})",
              f"{_f:+.2f}" if pd.notna(_f) else "n/a")
    c2.metric(f"{ccy} stretch ({hz})",
              f"{_v:+.2f}" if pd.notna(_v) else "n/a")
    st.write("**Pillars:**")
    _pt4 = pillars_v.set_index("ccy")
    if ccy in _pt4.index:
        st.dataframe(_pt4.loc[[ccy]], use_container_width=True)
    if ccy in warns:
        st.write("**Warnings:**")
        for n in warns[ccy]:
            st.markdown(f"- {n}")

st.markdown("---")
with st.expander("Methodology — what goes into each pillar and leg"):
    st.markdown(
        """
Every currency is scored against **its own history** on the lookbacks shown via the
sidebar toggle: **Structural (~10y)** is the level anchor; **Regime (~2y)** shows which
way it has moved lately; **Secular (~15y)** stretches the anchor across a full
generation of cycles (rate pillars: EUR yield data begins 2004 and US TIPS 2003, so
secular history starts later than structural). Read structural and regime as a pair. Each raw input is turned into a z-score
(distance from its own norm), signed so a positive value points to the axis's positive
pole, then averaged into pillars and the two axes.

### Axis 1 · Fundamental trajectory  (deteriorating ↔ improving)

- **Growth (A)** — business-confidence level (OECD BCICP) and its 3-month slope, real
  GDP growth (YoY, from OECD/FRED chain-volume levels), and the 3-month change in the
  unemployment rate (FRED · Eurostat · ONS · OECD, each currency on its freshest source).
- **Inflation (B)** — headline CPI YoY minus the central bank's target (the inflation
  *gap*), plus 3-month inflation *momentum*. Above-target, accelerating inflation reads
  as hawkish support — **until** the hike-feasibility overlay judges the central bank
  trapped (see below), at which point it fades and reverses.
- **External (C)** — current-account balance (% GDP, flow) and net international
  investment position (% GDP, stock). The flow says who's under pressure now; the stock
  says who's a creditor vs. debtor. (OECD BOP / IIP, Eurostat for the euro area.)
- **Real 10Y (D)** — the 10-year real yield (US via TIPS; others as 10Y nominal minus
  CPI). This is the market-priced long end — a term-premium / market-tolerance read, not
  the budget deficit. Its sign is set by the **reward-vs-stress overlay**: if the currency
  rises with its yields it's being rewarded (support holds); if it falls as yields rise
  it's the GBP-2022 stress regime and the signal flips to a drag.

### Axis 2 · Valuation & policy stretch  (cheap ↔ maxed-out)

- **Valuation (G)** — the real effective exchange rate (BIS REER): how rich or cheap the
  currency is versus its own history. This is the direct cheap/expensive read, and it
  **leads Axis 2 at roughly 2:1** over policy — because policy's own-history baseline
  misreads early-cycle currencies (e.g. the yen, with the BoJ hiking off decades of NIRP)
  as "stretched" when they aren't.
- **Policy (E)** — the real policy rate (policy rate minus CPI) and the priced path
  (2Y minus policy, i.e. how much tightening/easing the market has discounted). High and
  restrictive = closer to the ceiling = more stretched.

### Carry (pairwise, shown as the grid — not a per-currency pillar)

Carry is inherently relative, so it lives in the **8×8 grid** rather than the map: the
2-year rate differential of each currency against each other one (nominal, and real =
after stripping inflation). The dollar is one leg of eight here, not a hub.

### Overlays (objective, and surfaced as warnings when they bend a signal)

- **Yield reward vs. stress** — rolling correlation of FX moves and yield moves; sets the
  Real-10Y sign.
- **Hike feasibility** — growth health minus how restrictive policy already is; when a
  currency has above-target inflation but soft growth and a maxed central bank, its
  inflation tailwind fades to a drag (the "policy trap").
- **Carry-to-vol** — real carry divided by realized FX volatility, percentiled; a high
  reading flags crowded carry that's fragile to a volatility spike.
- **Speculative positioning** — CFTC TFF (futures **and options combined**): leveraged-fund
  and asset-manager net positions as % of open interest, z-scored per currency vs its own
  10-year history. The *listed-derivatives slice only* (no OTC/corporate/real-money flows) —
  used strictly as a crowding and unwind-risk flag, never in the axis composites. A ring on
  the map marks a crowded currency; the Positioning tab shows the full read.

### History (trails & time dial)

The map's month-end positions are recomputed back through the data (~2008+) with **as-of**
overlay multipliers, then appended daily. Trails show each currency's recent path; the
sidebar dial redraws the map at any past month-end. One honesty note: history is computed
from **today's data vintage** (revised macro, CPI aligned to its reference month) — "as
we'd compute it now", not "as you'd have seen it then". Fine for trails and pattern-matching;
a point-in-time backtest needs additional lag adjustments.

Whenever an overlay bends a signal, the **Currency Notes** panel says so in plain
language — including a note when a reading is confounded (e.g. a strong-dollar tape) or a
data source is missing, so nothing ambiguous hides inside a single number.

*Sources: FRED · OECD · BIS · Eurostat · Japan e-Stat · UK ONS · CFTC · Yahoo Finance ·
national debt offices for yield curves. Full mapping in `docs/DATA_SOURCES.md`.*
        """
    )

with t_pos:
    if HIST_MODE:
        st.caption(f"As of {asof_sel} — the last weekly CFTC report on or before "
                   "that date, with its own 13-week path.")
    if pos_v is not None and "lev_z" in getattr(pos_v, "columns", []):
        st.pyplot(positioning_fig(pos_v), use_container_width=True)
        st.caption(
            "**Scope, honestly:** CFTC Traders-in-Financial-Futures, futures **and "
            "options combined** — the listed-derivatives slice (CME/ICE). It does not "
            "see OTC forwards/swaps, corporate hedging, or real-money cash flows, so "
            "it is *speculative* positioning, not the whole market. Used as a "
            "crowding / unwind-risk conditioner only — it never enters the axis "
            "composites. Net positions are % of open interest, z-scored against each "
            "currency's own 10-year history; USD row = ICE Dollar Index. Weekly data, "
            "Tuesday-dated, published Fridays (3-day lag).")
        _pcols = ["ccy", "pos_label", "lev_z", "lev_z_13w", "am_z", "am_z_13w",
                  "lev_pct_oi", "am_pct_oi", "pos_date"]
        _pt = pos_v[[c for c in _pcols if c in pos_v.columns]].dropna(
            subset=["lev_z"])
        st.dataframe(_pt, use_container_width=True, hide_index=True)
    else:
        st.info("No positioning data in this snapshot — the TFF cache is absent "
                "or the overlay hasn't been rebuilt yet.")
