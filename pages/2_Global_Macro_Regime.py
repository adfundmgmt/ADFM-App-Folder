from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.global_macro import (
    COUNTRIES, INDICATORS, clean, comparison_period, country_rows,
    load_economics, load_equities, load_yields,
)
from adfm_core.ui import PageHeader, inject_explorer_style, render_page_header, render_footer

st.set_page_config(page_title="Global Macro Regime", layout="wide", initial_sidebar_state="collapsed")
inject_explorer_style(max_width_px=1700)
render_page_header(PageHeader("Global Macro Regime", "G20 · Country leadership, sovereign yields and economic momentum"))

metric = st.segmented_control("Market / economy", ["Equities", "10Y yields", *INDICATORS], default="Equities", key="gm_metric", label_visibility="collapsed") or "Equities"
today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
a, b, c = st.columns([1, 1, 2])
view, horizon, mode = "Level", "1M", "Comparable period"
with a:
    if metric == "Equities":
        horizon = st.selectbox("Performance window", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=1)
    else:
        view = st.selectbox("Map measure", ["Level", "Change"], index=1 if metric == "10Y yields" else 0, key=f"gm_view_{metric}")
with b:
    if metric == "10Y yields" and view == "Change":
        horizon = st.selectbox("Change window", ["1M", "3M", "6M", "1Y"], index=1)
    elif metric == "Equities":
        st.caption("Local-currency index returns\n\nDaily observations through the prior date")
    elif metric != "10Y yields":
        st.caption("Annual data" + (" · change versus prior year" if view == "Change" else " · latest comparable year"))
with c:
    if metric != "Equities":
        mode = st.selectbox("Observation alignment", ["Comparable period", "Latest per country"], help="Comparable period uses the latest completed period covered by at least 80% of fresh reporting countries. Missing countries stay gray.")

with st.spinner(f"Loading {metric.lower()}…"):
    if metric == "Equities":
        series, errors = load_equities()
    elif metric == "10Y yields":
        series, errors = load_yields()
    else:
        series, errors = load_economics(metric)
frequency = "M" if metric == "10Y yields" else "Y"
period = comparison_period(series, today, frequency) if metric != "Equities" and mode == "Comparable period" else None
data = country_rows(series, today, metric, view, horizon, period)
# If no defensible common period exists, never silently switch to mixed dates.
if metric != "Equities" and mode == "Comparable period" and period is None:
    data["Value"] = np.nan
    data["Status"] = "No comparable period"

is_change = metric == "Equities" or view == "Change"
unit = "bp" if metric == "10Y yields" and is_change else "pp" if metric in INDICATORS and is_change else "%"
measure = f"{horizon} return" if metric == "Equities" else f"{horizon} yield change" if metric == "10Y yields" and is_change else "Annual change" if is_change else metric
available = data.loc[data["Value"].notna()].copy()
period_note = f" · {period}" if period else " · dates vary by country" if metric != "Equities" else ""
if metric == "Equities":
    st.caption(f"{len(available)}/19 countries observed · {horizon} local-currency performance · green = rising; red = falling · gray = unavailable, stale or outside G20")
elif metric == "10Y yields":
    st.caption(f"{len(available)}/19 countries observed · 10Y monthly averages, not live yields{period_note} · blue = lower; orange = higher · gray = unavailable")
else:
    extra = " · ILO modeled estimates" if metric == "Unemployment" else ""
    st.caption(f"{len(available)}/19 countries observed · annual {metric.lower()}{extra}{period_note} · gray = unavailable")
if available.empty:
    st.warning("No observations meet this view's coverage and freshness requirements. Unavailable countries remain gray; try another metric.")

# A full-country gray trace retains missing-country hover without inventing values.
fig = go.Figure(go.Choropleth(
    locations=data["ISO"], z=[0] * len(data), locationmode="ISO-3",
    colorscale=[[0, "#44484e"], [1, "#44484e"]], showscale=False,
    text=data["Country"], customdata=data[["Status", "Period"]].fillna("").to_numpy(),
    marker_line_color="#91969d", marker_line_width=.55,
    hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Observation: %{customdata[1]}<extra></extra>",
))
if not available.empty:
    if metric in ("10Y yields", "Inflation"):
        scale = [[0, "#598ace"], [.5, "#e0e4e8"], [1, "#df9653"]]
    elif metric == "Unemployment":
        scale = [[0, "#72b3a0"], [.5, "#e0e4e8"], [1, "#c56d79"]]
    else:
        scale = [[0, "#c56d79"], [.5, "#e0e4e8"], [1, "#72b3a0"]]
    bounds = {}
    if is_change or metric == "GDP growth":
        limit = max(float(available["Value"].abs().max()), .01)
        bounds = dict(zmin=-limit, zmax=limit, zmid=0)
    fig.add_trace(go.Choropleth(
        locations=available["ISO"], z=available["Value"], locationmode="ISO-3",
        text=available["Country"], colorscale=scale, **bounds,
        customdata=available[["Series", "Period", "Baseline"]].to_numpy(),
        marker_line_color="#bdc3cb", marker_line_width=.65,
        colorbar=dict(title=dict(text=unit), thickness=10, len=.65, tickfont=dict(color="#e1e4e8")),
        hovertemplate=f"<b>%{{text}}</b><br>%{{customdata[0]}}<br>{measure}: %{{z:.2f}} {unit}<br>Observation: %{{customdata[1]}}<br>Baseline: %{{customdata[2]}}<extra></extra>",
    ))
fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False,
                showland=True, landcolor="#30343a", showcountries=True, countrycolor="#777d85",
                countrywidth=.45, showocean=True, oceancolor="#0b0e12", bgcolor="#0b0e12",
                lataxis_range=[-58, 85])
fig.update_layout(height=580, margin=dict(l=0, r=0, t=12, b=4), paper_bgcolor="#0b0e12",
                  font=dict(color="#e1e4e8"), dragmode=False)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False}, key="gm_world_map")

# Numeric values stay numeric so every column sorts correctly.
value_label = f"{measure} ({unit})"
ranked = data.sort_values("Value", ascending=False, na_position="last").copy()
ranked.insert(0, "Rank", ranked["Value"].rank(ascending=False, method="min").astype("Int64"))
ranked = ranked.rename(columns={"Value": value_label, "Period": "Observation", "Level": "Index level" if metric == "Equities" else "Level (%)"})
level_label = "Index level" if metric == "Equities" else "Level (%)"
columns = ["Rank", "Country", value_label]
if is_change:
    columns += [level_label]
columns += ["Observation", "Baseline", "Series"]
if metric == "Equities":
    columns += ["Currency"]
columns += ["Status", "Source"]
st.dataframe(ranked[columns], hide_index=True, width="stretch", height=38 + 35 * 19,
             column_config={value_label: st.column_config.NumberColumn(format="%.2f"),
                            level_label: st.column_config.NumberColumn(format="%.2f"),
                            "Source": st.column_config.LinkColumn(display_text="Source"),
                            "Rank": st.column_config.NumberColumn(help="Descending numeric rank. A higher yield or inflation rank does not imply a stronger economy.")})

selected = st.selectbox("Country history", [c.name for c in COUNTRIES], index=18)
country = next(c for c in COUNTRIES if c.name == selected)
history = clean(series.get(country.iso, pd.Series(dtype=float)))
if metric == "Equities":
    history = history.loc[history.index < today]
else:
    history = history.loc[history.index.to_period(frequency) < today.to_period(frequency)]
if not history.empty:
    plot_history = history.copy()
    if metric != "Equities":
        plot_history.index = plot_history.index.to_period(frequency).to_timestamp()
        plot_history = plot_history.asfreq("MS" if frequency == "M" else "YS")
    line = go.Figure(go.Scatter(x=plot_history.index, y=plot_history, mode="lines", line=dict(color="#395b78", width=2), connectgaps=False))
    line.update_layout(height=250, margin=dict(l=0, r=20, t=10, b=0), template="plotly_white",
                       yaxis_title=country.index if metric == "Equities" else metric + " (%)",
                       xaxis_title=None, showlegend=False)
    st.plotly_chart(line, width="stretch", config={"displayModeBar": False})
    st.caption(f"{selected} · {'index level in ' + country.currency if metric == 'Equities' else 'monthly average' if metric == '10Y yields' else 'annual observations'} · history through {history.index[-1]:%Y-%m-%d}")
else:
    st.caption(f"No {metric.lower()} history available for {selected}.")

with st.expander("Sources and definitions"):
    st.markdown("G20's 19 individual countries. The EU and African Union are regional members and are not painted as countries. Other countries remain gray.")
    st.markdown("**Equities:** Yahoo Finance national headline indices in local currency, using raw published index closes through the prior UTC date. Calendar windows use the last available observation on or before the anchor; YTD starts at prior year-end. Quotes or baselines over seven days old are excluded. DAX and Ibovespa include dividends; most other headline indices are price indices. Local inflation and FX can materially alter the economic meaning of returns.")
    st.markdown("**Yields:** [OECD long-term interest rates](https://www.oecd.org/en/data/indicators/long-term-interest-rates.html), delivered by FRED. Comparable 10-year monthly averages for 11 countries, without ETF or policy-rate substitutes. Changes are basis points between exact months. Observations more than four months behind are excluded. Falling yields can reflect disinflation or growth stress; rising yields can reflect stronger growth, inflation or fiscal risk.")
    st.markdown("**Economy:** [World Bank GDP growth](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG), [unemployment](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS) and [CPI inflation](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG). Annual data; unemployment is an ILO modeled estimate. GDP is real annual growth, unemployment is a labor-force percentage, and CPI inflation is annual price growth. Changes are percentage points versus the exact preceding year. Data more than two years behind are excluded. Current-year observations are excluded; historical data may be revised.")
    st.markdown("**Alignment:** Comparable period uses the latest completed month/year shared by at least 80% of fresh reporting countries. Latest per country permits different observation dates, shown individually. Gray never means zero. Color scales use actual values, with changes centered on zero. Rank is descending numeric value, not an economic score.")
    if errors:
        st.dataframe(pd.DataFrame(errors.items(), columns=["Provider / series", "Load status"]), hide_index=True)
render_footer()
