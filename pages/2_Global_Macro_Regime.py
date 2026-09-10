from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.global_macro import (
    COUNTRIES,
    INDICATORS,
    MARKET_HORIZONS,
    clean,
    comparison_period,
    country_rows,
    equity_matrix,
    load_economics,
    load_equities,
    load_yields,
)
from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header


st.set_page_config(page_title="Global Macro Regime", layout="wide", initial_sidebar_state="collapsed")
inject_explorer_style(max_width_px=1700)
render_page_header(
    PageHeader(
        "Global Macro Regime",
        "G20 · Fast market leadership with slower official macro context",
    )
)

now_ny = pd.Timestamp.now(tz="America/New_York")
today = now_ny.tz_localize(None)
metric = st.segmented_control(
    "Market / economy",
    ["Equities", "10Y yields", *INDICATORS],
    default="Equities",
    key="gm_metric",
    label_visibility="collapsed",
) or "Equities"


def tape_label(one_month, three_month):
    if not np.isfinite(one_month) or not np.isfinite(three_month):
        return "Unavailable"
    if one_month > 0 and three_month > 0:
        return "Leading"
    if one_month > 0 and three_month <= 0:
        return "Rebounding"
    if one_month <= 0 and three_month > 0:
        return "Fading"
    return "Struggling"


def render_world_map(data, metric_name, measure, unit, is_change, key):
    available = data.loc[data["Value"].notna()].copy()

    fig = go.Figure(
        go.Choropleth(
            locations=data["ISO"],
            z=[0] * len(data),
            locationmode="ISO-3",
            colorscale=[[0, "#44484e"], [1, "#44484e"]],
            showscale=False,
            text=data["Country"],
            customdata=data[["Status", "Observation"]].fillna("").to_numpy(),
            marker_line_color="#91969d",
            marker_line_width=0.55,
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}"
                "<br>Observation: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if not available.empty:
        if metric_name in ("10Y yields", "Inflation"):
            scale = [[0, "#598ace"], [0.5, "#e0e4e8"], [1, "#df9653"]]
        elif metric_name == "Unemployment":
            scale = [[0, "#72b3a0"], [0.5, "#e0e4e8"], [1, "#c56d79"]]
        else:
            scale = [[0, "#c56d79"], [0.5, "#e0e4e8"], [1, "#72b3a0"]]

        bounds = {}
        if is_change or metric_name == "GDP growth":
            limit = max(float(available["Value"].abs().max()), 0.01)
            bounds = dict(zmin=-limit, zmax=limit, zmid=0)

        fig.add_trace(
            go.Choropleth(
                locations=available["ISO"],
                z=available["Value"],
                locationmode="ISO-3",
                text=available["Country"],
                colorscale=scale,
                **bounds,
                customdata=available[["Series", "Observation", "Baseline"]]
                .fillna("")
                .to_numpy(),
                marker_line_color="#bdc3cb",
                marker_line_width=0.65,
                colorbar=dict(
                    title=dict(text=unit),
                    thickness=10,
                    len=0.65,
                    tickfont=dict(color="#e1e4e8"),
                ),
                hovertemplate=(
                    f"<b>%{{text}}</b><br>%{{customdata[0]}}"
                    f"<br>{measure}: %{{z:.2f}} {unit}"
                    "<br>Observation: %{customdata[1]}"
                    "<br>Baseline: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showland=True,
        landcolor="#30343a",
        showcountries=True,
        countrycolor="#777d85",
        countrywidth=0.45,
        showocean=True,
        oceancolor="#0b0e12",
        bgcolor="#0b0e12",
        lataxis_range=[-58, 85],
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=6, b=4),
        paper_bgcolor="#0b0e12",
        font=dict(color="#e1e4e8"),
        dragmode=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key=key,
    )


def render_country_history(series, selected_metric, frequency):
    selected = st.selectbox(
        "Country history",
        [c.name for c in COUNTRIES],
        index=18,
        key=f"gm_history_{selected_metric}",
    )
    country = next(c for c in COUNTRIES if c.name == selected)
    history = clean(series.get(country.iso, pd.Series(dtype=float)))

    if selected_metric != "Equities":
        history = history.loc[
            history.index.to_period(frequency) < pd.Timestamp(today).to_period(frequency)
        ]

    if history.empty:
        st.caption(f"No {selected_metric.lower()} history available for {selected}.")
        return

    plot_history = history.copy()
    if selected_metric != "Equities":
        plot_history.index = plot_history.index.to_period(frequency).to_timestamp()
        plot_history = plot_history.asfreq("MS" if frequency == "M" else "YS")

    line = go.Figure(
        go.Scatter(
            x=plot_history.index,
            y=plot_history,
            mode="lines",
            line=dict(color="#395b78", width=2),
            connectgaps=False,
        )
    )
    line.update_layout(
        height=245,
        margin=dict(l=0, r=20, t=8, b=0),
        template="plotly_white",
        yaxis_title=country.index if selected_metric == "Equities" else selected_metric + " (%)",
        xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(line, width="stretch", config={"displayModeBar": False})
    history_type = (
        "index level in " + country.currency
        if selected_metric == "Equities"
        else "monthly average"
        if selected_metric == "10Y yields"
        else "annual observations"
    )
    st.caption(
        f"{selected} · {history_type} · history through {history.index[-1]:%Y-%m-%d}"
    )


errors = {}

if metric == "Equities":
    a, b, c = st.columns([1, 1, 2])
    with a:
        horizon = st.selectbox(
            "Ranking / map window",
            list(MARKET_HORIZONS),
            index=2,
            key="gm_equity_horizon",
        )
    with b:
        basis = st.selectbox(
            "Return basis",
            ["Local currency", "USD-adjusted"],
            index=0,
            key="gm_equity_basis",
            help=(
                "USD-adjusted combines the local index return with the local currency's "
                "move versus USD. Countries without a usable FX series remain unavailable."
            ),
        )
    with c:
        st.caption(
            "Yahoo Finance daily history with the latest available 5-minute bar overlaid. "
            "Market data can be exchange-delayed. Cache refreshes every 2 minutes."
        )

    with st.spinner("Loading global equity and FX tape…"):
        equities, fx, errors = load_equities()
    matrix = equity_matrix(equities, fx, today)

    prefix = "Local " if basis == "Local currency" else "USD "
    selected_col = f"{prefix}{horizon}"
    one_month_col = f"{prefix}1M"
    three_month_col = f"{prefix}3M"
    matrix["Tape"] = [
        tape_label(x, y)
        for x, y in zip(matrix[one_month_col], matrix[three_month_col])
    ]

    valid = matrix.loc[matrix[selected_col].notna()].copy()
    if valid.empty:
        st.warning("No fresh equity observations are available for this view.")
    else:
        leader = valid.loc[valid[selected_col].idxmax()]
        laggard = valid.loc[valid[selected_col].idxmin()]
        breadth = float((valid[selected_col] > 0).mean() * 100)
        rebounds = int((matrix["Tape"] == "Rebounding").sum())
        struggling = int((matrix["Tape"] == "Struggling").sum())
        st.caption(
            f"{horizon} {basis.lower()} breadth: {breadth:.0f}% positive · "
            f"leader: {leader['Country']} {leader[selected_col]:+.2f}% · "
            f"laggard: {laggard['Country']} {laggard[selected_col]:+.2f}% · "
            f"rebounding: {rebounds} · struggling: {struggling}"
        )

    ranked = matrix.copy()
    ranked.insert(
        0,
        "Rank",
        ranked[selected_col].rank(ascending=False, method="min").astype("Int64"),
    )

    if basis == "Local currency":
        performance_columns = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y"]
    else:
        performance_columns = [f"USD {h}" for h in MARKET_HORIZONS]

    table_columns = [
        "Rank",
        "Country",
        "Tape",
        "Level",
        *performance_columns,
        "FX 1M",
        "Observation",
        "Index",
        "Currency",
        "Status",
    ]
    number_config = {
        "Level": st.column_config.NumberColumn(format="%.2f"),
        "FX 1M": st.column_config.NumberColumn(
            format="%.2f",
            help="Local currency return versus USD over one month. Positive = local currency strengthened.",
        ),
        "Rank": st.column_config.NumberColumn(
            help=f"Descending rank by {horizon} {basis.lower()} performance."
        ),
    }
    for col in performance_columns:
        number_config[col] = st.column_config.NumberColumn(format="%.2f")

    st.dataframe(
        ranked[table_columns],
        hide_index=True,
        width="stretch",
        height=38 + 35 * 19,
        column_config=number_config,
    )

    map_data = matrix[
        ["Country", "ISO", "Observation", "Status", "Index", selected_col]
    ].rename(columns={selected_col: "Value", "Index": "Series"})
    map_data["Baseline"] = ""
    st.caption(
        f"World map · {horizon} {basis.lower()} index performance · "
        "green = rising, red = falling, gray = unavailable"
    )
    render_world_map(
        map_data,
        "Equities",
        f"{horizon} return",
        "%",
        True,
        "gm_equity_world_map",
    )

    render_country_history(equities, "Equities", "D")

else:
    a, b, c = st.columns([1, 1, 2])
    view = "Change" if metric == "10Y yields" else "Level"
    horizon = "1M"
    with a:
        view = st.selectbox(
            "Map measure",
            ["Level", "Change"],
            index=1 if metric == "10Y yields" else 0,
            key=f"gm_view_{metric}",
        )
    with b:
        if metric == "10Y yields" and view == "Change":
            horizon = st.selectbox(
                "Change window",
                ["1M", "3M", "6M", "1Y"],
                index=0,
                key="gm_yield_horizon",
            )
        elif metric != "10Y yields":
            st.caption("Annual official data · latest completed observation only")
    with c:
        mode = st.selectbox(
            "Observation alignment",
            ["Latest per country", "Comparable period"],
            index=0,
            help=(
                "Latest per country maximizes freshness and prints each observation date. "
                "Comparable period uses the latest completed period covered by at least 80% "
                "of fresh reporting countries."
            ),
            key=f"gm_alignment_{metric}",
        )

    with st.spinner(f"Loading {metric.lower()}…"):
        if metric == "10Y yields":
            series, errors = load_yields()
        else:
            series, errors = load_economics(metric)

    frequency = "M" if metric == "10Y yields" else "Y"
    period = (
        comparison_period(series, today, frequency)
        if mode == "Comparable period"
        else None
    )
    data = country_rows(series, today, metric, view, horizon, period)
    if mode == "Comparable period" and period is None:
        data["Value"] = np.nan
        data["Status"] = "No comparable period"

    is_change = view == "Change"
    unit = (
        "bp"
        if metric == "10Y yields" and is_change
        else "pp"
        if metric in INDICATORS and is_change
        else "%"
    )
    measure = (
        f"{horizon} yield change"
        if metric == "10Y yields" and is_change
        else "Annual change"
        if is_change
        else metric
    )
    available = data.loc[data["Value"].notna()].copy()

    if metric == "10Y yields":
        st.caption(
            f"{len(available)}/19 countries observed · official 10Y government-bond "
            "monthly averages via OECD/FRED · use as macro context rather than a live yield monitor. "
            "Observation dates are shown explicitly."
        )
    else:
        extra = " · ILO modeled estimates" if metric == "Unemployment" else ""
        st.caption(
            f"{len(available)}/19 countries observed · World Bank annual {metric.lower()}"
            f"{extra} · observation years are shown explicitly."
        )

    if available.empty:
        st.warning(
            "No observations meet this view's coverage and freshness requirements. "
            "Unavailable countries remain gray."
        )

    value_label = f"{measure} ({unit})"
    ranked = data.sort_values("Value", ascending=False, na_position="last").copy()
    ranked.insert(
        0,
        "Rank",
        ranked["Value"].rank(ascending=False, method="min").astype("Int64"),
    )
    ranked = ranked.rename(
        columns={
            "Value": value_label,
            "Period": "Observation",
            "Level": "Level (%)",
        }
    )
    columns = ["Rank", "Country", value_label]
    if is_change:
        columns += ["Level (%)"]
    columns += ["Observation", "Baseline", "Series", "Status", "Source"]

    st.dataframe(
        ranked[columns],
        hide_index=True,
        width="stretch",
        height=38 + 35 * 19,
        column_config={
            value_label: st.column_config.NumberColumn(format="%.2f"),
            "Level (%)": st.column_config.NumberColumn(format="%.2f"),
            "Source": st.column_config.LinkColumn(display_text="Source"),
            "Rank": st.column_config.NumberColumn(
                help=(
                    "Descending numeric rank. A higher yield, inflation, or unemployment "
                    "rank does not by itself imply a stronger economy."
                )
            ),
        },
    )

    map_data = data.rename(columns={"Period": "Observation"}).copy()
    st.caption(
        f"World map · {measure.lower()} · gray = unavailable or outside the 19 G20 countries"
    )
    render_world_map(
        map_data,
        metric,
        measure,
        unit,
        is_change,
        f"gm_world_map_{metric}_{view}",
    )

    render_country_history(series, metric, frequency)


with st.expander("Sources, freshness and definitions"):
    st.markdown(
        "**Universe:** the G20's 19 individual countries. The EU and African Union are "
        "regional members and are not painted as countries. Other countries remain gray."
    )
    st.markdown(
        "**Equities:** Yahoo Finance national headline indices in local currency. Two years "
        "of daily history are overlaid with the latest available 5-minute observation, then "
        "cached for two minutes. Current-session observations are used when Yahoo supplies them; "
        "otherwise the latest close is used. Returns are point-to-point from the last valid "
        "observation at or before the requested baseline. YTD uses prior year-end. DAX and "
        "Ibovespa include dividends; most other headline indices are price indices."
    )
    st.markdown(
        "**USD-adjusted equities:** local index return compounded with the local currency's "
        "return versus USD. This makes large nominal equity gains caused by currency weakness "
        "easier to distinguish from genuine USD wealth creation. Missing FX remains missing."
    )
    st.markdown(
        "**Tape labels:** Leading = positive 1M and 3M; Rebounding = positive 1M but non-positive "
        "3M; Fading = non-positive 1M but positive 3M; Struggling = non-positive 1M and 3M. "
        "These are descriptive labels, not a composite score or recommendation."
    )
    st.markdown(
        "**10Y yields:** OECD long-term interest rates delivered by FRED. These refer to "
        "government bonds maturing around ten years and are monthly averages of market yields, "
        "not live quotes. Changes are basis points between exact months. Countries without a "
        "comparable series stay gray."
    )
    st.markdown(
        "**Economy:** World Bank real GDP growth, unemployment and CPI inflation. These are "
        "annual official series and therefore intentionally slower than the market layer. "
        "Unemployment is an ILO modeled estimate. Changes are percentage points versus the exact "
        "preceding year. Current-year observations are excluded and historical data may be revised."
    )
    st.markdown(
        "**Alignment:** Latest per country maximizes freshness and displays each observation date. "
        "Comparable period requires a period shared by at least 80% of fresh reporting countries. "
        "Gray never means zero."
    )
    if errors:
        st.dataframe(
            pd.DataFrame(errors.items(), columns=["Provider / series", "Load status"]),
            hide_index=True,
        )

render_footer()
