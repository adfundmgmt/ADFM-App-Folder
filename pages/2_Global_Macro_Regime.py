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
from adfm_core.palette import PASTEL, PASTEL_DIVERGING_SCALE, PASTEL_RATES_SCALE
from adfm_core.ui import PageHeader, inject_explorer_style, render_footer, render_page_header


st.set_page_config(
    page_title="Global Macro Regime",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_explorer_style(max_width_px=1700)
render_page_header(
    PageHeader(
        "Global Macro Regime",
        "G20 · Current market leadership and higher-frequency macro momentum",
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


def _map_scale(metric_name, is_change):
    if is_change:
        if metric_name in ("10Y yields", "Inflation", "Unemployment"):
            return PASTEL_RATES_SCALE
        return PASTEL_DIVERGING_SCALE
    if metric_name == "GDP growth":
        return PASTEL_DIVERGING_SCALE
    if metric_name == "10Y yields":
        return [
            [0.0, PASTEL["blue"]],
            [0.5, "#FBFBF8"],
            [1.0, PASTEL["coral"]],
        ]
    if metric_name in ("Inflation", "Unemployment"):
        return [
            [0.0, PASTEL["sage"]],
            [0.5, "#FBFBF8"],
            [1.0, PASTEL["rose"]],
        ]
    return PASTEL_DIVERGING_SCALE


def render_world_map(data, metric_name, measure, unit, is_change, key):
    available = data.loc[data["Value"].notna()].copy()

    fig = go.Figure(
        go.Choropleth(
            locations=data["ISO"],
            z=[0] * len(data),
            locationmode="ISO-3",
            colorscale=[[0, "#D9DDE3"], [1, "#D9DDE3"]],
            showscale=False,
            text=data["Country"],
            customdata=data[["Status", "Observation"]].fillna("").to_numpy(),
            marker_line_color="#FFFFFF",
            marker_line_width=0.8,
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}"
                "<br>Observation: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if not available.empty:
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
                colorscale=_map_scale(metric_name, is_change),
                **bounds,
                customdata=available[["Series", "Observation", "Baseline"]]
                .fillna("")
                .to_numpy(),
                marker_line_color="#FFFFFF",
                marker_line_width=0.8,
                colorbar=dict(
                    title=dict(text=unit, font=dict(color="#4B5563")),
                    thickness=10,
                    len=0.62,
                    outlinewidth=0,
                    tickfont=dict(color="#4B5563"),
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
        landcolor="#F3F4F6",
        showcountries=True,
        countrycolor="#FFFFFF",
        countrywidth=0.6,
        showocean=True,
        oceancolor="#FFFFFF",
        bgcolor="#FFFFFF",
        lataxis_range=[-58, 85],
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
        dragmode=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key=key,
    )


def render_country_history(series, selected_metric):
    selected = st.selectbox(
        "Country history",
        [c.name for c in COUNTRIES],
        index=18,
        key=f"gm_history_{selected_metric}",
    )
    country = next(c for c in COUNTRIES if c.name == selected)
    history = clean(series.get(country.iso, pd.Series(dtype=float)))

    if history.empty:
        st.caption(f"No {selected_metric.lower()} history available for {selected}.")
        return

    line = go.Figure(
        go.Scatter(
            x=history.index,
            y=history,
            mode="lines",
            line=dict(color=PASTEL["blue"], width=2),
            connectgaps=False,
        )
    )
    if selected_metric == "Equities":
        yaxis_title = country.index
        history_type = f"index level in {country.currency}"
    elif selected_metric == "10Y yields":
        yaxis_title = "10Y yield (%)"
        history_type = "monthly average"
    elif selected_metric == "GDP growth":
        yaxis_title = "q/q annualized (%)"
        history_type = "quarterly real GDP growth, annualized"
    elif selected_metric == "Unemployment":
        yaxis_title = "unemployment (%)"
        history_type = "latest seasonally adjusted monthly/quarterly rate"
    else:
        yaxis_title = "inflation YoY (%)"
        history_type = "monthly CPI year-over-year"

    line.update_layout(
        height=245,
        margin=dict(l=0, r=20, t=8, b=0),
        template="plotly_white",
        yaxis_title=yaxis_title,
        xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(line, width="stretch", config={"displayModeBar": False})
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
                "move versus USD."
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

    ranked = matrix.copy()
    ranked.insert(
        0,
        "Rank",
        ranked[selected_col].rank(ascending=False, method="min").astype("Int64"),
    )
    performance_columns = (
        ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y"]
        if basis == "Local currency"
        else [f"USD {h}" for h in MARKET_HORIZONS]
    )
    table_columns = [
        "Rank", "Country", "Tape", "Level", *performance_columns,
        "FX 1M", "Observation", "Index", "Currency", "Status",
    ]
    number_config = {
        "Level": st.column_config.NumberColumn(format="%.2f"),
        "FX 1M": st.column_config.NumberColumn(
            format="%.2f",
            help="Local currency return versus USD over one month. Positive = stronger local currency.",
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
    render_country_history(equities, "Equities")

else:
    a, b, c = st.columns([1, 1, 2])
    view = "Change" if metric == "10Y yields" else "Level"
    horizon = "1M"
    mode = "Latest per country"

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
        elif metric == "GDP growth":
            st.caption("Latest quarterly real GDP growth · q/q rate compounded to an annual rate")
        elif metric == "Unemployment":
            st.caption("Latest seasonally adjusted monthly rate · quarterly fallback where needed")
        elif metric == "Inflation":
            st.caption("Latest monthly CPI inflation · year-over-year rate")
    with c:
        if metric == "10Y yields":
            mode = st.selectbox(
                "Observation alignment",
                ["Latest per country", "Comparable period"],
                index=0,
                help=(
                    "Latest per country maximizes freshness. Comparable period uses the latest "
                    "completed month covered by at least 80% of fresh reporters."
                ),
                key="gm_alignment_yields",
            )
        else:
            st.caption(
                "Higher-frequency OECD releases via FRED. Each country's actual observation "
                "date is shown; unavailable or stale series remain gray."
            )

    with st.spinner(f"Loading {metric.lower()}…"):
        if metric == "10Y yields":
            series, errors = load_yields()
        else:
            series, errors = load_economics(metric)

    period = (
        comparison_period(series, today, "M")
        if metric == "10Y yields" and mode == "Comparable period"
        else None
    )
    data = country_rows(series, today, metric, view, horizon, period)
    if metric == "10Y yields" and mode == "Comparable period" and period is None:
        data["Value"] = np.nan
        data["Status"] = "No comparable period"

    is_change = view == "Change"
    if metric == "10Y yields":
        unit = "bp" if is_change else "%"
        measure = f"{horizon} yield change" if is_change else "10Y yield"
    elif metric == "GDP growth":
        unit = "pp" if is_change else "%"
        measure = "Change vs prior quarter" if is_change else "GDP q/q annualized"
    elif metric == "Unemployment":
        unit = "pp" if is_change else "%"
        measure = "Change vs prior release" if is_change else "Unemployment"
    else:
        unit = "pp" if is_change else "%"
        measure = "Change vs prior release" if is_change else "Inflation YoY"

    available = data.loc[data["Value"].notna()].copy()
    if metric == "10Y yields":
        st.caption(
            f"{len(available)}/19 countries observed · official 10Y government-bond monthly "
            "averages via OECD/FRED · observation dates shown explicitly."
        )
    elif metric == "GDP growth":
        st.caption(
            f"{len(available)}/19 countries observed · latest seasonally adjusted quarterly "
            "real GDP growth, compounded to an annual rate."
        )
    elif metric == "Unemployment":
        st.caption(
            f"{len(available)}/19 countries observed · latest seasonally adjusted unemployment "
            "rate, monthly where available with quarterly fallback."
        )
    else:
        st.caption(
            f"{len(available)}/19 countries observed · latest monthly CPI year-over-year inflation."
        )

    if available.empty:
        st.warning(
            "No observations meet this view's coverage and freshness requirements. "
            "Unavailable countries remain gray."
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
                help="Descending numeric rank. Rank is descriptive, not an economic score."
            ),
        },
    )
    render_country_history(series, metric)


with st.expander("Sources, freshness and definitions"):
    st.markdown(
        "**Universe:** the G20's 19 individual countries. The EU and African Union are "
        "regional members and are not painted as countries. Other countries remain unhighlighted."
    )
    st.markdown(
        "**Equities:** Yahoo Finance national headline indices in local currency. Two years "
        "of daily history are overlaid with the latest available 5-minute observation and cached "
        "for two minutes. Current-session observations are used when available."
    )
    st.markdown(
        "**USD-adjusted equities:** local index return compounded with the local currency's "
        "return versus USD. Missing FX remains missing."
    )
    st.markdown(
        "**Tape labels:** Leading = positive 1M and 3M; Rebounding = positive 1M but non-positive "
        "3M; Fading = non-positive 1M but positive 3M; Struggling = non-positive 1M and 3M."
    )
    st.markdown(
        "**10Y yields:** OECD long-term interest rates delivered by FRED. They are monthly "
        "averages rather than live sovereign yields. Changes are basis points between exact months."
    )
    st.markdown(
        "**GDP:** OECD quarterly real GDP growth delivered by FRED. The latest seasonally adjusted "
        "q/q growth rate is compounded for four quarters: (1 + q/q)^4 - 1. This is a run-rate "
        "annualization, not a forecast for full-year GDP."
    )
    st.markdown(
        "**Unemployment:** latest OECD seasonally adjusted rate, monthly when available with a "
        "quarterly fallback. The level is not annualized because unemployment is a point-in-time rate."
    )
    st.markdown(
        "**Inflation:** latest OECD monthly CPI year-over-year rate. It is already expressed over a "
        "12-month interval, so no additional annualization is applied."
    )
    st.markdown(
        "**Freshness:** each country's observation date is displayed. GDP older than roughly two "
        "quarters and monthly labor/inflation data older than roughly five months are treated as stale "
        "and remain gray."
    )
    if errors:
        st.dataframe(
            pd.DataFrame(errors.items(), columns=["Provider / series", "Load status"]),
            hide_index=True,
        )

render_footer()
