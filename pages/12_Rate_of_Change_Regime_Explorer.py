import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, render_footer, render_page_header, render_status_line
from plotly.subplots import make_subplots

from adfm_core.data_integrity import DataIntegrityPolicy, build_data_quality_report
from adfm_core.market_data import fetch_daily_ohlcv
from adfm_core.rate_of_change import (
    add_trading_session_axis,
    compute_features,
    make_date_ticks,
    padded_range,
)
TITLE = "Rate of Change Regime Explorer"
SUBTITLE = "Track price trend, rate of change, and acceleration through a clean regime lens."

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


TIMEFRAME_MAP = {
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
    "25Y": "25y",
    "Max": "max",
}

ROC_PERIODS = {
    "10D": 10,
    "20D": 20,
    "63D": 63,
    "126D": 126,
    "252D": 252,
}

PASTEL_GREEN = PASTEL["sage"]
PASTEL_RED = PASTEL["rose"]
PASTEL_GREY = "#8b949e"

SMA_COLORS = {
    "SMA_21": PASTEL["blue"],
    "SMA_50": PASTEL["coral"],
    "SMA_100": PASTEL["lavender"],
    "SMA_200": PASTEL["slate_blue"],
}


st.markdown(
    """
    <style>
        .tool-divider {
            margin-top: 1.25rem;
            margin-bottom: 1.25rem;
            border-top: 1px solid #000000;
        }

        .chart-wrap {
            margin-top: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(
    ticker: str,
    period: str,
) -> pd.DataFrame:
    """Load one completed daily OHLCV history using the shared data contract."""
    symbol = ticker.strip().upper()
    frames, dropped = fetch_daily_ohlcv((symbol,), period)
    if symbol in frames:
        return frames[symbol]
    reason = "No valid OHLCV data returned"
    if not dropped.empty:
        reason = str(dropped.iloc[0]["Reason"])
    raise RuntimeError(f"Failed to fetch {symbol}: {reason}")


with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Rate-of-change regime explorer for identifying when price momentum is accelerating, fading, or transitioning.

        **What this tab shows**

        - Price trend with 21, 50, 100, and 200-day simple moving averages.
        - Rolling rate of change over the selected lookback window.
        - Second-derivative acceleration to flag turning points in momentum.
        - Positive and negative inflection markers when acceleration crosses the zero line.

        **Data source**

        - Yahoo Finance daily OHLCV history via `yfinance`.

        **Chart construction**

        - The x-axis is plotted by trading session, then labeled with calendar dates.
        - Weekends, holidays, and missing sessions are compressed automatically.
        - Price, ROC, and acceleration therefore share the same observation-by-observation alignment.
        """
    )

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Display Controls")
    show_inflections = st.toggle("Show inflection markers", value=True)


render_page_header(
    PageHeader(
        title=TITLE,
        description=SUBTITLE,
        eyebrow="ADFM Equity Leadership",
    )
)


input_col_1, input_col_2, input_col_3, input_col_4 = st.columns([1.55, 1.0, 1.0, 1.0])

with input_col_1:
    ticker = st.text_input("Ticker symbol", value="^SPX").upper().strip()

with input_col_2:
    timeframe = st.selectbox("Analysis window", list(TIMEFRAME_MAP.keys()), index=3)

with input_col_3:
    roc_label = st.selectbox("ROC period", list(ROC_PERIODS.keys()), index=2)

with input_col_4:
    chart_view = st.selectbox("Chart view", ["Candlestick", "Line"], index=0)


if not ticker:
    st.warning("Enter a ticker symbol.")
    st.stop()


try:
    df = fetch_history(ticker, TIMEFRAME_MAP[timeframe])
except Exception as e:
    st.error(str(e))
    st.stop()

quality = build_data_quality_report(
    {ticker.strip().upper(): df},
    ticker.strip().upper(),
    policy=DataIntegrityPolicy(min_valid_sessions=1, max_stale_sessions=0),
)
render_status_line(
    data_through=quality.data_through.date().isoformat() if quality.data_through is not None else "unavailable",
    source="Yahoo Finance",
    data_quality="complete daily sessions" if quality.benchmark_ready else quality.reason_for(ticker.strip().upper()),
)


if len(df) < 60:
    st.warning("The selected window has fewer than 60 observations, so derivative readings may be unstable.")


feat = compute_features(df, ROC_PERIODS[roc_label])
feat = feat.dropna(subset=["ROC", "Second_Derivative"], how="any")

if feat.empty:
    st.warning("Data became empty after indicator calculations. Try a longer analysis window.")
    st.stop()


feat = add_trading_session_axis(feat)

x_plot = feat["Session"]
custom_dates = feat["Date_Label"].to_numpy().reshape(-1, 1)

price_series_for_range = []

for col in ["Open", "High", "Low", "Close", "SMA_21", "SMA_50", "SMA_100", "SMA_200"]:
    if col in feat.columns:
        price_series_for_range.append(feat[col])

price_y_range = padded_range(price_series_for_range, pad_pct=0.035)
roc_y_range = padded_range([feat["ROC"]], pad_pct=0.12, include_zero=True)
acceleration_y_range = padded_range([feat["Second_Derivative"]], pad_pct=0.15, include_zero=True)

x_range = [-0.5, len(feat) - 0.5]
tickvals, ticktext = make_date_ticks(feat.index, feat["Session"], max_ticks=11)


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    row_heights=[0.52, 0.24, 0.24],
)


if chart_view == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=x_plot,
            open=feat["Open"],
            high=feat["High"],
            low=feat["Low"],
            close=feat["Close"],
            name="Price",
            customdata=custom_dates,
            increasing_line_color=PASTEL_GREEN,
            decreasing_line_color=PASTEL_RED,
            increasing_fillcolor="rgba(82, 183, 136, 0.60)",
            decreasing_fillcolor="rgba(232, 93, 93, 0.60)",
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
else:
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=feat["Close"],
            mode="lines",
            name="Price",
            customdata=custom_dates,
            line=dict(color="#111827", width=2.2),
            hovertemplate="%{customdata[0]}<br>Close: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


for ma in ["SMA_21", "SMA_50", "SMA_100", "SMA_200"]:
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=feat[ma],
            mode="lines",
            name=ma.replace("_", " "),
            customdata=custom_dates,
            line=dict(color=SMA_COLORS[ma], width=1.6),
            hovertemplate="%{customdata[0]}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


fig.add_trace(
    go.Scatter(
        x=x_plot,
        y=feat["ROC"],
        mode="lines",
        name=f"ROC {roc_label}",
        customdata=custom_dates,
        line=dict(color="#4c78a8", width=2.0),
        hovertemplate="%{customdata[0]}<br>ROC: %{y:.2%}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color=PASTEL_GREY,
    row=2,
    col=1,
)


bar_colors = np.where(feat["Second_Derivative"] >= 0, PASTEL_GREEN, PASTEL_RED)

fig.add_trace(
    go.Bar(
        x=x_plot,
        y=feat["Second_Derivative"],
        marker_color=bar_colors,
        name="Acceleration",
        customdata=custom_dates,
        width=0.85,
        hovertemplate="%{customdata[0]}<br>Acceleration: %{y:.2%}<extra></extra>",
    ),
    row=3,
    col=1,
)

fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color=PASTEL_GREY,
    row=3,
    col=1,
)


if show_inflections:
    pos_marks = feat[feat["Pos_Inflect"]]
    neg_marks = feat[feat["Neg_Inflect"]]

    fig.add_trace(
        go.Scatter(
            x=pos_marks["Session"],
            y=pos_marks["Second_Derivative"],
            mode="markers",
            name="Positive inflection",
            customdata=pos_marks["Date_Label"].to_numpy().reshape(-1, 1),
            marker=dict(
                color=PASTEL_GREEN,
                size=8,
                symbol="triangle-up",
                line=dict(width=0.8, color="#ffffff"),
            ),
            hovertemplate="%{customdata[0]}<br>Positive acceleration inflection<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=neg_marks["Session"],
            y=neg_marks["Second_Derivative"],
            mode="markers",
            name="Negative inflection",
            customdata=neg_marks["Date_Label"].to_numpy().reshape(-1, 1),
            marker=dict(
                color=PASTEL_RED,
                size=8,
                symbol="triangle-down",
                line=dict(width=0.8, color="#ffffff"),
            ),
            hovertemplate="%{customdata[0]}<br>Negative acceleration inflection<extra></extra>",
        ),
        row=3,
        col=1,
    )


fig.update_layout(
    height=900,
    autosize=True,
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=42, r=18, t=64, b=72),
    title=dict(
        text=f"<b>{ticker}</b> · Price, ROC and Acceleration",
        x=0.015,
        xanchor="left",
        y=0.975,
        font=dict(size=20, color="#111827", family="Arial, sans-serif"),
    ),
    font=dict(color="#334155", family="Arial, sans-serif"),
    hovermode="x unified",
    bargap=0,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.105,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0)",
    ),
    xaxis_rangeslider_visible=False,
)


fig.update_xaxes(
    range=x_range,
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    constrain="domain",
    automargin=False,
    showgrid=True,
    gridcolor="rgba(226, 232, 240, 0.55)",
    showline=True,
    linewidth=1,
    linecolor="#d7dde8",
    rangeslider_visible=False,
    fixedrange=False,
)


fig.update_yaxes(
    range=price_y_range,
    title_text="Price",
    row=1,
    col=1,
    fixedrange=False,
    automargin=True,
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)

fig.update_yaxes(
    range=roc_y_range,
    title_text=f"ROC {roc_label}",
    row=2,
    col=1,
    fixedrange=False,
    automargin=True,
    tickformat=".1%",
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)

fig.update_yaxes(
    range=acceleration_y_range,
    title_text="Acceleration",
    row=3,
    col=1,
    fixedrange=False,
    automargin=True,
    tickformat=".2%",
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)


st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
    },
)
st.markdown("</div>", unsafe_allow_html=True)

render_footer()
