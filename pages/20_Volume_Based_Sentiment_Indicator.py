def build_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    ma_period: int,
    high_z: float,
    low_z: float,
    show_markers: bool,
    show_last_price: bool,
    vol_opacity: float,
) -> go.Figure:
    rangebreaks = build_rangebreaks(df.index)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.69, 0.17, 0.14],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]],
    )

    stress_windows = df[df["Vol_Z"] >= high_z].copy()

    for idx in stress_windows.index:
        fig.add_vrect(
            x0=idx - pd.Timedelta(hours=12),
            x1=idx + pd.Timedelta(hours=12),
            fillcolor="rgba(239, 83, 80, 0.06)",
            line_width=0,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLORS["up"],
            increasing_fillcolor=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            decreasing_fillcolor=COLORS["down"],
            name=symbol,
            showlegend=False,
            whiskerwidth=0.5,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if show_markers:
        highs_sparse = sparse_signal_points(df, "Is_High", gap_days=18)
        lows_sparse = sparse_signal_points(df, "Is_Low", gap_days=18)

        if not highs_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=highs_sparse.index,
                    y=highs_sparse["Low"] * 0.992,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color="rgba(239,83,80,0.92)",
                        line=dict(width=1, color="white"),
                    ),
                    name="Stress signal",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<br>"
                        "Percentile: %{customdata[2]:.0f}<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [highs_sparse["Close"], highs_sparse["Vol_Z"], highs_sparse["Vol_PctRank"]]
                    ),
                ),
                row=1,
                col=1,
            )

        if not lows_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=lows_sparse.index,
                    y=lows_sparse["High"] * 1.008,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=7,
                        color="rgba(251,140,0,0.85)",
                        line=dict(width=1, color="white"),
                    ),
                    name="Quiet signal",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<br>"
                        "Percentile: %{customdata[2]:.0f}<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [lows_sparse["Close"], lows_sparse["Vol_Z"], lows_sparse["Vol_PctRank"]]
                    ),
                ),
                row=1,
                col=1,
            )

    if show_last_price and len(df) > 0:
        last_close = float(df["Close"].iloc[-1])
        fig.add_hline(
            y=last_close,
            line_width=1,
            line_dash="dot",
            line_color=COLORS["last"],
            row=1,
            col=1,
            annotation_text=f"Last {last_close:,.2f}",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["last"], size=11),
        )

    bar_colors = np.where(
        df["Vol_Z"] >= high_z,
        COLORS["vol_stress"],
        np.where(
            df["Vol_Z"] <= low_z,
            COLORS["vol_quiet"],
            f"rgba(41, 98, 255, {vol_opacity})",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            marker_color=bar_colors,
            name="Volume",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Vol_MA"],
            mode="lines",
            line=dict(color=COLORS["vol_ma"], width=2.0),
            name=f"{ma_period}D volume average",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume MA: %{y:,.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Vol_Z"],
            mode="lines",
            line=dict(color=COLORS["zscore"], width=1.8),
            name="Volume Z-score",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume Z: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=high_z,
        line=dict(color=COLORS["stress_line"], width=1.15, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0,
        line=dict(color=COLORS["zero_line"], width=1.0, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=low_z,
        line=dict(color=COLORS["quiet_line"], width=1.15, dash="dot"),
        row=3,
        col=1,
    )

    for r in [1, 2, 3]:
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(120,120,120,0.12)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            fixedrange=False,
            automargin=True,
            row=r,
            col=1,
        )

    fig.update_xaxes(
        type="date",
        showgrid=True,
        gridcolor="rgba(120,120,120,0.10)",
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        rangebreaks=rangebreaks,
        tickformat="%b %Y",
    )

    fig.update_yaxes(title_text="Price", nticks=10, row=1, col=1)
    fig.update_yaxes(title_text="Volume", nticks=5, row=2, col=1)
    fig.update_yaxes(title_text="Z", nticks=5, row=3, col=1)

    fig.update_layout(
        height=900,
        title=None,
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        hovermode="x unified",
        margin=dict(l=42, r=18, t=28, b=10),
        font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.88)",
            borderwidth=0,
            font=dict(size=11, color="#4b5563"),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            traceorder="normal",
        ),
        bargap=0.08,
    )

    return fig
