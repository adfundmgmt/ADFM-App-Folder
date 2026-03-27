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

    BBG = {
        "bg": "#0b0f14",
        "panel": "#0b0f14",
        "grid": "rgba(255,255,255,0.08)",
        "text": "#E5E7EB",
        "subtle": "#9CA3AF",
        "up": "#00C176",
        "down": "#FF5B5B",
        "volume": "rgba(120, 144, 156, 0.55)",
        "volume_hi": "#F59E0B",
        "ma": "#4FC3F7",
        "z": "#D1D5DB",
        "stress": "#FF9F43",
        "quiet": "#60A5FA",
        "last": "#F3F4F6",
        "zero": "rgba(255,255,255,0.25)",
    }

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.74, 0.16, 0.10],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=BBG["up"],
            increasing_fillcolor=BBG["up"],
            decreasing_line_color=BBG["down"],
            decreasing_fillcolor=BBG["down"],
            whiskerwidth=0.35,
            name=symbol,
            showlegend=False,
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
        highs_sparse = sparse_signal_points(df, "Is_High", gap_days=20)
        lows_sparse = sparse_signal_points(df, "Is_Low", gap_days=20)

        if not highs_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=highs_sparse.index,
                    y=highs_sparse["High"] * 1.01,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=6,
                        color=BBG["stress"],
                        line=dict(width=0.5, color=BBG["bg"]),
                    ),
                    name="High Vol",
                    showlegend=True,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([highs_sparse["Close"], highs_sparse["Vol_Z"]]),
                ),
                row=1,
                col=1,
            )

        if not lows_sparse.empty:
            fig.add_trace(
                go.Scatter(
                    x=lows_sparse.index,
                    y=lows_sparse["Low"] * 0.99,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=6,
                        color=BBG["quiet"],
                        line=dict(width=0.5, color=BBG["bg"]),
                    ),
                    name="Low Vol",
                    showlegend=True,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        "Close: %{customdata[0]:.2f}<br>"
                        "Volume Z: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.column_stack([lows_sparse["Close"], lows_sparse["Vol_Z"]]),
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
            line_color=BBG["last"],
            row=1,
            col=1,
            annotation_text=f"{last_close:,.2f}",
            annotation_position="top right",
            annotation_font=dict(color=BBG["last"], size=11),
        )

    bar_colors = np.where(
        df["Vol_Z"] >= high_z,
        BBG["volume_hi"],
        np.where(
            df["Vol_Z"] <= low_z,
            "rgba(96, 165, 250, 0.80)",
            BBG["volume"],
        ),
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Vol_Display"],
            marker_color=bar_colors,
            name="Volume",
            showlegend=True,
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
            line=dict(color=BBG["ma"], width=1.6),
            name=f"{ma_period}D Vol MA",
            showlegend=True,
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
            line=dict(color=BBG["z"], width=1.5),
            name="Vol Z",
            showlegend=True,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume Z: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=high_z,
        line=dict(color=BBG["stress"], width=1, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0,
        line=dict(color=BBG["zero"], width=1, dash="dot"),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=low_z,
        line=dict(color=BBG["quiet"], width=1, dash="dot"),
        row=3,
        col=1,
    )

    for r in [1, 2, 3]:
        fig.update_yaxes(
            showgrid=True,
            gridcolor=BBG["grid"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            automargin=True,
            tickfont=dict(color=BBG["subtle"], size=11),
            title_font=dict(color=BBG["subtle"], size=11),
            row=r,
            col=1,
        )

    fig.update_xaxes(
        type="date",
        showgrid=True,
        gridcolor=BBG["grid"],
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        rangebreaks=rangebreaks,
        tickformat="%b '%y",
        tickfont=dict(color=BBG["subtle"], size=11),
    )

    fig.update_yaxes(title_text=f"{symbol} Price", nticks=10, row=1, col=1)
    fig.update_yaxes(title_text="Volume", nticks=4, row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", nticks=4, row=3, col=1)

    fig.update_layout(
        height=920,
        title=dict(
            text=f"{symbol} <span style='font-size:13px;color:{BBG['subtle']}'>{vol_label}</span>",
            x=0.01,
            y=0.985,
            xanchor="left",
            yanchor="top",
            font=dict(size=20, color=BBG["text"], family="Arial, sans-serif"),
        ),
        plot_bgcolor=BBG["panel"],
        paper_bgcolor=BBG["bg"],
        hovermode="x unified",
        margin=dict(l=48, r=20, t=56, b=18),
        font=dict(family="Arial, sans-serif", size=12, color=BBG["text"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color=BBG["subtle"]),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            traceorder="normal",
        ),
        bargap=0.18,
    )

    return fig
