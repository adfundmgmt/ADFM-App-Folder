# =========================
# Trailing 252-Day Rolling Analog Overlay
# Append-only block
# =========================

st.markdown("<hr style='margin-top:18px; margin-bottom:12px;'>", unsafe_allow_html=True)
st.subheader("Trailing 252-Day Rolling Analogs")

TRAILING_WINDOW = 252
ROLLING_CORR_CUTOFF = 0.90

close_series = raw["Close"].dropna().copy()

def forward_return(prices: pd.Series, end_exclusive: int, horizon: int) -> float:
    if end_exclusive + horizon - 1 >= len(prices):
        return np.nan
    start_px = float(prices.iloc[end_exclusive - 1])
    end_px = float(prices.iloc[end_exclusive + horizon - 1])
    if start_px == 0:
        return np.nan
    return end_px / start_px - 1

if len(close_series) < TRAILING_WINDOW * 2:
    st.info(
        f"Need at least {TRAILING_WINDOW * 2} trading days of history to run a clean "
        f"rolling {TRAILING_WINDOW}-day analog scan."
    )
else:
    current_window = close_series.iloc[-TRAILING_WINDOW:].copy()
    current_252 = cumret(current_window)
    current_252.index = np.arange(1, len(current_252) + 1)

    current_start_loc = len(close_series) - TRAILING_WINDOW
    rolling_matches = []

    for start_loc in range(0, len(close_series) - TRAILING_WINDOW + 1):
        end_loc = start_loc + TRAILING_WINDOW

        # Exclude windows that overlap with the current trailing 252-day window
        if end_loc > current_start_loc:
            continue

        hist_window = close_series.iloc[start_loc:end_loc].copy()
        if len(hist_window) != TRAILING_WINDOW or hist_window.isna().any():
            continue

        hist_252 = cumret(hist_window)
        rho = safe_corr(current_252.values, hist_252.values)

        if not np.isfinite(rho) or rho < ROLLING_CORR_CUTOFF:
            continue

        start_dt = pd.Timestamp(hist_window.index[0]).normalize()
        end_dt = pd.Timestamp(hist_window.index[-1]).normalize()

        rolling_matches.append(
            {
                "Match Year": int(end_dt.year),
                "Start Date": start_dt,
                "End Date": end_dt,
                "Correlation": float(rho),
                "Matched 252D Return": float(hist_252.iloc[-1]),
                "Next 21D": forward_return(close_series, end_loc, 21),
                "Next 63D": forward_return(close_series, end_loc, 63),
                "Next 126D": forward_return(close_series, end_loc, 126),
                "_start_loc": int(start_loc),
                "_end_loc": int(end_loc),
            }
        )

    if not rolling_matches:
        st.info(f"No rolling {TRAILING_WINDOW}-day windows cleared ρ >= {ROLLING_CORR_CUTOFF:.2f}.")
    else:
        rolling_df = pd.DataFrame(rolling_matches).sort_values(
            ["Correlation", "End Date"], ascending=[False, True]
        )

        # Keep the single best rolling match per year so the overlay does not get cluttered
        best_by_year = (
            rolling_df.sort_values(["Match Year", "Correlation", "End Date"], ascending=[True, False, True])
            .groupby("Match Year", as_index=False)
            .head(1)
            .sort_values("Correlation", ascending=False)
            .reset_index(drop=True)
        )

        overlay_df = best_by_year.head(top_n).copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Above 0.90", f"{len(best_by_year)}")
        c2.metric("Best Rolling ρ", f"{best_by_year['Correlation'].max():.2f}")
        c3.metric(
            f"Current Trailing {TRAILING_WINDOW}D Return",
            f"{current_252.iloc[-1]:.2%}"
        )

        palette_252 = distinct_palette(len(overlay_df))
        fig2, ax2 = plt.subplots(figsize=(14, 7))

        for idx, row in overlay_df.reset_index(drop=True).iterrows():
            hist_window = close_series.iloc[int(row["_start_loc"]):int(row["_end_loc"])].copy()
            hist_252 = cumret(hist_window)
            hist_252.index = np.arange(1, len(hist_252) + 1)

            ax2.plot(
                hist_252.index,
                hist_252.values,
                "--",
                lw=2.2,
                alpha=0.95,
                color=palette_252[idx],
                label=f"{int(row['Match Year'])} (ρ={row['Correlation']:.2f})",
            )

        ax2.plot(
            current_252.index,
            current_252.values,
            color="black",
            lw=3.2,
            label=f"Current Trailing {TRAILING_WINDOW}D",
        )

        ax2.set_title(
            f"{ticker} – Current Trailing {TRAILING_WINDOW}D vs Rolling Historical Matches",
            fontsize=16,
            weight="bold",
        )
        ax2.set_xlabel("Trading Day in 252-Day Window", fontsize=13)
        ax2.set_ylabel("Cumulative Return", fontsize=13)
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.set_xlim(1, TRAILING_WINDOW)

        overlay_arrays = [current_252.values]
        for _, row in overlay_df.iterrows():
            hist_window = close_series.iloc[int(row["_start_loc"]):int(row["_end_loc"])].copy()
            hist_252 = cumret(hist_window)
            overlay_arrays.append(hist_252.values)

        all_y2 = np.hstack(overlay_arrays)
        ymin2, ymax2 = float(np.min(all_y2)), float(np.max(all_y2))
        pad2 = 0.06 * (ymax2 - ymin2) if ymax2 > ymin2 else 0.02
        ax2.set_ylim(ymin2 - pad2, ymax2 + pad2)

        span2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        raw_step2 = max(span2 / 12, 0.0025)
        candidates2 = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
        step2 = float(candidates2[np.argmin(np.abs(candidates2 - raw_step2))])

        ax2.yaxis.set_major_locator(MultipleLocator(step2))
        ax2.yaxis.set_minor_locator(MultipleLocator(step2 / 2))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.grid(True, ls=":", lw=0.7, color="#888")
        ax2.legend(loc="best", frameon=False, ncol=2, fontsize=11)

        plt.tight_layout()
        st.pyplot(fig2)

        table_df = best_by_year.copy()
        table_df["Start Date"] = pd.to_datetime(table_df["Start Date"]).dt.strftime("%Y-%m-%d")
        table_df["End Date"] = pd.to_datetime(table_df["End Date"]).dt.strftime("%Y-%m-%d")
        table_df["Correlation"] = table_df["Correlation"].map(lambda x: f"{x:.3f}")
        table_df["Matched 252D Return"] = table_df["Matched 252D Return"].map(lambda x: f"{x:.2%}")
        table_df["Next 21D"] = table_df["Next 21D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        table_df["Next 63D"] = table_df["Next 63D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
        table_df["Next 126D"] = table_df["Next 126D"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")

        table_df = table_df[
            [
                "Match Year",
                "Start Date",
                "End Date",
                "Correlation",
                "Matched 252D Return",
                "Next 21D",
                "Next 63D",
                "Next 126D",
            ]
        ]

        st.markdown("**Best rolling 252-day match per year, filtered at ρ >= 0.90**")
        st.dataframe(table_df, use_container_width=True, hide_index=True)
