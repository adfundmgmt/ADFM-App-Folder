def methodology(primary,comparison,primary_implied,comparison_implied,rvol_window,normalization_window):
    return f"""
        **Synthetic VIX calculation**

        - Daily price action is measured with log returns.
        - The selected {rvol_window}-session rolling standard deviation is annualized by multiplying by the square root of 252 and shown in percent units.
        - This is a realized-volatility estimate. It is comparable across liquid assets, but it is not an options-implied volatility index and does not contain a forward volatility risk premium.

        **Normalization and comparison**

        - Each z-score compares today's synthetic VIX with the mean and sample standard deviation of up to {normalization_window} prior observations. Excluding today keeps the calculation causal.
        - The ratio divides {primary} synthetic VIX by {comparison} synthetic VIX on overlapping dates.
        - The two asset percentiles and the ratio percentile rank each latest reading against all earlier observations in the loaded history; ties receive half credit. The current observation is excluded from its own reference set.
        - The 5D ratio change is the point-to-point change from five valid ratio observations earlier.
        - `{primary_implied or 'Primary implied volatility'}` divided by `{comparison_implied or 'comparison implied volatility'}` is shown beside the realized ratio. Both implied series are plotted as reported by Yahoo Finance and are not transformed into realized-volatility estimates.

        **Fixed-window diagnostics**

        - Relative-volatility acceleration divides the 5-session {primary}/{comparison} RVOL ratio by the 21-session ratio. A reading above 1.0 means short-term relative stress is running above the recent regime.
        - SOXX/NDX and QEW/QQQ divide 21-session annualized realized volatility for those fixed benchmark pairs. QEW is used as the Nasdaq-100 equal-weight proxy and QQQ as the cap-weight proxy.
        - The downside-semivolatility ratio uses the annualized sample standard deviation of negative log-return sessions observed within each trailing 21-session window. It requires at least two negative sessions per asset; sparse windows remain unavailable.
        - No optional series is filled or fabricated. Missing implied-volatility or ETF history produces `N/A` diagnostics while the selected pair continues to render.

        Thin trading, stale observations, leverage, market-hour differences, and overnight gaps can make comparisons less representative. This dashboard is an analytical tool, not an investment recommendation.
        """
