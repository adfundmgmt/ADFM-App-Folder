# ------------------------------- Panels v2 ---------------------------------
# Drop-in replacement for your current "Plots grid" block. Uses same variables:
# df, events, comps, fwd_days, PASTELS, TEXT_COLOR, GRID_COLOR, BAR_EDGE, day_word, winrate

from matplotlib.colors import LinearSegmentedColormap

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.45)

# Small helpers (local, no side effects)
def safe_group_lists(g, col, order):
    data = []
    labels = []
    for k in order:
        arr = pd.to_numeric(g.get_group(k)[col], errors="coerce").dropna().values if k in g.groups else np.array([])
        if arr.size > 0:
            data.append(arr)
            labels.append(k)
    return data, labels

def ensure_fwd_cols(df_, horizons):
    # compute forward returns in percent if missing
    for h in horizons:
        col_next = f"spx_next{h}"
        col_ret = f"spx_fwd{h}_ret"
        if col_ret not in df_.columns:
            df_[col_next] = df_["spx_close"].shift(-h)
            df_[col_ret] = (df_[col_next] / df_["spx_close"] - 1.0) * 100.0
    return df_

# 1) Boxplot: forward returns by VIX base bucket (distribution, not just hit rate)
ax1 = axes[0, 0]
order_base = ["0-12","12-16","16-20","20-24","24-30","30+"]
if not events.empty:
    g = events.groupby("vix_base_bucket")
    data, labels = safe_group_lists(g, f"spx_fwd{fwd_days}_ret", order_base)
else:
    data, labels = [], []
bp = ax1.boxplot(
    data if data else [np.array([np.nan])],
    labels=labels if labels else ["no data"],
    patch_artist=True,
    notch=True,
    showfliers=False
)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PASTELS[i % len(PASTELS)])
    patch.set_edgecolor(BAR_EDGE)
for med in bp["medians"]:
    med.set_color("#444444")
ax1.axhline(0, color="#888888", linewidth=1)
ax1.set_title(f"Distribution by VIX Base, {fwd_days}-Day %", color=TEXT_COLOR, fontsize=12, pad=8)
ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

# 2) Heatmap: win rate by Base Bucket × Spike Magnitude (interaction)
ax2 = axes[0, 1]
order_mag = ["Moderate (20-30%)","Large (30-50%)","Extreme (50%+)"]
if not events.empty:
    pivot = events.pivot_table(
        index="vix_base_bucket", columns="spike_mag",
        values=f"spx_fwd{fwd_days}_ret",
        aggfunc=lambda s: winrate(pd.Series(s))
    ).reindex(index=order_base, columns=order_mag)
else:
    pivot = pd.DataFrame(np.nan, index=order_base, columns=order_mag)

# Light pastel custom cmap, 0..100
cmap = LinearSegmentedColormap.from_list("pastelWR", ["#EAF7FB", "#A8DADC", "#4DB6AC"])
im = ax2.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=100)

# Ticks and labels
ax2.set_xticks(range(len(order_mag)))
ax2.set_xticklabels(["20–30%", "30–50%", "50%+"], fontsize=10)
ax2.set_yticks(range(len(order_base)))
ax2.set_yticklabels(order_base, fontsize=10)
ax2.set_title("Win Rate by Base × Magnitude", color=TEXT_COLOR, fontsize=12, pad=8)
ax2.grid(False)

# Annotate cells
for i in range(len(order_base)):
    for j in range(len(order_mag)):
        val = pivot.values[i, j]
        txt = "" if np.isnan(val) else f"{val:.0f}%"
        ax2.text(j, i, txt, ha="center", va="center", color="#1f1f1f", fontsize=9)

# 3) ECDF curves: Bull vs Bear regime
ax3 = axes[0, 2]
if not events.empty:
    for name, color in [("Bull", PASTELS[0]), ("Bear", PASTELS[2])]:
        vals = pd.to_numeric(events.loc[events["regime"] == name, f"spx_fwd{fwd_days}_ret"], errors="coerce").dropna().values
        if vals.size > 0:
            x = np.sort(vals)
            y = np.arange(1, x.size + 1) / x.size
            ax3.plot(x, y, label=f"{name} (N={x.size})", linewidth=2, color=color)
ax3.axvline(0, color="#888888", linewidth=1)
ax3.set_title(f"ECDF of {fwd_days}-Day Returns by Regime", color=TEXT_COLOR, fontsize=12, pad=8)
ax3.set_xlabel("SPX Forward Return (%)", color=TEXT_COLOR)
ax3.set_ylabel("Cumulative Probability", color=TEXT_COLOR)
ax3.grid(color=GRID_COLOR, linewidth=0.6)
ax3.legend(frameon=False, fontsize=9)

# 4) Scatter: VIX base vs forward return, size = spike %, color = RSI state
ax4 = axes[1, 0]
if not events.empty:
    vals = events.copy()
    vals["SpikePct"] = vals["vix_pctchg"] * 100.0
    colors = np.where(vals["oversold"] == "Oversold", PASTELS[3], PASTELS[1])
    sizes = 10 + np.clip(vals["SpikePct"].abs(), 0, 100)  # cap size growth
    ax4.scatter(vals["vix_base"], vals[f"spx_fwd{fwd_days}_ret"], s=sizes, c=colors,
                alpha=0.85, edgecolors=BAR_EDGE, linewidths=0.3)
    # Simple fit line
    good = vals[["vix_base", f"spx_fwd{fwd_days}_ret"]].dropna()
    if len(good) > 5:
        m, b = np.polyfit(good["vix_base"], good[f"spx_fwd{fwd_days}_ret"], 1)
        xs = np.linspace(good["vix_base"].min(), good["vix_base"].max(), 100)
        ax4.plot(xs, m*xs + b, linewidth=1.5, color="#444444")
ax4.axhline(0, color="#888888", linewidth=1)
ax4.set_title("VIX Base vs Forward Return, size=Spike% color=RSI", color=TEXT_COLOR, fontsize=12, pad=8)
ax4.set_xlabel("VIX Level Before Spike", color=TEXT_COLOR)
ax4.set_ylabel(f"SPX {fwd_days}-Day Return (%)", color=TEXT_COLOR)
ax4.grid(color=GRID_COLOR, linewidth=0.6)

# 5) Event-study path: mean and 10/90 bands across horizons
ax5 = axes[1, 1]
horizons = [1, 2, 3, 5, 10, 20]
df = ensure_fwd_cols(df, horizons)
if not events.empty:
    # Build mask of event dates
    idx = events.index
    means, p10s, p90s = [], [], []
    for h in horizons:
        # Align to the same event dates
        x = df.loc[idx, f"spx_fwd{h}_ret"].astype(float)
        means.append(np.nanmean(x))
        p10s.append(np.nanpercentile(x.dropna(), 10) if x.notna().sum() else np.nan)
        p90s.append(np.nanpercentile(x.dropna(), 90) if x.notna().sum() else np.nan)
    ax5.plot(horizons, means, linewidth=2, marker="o")
    ax5.fill_between(horizons, p10s, p90s, alpha=0.25)
ax5.axhline(0, color="#888888", linewidth=1)
ax5.set_title("Event Study, Mean Path after Spike", color=TEXT_COLOR, fontsize=12, pad=8)
ax5.set_xlabel("Horizon (trading days)", color=TEXT_COLOR)
ax5.set_ylabel("Forward Return (%)", color=TEXT_COLOR)
ax5.grid(color=GRID_COLOR, linewidth=0.6)

# 6) Frequency over time: yearly count of qualifying spikes
ax6 = axes[1, 2]
if not events.empty:
    yr_counts = events.groupby(events.index.year).size()
    years = yr_counts.index.astype(int).tolist()
    counts = yr_counts.values.tolist()
    ax6.bar(years, counts, edgecolor=BAR_EDGE, color=PASTELS[9])
    ax6.set_xlim(min(years) - 0.5, max(years) + 0.5)
ax6.set_title("Yearly Count of ≥ Threshold Spikes", color=TEXT_COLOR, fontsize=12, pad=8)
ax6.set_xlabel("Year", color=TEXT_COLOR)
ax6.set_ylabel("Count", color=TEXT_COLOR)
ax6.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

st.pyplot(fig, clear_figure=True)
# ----------------------------- End Panels v2 -------------------------------
