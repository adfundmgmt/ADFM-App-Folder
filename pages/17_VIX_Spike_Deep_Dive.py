# ------------------------------- Dynamic Commentary (rich) -----------------
def find_analogs(evs: pd.DataFrame, latest_row: pd.Series, k: int = 10):
    # Feature set
    X = evs[["vix_base","vix_pctchg","rsi14","regime","is_october"]].copy()
    X = X.dropna()
    # Encode regime, month
    X["regime_bin"] = np.where(evs.loc[X.index, "regime"]=="Bull", 1.0, 0.0)
    X["oct_bin"] = np.where(evs.loc[X.index, "is_october"]=="October", 1.0, 0.0)
    X_feat = X[["vix_base","vix_pctchg","rsi14","regime_bin","oct_bin"]].copy()

    # Standardize
    mu = X_feat.mean()
    sd = X_feat.std().replace(0, 1.0)
    Z = (X_feat - mu)/sd

    # Latest vector
    lat = pd.DataFrame({
        "vix_base":[latest_row["vix_base"]],
        "vix_pctchg":[latest_row["vix_pctchg"]],
        "rsi14":[latest_row["rsi14"]],
        "regime_bin":[1.0 if latest_row["regime"]=="Bull" else 0.0],
        "oct_bin":[1.0 if latest_row["is_october"]=="October" else 0.0]
    })
    z_lat = (lat - mu)/sd

    # Weighted distance in z space
    weights = np.array([1.0, 1.2, 0.8, 0.6, 0.4])  # emphasize spike %, then base VIX
    diffs = (Z.values - z_lat.values.squeeze())**2 * weights
    dist = np.sqrt(diffs.sum(axis=1))
    nn_idx = np.argsort(dist)[:k]
    nn = Z.iloc[nn_idx].index
    return evs.loc[nn].copy().assign(dist=dist[nn_idx])

def dynamic_commentary():
    st.subheader("Dynamic Commentary")
    if events.empty:
        card_box("No qualifying VIX spike events in the filtered sample.")
        return

    latest = events.iloc[-1]
    dt = latest.name.strftime("%Y-%m-%d")
    base = latest["vix_base"]; spike_pct = latest["vix_pctchg"]*100.0
    spx = latest["spx_close"]; rsi = latest["rsi14"]
    regime = latest["regime"]; bucket = latest["vix_base_bucket"]
    mag = latest["spike_mag"]; over = latest["oversold"]; month = latest["is_october"]

    # Baseline and matched cohorts
    base_wr = 100.0 * (events[f"spx_fwd{fwd_days}_ret"] > 0).mean()
    cohort_mask = (
        (events["vix_base_bucket"] == bucket) &
        (events["spike_mag"] == mag) &
        (events["regime"] == regime)
    )
    cohort = events[cohort_mask]
    wr = 100.0 * (cohort[f"spx_fwd{fwd_days}_ret"] > 0).mean() if len(cohort) else np.nan
    avg = cohort[f"spx_fwd{fwd_days}_ret"].mean() if len(cohort) else np.nan
    med = cohort[f"spx_fwd{fwd_days}_ret"].median() if len(cohort) else np.nan
    p5  = np.percentile(cohort[f"spx_fwd{fwd_days}_ret"], 5) if len(cohort) else np.nan
    p95 = np.percentile(cohort[f"spx_fwd{fwd_days}_ret"],95) if len(cohort) else np.nan
    n   = int(len(cohort))

    # Effect size vs baseline
    p_cohort = wr/100.0 if n else np.nan
    se_cohort = np.sqrt(p_cohort*(1-p_cohort)/n) if n else np.nan
    wr_diff = wr - base_wr if n else np.nan

    # Analogs
    nn = find_analogs(events, latest, k=10) if len(events) >= 15 else pd.DataFrame()
    analog_stats = {}
    if not nn.empty:
        for d in [1,2,3,5]:
            col = f"spx_fwd{d}_ret"
            if col not in events.columns:
                events[f"spx_next{d}"] = events["spx_close"].shift(-d)
                events[col] = (events[f"spx_next{d}"]/events["spx_close"]-1.0)*100.0
            analog_stats[d] = {
                "avg": nn[col].mean(),
                "wr": 100.0*(nn[col] > 0).mean(),
                "n": int(nn.shape[0])
            }

    # Header card
    hdr = f"""
    <b>Latest event:</b> {dt}<br>
    VIX {latest['vix_close']:.2f} from base {base:.2f}, spike {spike_pct:.1f}%<br>
    SPX {spx:.0f}, RSI14 {rsi:.1f} ({over}), Regime {regime}, Month {month}<br>
    Setup: base {bucket}, spike {mag}
    """
    card_box(hdr)

    # Cohort stats card
    cohort_txt = f"""
    <b>Matched cohort</b> (same base bucket, spike mag, regime)<br>
    n={n}, WR {wr:.1f}%, Avg {avg:.2f}%, Median {med:.2f}%, 5th to 95th {p5:.2f}% to {p95:.2f}%<br>
    vs baseline WR {base_wr:.1f}%  |  ΔWR {wr_diff:.1f}pp  |  s.e.≈{(se_cohort*100.0 if n else np.nan):.1f}%
    """
    card_box(cohort_txt)

    # Analogs table
    if not nn.empty:
        st.markdown("**Nearest analogs (z-scaled distance)**")
        show_cols = ["dist", "vix_base", "vix_pctchg", "rsi14", f"spx_fwd{fwd_days}_ret"]
        atbl = nn[show_cols].copy()
        atbl.rename(columns={
            "dist":"Dist", "vix_base":"VIX_Base", "vix_pctchg":"VIX_Spike_%", "rsi14":"RSI14",
            f"spx_fwd{fwd_days}_ret":f"Fwd{fwd_days}_Ret_%"
        }, inplace=True)
        atbl["VIX_Spike_%"] = (atbl["VIX_Spike_%"]*100.0).round(1)
        st.dataframe(atbl.round(3), use_container_width=True)

        # Analog horizon summary
        rows = []
        for d, s in analog_stats.items():
            rows.append([f"T+{d}", f"{s['wr']:.1f}%", f"{s['avg']:.2f}%", s["n"]])
        an_df = pd.DataFrame(rows, columns=["Horizon","WR","Avg","n"])
        st.caption("Analog outcomes by horizon")
        st.dataframe(an_df, use_container_width=True)

dynamic_commentary()
