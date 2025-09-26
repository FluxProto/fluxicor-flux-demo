# pages/2_Synthetic_Data_Demo.py
import io, gzip
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- Page setup -------------------------------------
st.set_page_config(page_title="BTC Mining Data Demo", layout="wide")
st.title("🧪 Telemetry Generator")
st.caption("Generate realistic, shareable Bitcoin-mining telemetry for demos (no real hardware needed).")

# --------------------------- Controls ---------------------------------------
colA, colB, colC, colD = st.columns(4)
with colA:
    n_miners = st.slider("Miners", 2, 32, 8, 1)
with colB:
    duration_min = st.slider("Duration (minutes)", 5, 240, 30, 5)
with colC:
    freq_hz = st.selectbox("Sampling rate (Hz)", [1.0, 0.5, 0.2], index=0)
with colD:
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

st.divider()

# --------------------------- Generate ---------------------------------------
if st.button("Generate dataset", type="primary"):
    with st.spinner("Synthesizing telemetry…"):
        t0 = pd.Timestamp.utcnow().floor("s")
        seconds = int(duration_min * 60 * freq_hz)
        step = 1.0 / float(freq_hz)

        rng = np.random.default_rng(seed)
        miner_ids = [f"miner_{i:02d}" for i in range(n_miners)]

        rows = []
        for m in miner_ids:
            base_hash = rng.uniform(85, 110)              # TH/s
            watts_per_th = rng.uniform(9.5, 11.5)
            base_power = base_hash * watts_per_th
            base_temp = rng.uniform(55, 70)               # °C
            fan_base = rng.uniform(3200, 3800)

            # Random step events per miner (throttling, pool switch)
            num_steps = max(3, seconds // 1800 + 2) if seconds > 0 else 3
            step_times = np.sort(rng.choice(np.arange(max(1, seconds)), size=num_steps, replace=False))
            step_effects = rng.uniform(-0.15, 0.20, size=num_steps)
            step_idx = 0
            hash_level = base_hash

            for s in range(seconds):
                ts = t0 + pd.to_timedelta(s * step, unit="s")
                if step_idx < len(step_times) and s == step_times[step_idx]:
                    hash_level *= (1 + float(step_effects[step_idx]))
                    step_idx += 1

                # smooth drift + noise
                drift = 0.00002 * (s - seconds/2.0)
                hash_rate = max(5.0, hash_level * (1 + 0.003*np.sin(s/60.0) + 0.002*rng.normal() + drift))
                power = max(200.0, hash_rate * watts_per_th * (1 + 0.001*rng.normal()))
                temp = base_temp + 0.05*np.sin(s/120.0) + 0.02*rng.normal() + 0.002*(power/base_power - 1) * 100
                fan_rpm = fan_base + (temp - 60) * 40 + 30 * rng.normal()
                nonce_rate = hash_rate * 1e12 / 2**32
                accepted = rng.poisson(3 + 0.02*hash_rate)
                rejected = rng.binomial(accepted, p=min(0.07, 0.02 + 0.01*abs(rng.normal())))
                pool_diff = 1 + 0.2*np.sin(s/1800.0)

                rows.append((
                    ts, m, round(hash_rate, 3), round(power, 1), round(temp, 2), int(fan_rpm),
                    round(nonce_rate, 3), int(accepted), int(rejected), round(pool_diff, 3)
                ))

        cols = [
            "timestamp","miner_id","hash_ths","power_w","temp_c","fan_rpm",
            "nonce_rate","shares_accepted","shares_rejected","pool_diff"
        ]
        df = pd.DataFrame(rows, columns=cols)

    st.success(f"Generated {len(df):,} rows for {n_miners} miners at {freq_hz} Hz over {duration_min} minutes.")
    st.dataframe(df.head(200), use_container_width=True, height=350)

    # ----------------------- Downloads --------------------------------------
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "synthetic_mining_telemetry.csv", "text/csv")

    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", compresslevel=6) as f:
        f.write(csv_bytes)
    st.download_button("Download CSV.gz", gz_buf.getvalue(), "synthetic_mining_telemetry.csv.gz", "application/gzip")

    # ----------------------- KPIs -------------------------------------------
    st.subheader("KPIs (session)")
    total_hash_ths = df["hash_ths"].sum()
    total_power_w  = df["power_w"].sum()
    avg_hash       = df["hash_ths"].mean()
    avg_power      = df["power_w"].mean()
    avg_temp       = df["temp_c"].mean()
    eff_w_per_th   = (total_power_w / total_hash_ths) if total_hash_ths else float("nan")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Hashrate (TH/s)", f"{avg_hash:,.2f}")
    k2.metric("Avg Power (W)", f"{avg_power:,.0f}")
    k3.metric("Avg Temp (°C)", f"{avg_temp:,.1f}")
    k4.metric("Efficiency (W / TH)", f"{eff_w_per_th:,.2f}")

    st.divider()

    # ----------------------- Focus & Trends ---------------------------------
    miner_opts = ["(All)"] + sorted(df["miner_id"].unique().tolist())
    focus = st.selectbox("Focus", miner_opts, index=0)
    df_focus = df if focus == "(All)" else df[df["miner_id"] == focus]

    agg = (
        df_focus.groupby("timestamp", as_index=False)
        .agg({
            "hash_ths": "mean",
            "power_w": "mean",
            "temp_c": "mean",
            "shares_accepted": "sum",
            "shares_rejected": "sum"
        })
        .sort_values("timestamp")
    )

    st.subheader("Trends")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Hashrate (mean across selected miners)")
        st.line_chart(agg.set_index("timestamp")["hash_ths"], height=220)
        st.caption("Temperature")
        st.line_chart(agg.set_index("timestamp")["temp_c"], height=200)
    with c2:
        st.caption("Power (mean across selected miners)")
        st.line_chart(agg.set_index("timestamp")["power_w"], height=220)
        st.caption("Shares (accepted vs rejected)")
        shares = agg[["timestamp","shares_accepted","shares_rejected"]].set_index("timestamp")
        st.area_chart(shares, height=200)

    st.download_button(
        "Download current view (CSV)",
        df_focus.to_csv(index=False).encode("utf-8"),
        file_name=("synthetic_focus_all.csv" if focus == "(All)" else f"synthetic_{focus}.csv"),
        mime="text/csv"
    )

    # ----------------------- Estimated Savings ------------------------------
    st.subheader("Estimated Savings")

    with st.expander("Assumptions", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rate = st.number_input("Electricity rate ($/kWh)", min_value=0.00, value=0.12, step=0.01, format="%.2f")
        with c2:
            pue_current = st.number_input("Current PUE", min_value=1.00, value=1.35, step=0.01, format="%.2f")
        with c3:
            pue_target = st.number_input("Target PUE (after ops)", min_value=1.00, value=1.25, step=0.01, format="%.2f")
        with c4:
            annualize = st.checkbox("Show annualized projection", value=True)

        st.caption("Select the optimizations you plan to use:")
        o1, o2, o3 = st.columns(3)
        with o1:
            use_dedup = st.checkbox("Dedup/Compression (~5%)", value=True)
        with o2:
            use_quant = st.checkbox("Quantization/Low-precision (~10%)", value=True)
        with o3:
            use_sched = st.checkbox("Scheduling/DFS (~5%)", value=False)

    # Integrate energy from the synthetic trace: sum(power)*dt
    step_sec = 1.0 / float(freq_hz)
    baseline_kwh_it = df["power_w"].sum() * (step_sec / 3600.0) / 1000.0
    baseline_kwh_facility = baseline_kwh_it * float(pue_current)

    # Combined reduction from chosen optimizations
    reduction = 1.0
    if use_dedup: reduction *= (1 - 0.05)
    if use_quant: reduction *= (1 - 0.10)
    if use_sched: reduction *= (1 - 0.05)
    combined_reduction = 1 - reduction               # e.g., 0.186 ≈ 18.6%

    # PUE improvement benefit
    pue_gain = float(pue_current) / max(float(pue_target), 1.0)

    # Apply both effects
    target_kwh_facility = baseline_kwh_facility * (1 - combined_reduction) * pue_gain

    kwh_saved = max(0.0, baseline_kwh_facility - target_kwh_facility)
    cost_saved = kwh_saved * float(rate)

    # Annualization
    hours = float(duration_min) / 60.0
    scale = (8760.0 / hours) if (annualize and hours > 0) else 1.0

    kwh_saved_scaled = kwh_saved * scale
    cost_saved_scaled = cost_saved * scale
    baseline_scaled = baseline_kwh_facility * scale
    target_scaled = target_kwh_facility * scale

    m1, m2, m3 = st.columns(3)
    label_suffix = " / year" if (annualize and hours > 0) else " (this run)"
    m1.metric(f"Baseline energy{label_suffix}", f"{baseline_scaled:,.1f} kWh")
    m2.metric(f"Energy saved{label_suffix}", f"{kwh_saved_scaled:,.1f} kWh",
              delta=f"{combined_reduction*100:.1f}% + PUE×{pue_gain:.3f}")
    m3.metric(f"Cost saved{label_suffix}", f"${cost_saved_scaled:,.2f}")

    with st.expander("Details"):
        st.write({
            "combined_reduction_pct": round(combined_reduction*100, 3),
            "pue_gain_multiplier": round(pue_gain, 3),
            "baseline_kWh_facility_run": round(baseline_kwh_facility, 6),
            "target_kWh_facility_run": round(target_kwh_facility, 6),
            "run_duration_hours": round(hours, 3)
        })

    # ----------------------- Savings over time ------------------------------
    st.subheader("Savings over time")

    # Aggregate IT power per timestamp
    per_ts = (
        df.groupby("timestamp", as_index=False)["power_w"]
          .sum()
          .sort_values("timestamp")
          .rename(columns={"power_w": "power_it_w"})
    )

    # Baseline vs target facility power
    per_ts["power_baseline_w"] = per_ts["power_it_w"] * float(pue_current)
    per_ts["power_target_w"]   = per_ts["power_it_w"] * (1 - combined_reduction) * float(pue_target)
    per_ts["power_saved_w"]    = per_ts["power_baseline_w"] - per_ts["power_target_w"]

    st.caption("Instantaneous facility power (baseline vs optimized)")
    st.line_chart(per_ts.set_index("timestamp")[["power_baseline_w", "power_target_w"]], height=260)

    # Optional: cumulative energy and $
    show_cum = st.checkbox("Show cumulative energy (kWh) & $", value=True)
    if show_cum:
        step_sec = 1.0 / float(freq_hz)
        # Wh per step → kWh cumulative
        per_ts["kwh_baseline"] = (per_ts["power_baseline_w"] * (step_sec/3600.0)) / 1000.0
        per_ts["kwh_target"]   = (per_ts["power_target_w"]   * (step_sec/3600.0)) / 1000.0
        per_ts["kwh_saved"]    = per_ts["kwh_baseline"] - per_ts["kwh_target"]
        per_ts["kwh_baseline_cum"] = per_ts["kwh_baseline"].cumsum()
        per_ts["kwh_target_cum"]   = per_ts["kwh_target"].cumsum()
        per_ts["kwh_saved_cum"]    = per_ts["kwh_saved"].cumsum()

        st.caption("Cumulative facility energy (kWh)")
        st.area_chart(per_ts.set_index("timestamp")[["kwh_baseline_cum", "kwh_target_cum"]], height=260)
        st.caption(f"Total saved this run: **{per_ts['kwh_saved_cum'].iloc[-1]:,.2f} kWh**")

        # Cumulative $ saved
        per_ts["usd_saved_cum"] = per_ts["kwh_saved_cum"] * float(rate)
        st.caption("Cumulative cost savings ($)")
        st.line_chart(per_ts.set_index("timestamp")[["usd_saved_cum"]], height=220)
        st.caption(f"Total $ saved this run: **${per_ts['usd_saved_cum'].iloc[-1]:,.2f}**")

    # Export time series
    st.download_button(
        "Download time series (baseline, target, savings)",
        per_ts.to_csv(index=False).encode("utf-8"),
        file_name="facility_power_timeseries.csv",
        mime="text/csv"
    )

    # ----------------------- Column descriptions ----------------------------
    with st.expander("Column descriptions"):
        st.markdown("""
        - **timestamp**: UTC timestamp at sampling interval  
        - **miner_id**: simulated worker ID  
        - **hash_ths**: hash rate (TH/s) with drift/noise/events  
        - **power_w**: power draw (W) proportional to hash rate  
        - **temp_c**: temperature (°C) influenced by power + ambient drift  
        - **fan_rpm**: fan response to temperature (with noise)  
        - **nonce_rate**: rough proportional rate to hash_ths  
        - **shares_accepted/rejected**: Poisson/Binomial draws per tick  
        - **pool_diff**: slow sinusoidal variation in pool difficulty
        """)
else:
    st.info("Adjust parameters, then click **Generate dataset**.")
