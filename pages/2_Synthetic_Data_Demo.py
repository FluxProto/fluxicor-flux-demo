import io, gzip, time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Synthetic Data Demo", layout="wide")
st.title("🧪 Synthetic Telemetry Generator")
st.caption("Generate realistic, shareable mock Bitcoin-mining telemetry for demos (no real hardware needed).")

# ----- Controls
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

# Generation button
if st.button("Generate dataset", type="primary"):
    with st.spinner("Synthesizing telemetry…"):
        t0 = pd.Timestamp.utcnow().floor("s")
        seconds = int(duration_min * 60 * freq_hz)
        step = 1.0 / freq_hz

        rng = np.random.default_rng(seed)
        miner_ids = [f"miner_{i:02d}" for i in range(n_miners)]

        rows = []
        for m in miner_ids:
            base_hash = rng.uniform(85, 110)              # TH/s
            watts_per_th = rng.uniform(9.5, 11.5)
            base_power = base_hash * watts_per_th
            base_temp = rng.uniform(55, 70)               # °C
            fan_base = rng.uniform(3200, 3800)

            # Random step events per miner
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
                hash_rate = max(
                    5.0,
                    hash_level * (1 + 0.003*np.sin(s/60.0) + 0.002*rng.normal() + drift)
                )
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

    # Downloads
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "synthetic_mining_telemetry.csv", "text/csv")

    # Also provide a gzipped CSV
    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", compresslevel=6) as f:
        f.write(csv_bytes)
    st.download_button("Download CSV.gz", gz_buf.getvalue(), "synthetic_mining_telemetry.csv.gz", "application/gzip")

    # Quick summary stats
    st.subheader("Quick stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Miners", f"{n_miners}")
    with col3:
        st.metric("Duration", f"{duration_min} min")
    with col4:
        st.metric("Rate", f"{freq_hz} Hz")

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
