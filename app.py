
import json, base64
import streamlit as st
from flux_compress import flux_compress_text, estimate_savings, DEFAULT_SYMBOL_MAP

st.set_page_config(page_title="Flux Compression Demo", page_icon="⚡", layout="wide")

st.title("Flux Compression Demo")
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin:-10px 0 10px 0;">
  <div style="width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,#00e5ff 0%,#6fffd2 100%);display:flex;align-items:center;justify-content:center;">
    <span style="font-size:18px;font-weight:700;color:#0a1224;">F</span>
  </div>
  <div>
    <div style="font-size:20px;font-weight:800;letter-spacing:0.5px;color:#e6f7ff;line-height:1;">Fluxicore</div>
    <div style="font-size:12px;color:#9fd8e6;margin-top:2px;line-height:1;">Symbolic Compression - Power & Water Savings</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Inputs")
    kwh_cost = st.number_input("Electricity cost ($/kWh)", min_value=0.0, value=0.12, step=0.01)
    pue = st.number_input("PUE (Power Usage Effectiveness)", min_value=1.0, value=1.4, step=0.1)

    st.markdown("---")
    server_presets = {"GPU H100 (inference)":0.10,"GPU A100 (mixed)":0.12,"GPU L40S (vision)":0.09,"CPU-only analytics":0.03,"Custom…":None}
    choice = st.selectbox("Server type", list(server_presets.keys()))
    preset = server_presets[choice]
    if preset is None:
        gpu_kwh_per_gb = st.number_input("Compute intensity (kWh/GB)", min_value=0.0, value=0.08, step=0.01)
    else:
        gpu_kwh_per_gb = st.number_input("Compute intensity (kWh/GB)", min_value=0.0, value=preset, step=0.01)
    gb_per_day = st.number_input("Data processed per day (GB)", min_value=0.0, value=50.0, step=1.0)

    st.markdown("---")
    wue_l_per_kwh = st.number_input("WUE (liters/kWh)", min_value=0.0, value=0.7, step=0.1)
    water_cost_per_kgal = st.number_input("Water cost ($/kGal)", min_value=0.0, value=5.0, step=0.5)

    st.markdown("---")
    st.markdown("Existing optimizations")
    opt_dedupe = st.checkbox("Dedup/compression (~5%)", value=False)
    opt_quant = st.checkbox("Quantization/low-precision (~10%)", value=False)
    opt_sched = st.checkbox("Scheduling/DVFS (~5%)", value=False)
    baseline_reduction_pct = (0.05 if opt_dedupe else 0.0) + (0.10 if opt_quant else 0.0) + (0.05 if opt_sched else 0.0)
    st.caption(f"Baseline reduction applied before Flux: {baseline_reduction_pct*100:.0f}%")

    st.markdown("---")
    manual_override = st.checkbox("Manual override: set Flux savings %", value=False)
    override_pct = st.slider("Override savings (%)", 0, 90, 35, 1) if manual_override else None

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Input")
    sample = {"text":"This is a sample log describing system behavior. Operators noted recursion and attempted to stabilize the loop with a memory anchor. Responsibility was shared."}
    st.code(sample["text"][:1000], language="markdown")
    input_text = st.text_area("Or paste your own text", value=sample["text"], height=160)

with col2:
    st.subheader("Results")
    if input_text.strip():
        comp = flux_compress_text(input_text)
        orig_bytes = comp["original_bytes"]
        comp_bytes = comp["compressed_bytes"]
        ratio = (1 - comp_bytes / orig_bytes) if orig_bytes else 0.0

        st.metric("Compression Ratio", f"{ratio*100:.1f}%")
        st.text(f"Original: {orig_bytes:,} bytes")
        st.text(f"Compressed: {comp_bytes:,} bytes")

        with st.expander("Preview (first 800 chars)"):
            st.write("**Compressed (symbolic):**")
            st.code(comp["compressed_text"][:800], language="markdown")
            st.write("**Reconstructed:**")
            st.code(comp["reconstructed_text"][:800], language="markdown")

        savings = estimate_savings(
            compression_ratio=ratio,
            gb_per_day=gb_per_day,
            gpu_kwh_per_gb=gpu_kwh_per_gb,
            pue=pue,
            kwh_cost=kwh_cost,
            baseline_reduction_pct=baseline_reduction_pct,
            manual_savings_override_pct=(override_pct/100.0 if manual_override else None),
            wue_l_per_kwh=wue_l_per_kwh,
            water_cost_per_kgal=water_cost_per_kgal
        )

        st.subheader("Estimated Savings (Annual)")
        st.metric("kWh saved", f"{savings['kwh_year_saved']:,.0f}")
        st.metric("Electric $ saved", f"${savings['usd_electric_year_saved']:,.0f}")
        st.metric("Water saved", f"{savings['water_kgal_year_saved']:,.1f} kGal")
        st.metric("Water $ saved", f"${savings['usd_water_year_saved']:,.0f}")
        st.metric("Total $ saved", f"${savings['usd_year_saved']:,.0f}")

st.markdown('''
<style>
[data-testid="stMetricValue"] { text-shadow: 0 0 8px rgba(0, 229, 255, 0.35); }
</style>
''', unsafe_allow_html=True)
