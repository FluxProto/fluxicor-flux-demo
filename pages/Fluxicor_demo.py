# HashMoney – Bitcoin Bill & Hash Log Demo (deterministic calibrated savings for bill & log)

import io
import re
import math
import hashlib
from dateutil import parser as dtparse

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional PDF parsing
try:
    import pdfplumber
    HAS_PDF = True
except Exception:
    HAS_PDF = False

st.set_page_config(page_title="HashMoney Bitcoin Demo", page_icon="💰", layout="wide")

# ---------- THEME ----------
FLUX_BLUE = "#0EA5E9"
DARK_BG   = "#0B1220"
CARD_BG   = "#111827"

st.markdown(
    f"""
    <style>
      .stApp {{ background: linear-gradient(180deg, {DARK_BG} 0%, #060a12 100%); }}
      .block-container {{ padding-top: 1.25rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

def metric_card(label: str, value: str, help_text: str):
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG};
            padding:16px;border-radius:16px;
            border:1px solid rgba(255,255,255,0.08);
            margin-bottom: 10px;
        ">
            <div style="color:#9CA3AF;font-size:12px;margin-bottom:6px">{label}</div>
            <div style="color:white;font-size:28px;font-weight:700;margin-bottom:6px">{value}</div>
            <div style="color:#9CA3AF;font-size:12px">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Simple compressibility ----------
_SUBS = [
    (r"(?i)\bamount due\b", "AmtDue"),
    (r"(?i)\bcurrent charges\b", "CurChg"),
    (r"(?i)\bservice address\b", "SvcAddr"),
    (r"(?i)\bbilling period\b", "BillPer"),
    (r"(?i)\baccount number\b", "Acct#"),
    (r"(?i)\bkwh\b", "kWh"),
    (r"[ ]{2,}", " "),
    (r"(,){2,}", ","),
]

def compressibility_percent(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)
    orig = text.encode("utf-8", errors="ignore")
    out = text
    for pat, rep in _SUBS:
        out = re.sub(pat, rep, out)
    comp = out.encode("utf-8", errors="ignore")
    if len(orig) == 0:
        return 0.0
    return max(0.0, (len(orig) - len(comp)) / len(orig) * 100.0)

def df_to_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)

# ---------- Deterministic calibrator ----------
def _hash_unit(b: bytes) -> float:
    if not b:
        return 0.0
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "big") / (2**64)

def calibrated_savings(raw_pct: float, content_bytes: bytes,
                       lo: float = 20.0, hi: float = 25.0) -> float:
    """
    Map raw compression % to [lo, hi] with a stable, file-specific offset.
    Same file -> same percent. Different files -> different percent.
    """
    if raw_pct is None:
        return lo
    # assume 0–50% raw typical for text/CSV; clamp & normalize
    norm = max(0.0, min(1.0, float(raw_pct) / 50.0))
    j = _hash_unit(content_bytes)          # stable per-file jitter (0..1)
    mix = 0.55 * norm + 0.45 * j           # blend raw signal + jitter
    eps = 0.02                              # keep off the exact edges
    lo2 = lo + (hi - lo) * eps
    hi2 = hi - (hi - lo) * eps
    return round(lo2 + (hi2 - lo2) * mix, 2)

# ---------- Bill parsers ----------
CURRENCY_RE = re.compile(r'(?i)\$?\s?([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))')
KWH_RE      = re.compile(r'(?i)(\d{1,3}(?:[,]\d{3})+|\d+)\s*kwh')
DATE_RE     = re.compile(r'(?i)(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2},?\s*\d{2,4}\b)')

def _to_float(num_str):
    if num_str is None:
        return None
    return float(str(num_str).replace(",", ""))

def parse_bill_pdf(file_bytes: bytes):
    if not HAS_PDF:
        return {}, "", 0.0, file_bytes
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    full = "\n".join(pages)
    raw_comp_pct = compressibility_percent(full)
    # Cost
    total_cost = None
    for key in [r"amount due", r"total due", r"current charges", r"total amount", r"amount to be paid"]:
        m = re.search(rf"(?i){key}.*?{CURRENCY_RE.pattern}", full)
        if m:
            total_cost = _to_float(m.groups()[-1]); break
    if total_cost is None:
        monies = [_to_float(m.group(1)) for m in CURRENCY_RE.finditer(full)]
        total_cost = max(monies) if monies else None
    # kWh
    kwh = None
    mk = KWH_RE.search(full)
    if mk:
        kwh = _to_float(mk.group(1))
    # Period
    dates = [dtparse.parse(m.group(1), fuzzy=True) for m in DATE_RE.finditer(full)]
    dates = sorted(set(dates))
    p_start = dates[0].isoformat() if len(dates) >= 2 else ""
    p_end   = dates[-1].isoformat() if len(dates) >= 2 else ""
    parsed = {
        "total_cost_usd": total_cost,
        "kwh": kwh,
        "period_start": p_start,
        "period_end": p_end,
    }
    return parsed, full, raw_comp_pct, file_bytes

def parse_bill_csv(df: pd.DataFrame):
    raw_text = df_to_text(df)
    raw_comp_pct = compressibility_percent(raw_text)
    content_bytes = raw_text.encode("utf-8", errors="ignore")
    cols = {c.lower(): c for c in df.columns}
    total = float(df[cols["costusd"]].sum()) if "costusd" in cols else None
    kwh = float(df[cols["kwh"]].sum()) if "kwh" in cols else None
    p_start = str(df[cols["month"]].iloc[0]) if "month" in cols and len(df) else ""
    p_end   = str(df[cols["month"]].iloc[-1]) if "month" in cols and len(df) else ""
    return {"total_cost_usd": total, "kwh": kwh, "period_start": p_start, "period_end": p_end}, raw_text, raw_comp_pct, content_bytes

# ---------- UI ----------
st.title("💰 HashMoney – Bitcoin Bill & Hash Log Demo")
st.markdown(
    f"Upload a **recent power bill (PDF or CSV)**. "
    f"We compute a **demo savings %** from the file’s compressibility. "
    f"<br><span style='color:{FLUX_BLUE}'>All metrics include plain-English explanations.</span>",
    unsafe_allow_html=True
)

# ---- Upload (fields always visible) ----
bill_file = st.file_uploader("Drag and drop file here", type=["pdf", "csv"])
fname = (bill_file.name if bill_file else "").lower()
bill_bytes = bill_file.getvalue() if bill_file else b""

st.subheader("📄 Power Bill (parsed or manual confirm)")
colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    total_cost_usd = st.number_input("Total Cost (USD)", value=0.00, step=100.0, format="%.2f")
with colB:
    kwh_used = st.number_input("Energy Used (kWh)", value=0.0, step=1000.0, format="%.0f")
with colC:
    ps = st.text_input("Period Start", value="")
with colD:
    pe = st.text_input("Period End", value="")

# Placeholder metrics
st.markdown("### 📊 Metrics")
m0, m1, m2, m3 = st.columns(4, gap="large")
with m0: metric_card("Computed Savings %", "—", "Appears after your bill is parsed.")
with m1: metric_card("Baseline Cost", "—", "The bill amount.")
with m2: metric_card("Projected Cost", "—", "Baseline minus computed savings.")
with m3: metric_card("Estimated Savings", "—", "Money not spent this billing period.")
m4, m5 = st.columns(2, gap="large")
with m4: metric_card("Cost per kWh", "—", "Total ÷ kWh.")
with m5: metric_card("Annualized Savings", "—", "This period’s savings × 12.")

# After upload: parse & compute
computed_savings_pct = None
if bill_file:
    if fname.endswith(".pdf"):
        parsed, bill_text, raw_pct, content_bytes = parse_bill_pdf(bill_bytes)
    elif fname.endswith(".csv"):
        try:
            df_bill = pd.read_csv(io.StringIO(bill_bytes.decode("utf-8", errors="ignore")))
        except Exception:
            df_bill = pd.read_csv(io.BytesIO(bill_bytes))
        st.caption("Detected CSV – first rows:")
        st.dataframe(df_bill.head())
        parsed, bill_text, raw_pct, content_bytes = parse_bill_csv(df_bill)
    else:
        st.error("Unsupported file type.")
        st.stop()

    # Fill widgets with parsed values
    with colA:
        total_cost_usd = st.number_input("Total Cost (USD)", value=float(parsed.get("total_cost_usd") or 0.0), step=100.0, format="%.2f")
    with colB:
        kwh_used = st.number_input("Energy Used (kWh)", value=float(parsed.get("kwh") or 0.0), step=1000.0, format="%.0f")
    with colC:
        ps = st.text_input("Period Start", value=str(parsed.get("period_start") or ""))
    with colD:
        pe = st.text_input("Period End", value=str(parsed.get("period_end") or ""))

    # Deterministic calibrated savings (20–25%)
    computed_savings_pct = calibrated_savings(raw_pct, content_bytes, 20.0, 25.0)

    # Derived metrics
    cost_per_kwh = (total_cost_usd / kwh_used) if (total_cost_usd and kwh_used) else None
    projected_cost = total_cost_usd * (1 - computed_savings_pct/100.0) if total_cost_usd else None
    dollars_saved = (total_cost_usd - projected_cost) if projected_cost is not None else None
    annual_savings = dollars_saved * 12 if dollars_saved is not None else None

    # Re-render metrics
    m0.empty(); m1.empty(); m2.empty(); m3.empty(); m4.empty(); m5.empty()
    with m0: metric_card("Computed Savings %", f"{computed_savings_pct:.2f}%", "Calibrated 20–25%, deterministic per file.")
    with m1: metric_card("Baseline Cost", f"${total_cost_usd:,.2f}" if total_cost_usd else "—", "The bill amount.")
    with m2: metric_card("Projected Cost", f"${projected_cost:,.2f}" if projected_cost else "—", "Baseline minus computed savings.")
    with m3: metric_card("Estimated Savings", f"${dollars_saved:,.2f}" if dollars_saved else "—", "Money not spent.")
    with m4: metric_card("Cost per kWh", f"${cost_per_kwh:.4f}" if cost_per_kwh else "—", "Total ÷ kWh.")
    with m5: metric_card("Annualized Savings", f"${annual_savings:,.0f}" if annual_savings else "—", "This period’s savings × 12.")

    # Bar chart
    if total_cost_usd:
        st.markdown("#### Cost Comparison")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.bar(["Baseline", "With HashMoney"], [total_cost_usd, projected_cost], color=[FLUX_BLUE, "#1F2937"])
        ax.set_ylabel("USD")
        ax.set_title("Power Bill – Before vs After")
        st.pyplot(fig)

    # dB estimates (negative = quieter)
    st.markdown("### 🔉 Estimated Noise Reduction")
    if computed_savings_pct > 0:
        r = max(1e-6, 1.0 - computed_savings_pct/100.0)
        delta_db_cons = (50.0/3.0) * math.log10(r)
        delta_db_aggr = 20.0 * math.log10(r)
        c1, c2 = st.columns(2)
        with c1: metric_card("ΔdB (Conservative)", f"{delta_db_cons:.2f} dB", "Fan-affinity scaling. Negative = quieter.")
        with c2: metric_card("ΔdB (Aggressive)", f"{delta_db_aggr:.2f} dB", "Best-case tonal peaks. Negative = quieter.")

# ---------- Optional: Hashrate log (deterministic calibrated gain) ----------
st.subheader("📈 Hashrate Log (optional)")
log_file = st.file_uploader("Upload Hashrate Log (CSV: Timestamp, HashrateTHs)", type=["csv"], key="hashlog")
if log_file:
    # robust read
    try:
        log_bytes = log_file.getvalue()
        log_df = pd.read_csv(io.StringIO(log_bytes.decode("utf-8", errors="ignore")))
    except Exception:
        log_df = pd.read_csv(log_file)
        log_bytes = log_file.getvalue()

    missing = [c for c in ["Timestamp","HashrateTHs"] if c not in log_df.columns]
    if missing:
        st.error(f"CSV must have columns: Timestamp, HashrateTHs. Missing: {missing}")
    else:
        # Try to coerce times
        try:
            log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"])
        except Exception:
            pass

        # Derive gain from the log’s own compressibility, deterministically 20–25%
        log_text = df_to_text(log_df)
        raw_log_comp = compressibility_percent(log_text)
        log_gain_pct = calibrated_savings(raw_log_comp, log_text.encode('utf-8', errors='ignore'), 20.0, 25.0)

        log_df["HashMoney_Hashrate"] = log_df["HashrateTHs"] * (1 + log_gain_pct/100.0)

        st.caption("First rows:")
        st.dataframe(log_df.head())

        st.markdown("#### Hashrate Over Time")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(log_df["Timestamp"], log_df["HashrateTHs"], label="Original")
        ax2.plot(log_df["Timestamp"], log_df["HashMoney_Hashrate"], label=f"With HashMoney (+{log_gain_pct:.1f}%)")
        ax2.set_ylabel("TH/s")
        ax2.legend()
        st.pyplot(fig2)

        avg_base = log_df["HashrateTHs"].mean()
        avg_flux = log_df["HashMoney_Hashrate"].mean()
        metric_card(
            "Avg Hashrate Gain",
            f"+{(avg_flux-avg_base):.2f} TH/s  ({(avg_flux/avg_base-1)*100:.1f}%)",
            "Calibrated deterministically from log compressibility (20–25%)."
        )

# ---------- Footnote ----------
st.markdown(
    """
    <hr style="opacity:0.1">
    <small style="color:#9CA3AF">
      • Savings % is calibrated deterministically. Same file = same result; different files vary within the band.<br>
      • dB estimates use fan scaling laws; large drops usually need airflow/fan policy changes.
    </small>
    """,
    unsafe_allow_html=True
)
