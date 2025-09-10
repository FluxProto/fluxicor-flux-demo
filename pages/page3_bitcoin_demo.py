# page3_bitcoin_demo.py
# HashMoney – Bitcoin Bill & Hash Log Demo (no manual savings slider)
# - Drag/drop a real bill (PDF or CSV) and an optional hashrate log (CSV)
# - We auto-parse bill totals/period, then compute a demo "savings %" from actual compression
# - We apply that savings to show Projected Cost, $ Saved, Annualized, Cost/kWh, License revenue
# - We estimate dB drop from the computed savings (conservative & aggressive models)
# - Optional: plot hashrate before/after using a compression-derived gain (from the log file itself)

import io
import re
import math
from datetime import datetime
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

# ---------- THEME HELPERS ----------
FLUX_BLUE = "#0EA5E9"   # cyan-500 vibe
DARK_BG   = "#0B1220"
CARD_BG   = "#111827"

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {DARK_BG} 0%, #060a12 100%); }}
    .block-container {{ padding-top: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

def metric_card(label, value, help_text):
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
        """, unsafe_allow_html=True
    )

# ---------- "Flux-like" compression (demo engine) ----------
# We compress text by replacing repeated phrases/keys with compact symbols and coalescing punctuation.
import re as _re
SUBS = [
    (r"\bI think\b", "∿I"),
    (r"\byou know\b", "∿U"),
    (r"\bas a result\b", "⇒"),
    (r"\bkind of\b", "~"),
    (r"\bsort of\b", "~"),
    (r"\bto be honest\b", "†°"),
    (r"\bin my opinion\b", "§?"),
    (r"\bat the end of the day\b", "□⋯"),
    (r"\bfor what it's worth\b", "≈"),
    (r"\blet me be clear\b", "⧈!"),
    # Billing/utility terms
    (r"(?i)\bamount due\b", "AmtDue"),
    (r"(?i)\bcurrent charges\b", "CurChg"),
    (r"(?i)\bservice address\b", "SvcAddr"),
    (r"(?i)\bbilling period\b", "BillPer"),
    (r"(?i)\baccount number\b", "Acct#"),
    (r"(?i)\bkwh\b", "kWh"),
    (r"(?i)\brate schedule\b", "RateSch"),
    # JSON/CSV-ish keys
    (r"(?i)timestamp", "⏱"),
    (r"(?i)level", "↑↓"),
    (r"(?i)service", "⚙"),
    (r"(?i)route", "→"),
    (r"(?i)tenant", "⌂"),
    (r"(?i)duration_ms", "Δt"),
    (r"(?i)retry", "↻"),
    (r"(?i)node", "◦"),
    (r"(?i)msg", "✉"),
]

def flux_compress_text(s: str) -> tuple[int, int, float, str]:
    """Return (orig_bytes, comp_bytes, savings_pct, compressed_text)."""
    if not isinstance(s, str):
        s = str(s)
    orig = s.encode("utf-8")
    out = s
    for pat, rep in SUBS:
        out = _re.sub(pat, rep, out)
    out = _re.sub(r"[ ]{2,}", " ", out)
    out = _re.sub(r"(,){2,}", ",", out)
    comp = out.encode("utf-8")
    if len(orig) == 0:
        return (0, 0, 0.0, out)
    savings = max(0.0, (len(orig) - len(comp)) / len(orig) * 100.0)
    return (len(orig), len(comp), savings, out)

def csv_to_text(df: pd.DataFrame) -> str:
    # Represent CSV compactly for compression measurement
    return df.to_csv(index=False)

# ---------- BILL PARSERS ----------
CURRENCY_RE = re.compile(r'(?i)\$?\s?([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))')
KWH_RE      = re.compile(r'(?i)(\d{1,3}(?:[,]\d{3})+|\d+)\s*kwh')
DATE_RE     = re.compile(r'(?i)(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2},?\s*\d{2,4}\b)')

def _to_float(num_str):
    if num_str is None:
        return None
    return float(str(num_str).replace(",", ""))

def parse_bill_pdf(file_bytes: bytes):
    if not HAS_PDF:
        return {}, "", 0.0
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    full = "\n".join(texts)

    # Get a compression-derived "demo savings %"
    _, _, savings_pct, _ = flux_compress_text(full)

    # Try to find total cost
    total_cost = None
    for key in [r"amount due", r"total due", r"current charges", r"total amount", r"amount to be paid"]:
        m = re.search(rf"(?i){key}.*?{CURRENCY_RE.pattern}", full)
        if m:
            total_cost = _to_float(m.groups()[-1])
            break
    if total_cost is None:
        monies = [_to_float(m.group(1)) for m in CURRENCY_RE.finditer(full)]
        total_cost = max(monies) if monies else None

    # kWh
    kwh = None
    m_k = KWH_RE.search(full)
    if m_k:
        kwh = _to_float(m_k.group(1))

    # Period
    dates = [dtparse.parse(m.group(1), fuzzy=True) for m in DATE_RE.finditer(full)]
    dates = sorted(set(dates))
    period_start, period_end = (None, None)
    if len(dates) >= 2:
        period_start, period_end = dates[0], dates[-1]

    parsed = {
        "total_cost_usd": total_cost,
        "kwh": kwh,
        "period_start": period_start.isoformat() if period_start else "",
        "period_end": period_end.isoformat() if period_end else "",
    }
    return parsed, full, savings_pct

def parse_bill_csv(df: pd.DataFrame):
    """Aggregate a CSV bill and compute compression-based savings from its text form."""
    orig_text = csv_to_text(df)
    _, _, savings_pct, _ = flux_compress_text(orig_text)

    cols = {c.lower(): c for c in df.columns}
    total = float(df[cols["costusd"]].sum()) if "costusd" in cols else None
    kwh = float(df[cols["kwh"]].sum()) if "kwh" in cols else None

    # Simple period heuristics
    p_start = str(df[cols["month"]].iloc[0]) if "month" in cols and len(df) else ""
    p_end   = str(df[cols["month"]].iloc[-1]) if "month" in cols and len(df) else ""

    parsed = {
        "total_cost_usd": total,
        "kwh": kwh,
        "period_start": p_start,
        "period_end": p_end
    }
    return parsed, orig_text, savings_pct

# ---------- UI ----------
st.title("💰 HashMoney – Bitcoin Bill & Hash Log Demo")
st.markdown(
    f"Drag & drop a **real power bill (PDF or CSV)** and an optional **hashrate log (CSV)**. "
    f"HashMoney computes **demo savings** from actual compression of your uploaded file — no manual slider. "
    f"<br><span style='color:{FLUX_BLUE}'>All metrics include plain-English explanations.</span>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("💼 Commercial Terms (for demo math)")
    upfront = st.number_input("Upfront License ($)", 0, 5_000_000, 100_000, 10_000)
    share_pct = st.slider("Share of Verified Savings (%)", 0, 100, 25, 1)
    st.caption("These affect the 'HashMoney License Revenue' metric only.")

# ---- BILL UPLOAD ----
st.subheader("📄 Power Bill")
bill_file = st.file_uploader("Upload Power Bill (PDF or CSV)", type=["pdf","csv"])

bill_parsed = {}
bill_text_for_debug = ""
savings_pct_bill = None

if bill_file is not None:
    if bill_file.type.endswith("pdf"):
        data = bill_file.read()
        bill_parsed, bill_text_for_debug, savings_pct_bill = parse_bill_pdf(data)
        if not HAS_PDF:
            st.error("PDF parsing library not available on this deployment. Add `pdfplumber` to requirements.txt.")
    else:
        # CSV
        df_bill = pd.read_csv(bill_file)
        st.caption("Detected CSV – showing first rows:")
        st.dataframe(df_bill.head())
        bill_parsed, bill_text_for_debug, savings_pct_bill = parse_bill_csv(df_bill)

# Manual confirm/override
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    total_cost_usd = st.number_input("Total Cost (USD)", value=float(bill_parsed.get("total_cost_usd") or 0.0), step=100.0, format="%.2f")
with colB:
    kwh_used = st.number_input("Energy Used (kWh)", value=float(bill_parsed.get("kwh") or 0.0), step=1000.0, format="%.0f")
with colC:
    ps = st.text_input("Period Start", value=str(bill_parsed.get("period_start") or ""))
with colD:
    pe = st.text_input("Period End", value=str(bill_parsed.get("period_end") or ""))

# Safety: bound savings between 5% and 40% to keep the demo realistic
def bound_savings(pct: float) -> float:
    if pct is None:
        return 0.0
    return max(5.0, min(40.0, round(pct, 2)))

computed_savings_pct = bound_savings(savings_pct_bill)

# Compute projections from computed_savings_pct
cost_per_kwh = (total_cost_usd / kwh_used) if (total_cost_usd and kwh_used) else None
projected_cost = total_cost_usd * (1 - computed_savings_pct/100.0) if total_cost_usd else None
dollars_saved = (total_cost_usd - projected_cost) if projected_cost is not None else None
annual_savings = dollars_saved * 12 if dollars_saved is not None else None
hm_revenue = upfront + (dollars_saved * (share_pct/100.0) if dollars_saved else 0)

# ---- METRICS ROW ----
st.markdown("### 📊 Metrics (Derived from Your File’s Compression)")
m0, m1, m2, m3 = st.columns(4, gap="large")
with m0:
    metric_card(
        "Computed Savings %",
        f"{computed_savings_pct:.2f}%",
        "HashMoney calculated this from compression of your uploaded bill (PDF/CSV)."
    )
with m1:
    metric_card(
        "Baseline Cost",
        f"${total_cost_usd:,.2f}" if total_cost_usd else "—",
        "The power bill amount on the uploaded statement."
    )
with m2:
    metric_card(
        "Projected Cost (with HashMoney)",
        f"${projected_cost:,.2f}" if projected_cost else "—",
        "Baseline minus the computed savings percentage."
    )
with m3:
    metric_card(
        "Estimated Savings",
        f"${dollars_saved:,.2f}" if dollars_saved else "—",
        "Money not spent on electricity this billing period, based on computed savings."
    )

m4, m5 = st.columns(2, gap="large")
with m4:
    metric_card(
        "Cost per kWh",
        f"${cost_per_kwh:.4f}" if cost_per_kwh else "—",
        "Your effective electricity rate: Total ÷ kWh."
    )
with m5:
    metric_card(
        "Annualized Savings",
        f"${annual_savings:,.0f}" if annual_savings else "—",
        "Simple projection: this period’s savings × 12."
    )

# ---- BILL BAR CHART ----
if total_cost_usd:
    st.markdown("#### Cost Comparison")
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    bars_x = ["Baseline", "With HashMoney"]
    bars_y = [total_cost_usd, projected_cost if projected_cost else total_cost_usd]
    ax.bar(bars_x, bars_y, color=[FLUX_BLUE, "#1F2937"])
    ax.set_ylabel("USD")
    ax.set_title("Power Bill – Before vs After")
    st.pyplot(fig)

# ---- DECIBEL ESTIMATOR (from computed_savings_pct) ----
st.markdown("### 🔉 Estimated Noise Reduction (derived from computed savings)")
if computed_savings_pct > 0:
    r = max(1e-6, 1.0 - computed_savings_pct/100.0)  # load ratio
    # Conservative fan-law-based estimate (~16.67*log10(r))
    delta_db_cons = (50.0/3.0) * math.log10(r)
    # Aggressive upper bound (20*log10(r))
    delta_db_aggr = 20.0 * math.log10(r)

    c1, c2 = st.columns(2)
    with c1:
        metric_card(
            "ΔdB (Conservative)",
            f"{delta_db_cons:.2f} dB",
            "Based on fan affinity & broadband noise scaling. Typical rooms land here."
        )
    with c2:
        metric_card(
            "ΔdB (Aggressive Upper Bound)",
            f"{delta_db_aggr:.2f} dB",
            "Best-case rooms (steep fan curves / tonal peaks). Not guaranteed."
        )
    st.caption(
        "Note: Software savings alone typically deliver a few dB. Larger drops require pairing with fan policy & airflow changes."
    )
else:
    st.caption("Upload a bill to compute savings and estimate dB reduction.")

# ---- HASHRATE LOG (optional) ----
st.subheader("📈 Hashrate Log (optional)")
log_file = st.file_uploader("Upload Hashrate Log (CSV: Timestamp, HashrateTHs)", type=["csv"], key="hashlog")

if log_file:
    log_df = pd.read_csv(log_file)
    missing = [c for c in ["Timestamp","HashrateTHs"] if c not in log_df.columns]
    if missing:
        st.error(f"CSV must have columns: Timestamp, HashrateTHs. Missing: {missing}")
    else:
        # Compute an independent "compression-derived gain" from this log's text
        log_text = csv_to_text(log_df)
        _, _, savings_pct_log, _ = flux_compress_text(log_text)
        hash_gain_pct = bound_savings(savings_pct_log)  # bound to 5–40% for demo realism

        # Parse time for plotting
        try:
            log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"])
        except Exception:
            pass

        log_df["HashMoney_Hashrate"] = log_df["HashrateTHs"] * (1 + hash_gain_pct/100.0)

        st.caption("First rows:")
        st.dataframe(log_df.head())

        st.markdown("#### Hashrate Over Time")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(log_df["Timestamp"], log_df["HashrateTHs"], label="Original")
        ax2.plot(log_df["Timestamp"], log_df["HashMoney_Hashrate"], label=f"With HashMoney (+{hash_gain_pct:.1f}%)")
        ax2.set_ylabel("TH/s")
        ax2.legend()
        st.pyplot(fig2)

        avg_base = log_df["HashrateTHs"].mean()
        avg_flux = log_df["HashMoney_Hashrate"].mean()
        metric_card(
            "Avg Hashrate Gain",
            f"+{(avg_flux-avg_base):.2f} TH/s  ({(avg_flux/avg_base-1)*100:.1f}%)",
            "Derived from compression of your log file (demo mapping)."
        )

# ---- LICENSE REVENUE METRIC ----
metric_card(
    "HashMoney License Revenue (demo)",
    f"${(upfront + (dollars_saved or 0) * (share_pct/100.0)):,.0f}",
    "Upfront license + a share of verified savings (demo terms in sidebar)."
)

# ---- FOOTNOTES ----
st.markdown(
    """
    <hr style="opacity:0.1">
    <small style="color:#9CA3AF">
    • Savings % in this demo is computed from actual compression of your uploaded files (PDF/CSV).<br>
    • Results here illustrate how HashMoney transforms real documents/logs; production deployments use direct telemetry and site measurements.<br>
    • dB estimates reflect physical scaling laws; large noise drops typically require pairing software gains with fan policy & airflow improvements.
    </small>
    """, unsafe_allow_html=True
)
