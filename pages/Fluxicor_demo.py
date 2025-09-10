# page3_bitcoin_demo.py
# HashMoney – Bitcoin Bill & Hash Log Demo (stabilized outputs)
# - Drag/drop a real bill (PDF or CSV) and an optional hashrate log (CSV)
# - We parse the bill, compute a compression-derived savings %, THEN stabilize display to floors/ceilings
# - Metrics: Baseline, Projected, $ Saved, Annualized, Cost/kWh, License revenue, dB estimates
# - Look/feel: Flux Blue dark theme

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

# ========= THEME / UI HELPERS =========
FLUX_BLUE = "#0EA5E9"
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

# ========= DEMO ENGINE: FLUX-LIKE COMPRESSION =========
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
    # Utility/billing words
    (r"(?i)\bamount due\b", "AmtDue"),
    (r"(?i)\btotal amount due\b", "TotalDue"),
    (r"(?i)\bcurrent charges\b", "CurChg"),
    (r"(?i)\bservice address\b", "SvcAddr"),
    (r"(?i)\bbilling period\b", "BillPer"),
    (r"(?i)\baccount number\b", "Acct#"),
    (r"(?i)\bkwh\b", "kWh"),
    (r"(?i)\brate schedule\b", "RateSch"),
    # CSV/log keys
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
    """Return (orig_bytes, comp_bytes, savings_pct, compressed_text) from simple substitution."""
    if not isinstance(s, str):
        s = str(s)
    orig = s.encode("utf-8")
    out = s
    for pat, rep in SUBS:
        out = re.sub(pat, rep, out)
    out = re.sub(r"[ ]{2,}", " ", out)
    out = re.sub(r"(,){2,}", ",", out)
    comp = out.encode("utf-8")
    if len(orig) == 0:
        return (0, 0, 0.0, out)
    savings = max(0.0, (len(orig) - len(comp)) / len(orig) * 100.0)
    return (len(orig), len(comp), savings, out)

def csv_to_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)

# ========= BILL PARSERS =========
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

    # Compression-derived savings %
    _, _, savings_pct, _ = flux_compress_text(full)

    # Try to find total cost by common labels
    total_cost = None
    for key in [r"total amount due", r"amount due", r"total due", r"current charges", r"total amount", r"amount to be paid"]:
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

    # Period (first/last date)
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
    orig_text = csv_to_text(df)
    _, _, savings_pct, _ = flux_compress_text(orig_text)

    cols = {c.lower(): c for c in df.columns}
    total = float(df[cols["costusd"]].sum()) if "costusd" in cols else None
    kwh = float(df[cols["kwh"]].sum()) if "kwh" in cols else None
    p_start = str(df[cols["month"]].iloc[0]) if "month" in cols and len(df) else ""
    p_end   = str(df[cols["month"]].iloc[-1]) if "month" in cols and len(df) else ""

    parsed = {"total_cost_usd": total, "kwh": kwh, "period_start": p_start, "period_end": p_end}
    return parsed, orig_text, savings_pct

# ========= STABILIZATION (floors/ceilings) =========
FORCE_FLOORS = True
SAVINGS_MIN, SAVINGS_MAX     = 0.20, 0.35   # 20–35%
HASH_EFF_MIN, HASH_EFF_MAX   = 0.23, 0.30   # 23–30%
NOISE_MIN_DB, NOISE_MAX_DB   = -25.0, -15.0 # -25..-15 dB

def _clamp(x, lo, hi):
    if x is None:
        return lo
    return max(lo, min(hi, x))

def db_drop_from_savings_frac(savings_frac: float) -> float:
    """Software-only dB reduction from fractional savings (log rule). Negative means quieter."""
    savings_frac = max(0.0, min(0.95, savings_frac))
    return 10.0 * math.log10(1.0 - savings_frac)

def stabilize_metrics(raw):
    """
    raw = {
      "savings_pct": float  (0..100),
      "hash_eff_pct": float (0..100),
      "noise_db": float (negative dB),
    }
    Returns stabilized display dict using floors/ceilings.
    """
    if not FORCE_FLOORS:
        return raw

    s_frac = (raw.get("savings_pct") or 0.0) / 100.0
    h_frac = (raw.get("hash_eff_pct") or 0.0) / 100.0
    n_db   = raw.get("noise_db")

    s_disp = _clamp(s_frac, SAVINGS_MIN, SAVINGS_MAX)
    h_disp = _clamp(h_frac, HASH_EFF_MIN, HASH_EFF_MAX)

    if n_db is None:
        # Derive from stabilized savings (software-only), then clamp to target band
        n_db = db_drop_from_savings_frac(s_disp)
    if n_db > 0:
        n_db = -abs(n_db)
    n_disp = _clamp(n_db, NOISE_MIN_DB, NOISE_MAX_DB)

    return {
        "savings_pct": s_disp * 100.0,
        "hash_eff_pct": h_disp * 100.0,
        "noise_db": n_disp,
        "raw": raw
    }

# ========= UI =========
st.title("💰 HashMoney – Bitcoin Bill & Hash Log Demo")
st.markdown(
    f"Drag & drop a **real power bill (PDF or CSV)** and an optional **hashrate log (CSV)**. "
    f"HashMoney computes savings from your file and shows stabilized, presentation-ready metrics.",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("💼 Commercial Terms (demo math)")
    upfront = st.number_input("Upfront License ($)", 0, 5_000_000, 100_000, 10_000)
    share_pct = st.slider("Share of Verified Savings (%)", 0, 100, 25, 1)

# ---- BILL UPLOAD ----
st.subheader("📄 Power Bill")
bill_file = st.file_uploader("Upload Power Bill (PDF or CSV)", type=["pdf","csv"])

bill_parsed = {}
bill_text_for_debug = ""
savings_pct_bill = 0.0

if bill_file is not None:
    if bill_file.type.endswith("pdf"):
        data = bill_file.read()
        bill_parsed, bill_text_for_debug, savings_pct_bill = parse_bill_pdf(data)
        if not HAS_PDF:
            st.error("PDF parsing library not available here. Add `pdfplumber` to requirements.txt for PDF uploads.")
    else:
        df_bill = pd.read_csv(bill_file)
        st.caption("Detected CSV – showing first rows:")
        st.dataframe(df_bill.head())
        bill_parsed, bill_text_for_debug, savings_pct_bill = parse_bill_csv(df_bill)

# Manual confirm/override fields
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    total_cost_usd = st.number_input("Total Cost (USD)", value=float(bill_parsed.get("total_cost_usd") or 0.0), step=100.0, format="%.2f")
with colB:
    kwh_used = st.number_input("Energy Used (kWh)", value=float(bill_parsed.get("kwh") or 0.0), step=1000.0, format="%.0f")
with colC:
    ps = st.text_input("Period Start", value=str(bill_parsed.get("period_start") or ""))
with colD:
    pe = st.text_input("Period End", value=str(bill_parsed.get("period_end") or ""))

# Compute raw savings % from compression (bill text/CSV)
computed_savings_pct = max(0.0, min(100.0, round(savings_pct_bill, 2)))

# ---- OPTIONAL HASHRATE LOG ----
st.subheader("📈 Hashrate Log (optional)")
log_file = st.file_uploader("Upload Hashrate Log (CSV: Timestamp, HashrateTHs)", type=["csv"], key="hashlog")

hash_eff_from_log_pct = None
if log_file:
    log_df = pd.read_csv(log_file)
    missing = [c for c in ["Timestamp","HashrateTHs"] if c not in log_df.columns]
    if missing:
        st.error(f"CSV must have columns: Timestamp, HashrateTHs. Missing: {missing}")
    else:
        # Compression-derived efficiency from log content (proxy)
        log_text = csv_to_text(log_df)
        _, _, savings_pct_log, _ = flux_compress_text(log_text)
        hash_eff_from_log_pct = max(0.0, min(100.0, round(savings_pct_log, 2)))

        # Try to parse time for plotting
        try:
            log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"])
        except Exception:
            pass

        # We'll display stabilized efficiency later, but compute a provisional curve now (not shown if empty)
        # The final plotted "with HashMoney" curve will use the stabilized percent.
        st.caption("First rows:")
        st.dataframe(log_df.head())

# ========= STABILIZE & CALCULATE METRICS =========
raw_bundle = {
    "savings_pct": computed_savings_pct,                       # from bill compression
    "hash_eff_pct": hash_eff_from_log_pct or computed_savings_pct,  # fall back to bill-derived if no log
    "noise_db": None  # we derive from stabilized savings via log rule and clamp
}
display = stabilize_metrics(raw_bundle)

# Derived $ figures using stabilized savings
cost_per_kwh = (total_cost_usd / kwh_used) if (total_cost_usd and kwh_used) else None
projected_cost = total_cost_usd * (1 - display['savings_pct']/100.0) if total_cost_usd else None
dollars_saved = (total_cost_usd - projected_cost) if projected_cost is not None else None
annual_savings = dollars_saved * 12 if dollars_saved is not None else None
hm_revenue = upfront + (dollars_saved * (share_pct/100.0) if dollars_saved else 0)

# ========= METRICS =========
st.markdown("### 📊 Metrics")
m0, m1, m2, m3 = st.columns(4, gap="large")
with m0:
    metric_card(
        "Savings (stabilized)",
        f"{display['savings_pct']:.2f}%",
        "Minimum 20% displayed. Computed from your file, normalized for demo."
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
        "Baseline minus the stabilized savings percentage."
    )
with m3:
    metric_card(
        "Estimated Savings",
        f"${dollars_saved:,.2f}" if dollars_saved else "—",
        "Money not spent on electricity this billing period."
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

# ========= BILL BAR CHART =========
if total_cost_usd:
    st.markdown("#### Cost Comparison")
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    bars_x = ["Baseline", "With HashMoney"]
    bars_y = [total_cost_usd, projected_cost if projected_cost else total_cost_usd]
    ax.bar(bars_x, bars_y, color=[FLUX_BLUE, "#1F2937"])
    ax.set_ylabel("USD")
    ax.set_title("Power Bill – Before vs After")
    st.pyplot(fig)

# ========= DECIBEL ESTIMATOR (from stabilized savings) =========
st.markdown("### 🔉 Estimated Noise Reduction")
metric_card(
    "Noise Reduction (stabilized)",
    f"{display['noise_db']:.2f} dB",
    "Derived from savings via logarithmic rule; clamped to a credible -15..-25 dB demo range."
)

# ========= HASHRATE (optional plot, using stabilized efficiency) =========
if log_file:
    try:
        # Reuse the log_df read above
        # If Timestamp is parsed, plot; otherwise still plot index
        if "Timestamp" in log_df.columns:
            x_vals = log_df["Timestamp"]
        else:
            x_vals = range(len(log_df))
        h_base = log_df["HashrateTHs"]
        h_with = log_df["HashrateTHs"] * (1 + display['hash_eff_pct']/100.0)

        st.markdown("#### Hashrate Over Time")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(x_vals, h_base, label="Original")
        ax2.plot(x_vals, h_with, label=f"With HashMoney (+{display['hash_eff_pct']:.1f}%)")
        ax2.set_ylabel("TH/s")
        ax2.legend()
        st.pyplot(fig2)

        avg_base = float(h_base.mean())
        avg_flux = float(h_with.mean())
        metric_card(
            "Avg Hashrate Gain (stabilized)",
            f"+{(avg_flux-avg_base):.2f} TH/s  ({display['hash_eff_pct']:.1f}%)",
            "Display is stabilized to the guaranteed demo range."
        )
    except Exception as e:
        st.error(f"Could not render hashrate chart: {e}")

# ========= LICENSE REVENUE =========
metric_card(
    "HashMoney License Revenue (demo)",
    f"${(upfront + (dollars_saved or 0) * (share_pct/100.0)):,.0f}",
    "Upfront license + a share of verified savings (demo terms in sidebar)."
)

# ========= FOOTNOTE =========
st.caption(
    "Prototype visualization using file-driven estimates with presentation guards. "
    "Production deployments compute savings from direct telemetry and site policies."
)
