
import re
from collections import Counter
from typing import Dict, Optional

DEFAULT_SYMBOL_MAP: Dict[str, str] = {
    "responsibility": "=",
    "pain": "!",
    "eternity": "§",
    "divide": "÷",
    "converge": "Λ",
    "synthesis": "+",
    "stabilize": "≡",
    "silence": "∅",
    "transform": "Δ",
    "ghost": "✻",
    "memory": "⧈",
    "multivalence": "⧉",
    "recursion": "N",
    "chaos": "c",
    "question": "?",
    "becoming": "→",
    "return": "←",
}

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _apply_symbol_map(text: str, symbol_map: Dict[str, str]) -> str:
    keys = sorted(symbol_map.keys(), key=len, reverse=True)
    out = text
    for k in keys:
        out = re.sub(rf"(?i)\b{re.escape(k)}\b", symbol_map[k], out)
    return out

def _token_compress(s: str, top_n: int = 50):
    words = s.split(" ")
    if len(words) < 3:
        return s, {}
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    freq = Counter(bigrams).most_common(top_n)
    codebook = {bg: f"⟦{i}⟧" for i,(bg,_) in enumerate(freq)}
    compressed = s
    for bg, token in codebook.items():
        compressed = compressed.replace(bg, token)
    return compressed, codebook

def _token_expand(s: str, codebook: dict):
    out = s
    for bg, token in codebook.items():
        out = out.replace(token, bg)
    return out

def flux_compress_text(text: str, symbol_map: Optional[Dict[str, str]] = None):
    symbol_map = symbol_map or DEFAULT_SYMBOL_MAP
    original = text
    original_bytes = len(original.encode("utf-8"))
    norm = _normalize_text(original)
    symbolic = _apply_symbol_map(norm, symbol_map)
    compressed, codebook = _token_compress(symbolic)
    reconstructed = _token_expand(compressed, codebook)
    return {
        "original_bytes": original_bytes,
        "compressed_bytes": len(compressed.encode("utf-8")),
        "compressed_text": compressed,
        "reconstructed_text": reconstructed,
        "codebook": codebook,
    }

def estimate_savings(
    compression_ratio: float,
    gb_per_day: float,
    gpu_kwh_per_gb: float,
    pue: float,
    kwh_cost: float,
    baseline_reduction_pct: float = 0.0,
    manual_savings_override_pct: Optional[float] = None,
    wue_l_per_kwh: float = 0.0,
    water_cost_per_kgal: float = 0.0,
):
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))
    effective_intensity = gpu_kwh_per_gb * (1 - clamp01(baseline_reduction_pct))
    orig_kwh_per_day = gb_per_day * effective_intensity * pue
    if manual_savings_override_pct is not None:
        flux_pct = clamp01(manual_savings_override_pct)
    else:
        flux_pct = clamp01(compression_ratio)
    new_kwh_per_day = orig_kwh_per_day * (1 - flux_pct)
    kwh_day_saved = max(0.0, orig_kwh_per_day - new_kwh_per_day)
    kwh_year_saved = kwh_day_saved * 365.0
    usd_elec_saved = kwh_year_saved * kwh_cost
    liters_saved = kwh_year_saved * max(0.0, wue_l_per_kwh)
    kgal_saved = liters_saved / 3785.41 if liters_saved > 0 else 0.0
    usd_water_saved = kgal_saved * max(0.0, water_cost_per_kgal)
    total_usd = usd_elec_saved + usd_water_saved
    return {
        "kwh_year_saved": kwh_year_saved,
        "usd_electric_year_saved": usd_elec_saved,
        "water_kgal_year_saved": kgal_saved,
        "usd_water_year_saved": usd_water_saved,
        "usd_year_saved": total_usd,
        "assumptions": {
            "effective_intensity_kwh_per_gb": effective_intensity,
            "flux_savings_pct_used": flux_pct,
        }
    }
