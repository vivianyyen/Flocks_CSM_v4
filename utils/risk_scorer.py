"""
utils/risk_scorer.py
────────────────────────────────────────────────────────────────────────────────
Custom Risk / Impact-Level Scoring Engine
Formula:
    impact_score = w1(sector) + w2(country) + w3(attack_type) + w4(data_exposure)

Thresholds (normalised 0–1):
    > 0.65  → Critical
    > 0.50  → High
    > 0.35  → Medium
    ≤ 0.35  → Low

Max achievable score is ~0.815 (all components maxed), so thresholds are
calibrated to spread incidents meaningfully across all four bands.
────────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
from typing import Tuple

# ── Weights (must sum to 1.0) ─────────────────────────────────────────────────
W1_SECTOR        = 0.25
W2_COUNTRY       = 0.20
W3_ATTACK_TYPE   = 0.35   # highest — attack type drives severity most
W4_DATA_EXPOSURE = 0.20

assert abs(W1_SECTOR + W2_COUNTRY + W3_ATTACK_TYPE + W4_DATA_EXPOSURE - 1.0) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  W1 — SECTOR  (0.2 / 0.4 / 0.6 / 0.8)
# ══════════════════════════════════════════════════════════════════════════════
_SECTOR_TIER1 = [
    "government", "federal", "ministry", "parliament", "pdrm", "police",
    "financial service", "finance", "banking", "bank", "bursa", "bnm",
    "defense", "defence", "military", "armed forces",
    "healthcare", "hospital", "clinic", "health", "medical",
    "energy", "utility", "utilities", "power grid", "nuclear", "petronas",
    "water", "critical infrastructure",
]
_SECTOR_TIER2 = [
    "manufacturing", "construction", "transportation", "logistics",
    "supply chain", "building automation",
    "digital", "information technology", "it service", "cyber",
    "telecommunication", "telecom", "telco", "maxis", "celcom", "digi", "unifi",
    "media", "broadcast", "news agency",
]
_SECTOR_TIER3 = [
    "consumer", "retail", "e-commerce", "ecommerce", "shopping",
    "product", "service", "industrial", "plantation", "agriculture",
    "property", "real estate", "education", "university", "school",
    "tourism", "hotel", "restaurant",
]

def score_sector(text: str) -> float:
    t = str(text).lower()
    if any(kw in t for kw in _SECTOR_TIER1): return 0.8
    if any(kw in t for kw in _SECTOR_TIER2): return 0.6
    if any(kw in t for kw in _SECTOR_TIER3): return 0.4
    return 0.2


# ══════════════════════════════════════════════════════════════════════════════
#  W2 — COUNTRY  (0.3 / 0.5 / 0.7)
# ══════════════════════════════════════════════════════════════════════════════
_MALAYSIA_KW = [
    "malaysia", "malaysian", "kuala lumpur", " kl ", "putrajaya",
    "cyberjaya", "sabah", "sarawak", "penang", "johor", "selangor",
]
_SEA_KW = [
    "singapore", "indonesia", "thailand", "philippines", "vietnam",
    "myanmar", "cambodia", "laos", "brunei", "timor",
    "southeast asia", "asean",
]

def score_country(text: str) -> float:
    t = str(text).lower()
    if any(kw in t for kw in _MALAYSIA_KW): return 0.7
    if any(kw in t for kw in _SEA_KW):     return 0.5
    return 0.3


# ══════════════════════════════════════════════════════════════════════════════
#  W3 — ATTACK TYPE  (0.2 / 0.5 / 0.9)
# ══════════════════════════════════════════════════════════════════════════════
_ATTACK_CRITICAL = [
    "ransomware", "data breach", "databreach", "data leak", "dataleak",
    "supply chain attack", "supply chain",
    "advanced persistent threat", "apt",
    "zero-day", "zero day", "0day", "0-day",
    "business email compromise", "bec",
    "remote code execution", "rce",
    "unauthorized privileged access", "privilege escalation",
    "critical infrastructure attack",
    "cyberattack", "cyber attack", "hack", "hacked", "hacking",
    "breach", "breached", "leaked", "exposed database", "database exposed",
    "stolen data", "data stolen", "credentials stolen",
    "intrusion", "compromised", "unauthorized access",
]
_ATTACK_MEDIUM = [
    "phishing", "spear phishing", "smishing", "vishing",
    "malware", "trojan", "virus", "worm", "keylogger",
    "credential theft", "credential stuffing", "brute force",
    "distributed denial", "ddos", "dos attack",
    "insider threat", "web application attack",
    "sql injection", "xss", "cross-site",
    "api abuse", "api exploit", "spyware", "adware",
    "cloud misconfiguration", "misconfiguration",
    "scam", "fraud", "phishing scam", "online scam",
    "identity theft", "impersonation",
    "sim swap", "account takeover",
]
_ATTACK_LOW = [
    "website defacement", "defacement",
    "spam", "botnet", "port scan", "scanning",
    "cryptojacking", "crypto mining",
    "reconnaissance", "recon",
    "low-level social engineering",
]

def score_attack_type(text: str) -> float:
    t = str(text).lower()
    if any(kw in t for kw in _ATTACK_CRITICAL): return 0.9
    if any(kw in t for kw in _ATTACK_MEDIUM):   return 0.5
    if any(kw in t for kw in _ATTACK_LOW):       return 0.2
    return 0.3

def classify_attack(text: str) -> str:
    t = str(text).lower()
    if any(kw in t for kw in _ATTACK_CRITICAL): return "Critical Attack"
    if any(kw in t for kw in _ATTACK_MEDIUM):   return "Medium Attack"
    if any(kw in t for kw in _ATTACK_LOW):       return "Low-Level Attack"
    return "Unclassified"


# ══════════════════════════════════════════════════════════════════════════════
#  W4 — DATA EXPOSURE  (0.3 / 0.5 / 0.8)
# ══════════════════════════════════════════════════════════════════════════════
_EXPOSURE_HIGH = [
    "identity theft", "privacy breach", "personally identifiable", "pii",
    "financial loss", "financial fraud", "integrity loss",
    "regulatory", "legal consequence", "fine", "gdpr", "pdpa",
    "lawsuit", "litigation",
    "data leak", "data breach", "sensitive data", "confidential data",
    "medical record", "health record",
    "credit card", "bank account", "password exposed", "password leaked",
    "personal data", "personal information", "ic number", "passport",
    "phone number leaked", "email leaked", "credentials exposed",
    "millions of", "thousands of records", "records exposed",
]
_EXPOSURE_MED = [
    "reputational damage", "reputation", "brand damage",
    "customer trust", "public disclosure", "media coverage",
    "negative publicity", "embarrassment", "investigation",
    "suspended", "taken down", "disrupted", "service outage",
]
_EXPOSURE_LOW = [
    "competitive disadvantage", "minor disruption", "performance impact",
    "service degradation", "limited impact", "low impact",
    "website down", "temporarily unavailable",
]

def score_data_exposure(text: str) -> float:
    t = str(text).lower()
    if any(kw in t for kw in _EXPOSURE_HIGH): return 0.8
    if any(kw in t for kw in _EXPOSURE_MED):  return 0.5
    if any(kw in t for kw in _EXPOSURE_LOW):  return 0.3
    return 0.3


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════════════════════
def compute_impact_score(
    sector_text:  str,
    country_text: str,
    attack_text:  str,
    summary_text: str,
) -> Tuple[float, str, dict]:
    combined_sector   = f"{sector_text} {summary_text}"
    combined_country  = f"{country_text} {summary_text}"
    combined_attack   = f"{attack_text} {summary_text}"
    combined_exposure = summary_text

    s1 = score_sector(combined_sector)
    s2 = score_country(combined_country)
    s3 = score_attack_type(combined_attack)
    s4 = score_data_exposure(combined_exposure)

    total = W1_SECTOR*s1 + W2_COUNTRY*s2 + W3_ATTACK_TYPE*s3 + W4_DATA_EXPOSURE*s4

    # Thresholds calibrated to max achievable score of ~0.815
    if   total > 0.65: label = "Critical"
    elif total > 0.50: label = "High"
    elif total > 0.35: label = "Medium"
    else:              label = "Low"

    breakdown = {
        "sector_score":        round(s1, 3),
        "country_score":       round(s2, 3),
        "attack_type_score":   round(s3, 3),
        "data_exposure_score": round(s4, 3),
        "weighted_total":      round(total, 4),
        "attack_class":        classify_attack(combined_attack),
    }
    return round(total, 4), label, breakdown


# ══════════════════════════════════════════════════════════════════════════════
#  DATAFRAME-LEVEL SCORER
# ══════════════════════════════════════════════════════════════════════════════
def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _pick(cols):
        for c in cols:
            if c in df.columns: return c
        return None

    sector_col  = _pick(["sector", "category", "incident_category"])
    country_col = _pick(["country", "nation", "location", "region"])
    attack_col  = _pick(["incident_type", "attack_type", "type", "threat_type"])
    summary_col = _pick(["summary", "description", "content"])

    scores, labels, breakdowns = [], [], []

    for _, row in df.iterrows():
        sector  = str(row.get(sector_col,  "")) if sector_col  else ""
        country = str(row.get(country_col, "")) if country_col else ""
        attack  = str(row.get(attack_col,  "")) if attack_col  else ""
        summary = str(row.get(summary_col, "")) if summary_col else ""

        score, label, bd = compute_impact_score(sector, country, attack, summary)
        scores.append(score)
        labels.append(label)
        breakdowns.append(bd)

    df["risk_score"]          = scores
    df["severity"]            = labels
    df["sector_score"]        = [b["sector_score"]        for b in breakdowns]
    df["country_score"]       = [b["country_score"]       for b in breakdowns]
    df["attack_type_score"]   = [b["attack_type_score"]   for b in breakdowns]
    df["data_exposure_score"] = [b["data_exposure_score"] for b in breakdowns]
    df["attack_class"]        = [b["attack_class"]        for b in breakdowns]

    return df


def build_update_payload(row: pd.Series) -> dict:
    return {
        "severity":            row["severity"],
        "risk_score":          float(row["risk_score"]),
        "sector_score":        float(row["sector_score"]),
        "country_score":       float(row["country_score"]),
        "attack_type_score":   float(row["attack_type_score"]),
        "data_exposure_score": float(row["data_exposure_score"]),
        "attack_class":        row["attack_class"],
    }
