"""
utils/decision_analysis.py
────────────────────────────────────────────────────────────────────────────────
Multi-Criteria Decision Analysis — AHP (Analytic Hierarchy Process)
─────────────────────────────────────────────────────────────────────
Implements Saaty's (1980) Analytic Hierarchy Process to prioritise
cybersecurity incidents for analyst response.

Criteria (5 dimensions)
────────────────────────
  C1  risk_index       — Likelihood × Impact composite score
  C2  impact_score     — DL-weighted severity (0–1)
  C3  likelihood       — Historical frequency of this attack type
  C4  risk_mc_p95      — Monte Carlo worst-case estimate (P95)
  C5  sector_weight    — Criticality of the affected sector

AHP Steps
──────────
  1. Build 5×5 pairwise comparison matrix (expert judgement, Saaty scale 1–9)
  2. Normalise matrix column-by-column
  3. Derive priority weights via row averaging (principal eigenvector approx.)
  4. Check Consistency Ratio (CR) — must be < 0.10
  5. Score each incident as weighted sum of normalised criteria values
  6. Map scores to response priority tiers

Public API
──────────
  run_ahp(df)           → df enriched with ahp_score, priority_tier, recommendation
  ahp_weights()         → dict {criterion: weight}
  consistency_ratio()   → float
  weight_table()        → DataFrame for display
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ── Criteria ──────────────────────────────────────────────────────────────────
CRITERIA = [
    "risk_index",
    "impact_score",
    "likelihood",
    "risk_mc_p95",
    "sector_weight",
]

# ── AHP Pairwise Comparison Matrix (5 × 5) ───────────────────────────────────
# Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
# Row/Col order matches CRITERIA list above
_AHP_MATRIX = np.array([
    # risk   impact  like   mc95   sector
    [1,      3,      3,     2,     4    ],   # risk_index   — most important
    [1/3,    1,      2,     2,     3    ],   # impact_score
    [1/3,    1/2,    1,     2,     3    ],   # likelihood
    [1/2,    1/2,    1/2,   1,     2    ],   # risk_mc_p95
    [1/4,    1/3,    1/3,   1/2,   1    ],   # sector_weight — least important
], dtype=float)

# Saaty Random Index for n=5
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

# ── Sector criticality lookup ─────────────────────────────────────────────────
_SECTOR_WEIGHT_MAP = {
    "government": 1.0, "finance": 1.0, "banking": 1.0,
    "healthcare": 1.0, "energy": 1.0,  "defense": 1.0,
    "critical infrastructure": 1.0,
    "telecommunications": 0.75, "manufacturing": 0.75,
    "digital": 0.75, "it service": 0.75, "media": 0.75,
    "retail": 0.50, "education": 0.50, "tourism": 0.50,
}


# ═════════════════════════════════════════════════════════════════════════════
#  AHP core
# ═════════════════════════════════════════════════════════════════════════════

def _compute_weights(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Derive AHP priority weights and Consistency Ratio.
    Uses the normalised column averaging method (Saaty 1980).
    """
    n       = matrix.shape[0]
    col_sum = matrix.sum(axis=0)
    norm    = matrix / col_sum
    weights = norm.mean(axis=1)

    lam_max = float((matrix @ weights / weights).mean())
    ci      = (lam_max - n) / (n - 1)
    ri      = _RI.get(n, 1.12)
    cr      = ci / ri if ri > 0 else 0.0
    return weights, round(cr, 4)


_WEIGHTS, _CR = _compute_weights(_AHP_MATRIX)
_WEIGHT_DICT  = {c: round(float(w), 4) for c, w in zip(CRITERIA, _WEIGHTS)}


def ahp_weights() -> Dict[str, float]:
    """Return AHP criteria weights as a dict."""
    return _WEIGHT_DICT.copy()


def consistency_ratio() -> float:
    """Return the Consistency Ratio of the pairwise matrix."""
    return _CR


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _sector_weight(text: str) -> float:
    t = str(text).lower()
    for kw, w in _SECTOR_WEIGHT_MAP.items():
        if kw in t:
            return w
    return 0.35


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in any missing criteria columns with sensible defaults."""
    df = df.copy()
    defaults = {
        "risk_index":    0.20,
        "impact_score":  0.45,
        "likelihood":    0.40,
        "risk_mc_p95":   0.25,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    sector_col = next(
        (c for c in ("category", "sector", "incident_category") if c in df.columns),
        None,
    )
    df["sector_weight"] = (
        df[sector_col].apply(_sector_weight) if sector_col else 0.35
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Main AHP function
# ═════════════════════════════════════════════════════════════════════════════

def run_ahp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every incident using AHP.

    Adds columns:
      ahp_score       — weighted priority score (0–1, higher = more critical)
      priority_tier   — Critical / High / Medium / Monitor
      recommendation  — plain-language response guidance
    """
    df = _ensure_columns(df)

    X = df[CRITERIA].values.astype(float)

    # Min-max normalise each criterion to [0, 1]
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    scale   = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    X_norm  = (X - col_min) / scale

    # Weighted sum
    scores = X_norm @ _WEIGHTS

    # Re-normalise scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    df["ahp_score"] = np.round(scores, 4)

    # Priority tiers
    def _tier(s: float) -> str:
        if s >= 0.75: return "Critical"
        if s >= 0.50: return "High"
        if s >= 0.30: return "Medium"
        return "Monitor"

    def _recommend(s: float) -> str:
        if s >= 0.75:
            return "🔴 Immediate — escalate to SOC within 1 hour"
        if s >= 0.50:
            return "🟠 High Priority — investigate within 24 hours"
        if s >= 0.30:
            return "🟡 Medium — schedule review within 72 hours"
        return "🟢 Monitor — include in next weekly report"

    df["priority_tier"]    = df["ahp_score"].apply(_tier)
    df["recommendation"]   = df["ahp_score"].apply(_recommend)

    return df.sort_values("ahp_score", ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
#  Weight table for UI
# ═════════════════════════════════════════════════════════════════════════════

def weight_table() -> pd.DataFrame:
    rows = []
    for crit, w in _WEIGHT_DICT.items():
        rows.append({
            "Criterion":  crit.replace("_", " ").title(),
            "Weight":     w,
            "Weight (%)": f"{w * 100:.1f}%",
            "Meaning":    _criterion_meaning(crit),
        })
    return pd.DataFrame(rows).sort_values("Weight", ascending=False)


def _criterion_meaning(c: str) -> str:
    m = {
        "risk_index":    "Likelihood × Impact composite",
        "impact_score":  "Severity weighted by DL confidence",
        "likelihood":    "Historical frequency of this attack type",
        "risk_mc_p95":   "Worst-case Monte Carlo estimate (P95)",
        "sector_weight": "Criticality of affected sector",
    }
    return m.get(c, "")
