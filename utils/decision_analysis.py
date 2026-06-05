"""
utils/decision_analysis.py
────────────────────────────────────────────────────────────────────────────────
Multi-Criteria Decision Analysis (MCDA) Layer
───────────────────────────────────────────────
Implements two complementary MCDA methods:

  1. AHP  — Analytic Hierarchy Process (Saaty 1980)
     Derives criteria weights from a pairwise comparison matrix and
     computes a priority score for each incident.

  2. TOPSIS — Technique for Order Preference by Similarity to Ideal Solution
     Ranks incidents by their geometric distance from the ideal best
     (closest) and ideal worst (farthest) solution vectors.

Criteria used (6 dimensions)
──────────────────────────────
  C1  risk_index        (from risk_analysis.py)   — higher is worse
  C2  impact_score      (DL-weighted severity)    — higher is worse
  C3  likelihood        (historical frequency)    — higher is worse
  C4  dl_confidence     (model certainty)         — higher = more reliable
  C5  risk_mc_p95       (worst-case MC estimate)  — higher is worse
  C6  sector_weight     (criticality of sector)   — higher is worse

AHP Pairwise Matrix (expert-defined, 6×6)
────────────────────────────────────────────
Relative importance:  risk_index > impact > likelihood > mc_p95 > sector > confidence
(Consistency Ratio checked; CR < 0.10 is acceptable)

Public API
──────────
  run_ahp(df)          → df with ahp_score column + weight_dict
  run_topsis(df)       → df with topsis_score, topsis_rank columns
  prioritise(df)       → df sorted by combined priority, top recommendations
  ahp_weights()        → dict of criteria → weight
  consistency_ratio()  → float
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# ── AHP Pairwise Comparison Matrix (6 × 6) ───────────────────────────────────
# Criteria order: risk_index, impact_score, likelihood, risk_mc_p95,
#                 sector_weight, dl_confidence
# Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme importance
_AHP_MATRIX = np.array([
    # risk  impact  like   mc95   sector  conf
    [1,     3,      3,     2,     4,      5   ],   # risk_index
    [1/3,   1,      2,     2,     3,      4   ],   # impact_score
    [1/3,   1/2,    1,     2,     3,      4   ],   # likelihood
    [1/2,   1/2,    1/2,   1,     2,      3   ],   # risk_mc_p95
    [1/4,   1/3,    1/3,   1/2,   1,      2   ],   # sector_weight
    [1/5,   1/4,    1/4,   1/3,   1/2,    1   ],   # dl_confidence
], dtype=float)

# Saaty's Random Index (RI) for n = 1..10
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

CRITERIA = [
    "risk_index", "impact_score", "likelihood",
    "risk_mc_p95", "sector_weight", "dl_confidence",
]

# Benefit (True) = higher is BETTER; Cost (False) = lower is BETTER
BENEFIT = {
    "risk_index":    False,
    "impact_score":  False,
    "likelihood":    False,
    "risk_mc_p95":   False,
    "sector_weight": False,
    "dl_confidence": True,
}

# Sector criticality lookup (matches risk_scorer tiers)
_SECTOR_WEIGHT = {
    "government": 1.0, "finance": 1.0, "banking": 1.0,
    "healthcare": 1.0, "energy": 1.0, "defense": 1.0,
    "critical infrastructure": 1.0,
    "telecommunications": 0.75, "manufacturing": 0.75,
    "digital": 0.75, "it service": 0.75, "media": 0.75,
    "retail": 0.50, "education": 0.50, "tourism": 0.50,
}


# ── AHP helpers ───────────────────────────────────────────────────────────────

def _ahp_weights(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Derive priority weights from a pairwise matrix using the
    eigenvector method, and compute the Consistency Ratio (CR).
    Returns (weights_array, CR).
    """
    n = matrix.shape[0]
    # Normalise each column then average rows
    col_sum = matrix.sum(axis=0)
    norm    = matrix / col_sum
    weights = norm.mean(axis=1)

    # Consistency check
    lam_max = (matrix @ weights / weights).mean()
    ci      = (lam_max - n) / (n - 1)
    ri      = _RI.get(n, 1.24)
    cr      = ci / ri if ri > 0 else 0.0
    return weights, round(cr, 4)


_WEIGHTS, _CR = _ahp_weights(_AHP_MATRIX)
_WEIGHT_DICT  = dict(zip(CRITERIA, [round(float(w), 4) for w in _WEIGHTS]))


def ahp_weights() -> Dict[str, float]:
    return _WEIGHT_DICT.copy()


def consistency_ratio() -> float:
    return _CR


# ── Sector weight feature ─────────────────────────────────────────────────────

def _get_sector_weight(text: str) -> float:
    t = str(text).lower()
    for kw, w in _SECTOR_WEIGHT.items():
        if kw in t:
            return w
    return 0.35


def _add_sector_weight(df: pd.DataFrame) -> pd.DataFrame:
    sector_col = next(
        (c for c in ("category", "sector", "incident_category") if c in df.columns),
        None,
    )
    if sector_col:
        df["sector_weight"] = df[sector_col].apply(_get_sector_weight)
    else:
        df["sector_weight"] = 0.35
    return df


# ── Ensure required columns exist ─────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "risk_index":    0.20,
        "impact_score":  0.45,
        "likelihood":    0.40,
        "risk_mc_p95":   0.25,
        "dl_confidence": 0.60,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df = _add_sector_weight(df)
    return df


# ── AHP scoring ───────────────────────────────────────────────────────────────

def run_ahp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an AHP priority score for every row.
    Lower score = higher threat priority (cost criteria dominate).
    Adds column: ahp_score (0–1, lower = more critical)
    """
    df = _ensure_columns(df)

    X = df[CRITERIA].values.astype(float)

    # Normalise each criterion to [0, 1]
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng     = np.where(col_max - col_min == 0, 1, col_max - col_min)
    X_norm  = (X - col_min) / rng

    # Invert benefit criteria so that higher always means more critical
    for j, crit in enumerate(CRITERIA):
        if BENEFIT[crit]:
            X_norm[:, j] = 1 - X_norm[:, j]

    # Weighted sum → threat priority score
    scores = X_norm @ _WEIGHTS
    # Normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    df["ahp_score"] = np.round(scores, 4)
    return df


# ── TOPSIS ranking ────────────────────────────────────────────────────────────

def run_topsis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank incidents using TOPSIS.
    Adds columns: topsis_score (0–1, higher = more critical), topsis_rank
    """
    df = _ensure_columns(df)

    X = df[CRITERIA].values.astype(float)

    # Step 1: Normalise (vector normalisation)
    norms = np.linalg.norm(X, axis=0)
    norms = np.where(norms == 0, 1, norms)
    X_n   = X / norms

    # Step 2: Weighted normalised matrix
    X_w = X_n * _WEIGHTS

    # Step 3: Ideal best / worst
    ideal_best  = np.where(
        [BENEFIT[c] for c in CRITERIA], X_w.max(axis=0), X_w.min(axis=0)
    )
    ideal_worst = np.where(
        [BENEFIT[c] for c in CRITERIA], X_w.min(axis=0), X_w.max(axis=0)
    )

    # Step 4: Euclidean distances
    d_best  = np.sqrt(((X_w - ideal_best)  ** 2).sum(axis=1))
    d_worst = np.sqrt(((X_w - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Closeness coefficient (higher = closer to worst = more critical)
    denom = d_best + d_worst
    denom = np.where(denom == 0, 1e-9, denom)
    scores = d_worst / denom

    df["topsis_score"] = np.round(scores, 4)
    df["topsis_rank"]  = df["topsis_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    return df


# ── Combined prioritisation ───────────────────────────────────────────────────

def prioritise(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Run both AHP and TOPSIS, combine into a final priority score,
    and return the top_n most critical incidents with recommendations.
    """
    df = run_ahp(df)
    df = run_topsis(df)

    # Combined score: equal blend of AHP and TOPSIS
    df["priority_score"] = ((df["ahp_score"] + df["topsis_score"]) / 2).round(4)
    df["priority_rank"]  = df["priority_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Response recommendation
    def _recommend(row) -> str:
        s = float(row["priority_score"])
        if s >= 0.75:
            return "🔴 Immediate Response — escalate to SOC team within 1 hour"
        if s >= 0.55:
            return "🟠 High Priority — investigate within 24 hours"
        if s >= 0.35:
            return "🟡 Medium Priority — schedule review within 72 hours"
        return "🟢 Monitor — log and review in next weekly report"

    df["recommendation"] = df.apply(_recommend, axis=1)

    return df.sort_values("priority_score", ascending=False).head(top_n)


# ── Criteria weight table for UI ──────────────────────────────────────────────

def weight_table() -> pd.DataFrame:
    rows = []
    for crit, w in _WEIGHT_DICT.items():
        rows.append({
            "Criterion":   crit.replace("_", " ").title(),
            "Weight":      w,
            "Weight (%)":  f"{w * 100:.1f}%",
            "Direction":   "Benefit ↑" if BENEFIT[crit] else "Cost ↓",
        })
    return pd.DataFrame(rows).sort_values("Weight", ascending=False)
