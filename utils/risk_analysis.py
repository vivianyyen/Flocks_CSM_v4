"""
utils/risk_analysis.py
────────────────────────────────────────────────────────────────────────────────
Quantitative Risk Analysis Layer
──────────────────────────────────
Combines DL model confidence scores with historical frequency data to produce:

  1. Likelihood × Impact Risk Matrix (per incident)
  2. Sector-level Composite Risk Index
  3. Monte Carlo Simulation — uncertainty bands around each risk score
  4. Risk Trend over time (rolling 30-day window)

Formula
───────
  likelihood  = normalised historical frequency of attack_type in sector
                (0–1, derived from incident count in the last 90 days)
  impact      = DL model confidence-weighted severity
                (Critical=1.0, High=0.75, Medium=0.45, Low=0.20)
  risk_index  = likelihood × impact

Monte Carlo
───────────
  For each incident we add Gaussian noise (σ = 0.08) to both likelihood
  and impact, run N=1000 simulations, and report (P5, P50, P95) confidence
  intervals.  This gives analysts a probabilistic view rather than a
  single deterministic score — aligned with ISO 31000 risk principles.

Public API
──────────
  enrich_with_risk(df, dl_predictions) → df with new columns
  compute_sector_risk_index(df)        → sector_df sorted by risk
  monte_carlo_one(likelihood, impact)  → {p5, p50, p95, std}
  risk_matrix_quadrant(l, i)           → "Extreme"|"High"|"Moderate"|"Low"
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ── Severity → numeric impact mapping ────────────────────────────────────────
SEVERITY_IMPACT: Dict[str, float] = {
    "Critical": 1.00,
    "High":     0.75,
    "Medium":   0.45,
    "Low":      0.20,
}

# ── Monte Carlo config ────────────────────────────────────────────────────────
MC_RUNS       = 1_000
MC_SIGMA      = 0.08    # noise std-dev — represents real-world uncertainty
RNG           = np.random.default_rng(42)

# ── Risk matrix quadrant thresholds ──────────────────────────────────────────
#   Extreme  : L > 0.6  AND  I > 0.6
#   High     : L > 0.4  AND  I > 0.4
#   Moderate : L > 0.2  OR   I > 0.4
#   Low      : everything else


def risk_matrix_quadrant(likelihood: float, impact: float) -> str:
    """Return ISO-31000-style quadrant label."""
    l, i = float(likelihood), float(impact)
    if l > 0.6 and i > 0.6:
        return "Extreme"
    if l > 0.4 and i > 0.4:
        return "High"
    if l > 0.2 or i > 0.4:
        return "Moderate"
    return "Low"


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def monte_carlo_one(
    likelihood: float,
    impact: float,
    n_runs: int = MC_RUNS,
) -> Dict[str, float]:
    """
    Simulate uncertainty around a single risk point.
    Returns P5, P50 (median), P95, and std of risk_index distribution.
    """
    l_samples = np.clip(
        RNG.normal(likelihood, MC_SIGMA, n_runs), 0.0, 1.0
    )
    i_samples = np.clip(
        RNG.normal(impact, MC_SIGMA, n_runs), 0.0, 1.0
    )
    risk_samples = l_samples * i_samples
    return {
        "p5":  float(np.percentile(risk_samples, 5)),
        "p50": float(np.percentile(risk_samples, 50)),
        "p95": float(np.percentile(risk_samples, 95)),
        "std": float(np.std(risk_samples)),
    }


# ── Likelihood estimation from historical data ────────────────────────────────
def _compute_likelihood_map(df: pd.DataFrame) -> Dict[str, float]:
    """
    Build a normalised likelihood score for each (attack_class, sector) pair
    based on how often that combination appears in the last 90 days.
    Returns a flat dict keyed by 'attack_class||sector'.
    """
    date_col = next(
        (c for c in ("incident_date", "publication_date", "created_at") if c in df.columns),
        None,
    )

    if date_col is None:
        return {}

    df = df.copy()
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    cutoff     = df["_dt"].max() - pd.Timedelta(days=90)
    recent     = df[df["_dt"] >= cutoff]

    if recent.empty:
        return {}

    attack_col = next(
        (c for c in ("attack_class", "incident_type", "type") if c in recent.columns),
        None,
    )
    sector_col = next(
        (c for c in ("category", "sector", "incident_category") if c in recent.columns),
        None,
    )

    if attack_col is None or sector_col is None:
        return {}

    counts = (
        recent
        .groupby([attack_col, sector_col])
        .size()
        .reset_index(name="count")
    )
    max_count = counts["count"].max() or 1
    counts["likelihood"] = counts["count"] / max_count

    result = {}
    for _, row in counts.iterrows():
        key = f"{row[attack_col]}||{row[sector_col]}"
        result[key] = float(row["likelihood"])
    return result


# ── Main enrichment function ──────────────────────────────────────────────────
def enrich_with_risk(
    df: pd.DataFrame,
    dl_predictions: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Add risk analysis columns to df:
      - dl_label          : DL model severity prediction
      - dl_confidence     : model confidence (0–1)
      - impact_score      : numeric severity (0–1)
      - likelihood        : historical frequency score (0–1)
      - risk_index        : likelihood × impact
      - risk_mc_p5/p50/p95: Monte Carlo confidence interval
      - risk_quadrant     : ISO 31000 matrix quadrant
    """
    df = df.copy()
    n  = len(df)

    # ── 1. DL predictions ─────────────────────────────────────────────────────
    if dl_predictions and len(dl_predictions) == n:
        df["dl_label"]      = [p["label"]      for p in dl_predictions]
        df["dl_confidence"] = [p["confidence"] for p in dl_predictions]
    else:
        # fall back to existing severity column if present
        sev_col = next((c for c in ("severity",) if c in df.columns), None)
        df["dl_label"]      = df[sev_col].fillna("Medium") if sev_col else "Medium"
        df["dl_confidence"] = 0.6

    # ── 2. Impact score (confidence-weighted) ──────────────────────────────────
    def _impact(row) -> float:
        base  = SEVERITY_IMPACT.get(str(row["dl_label"]).strip(), 0.45)
        conf  = float(row.get("dl_confidence", 0.6))
        # Blend base impact with confidence — low-confidence predictions
        # are pulled toward 0.45 (medium) to reduce false alarms
        return round(base * conf + 0.45 * (1 - conf), 4)

    df["impact_score"] = df.apply(_impact, axis=1)

    # ── 3. Likelihood from historical frequency ───────────────────────────────
    likelihood_map = _compute_likelihood_map(df)

    attack_col = next(
        (c for c in ("attack_class", "incident_type", "type") if c in df.columns),
        None,
    )
    sector_col = next(
        (c for c in ("category", "sector", "incident_category") if c in df.columns),
        None,
    )

    def _likelihood(row) -> float:
        if attack_col and sector_col:
            key = f"{row.get(attack_col, '')}||{row.get(sector_col, '')}"
            if key in likelihood_map:
                return likelihood_map[key]
        # default mid-range likelihood when no history
        return 0.40

    df["likelihood"] = df.apply(_likelihood, axis=1)

    # ── 4. Risk index ─────────────────────────────────────────────────────────
    df["risk_index"] = (df["likelihood"] * df["impact_score"]).round(4)

    # ── 5. Monte Carlo (vectorised) ───────────────────────────────────────────
    mc_rows = [
        monte_carlo_one(row["likelihood"], row["impact_score"])
        for _, row in df.iterrows()
    ]
    df["risk_mc_p5"]  = [r["p5"]  for r in mc_rows]
    df["risk_mc_p50"] = [r["p50"] for r in mc_rows]
    df["risk_mc_p95"] = [r["p95"] for r in mc_rows]
    df["risk_mc_std"] = [r["std"] for r in mc_rows]

    # ── 6. ISO 31000 quadrant ─────────────────────────────────────────────────
    df["risk_quadrant"] = df.apply(
        lambda r: risk_matrix_quadrant(r["likelihood"], r["impact_score"]),
        axis=1,
    )

    return df


# ── Sector-level risk index ───────────────────────────────────────────────────
def compute_sector_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate risk_index by sector/category.
    Returns a DataFrame sorted descending by composite_risk.
    """
    sector_col = next(
        (c for c in ("category", "sector", "incident_category") if c in df.columns),
        None,
    )
    if sector_col is None or "risk_index" not in df.columns:
        return pd.DataFrame()

    agg = (
        df.groupby(sector_col)
        .agg(
            incident_count=("risk_index", "count"),
            mean_risk=("risk_index", "mean"),
            max_risk=("risk_index", "max"),
            mean_likelihood=("likelihood", "mean"),
            mean_impact=("impact_score", "mean"),
        )
        .reset_index()
    )

    # Composite = mean_risk * log(1 + count) — frequency amplifier
    agg["composite_risk"] = (
        agg["mean_risk"] * np.log1p(agg["incident_count"])
    ).round(4)

    agg = agg.sort_values("composite_risk", ascending=False).reset_index(drop=True)
    agg.rename(columns={sector_col: "sector"}, inplace=True)
    return agg


# ── Risk trend (rolling 30-day) ───────────────────────────────────────────────
def compute_risk_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily mean risk_index with a 7-day rolling average.
    Returns a DataFrame with columns: date, mean_risk, rolling_avg.
    """
    date_col = next(
        (c for c in ("incident_date", "publication_date") if c in df.columns),
        None,
    )
    if date_col is None or "risk_index" not in df.columns:
        return pd.DataFrame()

    trend = df.copy()
    trend["_date"] = pd.to_datetime(trend[date_col], errors="coerce", utc=True).dt.date
    trend = (
        trend.groupby("_date")["risk_index"]
        .mean()
        .reset_index()
        .rename(columns={"_date": "date", "risk_index": "mean_risk"})
        .sort_values("date")
    )
    trend["rolling_avg"] = trend["mean_risk"].rolling(7, min_periods=1).mean().round(4)
    return trend
