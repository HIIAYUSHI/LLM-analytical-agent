"""
evaluation_metrics.py
=====================
Research-Grade Evaluation Framework for LLM Data Analysis Agents

METRIC DEFINITIONS
------------------

1. FVR  — Factual Verification Rate
         The fraction of numerical claims in the LLM's output that can be
         traced back to values computed by the executed code ("ground truth").
         Range [0, 1].  Higher is better.
         Formula:  FVR = verified_claims / total_numerical_claims

2. HER  — Hallucination Error Rate  (complement of FVR)
         The fraction of numerical claims that have NO match in the ground
         truth.  HER = 1 - FVR.
         Range [0, 1].  Lower is better.

3. CCE  — Confidence Calibration Error
         Measures how well the model knows what it doesn't know.
         CCE = |assigned_confidence - actual_fvr|
         Range [0, 1].  Lower is better (0 = perfectly calibrated).

4. EER  — Execution Error Rate
         Fraction of queries where the generated code failed to execute.
         Range [0, 1].  Lower is better.

5. SCR  — Self-Correction Rate
         Fraction of initially failed queries that were recovered after
         the self-correction pass.
         Range [0, 1].  Higher is better.

6. QRT  — Query Response Time (seconds)
         Wall-clock seconds from query submission to final insight delivery.
         Lower is better.

7. OCS  — Output Consistency Score
         Stability of response length across repeated invocations of the same
         query.  Computed as 1 / (1 + Var(lengths)).
         Range (0, 1].  Higher is better.  Close to 1 = very stable outputs.

NOTE: ACS (Analytical Consistency Score) has been **retired** from this version
because its previous implementation was a hard-coded placeholder (always 1.0),
which produced misleading research data.  It will be reintroduced once a proper
semantic-similarity-based implementation is available.
"""

import numpy as np
import re
from typing import List, Dict


# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------

def _preprocess_text(text: str) -> str:
    """Remove comma-separators inside numbers (e.g. 1,450 -> 1450)."""
    return re.sub(r'(?<=\d),(?=\d)', '', text)


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from a string, skipping list-index integers 1–100."""
    clean = _preprocess_text(text)
    raw = re.findall(r'-?\d+\.?\d*', clean)
    results: List[float] = []
    for s in raw:
        try:
            v = float(s)
            # Skip common list-index integers (1, 2, … 100) that are almost
            # never actual data values.
            if v.is_integer() and 1.0 <= v <= 100.0:
                continue
            results.append(v)
        except ValueError:
            pass
    return results


def _flatten_ground_truth(stats: dict) -> List[float]:
    """Pull every numeric value out of the stats dict into a flat list."""
    values: List[float] = []
    for col_stats in stats.get("numeric_summary", {}).values():
        if isinstance(col_stats, dict):
            for v in col_stats.values():
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
    return values


def _is_supported(val: float, truth_values: List[float]) -> bool:
    """
    A claim is considered *supported* when either:
      (a) the absolute difference from any truth value is ≤ 0.05  (exact match
          within rounding), OR
      (b) val/100 is within 0.01 of any truth value (percentage representation).
    """
    return (
        any(abs(val - tv) <= 0.05 for tv in truth_values)
        or any(abs((val / 100.0) - tv) <= 0.01 for tv in truth_values)
    )


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_fvr_her(insights: str, stats: dict) -> Dict:
    """
    Compute Factual Verification Rate (FVR) and Hallucination Error Rate (HER).

    Parameters
    ----------
    insights    : The natural-language answer produced by the LLM.
    stats       : Dataset statistics dict (output of stats_engine.compute_dataset_statistics).

    Returns
    -------
    {
        "fvr": float,
        "her": float,
        "verified_claims":   int,
        "unverified_claims": int,
        "total_claims":      int,
        "claim_details": List[dict]   # ← NEW: per-claim audit trail
            Each dict has:
                "value"        : float  — the number found in the synthesis
                "verified"     : bool   — whether it matched a ground-truth value
                "matched_truth": float | None  — the closest truth value (if verified)
                "delta"        : float | None  — absolute difference from best match
    }
    """
    numbers = _extract_numbers(insights)
    truth   = _flatten_ground_truth(stats)

    if not numbers:
        return {
            "fvr": 1.0, "her": 0.0,
            "verified_claims": 0, "unverified_claims": 0, "total_claims": 0,
            "claim_details": [],
        }

    claim_details = []
    for v in numbers:
        if truth:
            # Find the closest truth value and record the delta
            best_truth = min(truth, key=lambda tv: abs(v - tv))
            direct_delta = abs(v - best_truth)
            pct_delta    = abs(v / 100.0 - best_truth) if best_truth != 0 else float("inf")
            is_verified  = _is_supported(v, truth)
            claim_details.append({
                "value":         v,
                "verified":      is_verified,
                "matched_truth": best_truth if is_verified else None,
                "closest_truth": best_truth,          # always set for display
                "delta":         round(min(direct_delta, pct_delta * 100), 4),
            })
        else:
            claim_details.append({
                "value":         v,
                "verified":      False,
                "matched_truth": None,
                "closest_truth": None,
                "delta":         None,
            })

    verified   = sum(1 for c in claim_details if c["verified"])
    unverified = len(claim_details) - verified
    fvr        = verified / len(claim_details)

    return {
        "fvr":               round(fvr, 4),
        "her":               round(1.0 - fvr, 4),
        "verified_claims":   verified,
        "unverified_claims": unverified,
        "total_claims":      len(claim_details),
        "claim_details":     claim_details,
    }


def compute_cce(assigned_confidence: float, actual_fvr: float) -> Dict[str, float]:
    """
    Confidence Calibration Error (CCE).

    Parameters
    ----------
    assigned_confidence : Model's self-reported confidence (0–1).
    actual_fvr          : The FVR computed for that response.

    Returns
    -------
    {"cce": float}   # [0, 1],  0 = perfect calibration
    """
    return {"cce": round(abs(assigned_confidence - actual_fvr), 4)}


def compute_ocs(outputs: List[str]) -> Dict[str, float]:
    """
    Output Consistency Score (OCS).

    Requires at least 2 outputs for the same query; returns 1.0 if only
    one sample is available (undefined variance).

    Parameters
    ----------
    outputs : List of text responses for the *same* query.

    Returns
    -------
    {"ocs": float}   # (0, 1],  1.0 = perfectly consistent length
    """
    if len(outputs) < 2:
        return {"ocs": 1.0}
    lengths  = [len(o) for o in outputs]
    variance = float(np.var(lengths))
    return {"ocs": round(1.0 / (1.0 + variance), 6)}


# ---------------------------------------------------------------------------
# Session-level aggregator
# ---------------------------------------------------------------------------

class ResearchMetricsTracker:
    """
    Accumulates per-query metrics across a session or benchmark run and
    exposes aggregated summaries for the Research Dashboard.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._records: List[dict] = []
        self._execution_attempts: int   = 0
        self._execution_failures: int   = 0
        self._correction_attempts: int  = 0
        self._correction_successes: int = 0

    # ------------------------------------------------------------------
    # Per-query recording
    # ------------------------------------------------------------------

    def record_query(
        self,
        query:               str,
        insights:            str,
        stats:               dict,
        assigned_confidence: float,
        execution_failed:    bool,
        needed_correction:   bool,
        correction_succeeded: bool,
        response_time_s:     float,
    ) -> dict:
        """
        Record all metrics for a single query and return the per-query dict.
        """
        fvr_her = compute_fvr_her(insights, stats)
        cce     = compute_cce(assigned_confidence, fvr_her["fvr"])

        record = {
            "query":                query,
            # Core quality metrics
            "fvr":                  fvr_her["fvr"],
            "her":                  fvr_her["her"],
            "cce":                  cce["cce"],
            # Execution health
            "execution_failed":     execution_failed,
            "needed_correction":    needed_correction,
            "correction_succeeded": correction_succeeded,
            # Latency
            "response_time_s":      round(response_time_s, 3),
            # Raw claim counts (useful for weighted averages in papers)
            "verified_claims":      fvr_her["verified_claims"],
            "unverified_claims":    fvr_her["unverified_claims"],
            "total_claims":         fvr_her["total_claims"],
            # Per-claim audit trail
            "claim_details":        fvr_her.get("claim_details", []),
        }
        self._records.append(record)

        self._execution_attempts += 1
        if execution_failed:
            self._execution_failures  += 1
        if needed_correction:
            self._correction_attempts += 1
            if correction_succeeded:
                self._correction_successes += 1

        return record

    # ------------------------------------------------------------------
    # Session-level summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Aggregate metrics across all recorded queries.

        Returns
        -------
        {
            "total_queries": int,
            "avg_fvr":       float,   # mean Factual Verification Rate
            "avg_her":       float,   # mean Hallucination Error Rate
            "avg_cce":       float,   # mean Confidence Calibration Error
            "eer":           float,   # Execution Error Rate
            "scr":           float,   # Self-Correction Rate
            "avg_qrt_s":     float,   # mean Query Response Time (seconds)
        }
        """
        n = len(self._records)
        if n == 0:
            return {k: 0.0 for k in
                    ["total_queries", "avg_fvr", "avg_her",
                     "avg_cce", "eer", "scr", "avg_qrt_s"]}

        avg = lambda key: round(float(np.mean([r[key] for r in self._records])), 4)

        eer = (self._execution_failures / self._execution_attempts
               if self._execution_attempts else 0.0)

        scr = (self._correction_successes / self._correction_attempts
               if self._correction_attempts else 0.0)

        return {
            "total_queries": n,
            "avg_fvr":       avg("fvr"),
            "avg_her":       avg("her"),
            "avg_cce":       avg("cce"),
            "eer":           round(eer, 4),
            "scr":           round(scr, 4),
            "avg_qrt_s":     avg("response_time_s"),
        }

    def records(self) -> List[dict]:
        """Return raw per-query records (useful for exporting to CSV)."""
        return list(self._records)


# ---------------------------------------------------------------------------
# Module-level singleton  (drop-in replacement for old metrics_tracker)
# ---------------------------------------------------------------------------
metrics_tracker = ResearchMetricsTracker()
