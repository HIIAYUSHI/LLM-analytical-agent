"""
research_pipeline.py
====================
Batch benchmark runner for offline evaluation of the analysis agent.

FIXES OVER ORIGINAL
--------------------
1. METRIC KEYS CORRECTED
   Old pipeline read hr, ngs, acs — legacy aliases that no longer exist.
   Canonical keys are: fvr, her, cce, qrt.

2. SQLITE LOADING FIXED
   Old code concatenated ALL tables with pd.concat(), producing a garbage
   DataFrame full of NaNs for any relational schema (e.g. Chinook, 11 tables).
   SQLite benchmarks now pass df=pd.DataFrame() and rely on db_path alone —
   which is what the agent actually uses for all SQL queries.

3. SELF-CORRECTION LOOP SIMPLIFIED
   Uses needs_correction flag directly instead of re-computing from it.

4. CSV EXPORT USES CANONICAL COLUMN NAMES.

5. EXPONENTIAL BACK-OFF ADDED
   Transient API errors (rate limits, 5xx) are retried up to MAX_RETRIES times
   with exponential back-off instead of failing permanently after one attempt.
"""

import os
import sqlite3
import time
from datetime import datetime

import pandas as pd

from agent import analyze

MAX_RETRIES  = 3
BACKOFF_BASE = 2.0   # seconds; actual wait = BACKOFF_BASE ** attempt


def _analyze_with_retry(question: str, df, db_path, correction_context=None) -> dict:
    """
    Wrap analyze() with exponential back-off for transient API failures.
    Retries up to MAX_RETRIES times when a Critical API Error is returned
    (network blip, rate limit, 5xx).  Execution errors that set needs_correction
    are not retried here — the existing self-correction loop handles those.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        result = analyze(
            question=question,
            df=df,
            db_path=db_path,
            correction_context=correction_context,
        )
        if not str(result.get("insights", "")).startswith("Critical API Error"):
            return result
        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE ** attempt
            print(f"    → API error on attempt {attempt}/{MAX_RETRIES}. "
                  f"Retrying in {wait:.0f}s…")
            time.sleep(wait)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark configuration
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    {
        "dataset_name": "Titanic (CSV)",
        "file_path":    "titanic.csv",
        "type":         "csv",
        "queries": [
            "What is the overall survival rate of passengers?",
            "Plot a bar chart showing the survival rate broken down by passenger class.",
            "Plot a histogram of the passenger ages.",
        ],
    },
    {
        "dataset_name": "Chinook (SQLite)",
        "file_path":    "Chinook_Sqlite.sqlite",
        "type":         "sqlite",
        "queries": [
            "Total revenue generated per country.",
            "Plot a bar chart of the average track price by genre.",
            "Show me the top 3 customers by total spending.",
        ],
    },
]


def _load_dataset(suite: dict) -> tuple[pd.DataFrame, str | None]:
    """
    Load a dataset entry.

    Returns (df, db_path).
    For SQLite: df is empty — the agent reads the DB directly via DB_PATH.
    Raises on failure so the caller can skip this suite gracefully.
    """
    file_path = suite["file_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    if suite["type"] == "csv":
        return pd.read_csv(file_path), None

    if suite["type"] == "sqlite":
        # Validate the file is a readable SQLite DB, then return empty df.
        with sqlite3.connect(file_path) as conn:
            conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return pd.DataFrame(), file_path

    raise ValueError(f"Unknown dataset type: '{suite['type']}'")


def run_pipeline() -> None:
    print("=" * 60)
    print("Starting Batch Research Pipeline")
    print("=" * 60)

    results = []

    for suite in BENCHMARKS:
        name = suite["dataset_name"]
        print(f"\n{'─' * 50}")
        print(f"Dataset: {name}")

        try:
            df, db_path = _load_dataset(suite)
        except Exception as exc:
            print(f"  ✗ Skipped — {exc}")
            continue

        queries = suite["queries"]
        totals  = {k: 0.0 for k in ("fvr", "her", "cce", "qrt")}
        success_count = 0

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query}")

            result = _analyze_with_retry(question=query, df=df, db_path=db_path)

            # One automatic retry on execution failure
            if result.get("needs_correction"):
                print("    → Attempt 1 failed. Triggering self-correction…")
                result = _analyze_with_retry(
                    question           = query,
                    df                 = df,
                    db_path            = db_path,
                    correction_context = {
                        "query":   query,
                        "code":    result.get("executed_code", ""),
                        "error":   result.get("error_trace", ""),
                        "attempt": result.get("attempt", 1),
                    },
                )

            metrics = result.get("metrics", {})
            for key in ("fvr", "her", "cce", "qrt"):
                totals[key] += metrics.get(key, 0.0)

            if not result.get("needs_correction"):
                success_count += 1

            print(
                f"    FVR={metrics.get('fvr', 0):.2%}  "
                f"HER={metrics.get('her', 0):.2%}  "
                f"CCE={metrics.get('cce', 0):.3f}  "
                f"QRT={metrics.get('qrt', 0):.2f}s"
            )

        n         = len(queries)
        stability = (success_count / n * 100) if n else 0.0
        results.append({
            "dataset":       name,
            "total_queries": n,
            "avg_fvr":       totals["fvr"] / n if n else 0.0,
            "avg_her":       totals["her"] / n if n else 0.0,
            "avg_cce":       totals["cce"] / n if n else 0.0,
            "avg_qrt_s":     totals["qrt"] / n if n else 0.0,
            "stability_pct": stability,
        })
        print(f"  Completed. Stability: {stability:.1f}%")

    if not results:
        print("\nNo datasets processed — check file paths in BENCHMARKS.")
        return

    summary_df = pd.DataFrame(results)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"research_summary_{timestamp}.csv"
    summary_df.to_csv(filename, index=False)

    print(f"\n{'=' * 60}")
    print(f"Batch complete. Results saved to '{filename}'")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
