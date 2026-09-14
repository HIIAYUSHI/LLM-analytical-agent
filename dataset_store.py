"""
dataset_store.py
================
Persistent cross-dataset benchmark registry.

Every time a user finishes working on a dataset (or runs the batch pipeline),
the session's aggregate metrics are written to a local JSON file:
    ./benchmark_registry.json

This lets the Research Dashboard compare ANY two datasets that have ever been
analysed — not just the ones loaded in the current session.

Schema of one registry entry
-----------------------------
{
    "dataset_name":   str,          # e.g. "titanic.csv"
    "file_type":      str,          # "csv" | "sqlite"
    "row_count":      int,
    "col_count":      int,
    "recorded_at":    str,          # ISO-8601 timestamp
    "total_queries":  int,
    "avg_fvr":        float,
    "avg_her":        float,
    "avg_cce":        float,
    "eer":            float,
    "scr":            float,
    "avg_qrt_s":      float,
    "ts_queries":     int,          # time-series queries detected
    "tabular_queries":int,          # table-output queries
    "query_records":  List[dict],   # per-query detail rows
}
"""

import json
import os
import tempfile
from datetime import datetime
from typing import List, Optional

REGISTRY_PATH = "./benchmark_registry.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_registry() -> List[dict]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_registry(registry: List[dict]) -> None:
    # Write to a temp file in the same directory, then atomically rename.
    # This prevents partial-write corruption when two sessions save concurrently.
    dir_name = os.path.dirname(os.path.abspath(REGISTRY_PATH))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(registry, f, indent=2, default=str)
        os.replace(tmp_path, REGISTRY_PATH)   # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Public API ─────────────────────────────────────────────────────────────────

def save_dataset_benchmark(
    dataset_name:    str,
    file_type:       str,
    row_count:       int,
    col_count:       int,
    summary:         dict,          # output of ResearchMetricsTracker.summary()
    query_records:   List[dict],    # output of ResearchMetricsTracker.records()
    ts_queries:      int = 0,
    tabular_queries: int = 0,
) -> dict:
    """
    Upsert a benchmark entry for *dataset_name*.

    If an entry for this dataset already exists it is REPLACED so the registry
    always reflects the most recent run.  Returns the saved entry.
    """
    entry = {
        "dataset_name":    dataset_name,
        "file_type":       file_type,
        "row_count":       row_count,
        "col_count":       col_count,
        "recorded_at":     datetime.now().isoformat(timespec="seconds"),
        "total_queries":   summary.get("total_queries", 0),
        "avg_fvr":         summary.get("avg_fvr", 0.0),
        "avg_her":         summary.get("avg_her", 0.0),
        "avg_cce":         summary.get("avg_cce", 0.0),
        "eer":             summary.get("eer", 0.0),
        "scr":             summary.get("scr", 0.0),
        "avg_qrt_s":       summary.get("avg_qrt_s", 0.0),
        "ts_queries":      ts_queries,
        "tabular_queries": tabular_queries,
        "query_records":   query_records,
    }

    registry = _load_registry()
    # Replace existing entry for this dataset, or append
    registry = [e for e in registry if e["dataset_name"] != dataset_name]
    registry.append(entry)
    _save_registry(registry)
    return entry


def load_all_benchmarks() -> List[dict]:
    """Return all saved benchmark entries, newest first."""
    registry = _load_registry()
    return sorted(registry, key=lambda e: e.get("recorded_at", ""), reverse=True)


def delete_benchmark(dataset_name: str) -> bool:
    """Remove a single entry.  Returns True if something was deleted."""
    registry = _load_registry()
    new_registry = [e for e in registry if e["dataset_name"] != dataset_name]
    if len(new_registry) < len(registry):
        _save_registry(new_registry)
        return True
    return False


def get_benchmark(dataset_name: str) -> Optional[dict]:
    """Fetch a single entry by name, or None."""
    for e in _load_registry():
        if e["dataset_name"] == dataset_name:
            return e
    return None
