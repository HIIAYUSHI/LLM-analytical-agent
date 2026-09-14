"""
stats_engine.py
===============
Compute rich dataset statistics used by RAG ingestion, the agent's
code-prompt context, and evaluation metrics.

Fix over original
-----------------
Removed deprecated ``infer_datetime_format=True`` (pandas ≥ 2.0).
Replaced with ``format="mixed"`` — the correct modern equivalent.
"""

import pandas as pd
import numpy as np
from typing import List


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Return columns that are datetime dtype or can be reliably parsed as dates.
    Tests up to 50 non-null rows to avoid false positives on ID or numeric strings,
    and to handle columns that are heavily null at the top of the frame.
    """
    hits: List[str] = []
    for col in df.columns:
        if "datetime" in str(df[col].dtype).lower():
            hits.append(col)
            continue
        if df[col].dtype != object:
            continue
        # Use up to 50 non-null values to avoid missing sparse datetime cols
        sample = df[col].dropna().head(50).astype(str)
        if sample.empty:
            continue
        date_like = sample.str.match(
            r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}|^\d{4}-\d{2}|^\w+ \d{4}$'
        )
        if date_like.sum() < max(1, len(sample) // 2):
            continue
        try:
            pd.to_datetime(sample, format="mixed", dayfirst=False)
            hits.append(col)
        except Exception:
            pass
    return hits


def compute_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Return a rich statistics dict for the entire DataFrame.

    Keys
    ----
    row_count, column_count, data_types,
    numeric_summary, categorical_summary,
    datetime_columns, datetime_ranges,
    missing_by_column, high_cardinality_cols
    """
    stats: dict = {
        "row_count":             int(len(df)),
        "column_count":          int(len(df.columns)),
        "data_types":            {col: str(dt) for col, dt in df.dtypes.items()},
        "numeric_summary":       {},
        "categorical_summary":   {},
        "datetime_columns":      [],
        "datetime_ranges":       {},
        "missing_by_column":     {},
        "high_cardinality_cols": [],
    }

    # Missing values (only columns with at least one NaN)
    stats["missing_by_column"] = {
        col: int(n)
        for col, n in df.isna().sum().items()
        if n > 0
    }

    # Numeric summary
    for col in df.select_dtypes(include=[np.number]).columns:
        clean = df[col].dropna()
        if clean.empty:
            continue
        stats["numeric_summary"][col] = {
            "min":    float(clean.min()),
            "max":    float(clean.max()),
            "mean":   float(clean.mean()),
            "median": float(clean.median()),
            "std":    float(clean.std()),
            "q25":    float(clean.quantile(0.25)),
            "q75":    float(clean.quantile(0.75)),
        }

    # Categorical summary (top 10 per column)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        clean = df[col].dropna()
        if clean.empty:
            continue
        stats["categorical_summary"][col] = clean.value_counts().head(10).to_dict()
        if clean.nunique() > 50:
            stats["high_cardinality_cols"].append(col)

    # Datetime ranges
    dt_cols = detect_datetime_columns(df)
    stats["datetime_columns"] = dt_cols
    for col in dt_cols:
        try:
            parsed = pd.to_datetime(
                df[col], format="mixed", dayfirst=False, errors="coerce"
            ).dropna()
            if not parsed.empty:
                stats["datetime_ranges"][col] = {
                    "min":       str(parsed.min().date()),
                    "max":       str(parsed.max().date()),
                    "n_periods": int(len(parsed.dt.to_period("M").unique())),
                }
        except Exception:
            pass

    return stats
