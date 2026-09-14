"""
rag.py
======
ChromaDB-backed RAG layer for dataset context retrieval.

Fixes over original
--------------------
• Removed deprecated ``infer_datetime_format=True`` → ``format="mixed"``.
• ``retrieve_context()`` logs failures instead of silently returning "".
  Silent swallowing made ChromaDB index corruption undiagnosable.
"""

import hashlib
import logging

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from stats_engine import compute_dataset_statistics, detect_datetime_columns

logger = logging.getLogger(__name__)

_client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)
_embedder   = SentenceTransformer(EMBEDDING_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest_dataframe_info(df: pd.DataFrame) -> int:
    """
    Index rich dataset context into ChromaDB.
    Clears all previous entries first, then re-indexes.
    Returns the number of documents stored.
    """
    existing = _collection.get()
    if existing and existing["ids"]:
        _collection.delete(ids=existing["ids"])

    stats   = compute_dataset_statistics(df)
    dt_cols = detect_datetime_columns(df)
    docs: list[str] = []

    # 1. Dataset-level overview
    docs.append(
        f"Dataset overview: {stats['row_count']} rows, {stats['column_count']} columns. "
        f"Numeric columns: {list(df.select_dtypes(include=np.number).columns)}. "
        f"Categorical columns: {list(df.select_dtypes(include=['object','category']).columns)}. "
        f"Datetime columns: {dt_cols}. "
        f"Missing value counts: {stats['missing_by_column']}."
    )

    # 2. Per-column metadata
    for col in df.columns:
        docs.append(
            f"Column '{col}': dtype={df[col].dtype}, "
            f"{int(df[col].nunique())} unique values, "
            f"{int(df[col].isna().sum())} missing."
        )

    # 3. Numeric stats — one document per column for targeted retrieval
    for col, s in stats["numeric_summary"].items():
        docs.append(
            f"Numeric stats for '{col}': min={s['min']:.4g}, max={s['max']:.4g}, "
            f"mean={s['mean']:.4g}, median={s['median']:.4g}, std={s['std']:.4g}, "
            f"Q1={s['q25']:.4g}, Q3={s['q75']:.4g}."
        )

    # 4. Categorical distributions
    for col, dist in stats["categorical_summary"].items():
        top_str = ", ".join(f"{k}={v}" for k, v in list(dist.items())[:5])
        docs.append(f"Categorical distribution for '{col}': top values are [{top_str}].")

    # 5. Datetime ranges
    for col, rng in stats["datetime_ranges"].items():
        docs.append(
            f"Datetime column '{col}': from {rng['min']} to {rng['max']}, "
            f"approx {rng['n_periods']} monthly periods."
        )

    # 6. Sample rows
    docs.append(f"Sample data (first 5 rows):\n{df.head(5).to_string(index=False)}")

    # 7. Top pairwise correlations
    num_df = df.select_dtypes(include=np.number)
    if len(num_df.columns) >= 2:
        try:
            corr  = num_df.corr().abs()
            upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
            pairs = (
                corr.where(upper)
                    .stack()
                    .sort_values(ascending=False)
                    .head(10)
            )
            lines = [f"{a}↔{b}: r={v:.2f}" for (a, b), v in pairs.items()]
            docs.append("Top numeric correlations: " + ", ".join(lines))
        except Exception as exc:
            logger.warning("Correlation computation failed: %s", exc)

    if not docs:
        return 0

    embeddings = _embedder.encode(docs).tolist()
    ids        = [hashlib.md5(t.encode()).hexdigest()[:16] for t in docs]

    # Deduplicate on hash collision
    seen, u_docs, u_emb, u_ids = set(), [], [], []
    for doc, emb, uid in zip(docs, embeddings, ids):
        if uid not in seen:
            seen.add(uid)
            u_docs.append(doc)
            u_emb.append(emb)
            u_ids.append(uid)

    _collection.add(documents=u_docs, embeddings=u_emb, ids=u_ids)
    return len(u_docs)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

_K_MAP = {"timeseries": 8, "tabular": 7, "visualization": 6, "scalar": 5}


def retrieve_context(query: str, k: int = 6, query_type: str = "scalar") -> str:
    """
    Return the k most relevant RAG chunks for *query*.

    Logs a warning on ChromaDB errors instead of silently returning "".
    """
    effective_k = _K_MAP.get(query_type, k)
    try:
        q_emb   = _embedder.encode([query]).tolist()
        results = _collection.query(
            query_embeddings=q_emb, n_results=min(effective_k, 10)
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as exc:
        logger.warning("RAG retrieval failed (query_type=%s): %s", query_type, exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def build_rag_synthesis_prompt(
    question:     str,
    raw_output:   str,
    rag_context:  str,
    query_type:   str,
    data_summary: dict,
) -> str:
    """
    Build the synthesis prompt that merges:
      - The user's question
      - Code-execution result (ground truth — all numbers must come from here)
      - RAG-retrieved context (column info, stats, sample rows)
      - Query-type-specific formatting instructions
    """
    bt = "```"

    header = (
        f"USER QUESTION: {question}\n\n"
        f"DATASET CONTEXT (RAG — use to interpret column names and domain meaning):\n"
        f"{rag_context}\n\n"
        f"CODE EXECUTION RESULT (ground truth — base ALL numbers on this):\n"
        f"{raw_output}\n\n"
        "RULES:\n"
        "• Use ONLY numbers from the execution result — never invent values.\n"
        "• Use RAG context to interpret column names and add domain meaning.\n"
        "• CURRENCY: Never use $ — write 'dollars' or 'USD'.\n"
    )

    fmt_map = {
        "timeseries": (
            "\nFORMAT — Time-Series:\n"
            "1. 2–3 sentence narrative (direction, magnitude, peak/trough).\n"
            "2. Markdown table of the result (max 20 rows, must include the date/time column).\n"
            "3. 1–2 sentences on seasonality or anomalies.\n"
            f"4. JSON chart spec at the very end:\n{bt}json\n"
            f'[{{"type":"line","x":"<exact_date_col_name>","y":"<exact_value_col_name>"}}]\n'
            f"{bt}\n"
            "CRITICAL: column names must EXACTLY match the result table."
        ),
        "visualization": (
            "\nFORMAT — Visualisation:\n"
            "1–2 sentences summarising what the chart shows.\n"
            f"JSON chart spec:\n{bt}json\n"
            f'[{{"type":"bar","x":"<exact_x_col>","y":"<exact_y_col>"}}]\n'
            f"{bt}\n"
            "Supported types: bar, scatter, histogram, box, line.\n"
            "CRITICAL: column names must EXACTLY match the result."
        ),
        "tabular": (
            "\nFORMAT — Tabular:\n"
            "• 1–2 sentence key takeaway ABOVE the table.\n"
            "• Clean markdown table — no prose listing of numbers.\n"
            "• Truncate at 20 rows and note it."
        ),
        "scalar": (
            "\nFORMAT — Scalar:\n"
            "Direct answer in 1–3 sentences. No padding."
        ),
    }

    return header + fmt_map.get(query_type, fmt_map["scalar"])
