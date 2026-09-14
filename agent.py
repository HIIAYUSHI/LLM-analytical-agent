"""
agent.py  ·  v4
===============
Core analysis agent.

v4 changes
----------
• RAG is NOW ACTUALLY USED in synthesis — retrieve_context() feeds real
  column semantics into the synthesis prompt via build_rag_synthesis_prompt()
• Time-series code prompt is bulletproof: the LLM is told EXACTLY which date
  column exists, must reset_index() if needed, and must name columns clearly
• Visualization engine: after code executes we inspect the RESULT DataFrame
  and AUTO-INFER the correct x/y column names, never relying on the LLM to
  guess them — this fixes the "Column 'index' not found" crash
• query_type="timeseries" and query_type="visualization" both go through the
  same auto-inference path so charts always render
• Token usage logged per call for research paper cost analysis
• Failure taxonomy: error types are classified (NameError, KeyError, etc.)
"""

import pandas as pd
import sqlite3
import time
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL
from stats_engine import compute_dataset_statistics, detect_datetime_columns
from evaluation_metrics import metrics_tracker, compute_fvr_her, compute_cce
from code_executor import execute_pandas_code
from rag import retrieve_context, build_rag_synthesis_prompt

client = Groq(api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# Query classifier
# ══════════════════════════════════════════════════════════════════════════════

_TS_RE = re.compile(
    r'\b(trend|over time|time.?series|monthly|weekly|daily|yearly|annually|'
    r'per month|per year|per week|per day|by month|by year|by week|by day|'
    r'growth|forecast|seasonalit|rolling|cumulative|timeline|progression)\b',
    re.IGNORECASE,
)
_VIZ_RE = re.compile(
    r'\b(plot|chart|graph|visuali[sz]e|histogram|bar chart|scatter|pie|show.{0,10}chart)\b',
    re.IGNORECASE,
)
_TAB_RE = re.compile(
    r'\b(top \d|bottom \d|rank|list|show me|breakdown|break down|compare|'
    r'by \w+|per \w+|each \w+|all \w+|summary)\b',
    re.IGNORECASE,
)


_SCALAR_RE = re.compile(
    r'\b(what is the|what\'s the|whats the|calculate the|find the|give me the|'
    r'tell me the|how many|how much|count the|total number|'
    r'average|mean|median|minimum|maximum|min |max |sum |'
    r'overall|single|one number|percentage of|ratio of|rate of)\b',
    re.IGNORECASE,
)


def classify_query(question: str) -> str:
    """
    Classify query into: 'timeseries' | 'visualization' | 'tabular' | 'scalar'

    Priority order (highest → lowest):
    1. timeseries  — explicit time/trend keywords
    2. visualization — explicit plot/chart/graph keywords
    3. scalar — single-value aggregation keywords that would yield one number
       (checked BEFORE tabular to prevent "what is the average X" → tabular)
    4. tabular — multi-row list/breakdown queries
    5. scalar (default fallback)
    """
    if _TS_RE.search(question):   return "timeseries"
    if _VIZ_RE.search(question):  return "visualization"
    # Check scalar BEFORE tabular — aggregation questions come first
    if _SCALAR_RE.search(question) and not _TAB_RE.search(question):
        return "scalar"
    if _TAB_RE.search(question):  return "tabular"
    return "scalar"


def classify_error(traceback_str: str) -> str:
    """Classify an execution error for failure taxonomy research logging."""
    tb = traceback_str.lower()
    if "keyerror"         in tb: return "KeyError (wrong column name)"
    if "nameerror"        in tb: return "NameError (undefined variable)"
    if "typeerror"        in tb: return "TypeError (wrong data type)"
    if "valueerror"       in tb: return "ValueError (data conversion)"
    if "attributeerror"   in tb: return "AttributeError (wrong method)"
    if "syntaxerror"      in tb: return "SyntaxError (malformed code)"
    if "operationalerror" in tb: return "SQL OperationalError"
    if "indexerror"       in tb: return "IndexError"
    return "Other"


# ══════════════════════════════════════════════════════════════════════════════
# Schema helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_sqlite_schema(db_path: str) -> str:
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        out = ""
        for (tbl,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info('{tbl}')")
            cols = cursor.fetchall()
            out += f"Table: {tbl}\nColumns: {', '.join(f'{c[1]} ({c[2]})' for c in cols)}\n\n"
        conn.close()
        return out
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Chart spec auto-inference
# ══════════════════════════════════════════════════════════════════════════════

def _infer_chart_spec(result_df: pd.DataFrame, query_type: str,
                      original_question: str) -> list[dict]:
    """
    Deterministically build chart specs from the ACTUAL result DataFrame.
    Never relies on the LLM to guess column names.

    Returns [] (no chart) when:
      • result has < 2 rows  — single-value aggregation, chart is meaningless
      • query_type is 'scalar'
      • result is empty / has no columns
    """
    if result_df is None or result_df.empty or len(result_df.columns) < 1:
        return []

    # scalar queries never produce charts
    if query_type == "scalar":
        return []

    cols     = list(result_df.columns)
    n_rows   = len(result_df)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(result_df[c])]
    non_num  = [c for c in cols if c not in num_cols]

    # ── CRITICAL GATE ─────────────────────────────────────────────────────────
    # A 1-row result means one aggregated value (average, count, total, etc.).
    # A chart of a single bar or single histogram bin is meaningless noise.
    if n_rows < 2:
        return []

    # ── Time-series ───────────────────────────────────────────────────────────
    if query_type == "timeseries":
        dt_candidates = []
        for c in cols:
            if "datetime" in str(result_df[c].dtype).lower():
                dt_candidates.append(c)
            elif result_df[c].dtype == object:
                try:
                    pd.to_datetime(result_df[c].dropna().head(5))
                    dt_candidates.append(c)
                except Exception:
                    pass

        if not dt_candidates and non_num:
            dt_candidates = [non_num[0]]

        if not dt_candidates and num_cols:
            return [{"type": "line", "x": "__index__", "y": num_cols[0],
                     "_note": "use_index_as_x"}]

        x_col = dt_candidates[0]
        y_col = num_cols[0] if num_cols else cols[-1]
        return [{"type": "line", "x": x_col, "y": y_col}]

    # ── Visualization / tabular ───────────────────────────────────────────────
    if len(cols) == 1:
        # Single column histogram only useful with multiple distinct values
        if n_rows >= 5 and num_cols:
            return [{"type": "histogram", "x": cols[0], "y": None}]
        return []

    if len(cols) >= 2:
        x_col = non_num[0] if non_num else cols[0]
        y_col = num_cols[0] if num_cols else cols[1]
        chart_type = "bar"
        if not non_num and len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
            chart_type = "scatter"
        return [{"type": chart_type, "x": x_col, "y": y_col}]

    return []


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def _dataset_ctx(df: pd.DataFrame, db_path, dt_cols: list) -> str:
    if db_path:
        return (
            f"SQLITE — `DB_PATH` pre-loaded (do NOT redefine).\n"
            f"Schema:\n{get_sqlite_schema(db_path)}\n"
            "Use: import sqlite3; conn=sqlite3.connect(DB_PATH); pd.read_sql_query(sql,conn).\n"
            "Assign final answer to `RESULT`."
        )
    dt_hint = (
        f"\nKNOWN DATETIME COLUMNS: {dt_cols}. "
        "Parse with pd.to_datetime(df[col], infer_datetime_format=True) before any time ops."
        if dt_cols else ""
    )
    return (
        f"DATASET — variable `df` already loaded.\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} cols.\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n\n"
        f"Sample (3 rows):\n{df.head(3).to_string()}"
        f"{dt_hint}\n"
        "Do NOT read any file. Assign final answer to `RESULT`."
    )


def generate_code_prompt(question: str, df: pd.DataFrame,
                          db_path, query_type: str, dt_cols: list) -> str:
    ctx = _dataset_ctx(df, db_path, dt_cols)

    type_rules = {
        "timeseries": (
            "TIME-SERIES CODE RULES — READ CAREFULLY:\n"
            f"• Datetime columns detected: {dt_cols if dt_cols else 'none — try parsing object cols'}.\n"
            "• Step 1: parse the date column → pd.to_datetime(df['<col>'], infer_datetime_format=True, errors='coerce').\n"
            "• Step 2: set it as the DataFrame index or keep it as a named column.\n"
            "• Step 3: resample or groupby + aggregate numeric values.\n"
            "• Step 4: call reset_index() so the date is a NAMED COLUMN, not the index.\n"
            "• RESULT must be a DataFrame with AT LEAST TWO columns: one date/period column + one numeric column.\n"
            "• Name the date column clearly (e.g. 'Month', 'Date', 'Period').\n"
            "• Name the numeric column clearly (e.g. 'Revenue', 'Count', 'Total').\n"
            "• Cap at 60 periods. NO plotting libraries."
        ),
        "visualization": (
            "CHART CODE RULES:\n"
            "• Aggregate data; cap at top 20 rows.\n"
            "• RESULT must be a clean DataFrame with clear column names.\n"
            "• NO plotting libraries."
        ),
        "tabular": (
            "TABULAR CODE RULES:\n"
            "• RESULT must be a well-named DataFrame.\n"
            "• Column names must be clear and human-readable.\n"
            "• NO plotting libraries."
        ),
        "scalar": (
            "SCALAR CODE RULES:\n"
            "• RESULT may be a single value, string, or small DataFrame.\n"
            "• NO plotting libraries."
        ),
    }

    return f"""You are an elite Data Scientist. Write Python code to answer the query.

USER QUERY: "{question}"

{ctx}

GENERAL RULES:
1. Output ONLY valid executable Python — no markdown fences, no prose.
2. Use pd.to_numeric(..., errors='coerce') and dropna() before arithmetic.
3. Never use $ — write "dollars" or "USD".
4. Never import matplotlib, seaborn, plotly, or any visualisation library.
5. ALWAYS call reset_index() before assigning to RESULT if you used resample() or set_index().

{type_rules.get(query_type, '')}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

_FAIL_METRICS = {
    "fvr": 0.0, "her": 1.0, "cce": 1.0,
    "eer": 1.0, "scr": 0.0, "qrt": 0.0,
    "hr":  1.0, "ngs": 0.0, "acs": 0.0,
    "tokens_used": 0,
}


def analyze(
    question:          str,
    df:                pd.DataFrame,
    db_path            = None,
    correction_context = None,
) -> dict:
    """
    Full analysis pipeline with RAG, auto chart inference, and research logging.

    Result dict keys
    ----------------
    insights, metrics, processing_time, correction_log,
    executed_code, needs_correction, query_type, chart_specs,
    rag_context_used, tokens_used, error_class,
    [error_trace], [attempt]
    """
    t0              = time.time()
    stats           = compute_dataset_statistics(df)
    dt_cols         = detect_datetime_columns(df)
    current_attempt = (correction_context.get("attempt", 1) + 1
                       if correction_context else 1)
    query_type      = classify_query(question)
    total_tokens    = 0

    # ── RAG retrieval (used for code context hints + synthesis) ───────────────
    rag_context = retrieve_context(question, query_type=query_type)

    # ── Code generation ───────────────────────────────────────────────────────
    code_prompt = generate_code_prompt(question, df, db_path, query_type, dt_cols)
    messages = [
        {"role": "system", "content": "You output only python code. No markdown fences."},
        {"role": "user",   "content": code_prompt},
    ]
    if rag_context:
        # Inject RAG as a system hint so the LLM knows column meanings
        messages[0]["content"] += (
            f"\n\nRAG CONTEXT (dataset knowledge):\n{rag_context}"
        )
    if correction_context:
        messages += [
            {"role": "assistant", "content": correction_context["code"]},
            {"role": "user", "content": (
                f"Execution failed with this error:\n{correction_context['error']}\n\n"
                "Fix the code. Match exact column names from the schema. "
                "Remember: call reset_index() after resample(). "
                "Output ONLY valid Python code, no markdown."
            )},
        ]

    try:
        resp           = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.1
        )
        generated_code = resp.choices[0].message.content
        total_tokens  += getattr(resp.usage, "total_tokens", 0)

        # Strip accidental markdown fences the model sometimes adds
        generated_code = re.sub(r'^```(?:python)?\s*', '', generated_code.strip())
        generated_code = re.sub(r'\s*```$', '', generated_code.strip())

        execution = execute_pandas_code(generated_code, df, db_path)

        corr_log = [{
            "attempt":       current_attempt,
            "code":          generated_code,
            "status":        execution["status"],
            "error_message": execution["output"] if execution["status"] == "error" else None,
            "error_class":   classify_error(str(execution.get("output", "")))
                             if execution["status"] == "error" else None,
        }]

        if execution["status"] == "error":
            err_class = classify_error(str(execution["output"]))
            metrics_tracker.record_query(
                query=question, insights="", stats=stats,
                assigned_confidence=0.0, execution_failed=True,
                needed_correction=True, correction_succeeded=False,
                response_time_s=time.time() - t0,
            )
            elapsed = round(time.time() - t0, 2)
            return {
                "insights": (
                    f"⚠️ **Execution Failed — Attempt {current_attempt}** "
                    f"[`{err_class}`]\n\n"
                    "Click **Trigger Self-Correction** to auto-fix."
                ),
                "metrics":           {**_FAIL_METRICS, "qrt": elapsed,
                                       "tokens_used": total_tokens},
                "processing_time":   elapsed,
                "correction_log":    corr_log,
                "executed_code":     generated_code,
                "needs_correction":  True,
                "error_trace":       execution["output"],
                "error_class":       err_class,
                "attempt":           current_attempt,
                "query_type":        query_type,
                "chart_specs":       [],
                "rag_context_used":  bool(rag_context),
                "tokens_used":       total_tokens,
            }

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return {
            "insights":          f"Critical API Error: {e}",
            "metrics":           {**_FAIL_METRICS, "qrt": elapsed},
            "processing_time":   elapsed,
            "correction_log":    [],
            "executed_code":     "API Failure",
            "needs_correction":  False,
            "query_type":        query_type,
            "chart_specs":       [],
            "rag_context_used":  False,
            "tokens_used":       0,
            "error_class":       "APIError",
        }

    # ── Auto-infer chart specs from the ACTUAL result ─────────────────────────
    result_obj  = execution["output"]
    chart_specs = []
    if isinstance(result_obj, pd.DataFrame) and query_type in (
        "timeseries", "visualization", "tabular"
    ):
        chart_specs = _infer_chart_spec(result_obj, query_type, question)

    # ── Synthesis with RAG context ────────────────────────────────────────────
    try:
        # Build readable version of result for synthesis
        if isinstance(result_obj, pd.DataFrame):
            # For TS/tabular: readable markdown table
            raw_md = result_obj.head(20).to_markdown(index=False)
            # Also give column names explicitly so LLM can reference them
            col_info = f"Result columns: {list(result_obj.columns)}"
            raw_output = f"{col_info}\n\n{raw_md}"
        else:
            raw_output = str(result_obj)
            if len(raw_output) > 2500:
                raw_output = raw_output[:2500] + "\n...[TRUNCATED]..."

        synth_prompt = build_rag_synthesis_prompt(
            question     = question,
            raw_output   = raw_output,
            rag_context  = rag_context,
            query_type   = query_type,
            data_summary = stats,
        )

        synth_resp  = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional data analyst."},
                {"role": "user",   "content": synth_prompt},
            ],
            temperature=0.1,
        )
        insights      = synth_resp.choices[0].message.content
        total_tokens += getattr(synth_resp.usage, "total_tokens", 0)

        # Enrich stats with dynamic result values for FVR cross-checking
        all_nums = (
            re.findall(r'-?\d+\.?\d*', str(result_obj)) +
            re.findall(r'-?\d+\.?\d*', question)
        )
        if all_nums:
            stats.setdefault("numeric_summary", {})["_dynamic"] = {
                f"v{i}": float(n)
                for i, n in enumerate(all_nums)
                if _safe_float(n) is not None
            }

        fvr_data  = compute_fvr_her(insights, stats)
        base_conf = max(1.0 - (current_attempt - 1) * 0.15, 0.0)
        cce_data  = compute_cce(base_conf, fvr_data["fvr"])

        metrics_tracker.record_query(
            query=question, insights=insights, stats=stats,
            assigned_confidence=base_conf, execution_failed=False,
            needed_correction=correction_context is not None,
            correction_succeeded=correction_context is not None,
            response_time_s=time.time() - t0,
        )

        elapsed = round(time.time() - t0, 2)
        return {
            "insights": insights,
            "metrics": {
                "fvr":          round(fvr_data["fvr"], 4),
                "her":          round(fvr_data["her"], 4),
                "cce":          round(cce_data["cce"], 4),
                "eer":          0.0,
                "scr":          1.0 if correction_context else 0.0,
                "qrt":          elapsed,
                "tokens_used":  total_tokens,
                # legacy aliases
                "hr":  round(fvr_data["her"], 4),
                "ngs": round(fvr_data["fvr"], 4),
                "acs": 1.0,
            },
            "processing_time":  elapsed,
            "correction_log":   corr_log,
            "executed_code":    generated_code,
            "needs_correction": False,
            "query_type":       query_type,
            "chart_specs":      chart_specs,
            "rag_context_used": bool(rag_context),
            "tokens_used":      total_tokens,
            "error_class":      None,
            "claim_details":    fvr_data.get("claim_details", []),
        }

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return {
            "insights":         f"Code executed, but synthesis failed: {e}",
            "metrics":          {**_FAIL_METRICS, "qrt": elapsed,
                                  "tokens_used": total_tokens},
            "processing_time":  elapsed,
            "correction_log":   corr_log,
            "executed_code":    generated_code,
            "needs_correction": False,
            "query_type":       query_type,
            "chart_specs":      chart_specs,
            "rag_context_used": bool(rag_context),
            "tokens_used":      total_tokens,
            "error_class":      None,
        }


def _safe_float(s):
    try:    return float(s)
    except: return None
