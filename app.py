"""
app.py  ·  Enterprise AI Data Analyst — Research Edition  v4
=============================================================
v4 changes
----------
• Visualization engine rebuilt: uses agent's deterministic chart_specs
  (auto-inferred from the ACTUAL result DataFrame) — no more "Column 'index'
  not found" errors
• __index__ special case handled: if TS result has the date as the index,
  we reset it before charting
• RAG status shown in UI per response ("RAG ✓ / ✗")
• Token usage shown in metrics strip
• Error taxonomy shown when execution fails
• Research dashboard extended: token cost chart, error taxonomy breakdown,
  RAG hit rate, query-type distribution
"""

import streamlit as st
import pandas as pd
import glob
import os
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import tempfile
import hashlib
from datetime import datetime

from agent import analyze
from rag import ingest_dataframe_info, retrieve_context
from evaluation_metrics import metrics_tracker
from dataset_store import save_dataset_benchmark, load_all_benchmarks, delete_benchmark
from hypothesis_engine import calculate_p_values


# ══════════════════════════════════════════════════════════════════════════════
# Universal cross-table JOIN resolver
# Works for ANY SQLite / .db file — CSV files always have one table so this
# path is never reached for them.
# ══════════════════════════════════════════════════════════════════════════════

def _get_pragma_fk_map(db_path: str) -> dict:
    """
    Read PRAGMA foreign_key_list for every table in the DB.
    Returns a dict keyed by (from_table, to_table) -> (from_col, to_col).
    Both directions are stored so lookup is always symmetric.
    """
    fk_map = {}
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        for tbl in tables:
            cursor.execute(f'PRAGMA foreign_key_list("{tbl}")')
            for row in cursor.fetchall():
                # row: (id, seq, ref_table, from_col, to_col, ...)
                _, _, ref_tbl, from_col, to_col = row[:5]
                fk_map[(tbl, ref_tbl)]  = (from_col, to_col)
                fk_map[(ref_tbl, tbl)]  = (to_col, from_col)
        conn.close()
    except Exception:
        pass
    return fk_map


def _infer_join_key(t_tbl, f_tbl, all_tables, db_path):
    """
    Return (t_key, f_key) columns to JOIN on, or (None, None).

    Strategy (most-reliable first):
    1. SQLite PRAGMA foreign_key_list - actual FK constraints.
    2. Shared column names ranked by ID-likeness (e.g. 'GenreId' > 'Name').
    Works for ANY schema - never hardcoded.
    """
    # 1. Real FK via PRAGMA
    if db_path:
        fk_map = _get_pragma_fk_map(db_path)
        if (t_tbl, f_tbl) in fk_map:
            return fk_map[(t_tbl, f_tbl)]

    # 2. Shared column heuristic
    if t_tbl not in all_tables or f_tbl not in all_tables:
        return None, None
    t_cols = set(all_tables[t_tbl].columns)
    f_cols = set(all_tables[f_tbl].columns)
    shared = t_cols & f_cols
    if not shared:
        return None, None

    def _id_score(name):
        n = name.lower()
        if n.endswith("id") or n.startswith("id"): return 3
        if "id" in n:                               return 2
        if "key" in n or "fk" in n or "code" in n: return 1
        return 0

    best = sorted(shared, key=lambda c: (-_id_score(c), c))
    jk   = best[0]
    return jk, jk


def _build_cross_table_df(t_tbl, t_col, f_tbl, f_col, all_tables, db_path):
    """
    Build a two-column DataFrame for hypothesis testing from two tables.
    Returns (combined_df, method_description_str).
    Raises ValueError with a human-readable message on failure.

    Priority:
    1. SQL JOIN via db_path (cleanest - uses actual relational keys).
    2. pandas merge in-memory (when tables already loaded, no db_path).
    3. Row-position alignment as last resort (with explicit warning).
    """
    t_key, f_key = _infer_join_key(t_tbl, f_tbl, all_tables, db_path)

    # -- SQL JOIN (preferred for any SQLite / .db file) -----------------------
    if db_path and t_key and f_key:
        sql = (
            f'SELECT t."{t_col}", f."{f_col}" '
            f'FROM "{t_tbl}" t '
            f'JOIN "{f_tbl}" f ON t."{t_key}" = f."{f_key}"'
        )
        try:
            conn   = sqlite3.connect(db_path)
            df_out = pd.read_sql_query(sql, conn)
            conn.close()
            if t_col == f_col and len(df_out.columns) == 2:
                df_out.columns = [f"{t_col}_target", f"{f_col}_feature"]
                t_use, f_use   = df_out.columns[0], df_out.columns[1]
            else:
                t_use, f_use = t_col, f_col
            if t_use not in df_out.columns or f_use not in df_out.columns:
                raise ValueError(f"JOIN result missing expected columns. Got: {list(df_out.columns)}")
            combined = df_out[[t_use, f_use]].copy()
            combined.columns = [t_col, f_col]
            combined = combined.dropna()
            if len(combined) < 3:
                raise ValueError(f"JOIN on '{t_key}' returned only {len(combined)} rows.")
            return combined, f"SQL JOIN on `{t_tbl}.{t_key}` = `{f_tbl}.{f_key}`"
        except Exception as e:
            raise ValueError(f"SQL JOIN failed: {e}")

    # -- Fix 5: Multi-hop JOIN (3-table chain) when no direct FK exists -------
    if db_path and not (t_key and f_key):
        fk_map = _get_pragma_fk_map(db_path)
        multi_sql = _build_multi_hop_join_sql(t_tbl, t_col, f_tbl, f_col, fk_map, all_tables)
        if multi_sql:
            try:
                conn     = sqlite3.connect(db_path)
                df_out   = pd.read_sql_query(multi_sql, conn)
                conn.close()
                if t_col not in df_out.columns or f_col not in df_out.columns:
                    raise ValueError(f"Multi-hop JOIN missing columns. Got: {list(df_out.columns)}")
                combined = df_out[[t_col, f_col]].dropna()
                if len(combined) < 3:
                    raise ValueError(f"Multi-hop JOIN returned only {len(combined)} rows.")
                return combined, f"SQL multi-hop JOIN (bridge table)"
            except Exception as e:
                raise ValueError(f"Multi-hop JOIN failed: {e}")

    # -- Pandas merge (in-memory tables, no db_path) --------------------------
    if t_key and f_key and t_tbl in all_tables and f_tbl in all_tables:
        try:
            t_sub  = all_tables[t_tbl][[t_col, t_key]] if t_key != t_col else all_tables[t_tbl][[t_col]]
            f_sub  = all_tables[f_tbl][[f_col, f_key]] if f_key != f_col else all_tables[f_tbl][[f_col]]
            merged = pd.merge(
                t_sub, f_sub,
                left_on=t_key, right_on=f_key, how="inner"
            )[[t_col, f_col]].dropna()
            if len(merged) < 3:
                raise ValueError(
                    f"Merge on '{t_key}' returned only {len(merged)} rows."
                )
            return merged, f"in-memory merge on `{t_key}`"
        except Exception as e:
            raise ValueError(f"In-memory merge failed: {e}")

    # -- Last resort: positional alignment (warn loudly) ----------------------
    t_s = all_tables.get(t_tbl, pd.DataFrame()).get(t_col)
    f_s = all_tables.get(f_tbl, pd.DataFrame()).get(f_col)
    if t_s is None:
        raise ValueError(f"Column '{t_col}' not found in table '{t_tbl}'.")
    if f_s is None:
        raise ValueError(f"Column '{f_col}' not found in table '{f_tbl}'.")
    n        = min(len(t_s), len(f_s))
    combined = pd.DataFrame({
        t_col: t_s.iloc[:n].values,
        f_col: f_s.iloc[:n].values,
    }).dropna()
    method   = (
        "⚠️ row-position alignment (no join key found — results may be unreliable). "
        "Add a FOREIGN KEY to your schema for reliable cross-table tests."
    )
    return combined, method


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Data Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px !important;
}
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important; color: #58a6ff !important;
}
[data-testid="stChatMessage"] {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] > div {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
.stCodeBlock { background: #0d1117 !important; border: 1px solid #30363d !important; }
.streamlit-expanderHeader {
    background: #161b22 !important; border-radius: 6px !important;
    font-size: 0.82rem !important; color: #8b949e !important;
}
h1,h2,h3 { color: #e6edf3 !important; }
.stAlert { border-radius: 8px !important; }
[data-baseweb="tab-list"] { background: #161b22 !important; border-radius: 8px; }
[data-baseweb="tab"] { color: #8b949e !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #58a6ff !important; }
.qtype-badge {
    display:inline-block; font-family:'IBM Plex Mono',monospace;
    font-size:0.66rem; padding:2px 8px; border-radius:4px; margin-right:6px;
}
.qt-timeseries   { background:#0d3349; color:#58a6ff; border:1px solid #1f6feb; }
.qt-visualization{ background:#1a2d1a; color:#3fb950; border:1px solid #238636; }
.qt-tabular      { background:#2d1f00; color:#d29922; border:1px solid #9e6a03; }
.qt-scalar       { background:#1f1235; color:#bc8cff; border:1px solid #6e40c9; }
.rag-badge {
    display:inline-block; font-family:'IBM Plex Mono',monospace;
    font-size:0.66rem; padding:2px 8px; border-radius:4px;
}
.rag-on  { background:#1a2d1a; color:#3fb950; border:1px solid #238636; }
.rag-off { background:#2d1a1a; color:#f85149; border:1px solid #da3633; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages": [], "current_file": None, "db_path": None,
    "df": None, "file_type": "csv",
    "all_tables": {},          # {table_name: DataFrame} — populated for SQLite/DB files
    "needs_correction": False, "correction_data": {},
    "ts_count": 0, "tab_count": 0,
    "total_tokens": 0, "rag_hits": 0, "rag_total": 0,
    "error_taxonomy": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

METRIC_HELP = {
    "FVR": "Factual Verification Rate — fraction of numerical claims traceable to ground truth. [0→1] ↑ better",
    "HER": "Hallucination Error Rate — 1−FVR. Fraction of unsupported claims. [0→1] ↓ better",
    "CCE": "Confidence Calibration Error — |model confidence − actual FVR|. [0→1] ↓ better",
    "QRT": "Query Response Time (wall-clock seconds). ↓ better",
    "TOK": "Total LLM tokens used for code generation + synthesis (cost proxy).",
}

def _dark(fig):
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font_color="#e6edf3", title_font_color="#58a6ff",
        legend=dict(bgcolor="#0d1117"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Bulletproof visualization engine
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_plot_df(executed_code: str, df_original, db_path) -> pd.DataFrame:
    """Re-execute the agent's code to recover the RESULT DataFrame for charting."""
    if not executed_code or executed_code in ("API Failure", ""):
        return pd.DataFrame()
    env = {
        "df": df_original.copy() if df_original is not None else pd.DataFrame(),
        "DB_PATH": db_path, "pd": pd, "sqlite3": sqlite3, "RESULT": None,
    }
    try:
        code = re.sub(r'^```(?:python)?\s*', '', executed_code.strip())
        code = re.sub(r'\s*```$', '', code.strip())
        exec(code, env)
        res = env.get("RESULT")
        if isinstance(res, pd.DataFrame):
            # Fix interval/category dtypes that break JSON serialization
            for col in res.columns:
                if any(t in str(res[col].dtype) for t in ["interval", "category"]):
                    res[col] = res[col].astype(str)
            return res
        elif isinstance(res, pd.Series):
            return res.reset_index()
    except Exception:
        pass
    return pd.DataFrame()


def _make_fig(chart_spec: dict, plot_df: pd.DataFrame):
    """
    Build a single Plotly figure from one chart spec dict and a DataFrame.
    Handles the __index__ sentinel (TS result where date was the index).
    """
    if plot_df is None or plot_df.empty:
        return None

    ctype = chart_spec.get("type", "bar").lower()
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")

    # ── Handle index-as-x case ────────────────────────────────────────────────
    if x_col == "__index__":
        plot_df = plot_df.reset_index()
        # The old index is now a column named 'index' or the original index name
        x_col = plot_df.columns[0]   # first col after reset is the former index

    # ── Validate columns ──────────────────────────────────────────────────────
    if x_col and x_col not in plot_df.columns:
        # Last-ditch: try to find a column whose name contains x_col (case-insensitive)
        matches = [c for c in plot_df.columns if x_col.lower() in c.lower()]
        if matches:
            x_col = matches[0]
        else:
            return None     # genuinely can't plot this

    if y_col and y_col not in plot_df.columns:
        num_cols = [c for c in plot_df.columns
                    if pd.api.types.is_numeric_dtype(plot_df[c]) and c != x_col]
        y_col = num_cols[0] if num_cols else None

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = None
    try:
        if ctype == "line":
            if y_col:
                # Ensure x is sorted for a clean line chart
                try:
                    plot_df = plot_df.copy()
                    plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="coerce")
                    plot_df = plot_df.sort_values(x_col)
                    plot_df[x_col] = plot_df[x_col].astype(str)  # plotly needs strings for axis
                except Exception:
                    pass
                fig = px.line(plot_df, x=x_col, y=y_col,
                              title=f"{y_col} over {x_col}", markers=True)
                fig.update_traces(line=dict(color="#58a6ff", width=2),
                                  marker=dict(size=5, color="#58a6ff"))

        elif ctype in ("bar", "bar_chart"):
            if y_col and y_col in plot_df.columns:
                agg = (plot_df if len(plot_df) <= 100
                       else plot_df.groupby(x_col)[y_col].mean().reset_index())
                agg = agg.sort_values(y_col, ascending=False).head(20)
                fig = px.bar(agg, x=x_col, y=y_col,
                             color=y_col, color_continuous_scale="Blues",
                             title=f"{y_col} by {x_col}")
            elif x_col:
                vc = plot_df[x_col].value_counts().reset_index().head(20)
                vc.columns = [x_col, "count"]
                fig = px.bar(vc, x=x_col, y="count",
                             title=f"Distribution of {x_col}")

        elif ctype in ("scatter", "scatter_plot"):
            if y_col and y_col in plot_df.columns:
                fig = px.scatter(plot_df, x=x_col, y=y_col,
                                 title=f"{y_col} vs {x_col}",
                                 trendline="ols" if len(plot_df) > 5 else None)

        elif ctype == "histogram":
            target = y_col if (y_col and y_col in plot_df.columns) else x_col
            if target:
                if len(plot_df.columns) == 2:
                    y_auto = [c for c in plot_df.columns if c != target][0]
                    fig = px.bar(plot_df, x=target, y=y_auto,
                                 title=f"Histogram of {target}")
                else:
                    fig = px.histogram(plot_df, x=target, nbins=30,
                                       title=f"Histogram of {target}")

        elif ctype in ("box", "box_plot"):
            if y_col and y_col in plot_df.columns:
                fig = px.box(plot_df, x=x_col, y=y_col,
                             title=f"{y_col} by {x_col}")
            elif x_col:
                fig = px.box(plot_df, y=x_col, title=f"Box plot of {x_col}")

        elif ctype == "pie":
            if y_col and y_col in plot_df.columns:
                fig = px.pie(plot_df, names=x_col, values=y_col,
                             title=f"{y_col} by {x_col}")

    except Exception as e:
        st.warning(f"Chart render error ({ctype}): {e}")
        return None

    return _dark(fig) if fig else None


# ══════════════════════════════════════════════════════════════════════════════
# MLR helpers — query rewriting + dedicated result card
# ══════════════════════════════════════════════════════════════════════════════

def _validate_mlr_columns(user_question: str, df, all_tables: dict) -> tuple:
    """
    Pre-flight check: do the question's domain terms match any column OR table
    in the loaded dataset? Blocks hallucinated analysis on wrong datasets.
    Returns (is_valid, error_message_or_empty_string).

    Conservative by design — only blocks when it is highly confident the query
    is for a completely different dataset. Errs on the side of allowing queries.
    """
    import re as _re

    # Build normalised set of ALL known schema terms:
    # column names + table names (both are valid references in a question)
    all_schema_terms = set()
    if all_tables:
        for tbl_name, tdf in all_tables.items():
            # Add table name itself (e.g. "invoice", "customer")
            all_schema_terms.add(tbl_name.lower().replace("_", " "))
            # Add singular form (e.g. "invoices" → matches "invoice")
            tbl_s = tbl_name.lower().rstrip("s")
            all_schema_terms.add(tbl_s)
            # Add column names
            for c in tdf.columns:
                all_schema_terms.add(c.lower().replace("_", " ").replace(".", " "))
    if df is not None:
        try:
            if len(df.columns) > 0:
                for c in df.columns:
                    all_schema_terms.add(c.lower().replace("_", " ").replace(".", " "))
        except Exception:
            pass

    if not all_schema_terms:
        return True, ""  # no schema loaded yet — always allow

    stops = {
        # English grammar / function words
        "is","are","do","does","did","was","were","be","been","being",
        "the","a","an","of","in","on","at","to","for","and","or","but",
        "by","with","from","that","this","these","those","it","its",
        "how","what","which","who","when","where","why","whether",
        "have","has","had","will","would","could","should","may","might",
        # UI command words — these are instructions not data references
        "show","list","give","find","get","plot","chart","display","print",
        "tell","make","create","generate","calculate","compute","return",
        "fetch","load","read","run","execute","perform","analyze","analyse",
        # Comparison / filter words — never column names
        "greater","lesser","higher","lower","more","less","than","equal",
        "above","below","between","over","under","within","outside",
        "largest","smallest","highest","lowest","biggest","most","least",
        # Statistical method words
        "significantly","simultaneously","together","predict","affect",
        "impact","influence","combined","effect","multiple","regression",
        "between","across","using","based","given","overall","each",
        "linear","logistic","multivariate","control","controlling","adjust",
        "average","mean","median","total","count","number","percent","rate",
        # Time words
        "year","month","week","day","date","time","period","quarter",
        # Generic English nouns that are never column references
        "data","value","values","result","results","report","information",
        "difference","relationship","correlation","distribution","trend",
        "amount","number","type","kind","category","group","level","range",
    }

    q_lower = user_question.lower()
    raw_words = _re.findall(r'\b[a-z]+\b', q_lower)
    words = [w for w in raw_words if w not in stops and len(w) > 3]

    # Build 1-grams and 2-grams as candidate schema references
    candidates = set(words)
    for i in range(len(words) - 1):
        candidates.add(f"{words[i]} {words[i+1]}")

    # Count how many candidates match any schema term (fuzzy: substring either way)
    matched = sum(
        1 for c in candidates
        if any(c in term or term in c for term in all_schema_terms)
    )

    # Only block when ALL of these are true:
    # 1. At least 4 distinct domain-length terms are mentioned  (raised from 3)
    # 2. Zero of them match anything in the schema
    # This is deliberately conservative — a false positive (blocking a valid
    # query) is worse than a false negative (letting a wrong-dataset query through,
    # which will just fail with a column-not-found error anyway).
    domain_terms = [c for c in candidates if len(c) > 4]
    if len(domain_terms) >= 4 and matched == 0:
        available = sorted(all_schema_terms)[:20]
        return False, (
            "**Dataset mismatch detected.**\n\n"
            f"The question refers to concepts "
            f"({', '.join(sorted(domain_terms)[:6])}) "
            "that do not appear in the currently loaded dataset.\n\n"
            f"**Schema available:** {', '.join(available)}...\n\n"
            "Please upload the correct dataset and try again."
        )
    return True, ""


def _build_mlr_query(user_question: str, df, db_path,
                     all_tables: dict) -> str:
    """
    Rewrite a free-form MLR question into an explicit, unambiguous OLS
    instruction. Injects full schema + anti-hallucination rules.
    """
    if db_path and all_tables:
        schema_lines = []
        all_col_names = []
        for tbl, tdf in all_tables.items():
            num_cols = [c for c in tdf.columns if pd.api.types.is_numeric_dtype(tdf[c])]
            cat_cols = [c for c in tdf.columns if not pd.api.types.is_numeric_dtype(tdf[c])]
            schema_lines.append(
                f"  Table '{tbl}' ({len(tdf)} rows):\n"
                f"    Numeric : {', '.join(num_cols) or 'none'}\n"
                f"    Text/Cat: {', '.join(cat_cols) or 'none'}"
            )
            all_col_names.extend(tdf.columns.tolist())
        schema_str = "\n".join(schema_lines)
        fk_map = _get_pragma_fk_map(db_path)
        fk_lines = [f"  {a}.{fc} → {b}.{tc}" for (a,b),(fc,tc) in fk_map.items() if a < b]
        fk_str = "\n".join(fk_lines) if fk_lines else "  (none declared)"
        data_src = (
            f"SQLite database. `DB_PATH` is pre-defined.\n"
            f"Use: import sqlite3; conn = sqlite3.connect(DB_PATH)\n\n"
            f"TABLES:\n{schema_str}\n\nFOREIGN KEY RELATIONSHIPS:\n{fk_str}"
        )
        db_rules = """
SQLITE RULES:
- JOIN tables as needed using the FK relationships listed above.
- Use CTEs to derive metrics that don\'t exist as raw columns.
- Load result into a DataFrame called reg_df.
- Encode text columns with pd.get_dummies(drop_first=True).
- Cast all columns to float before OLS.
"""
    else:
        cols_info = "\n".join(
            f"  {c}: {str(df[c].dtype)}  sample={df[c].dropna().unique()[:3].tolist()}"
            for c in df.columns
        )
        all_col_names = df.columns.tolist()
        data_src = f"CSV file. DataFrame already loaded as `df`.\nCOLUMNS:\n{cols_info}"
        db_rules = """
CSV RULES:
- Work on `df` directly. Do NOT read any file.
- Derive any needed metrics using groupby on df.
- Use reg_df = df.copy() as working DataFrame.
- Encode text columns with pd.get_dummies(drop_first=True).
"""

    col_list = ", ".join(all_col_names)
    return f"""You are an expert statistician. Write Python to run OLS Multiple Linear Regression.

USER QUESTION: "{user_question}"

DATA SOURCE:
{data_src}
{db_rules}
ANTI-HALLUCINATION — CRITICAL:
- ONLY use columns from this exact list: {col_list}
- If the question mentions variables NOT in this list, raise ValueError(
  "Columns not found in dataset: <list missing columns>") — do NOT invent data.
- NEVER use numpy.random, pd.DataFrame with hardcoded values, or any synthetic data.

OLS RULES:
1. import statsmodels.api as sm
2. Use EXACT column names from the list above. Approximate the closest real column if needed.
3. reg_df = reg_df.dropna()
4. Encode all text predictors with pd.get_dummies; ensure X is all-float.
5. X = sm.add_constant(X)
6. model = sm.OLS(y, X).fit()
7. RESULT = pd.DataFrame({{
       "Variable":    list(model.params.index),
       "Coefficient": [round(v,6) for v in model.params.values],
       "Std_Error":   [round(v,6) for v in model.bse.values],
       "t_value":     [round(v,6) for v in model.tvalues.values],
       "p_value":     [round(v,6) for v in model.pvalues.values],
   }})
8. _r2 = round(model.rsquared, 4)
9. _f_pval = round(model.f_pvalue, 6)
10. Output ONLY valid Python. No markdown, no prose, no comments."""

def _render_mlr_card(result: dict, question: str = "") -> None:
    """
    Render a Multiple Linear Regression result as a clean structured card.
    Reads pre-captured _mlr_coef_df / _mlr_r2 / _mlr_f_pval from result dict
    — never re-executes code.
    """
    import plotly.graph_objects as go

    coef_df = result.get("_mlr_coef_df")
    r2      = result.get("_mlr_r2")
    f_pval  = result.get("_mlr_f_pval")

    # If pre-capture failed (e.g. no DB_PATH at capture time), show plain insights
    if not isinstance(coef_df, pd.DataFrame) or coef_df.empty:
        st.markdown("**Regression executed but coefficient table could not be parsed.**")
        # Show insights but strip any code blocks from it
        raw = result.get("insights", "")
        clean = re.sub(r'```[\s\S]*?```', '', raw).strip()
        if clean:
            st.markdown(clean)
        return

    # Normalise column names (handle minor LLM variations)
    col_map = {}
    for c in coef_df.columns:
        cl = c.lower().replace(" ", "_")
        if "variable" in cl or cl == "index":          col_map[c] = "Variable"
        elif "coef" in cl:                             col_map[c] = "Coefficient"
        elif "std" in cl or "se" in cl or "err" in cl: col_map[c] = "Std_Error"
        elif "t_val" in cl or "tvalue" in cl or cl == "t": col_map[c] = "t_value"
        elif "p_val" in cl or "pvalue" in cl or cl == "p": col_map[c] = "p_value"
    coef_df = coef_df.rename(columns=col_map)

    # If Variable column still missing, use index
    if "Variable" not in coef_df.columns:
        coef_df = coef_df.reset_index().rename(columns={"index": "Variable"})

    has_pval = "p_value" in coef_df.columns
    has_coef = "Coefficient" in coef_df.columns

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        '<span style="font-size:1.35rem">📐</span>'
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:1.1rem;'
        'font-weight:700;color:#e6edf3">Multiple Linear Regression</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── p-value formatter (never shows 0.000000 for tiny values) ─────────────
    def _fmt_p(p):
        try:
            pf = float(p)
            if pf == 0 or pf < 1e-15: return "< 1e-15"
            if pf < 0.0001:           return f"{pf:.2e}"
            return f"{pf:.4f}"
        except Exception:
            return str(p)

    def _fmt_coef(v):
        """Show scientific notation for very small/large values."""
        try:
            vf = float(v)
            if vf == 0:              return "0.000000"
            if abs(vf) < 0.000001:   return f"{vf:.4e}"
            if abs(vf) > 1_000_000:  return f"{vf:.4e}"
            return f"{vf:.6f}"
        except Exception:
            return str(v)

    # ── Model fit banner ──────────────────────────────────────────────────────
    r2_str    = f"{r2:.4f}" if r2 is not None else "—"
    fpval_str = _fmt_p(f_pval) if f_pval is not None else "—"
    n_pred    = len(coef_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²  (variance explained)", r2_str,
              help="Fraction of outcome variance explained by all predictors. 1.0 = perfect fit.")
    c2.metric("Model F p-value", fpval_str,
              help="Is the overall model significantly better than predicting the mean?")
    c3.metric("Predictors (incl. const)", str(n_pred),
              help="Number of terms including intercept.")

    st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)

    # ── Coefficient table ─────────────────────────────────────────────────────
    def _sig_badge(p):
        try:
            pf = float(p)
            if pf < 0.001: return "🟢 p<0.001"
            if pf < 0.01:  return "🟢 p<0.01"
            if pf < 0.05:  return "🟡 p<0.05"
            return "⚪ n.s."
        except Exception:
            return "—"

    display_df = coef_df.copy()
    if has_pval:
        display_df["Significance"] = display_df["p_value"].apply(_sig_badge)
    # Format numeric columns properly — never show 0.000000 for tiny numbers
    for col, fmt_fn in [("Coefficient", _fmt_coef), ("Std_Error", _fmt_coef),
                        ("t_value", _fmt_coef), ("p_value", _fmt_p)]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_fn)

    st.markdown(
        '<span style="color:#8b949e;font-size:0.78rem;font-family:monospace">'
        '📋 Regression Coefficients</span>',
        unsafe_allow_html=True,
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Coefficient plot ──────────────────────────────────────────────────────
    if has_coef and len(coef_df) > 1:
        try:
            plot_df = coef_df.copy()
            # Skip intercept/const for cleaner plot
            if "Variable" in plot_df.columns:
                plot_df = plot_df[~plot_df["Variable"].str.lower().isin(
                    ["const", "intercept", "constant"]
                )]

            if not plot_df.empty:
                coefs = pd.to_numeric(plot_df["Coefficient"], errors="coerce")
                vars_ = (plot_df["Variable"].astype(str)
                         if "Variable" in plot_df.columns
                         else [str(i) for i in plot_df.index])
                pvals = (pd.to_numeric(plot_df["p_value"], errors="coerce")
                         if has_pval else pd.Series([0.05] * len(plot_df)))

                colors = ["#3fb950" if p < 0.05 else "#8b949e" for p in pvals]

                fig = go.Figure(go.Bar(
                    x=coefs, y=vars_, orientation="h",
                    marker_color=colors,
                    text=[_fmt_coef(c) for c in coefs],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Coefficients  (🟢 significant  ⚪ not significant)",
                    title_font=dict(color="#58a6ff", size=13),
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font_color="#e6edf3",
                    xaxis=dict(title="Coefficient Value", gridcolor="#21262d",
                               zerolinecolor="#30363d"),
                    yaxis=dict(gridcolor="#21262d", automargin=True),
                    height=max(260, len(plot_df) * 48 + 90),
                    margin=dict(l=10, r=80, t=40, b=40),
                )
                key = hashlib.md5(f"mlr_{question}".encode()).hexdigest()
                st.plotly_chart(fig, use_container_width=True, key=key)
        except Exception:
            pass

    # ── Plain-English verdict ─────────────────────────────────────────────────
    if has_pval and has_coef and r2 is not None:
        sig_vars, insig_vars = [], []
        for _, row in coef_df.iterrows():
            try:
                vname = str(row.get("Variable", row.name))
                if vname.lower() in ("const", "intercept", "constant"):
                    continue
                pf = float(row["p_value"])
                cf = float(row["Coefficient"])
                direction = "positively" if cf > 0 else "negatively"
                if pf < 0.05:
                    sig_vars.append(f"{vname} ({direction}, p={_fmt_p(pf)})")
                else:
                    insig_vars.append(vname)
            except Exception:
                pass

        r2_pct = f"{float(r2)*100:.1f}%" if r2 is not None else "—"
        if sig_vars:
            verdict = (f"The model explains **{r2_pct}** of the variance. "
                       f"Significant predictors: **{', '.join(sig_vars)}**.")
            if insig_vars:
                verdict += f" Non-significant: {', '.join(insig_vars)}."
        else:
            verdict = (f"The model explains **{r2_pct}** of the variance, "
                       f"but **none** of the predictors reached significance (p < 0.05).")

        st.markdown(
            f'<div style="background:#1a1f2e;border-left:3px solid #58a6ff;'
            f'padding:10px 16px;border-radius:6px;margin-top:8px">'
            f'<span style="color:#8b949e;font-size:0.72rem;font-family:monospace">'
            f'💡 Interpretation</span><br>'
            f'<span style="color:#e6edf3;font-size:0.9rem">{verdict}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _safe_is_num(x) -> bool:
    try: float(x); return True
    except Exception: return False


def render_insights_and_charts(
    insights_text: str,
    df,
    db_path        = None,
    executed_code: str = "",
    query_type:    str = "scalar",
    chart_specs:   list = None,
):
    """
    Render insights text (supports embedded markdown tables) and all charts.

    Chart specs come from the agent's deterministic _infer_chart_spec() output.
    Any JSON block in the LLM text is STRIPPED (the agent no longer needs to
    produce it — we use chart_specs directly).
    """
    # Strip any JSON chart block the synthesis LLM may have appended
    clean_text = re.sub(r'```(?:json)?\s*\[.*?\]\s*```', '', insights_text, flags=re.DOTALL)
    clean_text = re.sub(r'\[\s*\{[^}]*"type"\s*:.*?\}\s*\]', '', clean_text, flags=re.DOTALL)
    clean_text = clean_text.strip()

    st.markdown(clean_text)

    # Determine which specs to use
    # chart_specs come from agent's deterministic _infer_chart_spec() which
    # already gates on row count and query_type — trust them directly.
    specs_to_use = chart_specs or []

    # Fallback: only attempt if agent returned no specs AND query genuinely
    # expects a multi-row chart (timeseries or explicit visualization request).
    # Tabular scalar results (1 row) intentionally produce no chart.
    needs_chart = query_type in ("timeseries", "visualization")
    if needs_chart and not specs_to_use and executed_code:
        from agent import _infer_chart_spec
        fallback_df = _rebuild_plot_df(executed_code, df, db_path)
        if not fallback_df.empty and len(fallback_df) >= 2:
            specs_to_use = _infer_chart_spec(fallback_df, query_type, insights_text)

    if not specs_to_use:
        return

    # Rebuild plot_df once for all specs
    plot_df = _rebuild_plot_df(executed_code, df, db_path)
    if plot_df.empty:
        st.caption("ℹ️ Chart data unavailable (code re-execution returned empty result).")
        return

    st.markdown("#### 📈 Visualisation")
    n_charts = len(specs_to_use)
    cols_ui  = st.columns(min(n_charts, 2)) if n_charts > 1 else [st.container()]

    for i, spec in enumerate(specs_to_use):
        col_ctx = cols_ui[i % len(cols_ui)] if n_charts > 1 else cols_ui[0]
        with col_ctx:
            fig = _make_fig(spec, plot_df.copy())
            if fig:
                key = hashlib.md5(f"{insights_text}{i}".encode()).hexdigest()
                st.plotly_chart(fig, use_container_width=True, key=key)
            else:
                st.caption(
                    f"Could not render chart (type={spec.get('type')}, "
                    f"x={spec.get('x')}, y={spec.get('y')}). "
                    f"Available columns: {list(plot_df.columns)}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Per-response metric strip
# ══════════════════════════════════════════════════════════════════════════════

def render_metric_strip(metrics: dict, query_type: str, rag_used: bool,
                        tokens: int, error_class=None):
    qt_labels = {
        "timeseries": ("⏱ Time-Series", "qt-timeseries"),
        "visualization": ("📊 Visualisation", "qt-visualization"),
        "tabular": ("📋 Tabular", "qt-tabular"),
        "scalar": ("🔢 Scalar", "qt-scalar"),
    }
    label, css = qt_labels.get(query_type, ("🔢 Scalar", "qt-scalar"))
    rag_cls  = "rag-on"  if rag_used else "rag-off"
    rag_txt  = "RAG ✓"   if rag_used else "RAG ✗"

    st.markdown(
        f'<span class="qtype-badge {css}">{label}</span>'
        f'<span class="rag-badge {rag_cls}">{rag_txt}</span>',
        unsafe_allow_html=True,
    )

    if error_class:
        st.error(f"🔴 Error type: **{error_class}**")

    fvr = metrics.get("fvr", 0.0)
    her = metrics.get("her", 1.0)
    cce = metrics.get("cce", 0.0)
    qrt = metrics.get("qrt", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FVR ↑", f"{fvr:.1%}",  help=METRIC_HELP["FVR"])
    c2.metric("HER ↓", f"{her:.1%}",
              delta=f"-{her:.1%}" if her > 0 else "✓ 0%",
              delta_color="inverse", help=METRIC_HELP["HER"])
    c3.metric("CCE ↓", f"{cce:.4f}", help=METRIC_HELP["CCE"])
    c4.metric("QRT",   f"{qrt:.2f}s", help=METRIC_HELP["QRT"])
    c5.metric("Tokens", f"{tokens:,}", help=METRIC_HELP["TOK"])



# ══════════════════════════════════════════════════════════════════════════════
# Claim Inspector — shows which numbers were verified vs hallucinated
# ══════════════════════════════════════════════════════════════════════════════

def render_claim_inspector(claim_details: list, insights: str):
    """
    Render a collapsible panel showing every numerical claim extracted from
    the LLM's synthesis, whether it was verified against the execution result,
    and what ground-truth value it matched (if any).

    Color coding:
      ✅ green  — verified: found a matching value in the execution result
      ❌ red    — unverified: no match found (potential hallucination)
    """
    if not claim_details:
        return

    verified_count   = sum(1 for c in claim_details if c["verified"])
    unverified_count = len(claim_details) - verified_count

    label = (
        f"🔍 Claim Inspector — "
        f"{verified_count} verified ✅  |  {unverified_count} unverified ❌  "
        f"(of {len(claim_details)} numerical claims)"
    )

    with st.expander(label, expanded=(unverified_count > 0)):
        st.caption(
            "Every number extracted from the AI's response is checked against "
            "the actual code-execution result. Green = found a matching value. "
            "Red = no match found in the ground truth."
        )

        # Build display rows
        rows = []
        for c in claim_details:
            if c["verified"]:
                status   = "✅ Verified"
                match    = f"{c['matched_truth']:g}" if c["matched_truth"] is not None else "—"
                delta    = f"{c['delta']:g}" if c["delta"] is not None else "—"
            else:
                status   = "❌ Unverified"
                closest  = c.get("closest_truth")
                match    = f"{closest:g} (no match)" if closest is not None else "No ground truth"
                delta    = f"{c['delta']:g}" if c["delta"] is not None else "—"

            rows.append({
                "Claimed Value": f"{c['value']:g}",
                "Status":        status,
                "Closest Ground Truth": match,
                "Abs. Difference": delta,
            })

        import pandas as _pd
        df = _pd.DataFrame(rows)

        # Colour-code the Status column using Styler
        def _colour(val):
            return "color: #3fb950; font-weight:600" if "Verified" in val else "color: #f85149; font-weight:600"

        styled = df.style.applymap(_colour, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Also highlight unverified claims inline in the synthesis text
        if unverified_count > 0:
            st.markdown("**Unverified values (highlighted in synthesis):**")
            highlighted = insights
            for c in claim_details:
                if not c["verified"]:
                    val_str = str(c["value"]).rstrip("0").rstrip(".")
                    # Wrap in a red highlight — use HTML mark tag
                    highlighted = highlighted.replace(
                        val_str,
                        f'<mark style="background:#3d1010;color:#f85149;'
                        f'border-radius:3px;padding:0 3px">{val_str}</mark>',
                        1,  # replace first occurrence only
                    )
            st.markdown(highlighted, unsafe_allow_html=True)


def _format_pval(p: float) -> str:
    """Display p-values clearly: scientific notation when very small."""
    if p == 0.0:
        return "< 1e-15"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _generate_hypothesis_card_text(hyp_result: dict, question: str = "") -> tuple:
    """
    Return (h0_text, h1_text, insight_text) — all LLM-generated from the
    original user question so they use plain English, not raw column names.
    Falls back to the engine-generated h0/h1 if the LLM call fails.
    Max 2 sentences for insight — no padding, no repetition.
    """
    try:
        from groq import Groq
        from config import GROQ_API_KEY, LLM_MODEL

        sig   = hyp_result.get("significant", False)
        test  = hyp_result.get("test", "")
        stat  = hyp_result.get("statistic", 0)
        pval  = hyp_result.get("p_value", 1.0)
        alpha = hyp_result.get("alpha", 0.05)
        n     = hyp_result.get("n", "?")
        h0_raw = hyp_result.get("h0", "")
        h1_raw = hyp_result.get("h1", "")

        verdict = "YES — statistically significant" if sig else "NO — not statistically significant"

        prompt = f"""You are a concise data analyst. The user ran this test:

User query: "{question}"
Test: {test} | n={n} | statistic={stat:.4f} | p-value={pval:.4g} | significant={sig}

Your job: return ONLY a JSON object with exactly these 3 keys, nothing else.
{{
  "h0": "<One sentence null hypothesis in plain English using the user's own terms from the query — no mu notation, no column names in quotes>",
  "h1": "<One sentence alternative hypothesis in plain English using the user's own terms — no mu notation>",
  "insight": "<Exactly 2 sentences. Sentence 1: what the result means in plain English (verdict={verdict}). Sentence 2: one concrete takeaway or caveat. No stat jargon, no repetition, no padding.>"
}}

Rules:
- Use the actual subject from the query (e.g. 'track duration', 'customer spending', 'unit price') — not raw column names
- h0 and h1 must be opposites and directly answer the user's question
- insight must be under 50 words total
- Return ONLY valid JSON — no markdown, no extra text"""

        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=220,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r'^```(?:json)?\s*', '', raw).rstrip('`').strip()
        data = json.loads(raw)
        return (
            data.get("h0", h0_raw),
            data.get("h1", h1_raw),
            data.get("insight", ""),
        )
    except Exception:
        return hyp_result.get("h0", ""), hyp_result.get("h1", ""), ""


def _render_hypothesis_card(hyp_result: dict, question: str = ""):
    """Render a self-contained hypothesis-test result block."""
    if "error" in hyp_result:
        st.error(f"**Hypothesis test error:** {hyp_result['error']}")
        return

    sig   = hyp_result["significant"]
    pval  = hyp_result["p_value"]
    stat  = hyp_result["statistic"]
    test  = hyp_result["test"]
    alpha = hyp_result.get("alpha", 0.05)
    n     = hyp_result.get("n", "?")

    stat_labels = {
        "Pearson Correlation": "r",
        "Spearman Correlation": "ρ",
        "T-Test": "t",
        "ANOVA": "F",
        "Mann-Whitney U": "U",
        "Chi-Square": "χ²",
    }
    stat_lbl = stat_labels.get(test, "stat")

    st.markdown(f"#### 🔬 Hypothesis Test — `{test}`")
    st.caption(f"n = {n:,} observations · alpha = {alpha}")

    # Generate query-aware H0/H1 and compact insight in one LLM call
    with st.spinner("Interpreting result…"):
        h0_text, h1_text, insight = _generate_hypothesis_card_text(hyp_result, question)

    # H0 / H1 block
    st.markdown(
        f"""
<div style="background:#1a2a1a;border-left:3px solid #238636;
            padding:8px 14px;border-radius:6px;margin-bottom:5px">
<span style="color:#8b949e;font-size:0.72rem;font-family:monospace">H₀ — null</span><br>
<span style="color:#e6edf3;font-size:0.9rem">{h0_text}</span>
</div>
<div style="background:#1a1a2a;border-left:3px solid #1f6feb;
            padding:8px 14px;border-radius:6px;margin-bottom:10px">
<span style="color:#8b949e;font-size:0.72rem;font-family:monospace">H₁ — alternative</span><br>
<span style="color:#e6edf3;font-size:0.9rem">{h1_text}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    # Stats row
    ca, cb, cc = st.columns(3)
    ca.metric(stat_lbl, f"{stat:.4f}")
    cb.metric("p-value", _format_pval(pval))
    cc.metric("alpha", str(alpha))

    # Verdict
    if sig:
        st.success(f"**Reject H₀** — p = {_format_pval(pval)} < {alpha}\n\n{hyp_result['interpretation']}")
    else:
        st.warning(f"**Fail to reject H₀** — p = {_format_pval(pval)} ≥ {alpha}\n\n{hyp_result['interpretation']}")

    # Cross-table join note
    if hyp_result.get("_cross_table"):
        method = hyp_result.get("_join_method", "unknown")
        n_rows = hyp_result.get("_n_aligned", "?")
        icon   = "✅" if "JOIN" in method or "merge" in method else "⚠️"
        label  = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)
        st.caption(f"{icon} Cross-table — {method} · {label} rows used")

    # Guard warnings — shown as callouts, always visible
    guard_warnings = hyp_result.get("_guard_warnings", [])
    for w in guard_warnings:
        if "🚨" in w:
            st.error(f"**Statistical integrity issue:** {w}")
        elif "⚠️" in w:
            st.warning(w)
        elif "ℹ️" in w:
            st.info(w)

    # Compact insight
    if insight:
        st.markdown(
            f"""<div style="background:#1a1f2e;border-left:3px solid #58a6ff;
                    padding:8px 14px;border-radius:6px;margin-top:8px">
<span style="color:#8b949e;font-size:0.72rem;font-family:monospace">💡 Insight</span><br>
<span style="color:#e6edf3;font-size:0.88rem">{insight}</span>
</div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Chat-side hypothesis engine — universal detection + aggregation + testing
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 1: Detection ────────────────────────────────────────────────────────
# Catches BOTH explicit stat keywords AND natural comparative language so that
# queries like "do albums with more tracks sell more?" are recognised without
# needing words like "significant" or "ANOVA".

# Tier A — explicit statistical terms (high confidence)
_HYP_EXPLICIT = re.compile(
    r'\b(hypothesis|t[\-\s]?test|ttest|anova|pearson|spearman|mann[\-\s]?whitney|'
    r'chi[\-\s]?square|chi2|correlat(e|ion)|significant(ly)?|p[\-\s]?value|'
    r'null hypothesis|alternative hypothesis|statistically|normality|'
    r'test (if|whether|the)|is there a (significant|correlation)|'
    r'distribution test|regression|effect of|relationship between|'
    r'associated with|dependent on|independent of)\b',
    re.IGNORECASE,
)

# Tier B — natural comparative language (medium confidence — used with context check)
_HYP_COMPARATIVE = re.compile(
    r'\b('
    # "more/less/higher/lower [adverb] than/compared"
    # BUT NOT "greater than 10" (numeric filters) — use negative lookahead (?!\s*\d)
    r'(more|less|higher|lower|greater|fewer|larger|smaller|faster|slower)'
    r'(\s+\w+)?\s+(than(?!\s*[$€£¥₹]?\d)|compared)|'
    # "purchased/bought more often / more frequently"
    r'(purchased|bought|sold|used)\s+(more|less)\s+(often|frequently|commonly)|'
    # "differ(ent) across/between/by"
    r'differ(ent|ence|s)?\s+(across|between|by|among|from)|'
    # "do X have/spend/generate/prefer/account more than Y"
    r'(spend|generate|account for|prefer|purchase|buy|earn|cost|take)'
    r'\s+(more|less|higher|lower|differently)|'
    # "effect/impact on", "influence on"
    r'(effect|impact|influence|role)\s+(of|on)|'
    # "proportion of"
    r'proportion\s+of|'
    # "disproportionate(ly)", "concentration", "skewed"
    r'disproportionate(ly)?|concentration|skewed\s+toward|'
    # "increased/decreased over time"
    r'(increase|decrease|grow|decline)(d|s|ing)?\s+(over|with|across)|'
    # "do X outperform / outsell Y"
    r'(outperform|outsell|outspend|dominate)'
    r')\b',
    re.IGNORECASE,
)

# Tier C — structural question patterns implying comparison
_HYP_STRUCTURAL = re.compile(
    r'\b('
    r'do (customers|users|artists|albums|tracks|genres|employees|countries|regions)'
    r'\s+(from|with|who|that|in)\s+\w+\s+(spend|generate|prefer|buy|have|earn)|'
    r'(is|are) (there|the) (a )?(difference|variation|gap|disparity|effect)\s+(in|between|across|among)|'
    # "which X has/generates more/higher/significantly" — NOT simple ranking like "which has the most"
    r'which (group|category|segment|country|genre|artist|employee)\s+(has|generates|spends|earns)'
    r'\s+(more|less|higher|lower|significantly|differently)'
    r')\b',
    re.IGNORECASE,
)


def is_hypothesis_query(question: str) -> bool:
    """
    Universal hypothesis detector — 3-tier matching.
    Returns True if the question is asking for statistical comparison/testing,
    regardless of whether explicit stat keywords are used.
    """
    q = question.strip()
    # Tier A: explicit stat terms → always hypothesis
    if _HYP_EXPLICIT.search(q):
        return True
    # Tiers B+C: comparative language → hypothesis if BOTH fire, or B alone
    # for strong patterns (avoids false positives on simple descriptive questions)
    b = bool(_HYP_COMPARATIVE.search(q))
    c = bool(_HYP_STRUCTURAL.search(q))
    return b or c


# ── MLR detector — must be checked BEFORE is_hypothesis_query ────────────────
# Queries asking about multiple predictors simultaneously need Multiple Linear
# Regression, not pairwise hypothesis tests. These are NORMAL queries (agent
# writes statsmodels/sklearn OLS code). The key signals are:
#   • 3+ columns listed as predictors
#   • words like "simultaneously", "combined effect", "together", "predict",
#     "control for", "adjust for", "multiple regression", "multivariate"
# We check this FIRST so that stat words like "significantly" don't
# accidentally route MLR queries into the hypothesis engine.

_MLR_SIMULTANEOUS = re.compile(
    r'\b('
    # Direct regression vocabulary
    r'simultaneously|multiple (linear |logistic )?regression|multivariate|'
    r'control(ling)? for|adjust(ing)? for|'
    # "together predict/affect/explain"
    r'together (predict|affect|explain|determine)|'
    # "combined effect of X and Y" — note: "combined effect" alone also matches
    r'combined effect|'
    # "effect of X and Y on Z"
    r'effect of .{3,80} (and|,) .{3,80} on|'
    # "impact/influence of X and Y on Z"
    r'(impact|influence) of .{3,80} (and|,) .{3,80} on|'
    # "X and Y together affect/predict/explain Z"
    r'\w[\w\s]+ and \w[\w\s]+ (together|simultaneously) (affect|predict|explain|determine)|'
    # "affect/predict Z given X and Y"
    r'(affect|impact|influence|predict)\s+\w[\w\s]+ (given|using|with|based on)\s+\w[\w\s]+ and|'
    # "how do X and Y affect Z"
    r'how (do|does) .{3,40} and .{3,40} (affect|impact|influence|predict)'
    r')\b',
    re.IGNORECASE,
)

# Three-or-more item enumeration before a causal/predictive keyword
_MLR_ENUMERATION = re.compile(
    r'(\w[\w\s]+,\s*\w[\w\s]+,\s*(and\s+)?\w[\w\s]+'
    r'\s*(simultaneously|together|combined|on|affect|predict|explain|impact))',
    re.IGNORECASE,
)

# Two-item "X and Y" patterns that imply multi-predictor regression
_MLR_TWO_PREDICTOR = re.compile(
    r'\b('
    # "X and Y on Z" structure
    r'(effect|impact|influence|role) of \w[\w\s]+ and \w[\w\s]+ on|'
    # "X and Y together"
    r'\w[\w\s]+ and \w[\w\s]+ together (affect|predict|explain|determine)|'
    # "how X and Y (together) affect Z"
    r'how (do|does) \w[\w\s]+ and \w[\w\s]+ (together )?(affect|predict|impact|influence)|'
    # "combined effect of X and Y"
    r'combined effect of \w[\w\s]+ and|'
    # "predict Z from/using X and Y"
    r'predict \w[\w\s]+ (from|using|with) \w[\w\s]+ and \w'
    r')\b',
    re.IGNORECASE,
)


def is_mlr_query(question: str) -> bool:
    """
    Returns True if the query is asking for multiple linear regression
    (multiple predictors → one outcome, evaluated simultaneously).
    These should be routed to normal analysis, NOT the hypothesis engine.
    """
    q = question.strip()
    return (bool(_MLR_SIMULTANEOUS.search(q)) or
            bool(_MLR_ENUMERATION.search(q)) or
            bool(_MLR_TWO_PREDICTOR.search(q)))


# ── Layer 2: Planning — LLM decides if aggregation SQL is needed ──────────────

def _plan_hypothesis(question: str, all_tables: dict, db_path: str | None) -> dict:
    """
    Ask the LLM to produce a full test plan:
      - If the query needs raw columns only  → returns {mode:"direct", ...}
      - If the query needs pre-aggregation   → returns {mode:"sql", agg_sql:..., ...}

    This handles complex queries like:
      "Do albums with more tracks generate higher sales?"
      → needs: SELECT AlbumId, COUNT(*) as track_count, SUM(revenue) as sales FROM ...
      → THEN run Pearson correlation on (track_count, sales)

    Works for ANY dataset — the LLM reads the full schema and writes the SQL.
    """
    from groq import Groq
    from config import GROQ_API_KEY, LLM_MODEL

    # Hard fallback: if an MLR query somehow bypasses the routing check,
    # refuse here instead of letting the LLM hallucinate table names.
    if is_mlr_query(question):
        return {"error": (
            "This query involves multiple simultaneous predictors and requires "
            "Multiple Linear Regression — not a pairwise hypothesis test. "
            "Please rephrase as: 'Run a regression of [outcome] on [feature1], "
            "[feature2], and [feature3]' to route it to the analysis engine."
        )}

    # Build full schema string for the LLM
    schema_lines = []
    if all_tables:
        for tbl, tdf in all_tables.items():
            cols = ", ".join(
                f"{c} ({str(tdf[c].dtype)})" for c in tdf.columns
            )
            schema_lines.append(f"  Table \"{tbl}\": {cols}")
    schema_str = "\n".join(schema_lines) if schema_lines else "  (CSV — single flat table)"

    db_note = (
        "Database type: SQLite. Use standard SQLite SQL."
        if db_path else
        "Data is a CSV loaded as a pandas DataFrame called \'df\'."
    )

    prompt = f"""You are a statistics assistant planning a hypothesis test.

{db_note}

Schema:
{schema_str}

User question: "{question}"

CRITICAL RULE BEFORE YOU DECIDE:
Statistical tests (t-test, ANOVA, Mann-Whitney) require ONE ROW PER OBSERVATION,
not one row per group. Grouping away rows with GROUP BY will almost always produce
too few rows and cause an error.

BAD SQL (only 2 rows - will always fail for group comparison):
  SELECT Country, AVG(Total) FROM Invoice GROUP BY Country

GOOD SQL (one invoice row per country - many rows, each is one data point):
  SELECT i."Total" AS invoice_total, c."Country"
  FROM "Invoice" i JOIN "Customer" c ON i."CustomerId" = c."CustomerId"

Use "direct" mode whenever the two needed columns exist in the schema and can be
accessed via a JOIN without collapsing rows. Only use "sql" mode to DERIVE columns
that do not exist anywhere in the schema (e.g. count of tracks per album, purchase
frequency per track - values you must compute via COUNT or SUM per entity).

Respond ONLY with a JSON object, no markdown, no explanation.

For group comparison questions (t-test, ANOVA, chi-square, Mann-Whitney) where
the needed columns exist or can be joined row-by-row:
{{
  "mode": "direct",
  "target_col":    "<numeric outcome column name>",
  "target_table":  "<table name, or dataset for CSV>",
  "feature_col":   "<group/predictor column name>",
  "feature_table": "<table name, or dataset for CSV>",
  "filter_values": ["<exact group label 1>", "<exact group label 2>"] or null,
  "force_test":    "<auto|ttest|anova|pearson|spearman|mannwhitney|chi2>",
  "alpha":         0.05,
  "reasoning":     "<one sentence: which columns and why>"
}}

For correlation/regression where a derived metric per entity is genuinely needed
(e.g. track_count per album that does not exist as a column):
{{
  "mode": "sql",
  "agg_sql":    "<SQLite SELECT — see CRITICAL SQL RULE below>",
  "target_col": "<derived numeric column name>",
  "feature_col": "<derived predictor column name>",
  "force_test": "<pearson|spearman|auto>",
  "alpha":      0.05,
  "reasoning":  "<one sentence: what derived metric and why it cannot be answered with raw columns>"
}}

CRITICAL SQL RULE FOR AGGREGATION:
NEVER use COUNT(*) or any aggregation on a JOIN result — this counts cartesian product rows,
not distinct entity rows, making both columns identical (r=1.0).

ALWAYS use separate subqueries (WITH clauses) — one per metric, each counting its OWN source table:

BAD (r=1.0 bug — JOIN multiplies rows, COUNT(*) counts the same thing twice):
  SELECT t."TrackId", COUNT(*) as PlaylistCount, COUNT(*) as TotalPurchases
  FROM "PlaylistTrack" pt JOIN "InvoiceLine" il ON pt."TrackId" = il."TrackId"
  GROUP BY t."TrackId"

GOOD (separate CTEs — each counts its own source independently):
  WITH playlist_counts AS (
    SELECT "TrackId", COUNT(*) as PlaylistCount
    FROM "PlaylistTrack" GROUP BY "TrackId"
  ),
  purchase_counts AS (
    SELECT "TrackId", COUNT(*) as TotalPurchases
    FROM "InvoiceLine" GROUP BY "TrackId"
  )
  SELECT p.PlaylistCount, c.TotalPurchases
  FROM playlist_counts p JOIN purchase_counts c ON p."TrackId" = c."TrackId"

Additional rules:
- filter_values: if the user names specific groups to compare (e.g. "USA vs Canada", "Rock vs Jazz"),
  set filter_values to exactly those group labels as they appear in the data (e.g. ["USA", "Canada"]).
  Set to null if the question compares ALL groups (e.g. "does spending differ across countries?").
- In direct mode, target_table and feature_table can differ (cross-table JOIN is handled automatically)
- In sql mode, agg_sql must be complete valid SQLite with double-quoted identifiers
- Default alpha = 0.05
- CRITICAL: feature_col and target_col must ALWAYS be a single string — NEVER a list or array.
  If the user asks about multiple predictors simultaneously, that is a regression task,
  not a hypothesis test. In that case return {{"error": "mlr"}} and nothing else."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = re.sub(r'^```(?:json)?\s*', '', raw).rstrip('`').strip()
        plan   = json.loads(raw)

        # Normalise feature_col: if it's a comma-separated string, split to list
        fc = plan.get("feature_col", "")
        if isinstance(fc, str) and "," in fc:
            plan["feature_col"] = [c.strip().strip('"').strip("'") for c in fc.split(",") if c.strip()]

        return plan
    except Exception as e:
        return {"error": f"Planning failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Universal pre/post-test guards  (work for ANY dataset, never hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

def _profile_column(series: pd.Series) -> dict:
    """
    Return a lightweight profile of a column used by all guards below.
    Works on any numeric or categorical Series.
    """
    s = series.dropna()
    n_unique   = s.nunique()
    n_total    = len(s)
    dtype_kind = s.dtype.kind          # 'f'=float, 'i'=int, 'O'=object, etc.
    is_numeric = dtype_kind in ('f', 'i', 'u')

    profile = {
        "n_unique":   n_unique,
        "n_total":    n_total,
        "dtype":      str(s.dtype),
        "is_numeric": is_numeric,
        "is_id_like": False,
        "is_binary_numeric": False,
        "zero_variance": False,
        "is_count_like": False,
    }

    if is_numeric:
        profile["zero_variance"]      = (s.std() == 0) if n_total > 1 else True
        profile["is_binary_numeric"]  = (n_unique <= 2)
        # Heuristic: if all values are non-negative integers and range ≤ 1000, likely counts
        profile["is_count_like"]      = (
            dtype_kind in ('i', 'u') or
            (is_numeric and (s % 1 == 0).all() and s.min() >= 0)
        )
        # ID-like: integer, many unique values, name suggests an identifier
        profile["is_id_like"]         = (
            is_numeric and n_unique > 0.9 * n_total and n_total > 10
        )

    return profile


def _guard_column_types(combined: pd.DataFrame, t_col: str, f_col: str,
                         force_test: str) -> tuple[str, list[str]]:
    """
    Fix 1 — Binary/categorical numeric columns.
    Detects when a "numeric" column is effectively categorical (e.g. price with
    2 values, ID columns) and adjusts the test type accordingly.

    Returns (adjusted_force_test, list_of_warning_strings).
    Works universally: thresholds are relative, not dataset-specific.
    """
    warnings = []
    ft = force_test

    t_prof = _profile_column(combined[t_col])
    f_prof = _profile_column(combined[f_col])

    # ── Outcome column checks ─────────────────────────────────────────────────

    # If the outcome column (target) has very few unique numeric values,
    # Pearson/ANOVA will produce misleading results (e.g. F=inf, r=0.93 on binary).
    if t_prof["is_numeric"] and t_prof["is_binary_numeric"] and ft in ("pearson", "spearman", "auto"):
        warnings.append(
            f"⚠️ '{t_col}' has only {t_prof['n_unique']} distinct numeric value(s) — "
            f"it behaves like a categorical variable. Switching to Chi-Square test."
        )
        ft = "chi2"

    if t_prof["is_numeric"] and t_prof["is_binary_numeric"] and ft in ("anova",):
        warnings.append(
            f"⚠️ '{t_col}' has only {t_prof['n_unique']} distinct numeric value(s). "
            f"ANOVA on a binary variable can produce F=∞. Switching to Chi-Square."
        )
        ft = "chi2"

    # ── Feature (group) column checks ────────────────────────────────────────

    # If the group column looks like a raw numeric ID (e.g. EmployeeId=1..8),
    # Pearson/Spearman is meaningless — the numbers are labels, not quantities.
    if (f_prof["is_numeric"] and not f_prof["is_count_like"] and
            f_prof["n_unique"] <= 30 and ft in ("pearson", "spearman", "auto")):
        # Small number of unique integer values → likely categorical ID
        warnings.append(
            f"⚠️ '{f_col}' has {f_prof['n_unique']} unique integer values and appears "
            f"to be a categorical identifier (e.g. an ID or code). "
            f"Switching from Pearson/Spearman to ANOVA."
        )
        ft = "anova"

    # ── Zero-variance check (causes F=inf) ────────────────────────────────────
    if t_prof["zero_variance"]:
        warnings.append(
            f"⚠️ '{t_col}' has zero variance (all values identical). "
            f"No statistical test is meaningful on this column."
        )

    return ft, warnings


def _guard_aggregation_sql(agg_sql: str, t_col: str, f_col: str,
                            combined: pd.DataFrame) -> list[str]:
    """
    Fix 2 — SQL aggregation self-reference detector.
    Catches the common bug where the LLM uses COUNT(*) or the same expression
    for both the outcome and predictor columns, producing r=1.0 artificially.

    Works on any SQL string — no dataset knowledge needed.
    """
    warnings = []

    # Check 1: perfect correlation — definitive sign of identical columns
    if len(combined) >= 3:
        t_s = combined[t_col].dropna()
        f_s = combined[f_col].dropna()
        if len(t_s) == len(f_s) and len(t_s) > 2:
            try:
                r = t_s.corr(f_s)
                if abs(r) > 0.9999:
                    warnings.append(
                        f"🚨 r={r:.4f} — near-perfect correlation detected in aggregated data. "
                        f"This almost always means the SQL computed the same quantity for both columns. "
                        f"Inspect the aggregation SQL and ensure '{t_col}' and '{f_col}' "
                        f"come from different source tables or use different aggregation functions."
                    )
            except Exception:
                pass

    # Check 2: same aggregation expression used for both columns in SQL
    sql_upper = agg_sql.upper()
    # Find all COUNT/SUM/AVG/MAX/MIN expressions
    agg_exprs = re.findall(r'(COUNT\s*\([^)]+\)|SUM\s*\([^)]+\)|AVG\s*\([^)]+\))', sql_upper)
    if len(agg_exprs) >= 2 and len(set(agg_exprs)) == 1:
        warnings.append(
            f"⚠️ The aggregation SQL uses the identical expression '{agg_exprs[0]}' "
            f"for both columns. This will produce r=1.0. Each column must aggregate "
            f"a different quantity (e.g. COUNT(tracks) vs SUM(invoice_total))."
        )

    return warnings


def _guard_test_for_aggregated_data(combined: pd.DataFrame, t_col: str,
                                     f_col: str, force_test: str) -> tuple[str, list[str]]:
    """
    Fix 3 — Auto-select Spearman for aggregated count/skewed data.
    When data comes from a GROUP BY aggregation, the distribution is typically
    right-skewed and non-normal. Pearson assumes normality; Spearman does not.

    Heuristic: if both columns are integer-valued, non-negative, and skewed
    (skewness > 1.0), switch Pearson → Spearman.
    Works universally by measuring the actual distribution.
    """
    warnings = []
    ft = force_test

    if ft not in ("pearson", "auto"):
        return ft, warnings

    t_s = combined[t_col].dropna()
    f_s = combined[f_col].dropna()

    t_prof = _profile_column(t_s)
    f_prof = _profile_column(f_s)

    # Both count-like and at least one is skewed → use Spearman
    if t_prof["is_count_like"] and f_prof["is_count_like"]:
        try:
            t_skew = abs(float(t_s.skew()))
            f_skew = abs(float(f_s.skew()))
            if max(t_skew, f_skew) > 1.0:
                old_ft = ft
                ft = "spearman"
                warnings.append(
                    f"ℹ️ Both columns contain count/integer data with high skewness "
                    f"(max skew={max(t_skew, f_skew):.2f}). "
                    f"Switching from {'auto→' if old_ft=='auto' else 'Pearson→'}Spearman "
                    f"for robustness (Spearman does not assume normality)."
                )
        except Exception:
            pass

    return ft, warnings


def _guard_degenerate_result(result: dict, t_col: str, f_col: str) -> list[str]:
    """
    Fix 4 — Post-test degenerate result detector.
    Catches mathematically impossible or suspicious results AFTER the test runs,
    and adds actionable explanations.

    Works on any test result dict from calculate_p_values().
    """
    warnings = []

    stat = result.get("statistic") or result.get("f_stat") or result.get("r") or 0
    pval = result.get("p_value", 1.0)

    # F=inf or stat=inf
    if stat == float("inf") or (isinstance(stat, float) and stat != stat):  # inf or NaN
        warnings.append(
            f"🚨 Test statistic is {stat} — this indicates zero within-group variance "
            f"(all values of '{t_col}' within at least one group of '{f_col}' are identical). "
            f"This is a data characteristic, not a real effect. "
            f"Consider: (1) using Chi-Square if '{t_col}' is effectively categorical, "
            f"(2) checking whether '{t_col}' has enough distinct values for this test."
        )

    # r = 1.0 exactly (perfect correlation — almost never real)
    if result.get("test_type") in ("pearson", "spearman") and abs(float(stat or 0)) > 0.9999:
        warnings.append(
            f"🚨 r={stat:.4f} — near-perfect correlation is almost never real in observed data. "
            f"Most likely cause: the two columns measure the same underlying quantity "
            f"(e.g. both computed from the same COUNT(*) in a GROUP BY query). "
            f"Verify that '{t_col}' and '{f_col}' are genuinely independent measurements."
        )

    # p = 1.0 exactly (extremely suspicious for ANOVA/t-test)
    if pval == 1.0 and result.get("test_type") in ("anova", "ttest", "mannwhitney"):
        warnings.append(
            f"⚠️ p=1.0000 exactly — this is statistically implausible. "
            f"Possible causes: (1) some groups have only 1 observation (check group sizes), "
            f"(2) between-group variance is near-zero due to data quality issues."
        )

    return warnings


def _build_multi_hop_join_sql(t_tbl: str, t_col: str, f_tbl: str, f_col: str,
                               fk_map: dict, all_tables: dict) -> str | None:
    """
    Fix 5 — Multi-hop (3+ table) JOIN resolver.
    When two tables have no direct FK relationship, find a bridge table
    and construct a 2-hop JOIN: t_tbl → bridge → f_tbl.

    Fully generic — reads the FK map, tries all possible bridge tables.
    Returns a complete SQL string or None if no path found.
    """
    all_table_names = list(all_tables.keys())

    for bridge in all_table_names:
        if bridge in (t_tbl, f_tbl):
            continue
        # Check: t_tbl → bridge AND bridge → f_tbl
        hop1 = fk_map.get((t_tbl, bridge)) or fk_map.get((bridge, t_tbl))
        hop2 = fk_map.get((bridge, f_tbl)) or fk_map.get((f_tbl, bridge))
        if hop1 and hop2:
            # Determine join columns for each hop
            if (t_tbl, bridge) in fk_map:
                t_key, b_key1 = fk_map[(t_tbl, bridge)]
            else:
                b_key1, t_key = fk_map[(bridge, t_tbl)]

            if (bridge, f_tbl) in fk_map:
                b_key2, f_key = fk_map[(bridge, f_tbl)]
            else:
                f_key, b_key2 = fk_map[(f_tbl, bridge)]

            sql = (
                f'SELECT t."{t_col}", f."{f_col}" '
                f'FROM "{t_tbl}" t '
                f'JOIN "{bridge}" b ON t."{t_key}" = b."{b_key1}" '
                f'JOIN "{f_tbl}" f ON b."{b_key2}" = f."{f_key}"'
            )
            return sql

    return None


# ── Layer 3: Execution — build DataFrame from plan and run the test ───────────

def _retry_agg_sql_with_cte(plan: dict, all_tables: dict, db_path: str | None) -> dict | None:
    """
    Called when Guard 2 detects r=1.0 (SQL self-reference bug).
    Re-asks the LLM to regenerate the aggregation SQL using separate CTEs —
    one per metric, each counting its own source table independently.
    Returns a new plan dict with corrected agg_sql, or None on failure.
    Works universally for any schema.
    """
    if not db_path:
        return None
    try:
        from groq import Groq
        from config import GROQ_API_KEY, LLM_MODEL

        schema_lines = []
        for tbl, tdf in (all_tables or {}).items():
            cols = ", ".join(f"{c} ({tdf[c].dtype})" for c in tdf.columns)
            schema_lines.append(f'  Table "{tbl}": {cols}')
        schema_str = "\n".join(schema_lines)

        t_col = plan.get("target_col", "")
        f_col = plan.get("feature_col", "")
        reasoning = plan.get("reasoning", "")

        prompt = f"""You previously generated aggregation SQL that produced r=1.0 — a self-reference bug
caused by using COUNT(*) on a JOIN result, which makes both columns measure the same thing.

Schema:
{schema_str}

Task: Compute "{t_col}" and "{f_col}" per entity ({reasoning}).

RULES — you MUST follow these exactly:
1. Use WITH (CTE) syntax — one CTE per metric, each from its OWN source table
2. Each CTE does its own GROUP BY independently
3. Join the CTEs together at the end on the shared entity key
4. NEVER use COUNT(*) or any aggregation on a JOIN result
5. Output exactly TWO columns named "{t_col}" and "{f_col}"
6. No trailing semicolon

Respond ONLY with the raw SQL, no markdown, no explanation."""

        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        new_sql = resp.choices[0].message.content.strip()
        new_sql = re.sub(r'^```(?:sql)?\s*', '', new_sql).rstrip('`').strip()

        # Quick validation: run it and check columns exist
        conn     = sqlite3.connect(db_path)
        test_df  = pd.read_sql_query(new_sql, conn)
        conn.close()
        if t_col in test_df.columns and f_col in test_df.columns and len(test_df) >= 3:
            new_plan = dict(plan)
            new_plan["agg_sql"] = new_sql
            return new_plan
    except Exception:
        pass
    return None


def _execute_hypothesis_plan(plan: dict, all_tables: dict, flat_df: pd.DataFrame,
                              db_path: str | None) -> dict:
    """Execute hypothesis plan with all 5 universal guards."""
    ft    = plan.get("force_test", "auto")
    alpha = float(plan.get("alpha", 0.05))
    mode  = plan.get("mode", "direct")
    all_guard_warnings = []

    if mode == "sql":
        agg_sql   = plan.get("agg_sql", "")
        t_col     = plan.get("target_col", "")
        f_col     = plan.get("feature_col", "")
        reasoning = plan.get("reasoning", "")
        if not db_path:
            return {"error": "SQL aggregation mode requires a SQLite/DB file."}
        if not agg_sql:
            return {"error": "Planning returned sql mode but no agg_sql was provided."}
        try:
            conn     = sqlite3.connect(db_path)
            combined = pd.read_sql_query(agg_sql, conn)
            conn.close()
        except Exception as e:
            return {"error": f"Aggregation SQL failed: {e}\nSQL was:\n{agg_sql}"}
        if t_col not in combined.columns or f_col not in combined.columns:
            return {"error": (f"SQL ran but expected columns [\'{t_col}\', \'{f_col}\'] not found. "
                              f"Got: {list(combined.columns)}. SQL: {agg_sql}")}
        combined = combined[[t_col, f_col]].dropna()
        if len(combined) < 3:
            return {"error": (f"SQL returned only {len(combined)} row(s) — too few. "
                              "SQL likely grouped away individual observations with GROUP BY.")}
        # Guard 2: SQL self-reference detector (r=1.0 bug)
        sql_warns = _guard_aggregation_sql(agg_sql, t_col, f_col, combined)
        all_guard_warnings.extend(sql_warns)
        if any("🚨" in w for w in sql_warns):
            # Auto-retry with CTE-based SQL before hard-failing
            retry_plan = _retry_agg_sql_with_cte(plan, all_tables, db_path)
            if retry_plan and "error" not in retry_plan:
                all_guard_warnings.append("ℹ️ SQL auto-corrected to use separate subqueries.")
                retry_result = _execute_hypothesis_plan(retry_plan, all_tables, flat_df, db_path)
                retry_result.setdefault("_guard_warnings", [])
                retry_result["_guard_warnings"] = all_guard_warnings + retry_result["_guard_warnings"]
                return retry_result
            return {"error": sql_warns[0], "_guard_warnings": sql_warns, "_agg_sql": agg_sql}
        # Guard 3: Pearson→Spearman for skewed count data
        ft, w = _guard_test_for_aggregated_data(combined, t_col, f_col, ft)
        all_guard_warnings.extend(w)
        # Guard 1: binary/ID column type correction
        ft, w = _guard_column_types(combined, t_col, f_col, ft)
        all_guard_warnings.extend(w)
        result = calculate_p_values(combined, t_col, f_col, force_test=ft, alpha=alpha)
        # Guard 4: degenerate result
        all_guard_warnings.extend(_guard_degenerate_result(result, t_col, f_col))
        result.update({"_cross_table": False, "_join_method": f"aggregation SQL ({reasoning})",
                        "_n_aligned": len(combined), "_agg_sql": agg_sql,
                        "_guard_warnings": all_guard_warnings})
        return result

    # Direct mode
    t_col       = plan.get("target_col", "")
    t_tbl       = plan.get("target_table", "dataset")
    f_col       = plan.get("feature_col", "")
    f_tbl       = plan.get("feature_table", "dataset")
    filter_vals = plan.get("filter_values")
    cross = (t_tbl != f_tbl) and bool(all_tables)

    # ── Safety: if f_col is still a list here, it means is_mlr_query() missed it.
    # Return a clear error directing the user to rephrase rather than silently
    # running wrong pairwise tests.
    if isinstance(f_col, list):
        return {"error": (
            f"This query has {len(f_col)} feature columns {f_col} — "
            "testing multiple predictors simultaneously requires Multiple Linear Regression. "
            "Try: 'Run a regression of [outcome] on [feature1], [feature2], and [feature3]' "
            "and it will be handled by the analysis engine."
        )}

    # ── Fix 1: Fuzzy column name resolver ─────────────────────────────────────
    # LLM sometimes shortens column names (e.g. 'Country' instead of 'BillingCountry').
    # Try exact match first, then case-insensitive partial match against real columns.
    def _resolve_col(col_hint: str, src_df) -> str:
        if col_hint in src_df.columns:
            return col_hint
        # Case-insensitive exact
        cl = {c.lower(): c for c in src_df.columns}
        if col_hint.lower() in cl:
            return cl[col_hint.lower()]
        # Substring: find columns that contain the hint (or vice versa)
        hint_l = col_hint.lower().replace(".", "").replace("_", "").replace(" ", "")
        candidates = [c for c in src_df.columns
                      if hint_l in c.lower().replace("_","").replace(" ","")
                      or c.lower().replace("_","").replace(" ","") in hint_l]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Prefer longer match (more specific)
            return max(candidates, key=len)
        return col_hint   # give up, return original (will error below with clear message)

    if cross:
        # ── Resolve column names before attempting JOIN ────────────────────────
        # The LLM often uses the question's wording (e.g. "EmployeeId") instead of
        # the real schema column in the target table (e.g. "SupportRepId" in Customer).
        # Two-stage resolution:
        #   Stage 1: fuzzy name match against actual columns in the table
        #   Stage 2: FK-based remap — if column not in table, find an FK in that
        #            table that points to another table containing the column
        fk_map_cross = _get_pragma_fk_map(db_path) if db_path else {}

        def _resolve_cross_col(col_hint: str, tbl: str) -> tuple[str, str]:
            """Return (resolved_col, note). note is '' if no change needed."""
            src = all_tables.get(tbl)
            if src is None:
                return col_hint, ""

            # Stage 1: fuzzy name match
            resolved = _resolve_col(col_hint, src)
            if resolved in src.columns and resolved != col_hint:
                return resolved, f"'{col_hint}' → '{resolved}' in '{tbl}'"

            if resolved in src.columns:
                return resolved, ""   # exact match, no change

            # Stage 2: FK remapping
            # Find an FK in tbl that points to another table containing col_hint
            for (from_tbl, to_tbl), (from_col, to_col) in fk_map_cross.items():
                if from_tbl == tbl:
                    other = all_tables.get(to_tbl)
                    if other is not None and col_hint in other.columns:
                        note = (f"'{col_hint}' not in '{tbl}' — "
                                f"using FK column '{from_col}' "
                                f"(which links to {to_tbl}.{to_col})")
                        return from_col, note

            return col_hint, ""   # could not resolve — let _build_cross_table_df error clearly

        t_col_new, t_note = _resolve_cross_col(t_col, t_tbl)
        f_col_new, f_note = _resolve_cross_col(f_col, f_tbl)

        if t_note:
            all_guard_warnings.append(f"ℹ️ Column name adjusted: {t_note}")
            t_col = t_col_new
        if f_note:
            all_guard_warnings.append(f"ℹ️ Column name adjusted: {f_note}")
            f_col = f_col_new

        try:
            combined, method = _build_cross_table_df(t_tbl, t_col, f_tbl, f_col, all_tables, db_path)
        except ValueError as e:
            return {"error": str(e)}
    else:
        src = all_tables.get(t_tbl, flat_df) if all_tables else flat_df
        if src is None:
            return {"error": f"Table \'{t_tbl}\' not found."}

        # Resolve both column names fuzzy
        t_col_resolved = _resolve_col(t_col, src)
        f_col_resolved = _resolve_col(f_col, src)

        if t_col_resolved not in src.columns:
            avail = [c for c in src.columns if src[c].dtype in (float, int) or "object" in str(src[c].dtype)]
            return {"error": f"Column \'{t_col}\' not found in \'{t_tbl}\'. Available: {list(src.columns)[:10]}"}
        if f_col_resolved not in src.columns:
            return {"error": f"Column \'{f_col}\' not found in \'{f_tbl}\'. Available: {list(src.columns)[:10]}"}

        # Rename if resolved to a different name
        if t_col_resolved != t_col or f_col_resolved != f_col:
            all_guard_warnings.append(
                f"ℹ️ Column name adjusted: \'{t_col}\'→\'{t_col_resolved}\', \'{f_col}\'→\'{f_col_resolved}\'"
            )
        t_col = t_col_resolved
        f_col = f_col_resolved

        # ── Fix 3: Null-based binary split ────────────────────────────────────
        # If feature column is mostly NaN and user wants "has X vs no X",
        # create a binary column instead of dropping NaN rows.
        raw_series = src[f_col]
        null_frac = raw_series.isna().mean()
        if null_frac > 0.3:
            # Create binary: "has_value" vs "no_value"
            binary_col = f_col + "__binary"
            src = src.copy()
            src[binary_col] = raw_series.notna().map({True: f"has_{f_col}", False: f"no_{f_col}"})
            all_guard_warnings.append(
                f"ℹ️ \'{f_col}\' is {null_frac:.0%} null. Created binary split: "
                f"\'{f'has_{f_col}'}\'  vs  \'{f'no_{f_col}'}\' ({(~raw_series.isna()).sum()} vs {raw_series.isna().sum()} rows)."
            )
            f_col = binary_col

        combined = src[[t_col, f_col]].dropna()
        method   = "single-table"

    # ── Fix 2: Smart group filter — handles derived categories like "Non-USA" ─
    if filter_vals and isinstance(filter_vals, list) and len(filter_vals) >= 2:
        actual_vals = combined[f_col].unique().tolist()

        # Find which filter_vals actually exist in the data (case-insensitive)
        val_map = {str(v).strip().lower(): v for v in actual_vals}
        matched   = []
        derived   = []   # values that don\'t exist — treated as "everything else"
        for fv in filter_vals:
            if fv in actual_vals:
                matched.append(fv)
            elif str(fv).strip().lower() in val_map:
                matched.append(val_map[str(fv).strip().lower()])
            else:
                derived.append(fv)

        if matched and derived:
            # e.g. filter_vals=["USA", "Non-USA"] → matched=["USA"], derived=["Non-USA"]
            # Create new binary column: matched[0] vs derived label
            binary_label = derived[0]   # e.g. "Non-USA"
            pivot_col = f_col + "__pivot"
            combined = combined.copy()
            combined[pivot_col] = combined[f_col].apply(
                lambda x: matched[0] if x in matched else binary_label
            )
            f_col  = pivot_col
            method = f"{method} · {matched[0]} vs {binary_label} (binary split)"
            all_guard_warnings.append(
                f"ℹ️ \'{binary_label}\' is not a real value in the data — "
                f"created binary split: {matched[0]} ({(combined[f_col]==matched[0]).sum()} rows) "
                f"vs {binary_label} ({(combined[f_col]==binary_label).sum()} rows)."
            )
        elif matched:
            # All filter values exist — standard filter
            combined = combined[combined[f_col].isin(matched)]
            method = f"{method} · filtered to {matched}"
        # If nothing matched at all, skip filter and run on full data
        if len(combined) < 3:
            return {"error": (
                f"After applying filter {filter_vals}, only {len(combined)} rows remain. "
                f"Check that the group labels match values in \'{f_col}\'. "
                f"Actual values (sample): {actual_vals[:10]}"
            )}

    # Auto-fix ttest→anova
    n_groups = combined[f_col].nunique() if f_col in combined.columns else 0
    if ft == "ttest" and n_groups > 2:
        ft = "anova"; method += " [auto-upgraded to ANOVA: >2 groups]"
    elif ft == "auto" and n_groups == 2:
        ft = "ttest"

    # Guard 1: binary/ID column type correction
    ft, w = _guard_column_types(combined, t_col, f_col, ft)
    all_guard_warnings.extend(w)

    result = calculate_p_values(combined, t_col, f_col, force_test=ft, alpha=alpha)

    # Guard 4: degenerate result
    all_guard_warnings.extend(_guard_degenerate_result(result, t_col, f_col))

    result.update({"_cross_table": cross, "_target_table": t_tbl, "_feature_table": f_tbl,
                   "_n_aligned": len(combined), "_join_method": method,
                   "_guard_warnings": all_guard_warnings})
    return result

# ── Thin wrappers kept for backward-compat with existing call sites ───────────

def _parse_hypothesis_from_chat(question: str, col_options: list, all_tables: dict) -> dict:
    """Legacy shim — planning is now done inside _run_hypothesis_from_chat."""
    return {"_use_planner": True, "_question": question}


def _run_hypothesis_from_chat(parsed: dict, all_tables: dict, flat_df: pd.DataFrame,
                               db_path: str = None) -> dict:
    """
    Universal entry point. If parsed contains _use_planner, calls the full
    plan → execute pipeline. Otherwise falls back to legacy direct mode.
    """
    question = parsed.get("_question", "")

    if parsed.get("_use_planner") and question:
        plan = _plan_hypothesis(question, all_tables, db_path)
        if "error" in plan:
            return {"error": plan["error"]}
        return _execute_hypothesis_plan(plan, all_tables, flat_df, db_path)

    # Legacy direct path (sidebar manual selectors still use this)
    ft    = parsed.get("force_test", "auto")
    alpha = float(parsed.get("alpha", 0.05))
    t_col = parsed.get("target_col", "")
    t_tbl = parsed.get("target_table", "dataset")
    f_col = parsed.get("feature_col", "")
    f_tbl = parsed.get("feature_table", "dataset")
    cross = (t_tbl != f_tbl) and bool(all_tables)

    if cross:
        try:
            combined, method = _build_cross_table_df(
                t_tbl, t_col, f_tbl, f_col, all_tables, db_path
            )
        except ValueError as e:
            return {"error": str(e)}
    else:
        src = all_tables.get(t_tbl, flat_df) if all_tables else flat_df
        if not t_col or t_col not in src.columns:
            return {"error": f"Column \'{t_col}\' not found."}
        if not f_col or f_col not in src.columns:
            return {"error": f"Column \'{f_col}\' not found."}
        combined = src[[t_col, f_col]].dropna()
        method   = "single-table"

    result = calculate_p_values(combined, t_col, f_col, force_test=ft, alpha=alpha)
    result["_cross_table"]  = cross
    result["_target_table"] = t_tbl
    result["_feature_table"]= f_tbl
    result["_n_aligned"]    = len(combined)
    result["_join_method"]  = method if cross else "single-table"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 Research Analyst")
    st.caption("Enterprise AI · v4.0")
    st.divider()

    app_mode = st.radio(
        "Navigation",
        ["💬 Interactive Chat", "📊 Research Metrics Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload Dataset", type=["csv", "db", "sqlite"],
        help="CSV · SQLite (.db / .sqlite)",
    )

    if st.session_state.df is not None:
        st.divider()
        _df = st.session_state.df
        _all_tables: dict = st.session_state.get("all_tables", {})
        st.caption("📁 Active dataset")
        st.code(st.session_state.current_file or "—", language=None)
        if _all_tables:
            st.caption(f"{len(_all_tables)} tables · {len(_df):,} total rows · {len(_df.columns)} cols")
            with st.expander("📋 Tables"):
                for tname, tdf in _all_tables.items():
                    st.caption(f"**{tname}** — {len(tdf):,} rows · {len(tdf.columns)} cols")
        else:
            st.caption(f"{len(_df):,} rows · {len(_df.columns)} cols")

        sm = metrics_tracker.summary()
        if sm["total_queries"] > 0:
            st.caption(
                f"Session: {sm['total_queries']} queries · "
                f"Avg FVR {sm['avg_fvr']:.0%} · "
                f"{st.session_state.total_tokens:,} tokens"
            )

        st.divider()
        st.caption("💾 Persist to registry")
        if st.button("Save Benchmark", use_container_width=True):
            summary = metrics_tracker.summary()
            if summary["total_queries"] == 0:
                st.warning("Ask at least one question first.")
            else:
                save_dataset_benchmark(
                    dataset_name    = st.session_state.current_file,
                    file_type       = st.session_state.file_type,
                    row_count       = len(_df),
                    col_count       = len(_df.columns),
                    summary         = summary,
                    query_records   = metrics_tracker.records(),
                    ts_queries      = st.session_state.ts_count,
                    tabular_queries = st.session_state.tab_count,
                )
                st.success("✅ Saved!")

        # ── Hypothesis Testing ─────────────────────────────────────────────
        st.divider()
        st.caption("🔬 Hypothesis Testing")

        # Build a flat list of "Table · column" labels and a lookup map
        # For CSV files, use a single pseudo-table named after the file.
        if _all_tables:
            # SQLite/DB: every column from every table is available
            col_options: list[str] = []          # display labels
            col_lookup:  dict[str, tuple] = {}   # label → (table_name, col_name, series)
            for tname, tdf in _all_tables.items():
                for col in tdf.columns:
                    label = f"{tname} · {col}"
                    col_options.append(label)
                    col_lookup[label] = (tname, col, tdf[col])
        else:
            # CSV: single table, simpler labels
            col_options = list(_df.columns)
            col_lookup  = {col: ("dataset", col, _df[col]) for col in _df.columns}

        # Partition into numeric and all options for the two selectors
        def _is_numeric_label(label: str) -> bool:
            _, _, series = col_lookup[label]
            return pd.api.types.is_numeric_dtype(series)

        numeric_options = [l for l in col_options if _is_numeric_label(l)]

        if len(col_options) < 2 or not numeric_options:
            st.info("Need ≥ 2 columns (at least 1 numeric) for hypothesis testing.")
        else:
            target_label = st.selectbox(
                "Target (numeric)",
                numeric_options,
                key="hyp_target",
                help="The numeric outcome you want to measure or explain.",
            )
            feature_options = [l for l in col_options if l != target_label]
            feature_label = st.selectbox(
                "Feature (any table · any column)",
                feature_options,
                key="hyp_feature",
                help="The column you think might influence the target. Can be from a different table.",
            )

            test_override = st.selectbox(
                "Test (optional override)",
                ["auto", "ttest", "anova", "pearson", "spearman", "mannwhitney", "chi2"],
                key="hyp_test_override",
                help="'auto' picks the best test based on column types. Override only if needed.",
            )
            alpha_val = st.select_slider(
                "Significance level (alpha)",
                options=[0.01, 0.05, 0.10],
                value=0.05,
                key="hyp_alpha",
            )

            if st.button("Run Hypothesis Test", use_container_width=True):
                t_tbl, t_col, _ = col_lookup[target_label]
                f_tbl, f_col, _ = col_lookup[feature_label]
                _db_path        = st.session_state.get("db_path")
                cross_table     = (t_tbl != f_tbl)

                if cross_table:
                    # Universal path: PRAGMA FK -> shared-col heuristic -> positional
                    try:
                        combined_df, method = _build_cross_table_df(
                            t_tbl, t_col, f_tbl, f_col, _all_tables, _db_path
                        )
                        icon = "✅" if "JOIN" in method or "merge" in method else "⚠️"
                        st.caption(
                            f"{icon} Cross-table: **{t_tbl}**.`{t_col}` vs "
                            f"**{f_tbl}**.`{f_col}` — {method} "
                            f"({len(combined_df):,} rows)"
                        )
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()
                else:
                    src = _all_tables.get(t_tbl, _df) if _all_tables else _df
                    if t_col not in src.columns or f_col not in src.columns:
                        st.error(f"Column not found in table '{t_tbl}'.")
                        st.stop()
                    combined_df = src[[t_col, f_col]].dropna()

                if len(combined_df) < 3:
                    st.error("Not enough rows. Try columns with more data.")
                    st.stop()

                hyp_result = calculate_p_values(
                    combined_df, t_col, f_col,
                    force_test=test_override, alpha=alpha_val,
                )
                _render_hypothesis_card(hyp_result, question=f"Test whether {t_col} differs by {f_col}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════════════════════

if app_mode == "💬 Interactive Chat":

    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            # Reset session
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.current_file = uploaded_file.name

            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            st.session_state.file_type = ext

            if ext == "csv":
                df = pd.read_csv(uploaded_file)
                st.session_state.db_path = None
            elif ext in ("db", "sqlite"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                st.session_state.db_path = tmp_path
                conn   = sqlite3.connect(tmp_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                # Load ALL tables into all_tables dict
                all_tables = {}
                for tbl in tables:
                    try:
                        all_tables[tbl] = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn)
                    except Exception:
                        pass
                conn.close()
                st.session_state.all_tables = all_tables
                # df used for RAG/chat = union of all tables (best-effort concat)
                df = (pd.concat(list(all_tables.values()), ignore_index=True)
                      if all_tables else pd.DataFrame())
            else:
                df, st.session_state.db_path = pd.DataFrame(), None

            st.session_state.df = df
            with st.spinner("Ingesting dataset into RAG store…"):
                n_docs = ingest_dataframe_info(df)
            st.toast(f"RAG ready — {n_docs} documents indexed ✓", icon="🧠")
        else:
            df = st.session_state.df

        # Dataset preview
        with st.expander("👀 Dataset Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Rows",    f"{len(df):,}")
            cb.metric("Columns", str(len(df.columns)))
            cc.metric("Missing", f"{df.isna().sum().sum():,}")
            from stats_engine import detect_datetime_columns
            dt = detect_datetime_columns(df)
            cd.metric("DateTime Cols", str(len(dt)),
                      help=f"{dt}" if dt else "None detected")

        st.markdown("---")

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    res = msg["content"]
                    # ── Hypothesis replay ────────────────────────────────────
                    if isinstance(res, dict) and res.get("_type") == "hypothesis":
                        _render_hypothesis_card(res["result"], question=res.get("_question", ""))
                    elif isinstance(res, dict) and res.get("_type") == "hypothesis_error":
                        st.error(res["error"])
                    # ── Normal analysis replay ───────────────────────────────
                    else:
                        render_insights_and_charts(
                            res["insights"], df, st.session_state.db_path,
                            res.get("executed_code", ""),
                            res.get("query_type", "scalar"),
                            res.get("chart_specs", []),
                        )
                        with st.expander("🔍 Execution Log"):
                            for entry in res.get("correction_log", []):
                                st.markdown(f"**Attempt {entry['attempt']}**"
                                            + (f" — `{entry['error_class']}`"
                                               if entry.get("error_class") else ""))
                                st.code(entry["code"], language="python")
                                if entry["status"] == "error":
                                    st.error(entry["error_message"])
                                else:
                                    st.success("✅ Success")
                        if not res.get("needs_correction"):
                            render_metric_strip(
                                res["metrics"],
                                res.get("query_type", "scalar"),
                                res.get("rag_context_used", False),
                                res.get("tokens_used", 0),
                                res.get("error_class"),
                            )
                            render_claim_inspector(
                                res.get("claim_details", []),
                                res.get("insights", ""),
                            )

        # Self-correction
        if st.session_state.needs_correction:
            st.warning("⚠️ Code crashed. Click below to auto-correct.")
            if st.button("🔧 Trigger Self-Correction", type="primary"):
                with st.chat_message("assistant"):
                    with st.spinner("Rewriting code…"):
                        result = analyze(
                            question           = st.session_state.correction_data["query"],
                            df                 = df,
                            db_path            = st.session_state.db_path,
                            correction_context = st.session_state.correction_data,
                        )
                        st.session_state.needs_correction = result.get("needs_correction", False)
                        if st.session_state.needs_correction:
                            st.session_state.correction_data.update({
                                "code": result["executed_code"],
                                "error": result["error_trace"],
                                "attempt": result["attempt"],
                            })
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        st.rerun()

        # New query
        if query := st.chat_input("Ask anything about your data…"):
            st.session_state.needs_correction = False
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):

                # ── Dataset mismatch check (MLR only) ────────────────────────
                # Hypothesis tests that reference non-existent columns will fail
                # gracefully with a "column not found" error from the engine.
                # Normal analysis queries are handled by the code executor which
                # works with table names and free text — no column validation needed.
                # Only MLR needs this gate because the LLM will silently invent data
                # if the requested columns don't exist in the schema.
                _at = st.session_state.get("all_tables", {})
                if is_mlr_query(query):
                    _valid, _val_err = _validate_mlr_columns(query, df, _at)
                    if not _valid:
                        st.error(_val_err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": {"insights": _val_err, "metrics": {},
                                        "correction_log": [], "executed_code": "",
                                        "needs_correction": False, "query_type": "scalar",
                                        "chart_specs": [], "rag_context_used": False,
                                        "tokens_used": 0, "error_class": None}
                        })
                        st.stop()

                # ── MLR / multi-feature → regression analysis (MUST come first) ──
                if is_mlr_query(query):
                    st.caption("📐 Running Multiple Linear Regression…")

                    mlr_query = _build_mlr_query(
                        query, df, st.session_state.db_path, _at
                    )

                    with st.spinner("Fitting regression model…"):
                        result = analyze(mlr_query, df, st.session_state.db_path)

                    qt = "tabular"
                    st.session_state.total_tokens += result.get("tokens_used", 0)
                    st.session_state.rag_total    += 1
                    if result.get("rag_context_used"):
                        st.session_state.rag_hits += 1

                    if result.get("error_class"):
                        ec = result["error_class"]
                        st.session_state.error_taxonomy[ec] = \
                            st.session_state.error_taxonomy.get(ec, 0) + 1

                    st.session_state.needs_correction = result.get("needs_correction", False)

                    # ── Capture RESULT, _r2, _f_pval NOW while DB_PATH is live ──────
                    # We extract these here instead of re-executing inside _render_mlr_card
                    # so the card never needs to re-open the database connection.
                    if not st.session_state.needs_correction:
                        _executed = result.get("executed_code", "")
                        _env = {
                            "df": df,
                            "DB_PATH": st.session_state.db_path,
                            "pd": pd, "sqlite3": sqlite3,
                            "RESULT": None, "_r2": None, "_f_pval": None,
                        }
                        try:
                            _code = re.sub(r'^```(?:python)?\s*', '', _executed.strip())
                            _code = re.sub(r'\s*```$', '', _code.strip())
                            exec(_code, _env)
                        except Exception:
                            pass
                        result["_mlr_coef_df"] = _env.get("RESULT")
                        result["_mlr_r2"]      = _env.get("_r2")
                        result["_mlr_f_pval"]  = _env.get("_f_pval")

                    if st.session_state.needs_correction:
                        st.session_state.correction_data = {
                            "query": mlr_query,
                            "code":  result["executed_code"],
                            "error": result["error_trace"],
                            "attempt": result["attempt"],
                        }
                        render_insights_and_charts(
                            result["insights"], df, st.session_state.db_path,
                            result.get("executed_code", ""), qt, [],
                        )
                    else:
                        _render_mlr_card(result, question=query)

                    with st.expander("🔍 Execution Log"):
                        for entry in result.get("correction_log", []):
                            st.markdown(f"**Attempt {entry['attempt']}**"
                                        + (f" — `{entry['error_class']}`"
                                           if entry.get("error_class") else ""))
                            st.code(entry["code"], language="python")
                            if entry["status"] == "error":
                                st.error(entry["error_message"])
                            else:
                                st.success("✅ Success")

                    if not st.session_state.needs_correction:
                        render_metric_strip(
                            result["metrics"], qt,
                            result.get("rag_context_used", False),
                            result.get("tokens_used", 0),
                            result.get("error_class"),
                        )
                        render_claim_inspector(
                            result.get("claim_details", []),
                            result.get("insights", ""),
                        )

                    st.session_state.messages.append({"role": "assistant", "content": result})
                    if st.session_state.needs_correction:
                        st.rerun()

                # ── Hypothesis branch ─────────────────────────────────────────
                elif is_hypothesis_query(query):
                    _at = st.session_state.get("all_tables", {})

                    with st.spinner("Planning hypothesis test…"):
                        plan = _plan_hypothesis(query, _at, st.session_state.db_path)

                    if "error" in plan:
                        st.error(f"**Could not plan test:** {plan['error']}")
                        st.info(
                            "Examples that work well:\n"
                            "• 'Do USA customers spend more per invoice than Canada customers?'\n"
                            "• 'Pearson correlation between track length and unit price'\n"
                            "• 'Do some genres generate significantly more revenue than others?'"
                        )
                        hyp_msg = {"_type": "hypothesis_error", "error": plan["error"]}
                    else:
                        reasoning = plan.get("reasoning", "")
                        if reasoning:
                            st.caption(f"🧠 Plan: {reasoning}")

                        if plan.get("mode") == "sql" and plan.get("agg_sql"):
                            with st.expander("🔍 Aggregation SQL", expanded=False):
                                st.code(plan["agg_sql"], language="sql")

                        with st.spinner("Running test…"):
                            hyp_result = _execute_hypothesis_plan(
                                plan, _at, df, db_path=st.session_state.db_path
                            )

                        _render_hypothesis_card(hyp_result, question=query)
                        hyp_msg = {"_type": "hypothesis", "result": hyp_result, "_question": query}

                    st.session_state.messages.append(
                        {"role": "assistant", "content": hyp_msg}
                    )

                # ── Normal analysis branch ────────────────────────────────────
                else:
                    with st.spinner("Analysing…"):
                        result = analyze(query, df, st.session_state.db_path)

                    qt = result.get("query_type", "scalar")
                    if qt == "timeseries":  st.session_state.ts_count  += 1
                    elif qt == "tabular":   st.session_state.tab_count += 1

                    st.session_state.total_tokens += result.get("tokens_used", 0)
                    st.session_state.rag_total    += 1
                    if result.get("rag_context_used"):
                        st.session_state.rag_hits += 1

                    if result.get("error_class"):
                        ec = result["error_class"]
                        st.session_state.error_taxonomy[ec] = \
                            st.session_state.error_taxonomy.get(ec, 0) + 1

                    st.session_state.needs_correction = result.get("needs_correction", False)
                    if st.session_state.needs_correction:
                        st.session_state.correction_data = {
                            "query": query,
                            "code":  result["executed_code"],
                            "error": result["error_trace"],
                            "attempt": result["attempt"],
                        }

                    render_insights_and_charts(
                        result["insights"], df, st.session_state.db_path,
                        result.get("executed_code", ""),
                        qt,
                        result.get("chart_specs", []),
                    )

                    with st.expander("🔍 Execution Log"):
                        for entry in result.get("correction_log", []):
                            st.markdown(f"**Attempt {entry['attempt']}**"
                                        + (f" — `{entry['error_class']}`"
                                           if entry.get("error_class") else ""))
                            st.code(entry["code"], language="python")
                            if entry["status"] == "error":
                                st.error(entry["error_message"])
                            else:
                                st.success("✅ Success")

                    with st.expander("🧠 RAG Context Used"):
                        rc = retrieve_context(query, query_type=qt)
                        st.text(rc if rc else "No RAG context retrieved.")

                    # ── Claim Inspector ───────────────────────────────────
                    if not st.session_state.needs_correction:
                        render_claim_inspector(
                            result.get("claim_details", []),
                            result.get("insights", ""),
                        )

                    if not st.session_state.needs_correction:
                        render_metric_strip(
                            result["metrics"], qt,
                            result.get("rag_context_used", False),
                            result.get("tokens_used", 0),
                            result.get("error_class"),
                        )

                    st.session_state.messages.append({"role": "assistant", "content": result})
                    if st.session_state.needs_correction:
                        st.rerun()

    else:
        st.info("👈  Upload a CSV or SQLite dataset in the sidebar to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESEARCH METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif app_mode == "📊 Research Metrics Dashboard":
    st.header("📊 Research Metrics Dashboard")

    with st.expander("📖 Full Metric Glossary", expanded=False):
        st.markdown("""
| Acronym | Full Name | Definition | Range | Goal |
|---------|-----------|------------|-------|------|
| **FVR** | Factual Verification Rate | Fraction of numerical claims traceable to code-computed ground truth | 0→1 | ↑ |
| **HER** | Hallucination Error Rate | Fraction with no ground-truth match (= 1−FVR) | 0→1 | ↓ |
| **CCE** | Confidence Calibration Error | |model confidence − FVR| | 0→1 | ↓ |
| **EER** | Execution Error Rate | Fraction of queries where generated code crashed | 0→1 | ↓ |
| **SCR** | Self-Correction Rate | Fraction of failures recovered after self-correction | 0→1 | ↑ |
| **QRT** | Query Response Time | Wall-clock seconds per query | s | ↓ |
| **OCS** | Output Consistency Score | 1/(1+Var(response lengths)) | 0→1 | ↑ |
| **RAG Hit Rate** | RAG Retrieval Rate | Fraction of queries where RAG returned non-empty context | 0→1 | ↑ |
| **Tokens** | LLM Token Usage | Total tokens consumed (cost proxy) | count | ↓ |
""")

    st.divider()

    # ── Section A: Live session ────────────────────────────────────────────────
    st.subheader("🔴 Live Session")
    summary = metrics_tracker.summary()
    records = metrics_tracker.records()
    total_q = summary.get("total_queries", 0)

    # Count hypothesis tests from session messages (not tracked by metrics_tracker)
    total_hyp = sum(
        1 for msg in st.session_state.get("messages", [])
        if msg["role"] == "assistant"
        and isinstance(msg.get("content"), dict)
        and msg["content"].get("_type") in ("hypothesis", "hypothesis_error")
    )

    if total_q == 0 and total_hyp == 0:
        st.info("No queries yet. Ask questions in Interactive Chat.")
    else:
        rag_rate = (st.session_state.rag_hits / st.session_state.rag_total
                    if st.session_state.rag_total else 0.0)

        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        c1.metric("Queries",       str(total_q))
        c2.metric("Hyp Tests 🔬",  str(total_hyp))
        c3.metric("Avg FVR ↑",     f"{summary['avg_fvr']:.1%}")
        c4.metric("Avg HER ↓",     f"{summary['avg_her']:.1%}")
        c5.metric("Avg CCE ↓",     f"{summary['avg_cce']:.4f}")
        c6.metric("EER ↓",         f"{summary['eer']:.1%}")
        c7.metric("RAG Hit",       f"{rag_rate:.1%}")
        c8.metric("Tokens",        f"{st.session_state.total_tokens:,}")

        if records or total_hyp > 0:
            rec_df = pd.DataFrame(records) if records else pd.DataFrame()
            xs = list(range(1, len(rec_df) + 1))

            t1, t2, t3, t4, t5, t6 = st.tabs([
                "📈 FVR/HER Trend",
                "⏱ Latency",
                "💰 Token Cost",
                "🔴 Error Taxonomy",
                "🗂 Raw Records",
                "🔬 Hypothesis Tests",
            ])

            with t1:
                if rec_df.empty:
                    st.info("No analysis queries yet — run queries in Interactive Chat to see FVR/HER trends.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=xs, y=rec_df["fvr"], name="FVR",
                        mode="lines+markers", line=dict(color="#58a6ff", width=2)))
                    fig.add_trace(go.Scatter(x=xs, y=rec_df["her"], name="HER",
                        mode="lines+markers", line=dict(color="#f85149", width=2)))
                    fig.update_layout(title="FVR & HER per Query",
                                      xaxis_title="Query #", yaxis=dict(range=[0,1.05]))
                    st.plotly_chart(_dark(fig), use_container_width=True)

                    fig2 = px.bar(rec_df, x=xs, y="cce", title="CCE per Query",
                                  color="cce", color_continuous_scale="Oranges")
                    st.plotly_chart(_dark(fig2), use_container_width=True)

            with t2:
                if rec_df.empty:
                    st.info("No analysis queries yet — latency data will appear here.")
                else:
                    fig3 = px.bar(rec_df, x=xs, y="response_time_s",
                                  title="Query Response Time (s)",
                                  color="response_time_s", color_continuous_scale="Blues")
                    st.plotly_chart(_dark(fig3), use_container_width=True)

            with t3:
                # Token usage is stored in session state, not per-record yet
                # Show cumulative tokens if we can reconstruct from messages
                msgs = st.session_state.get("messages", [])
                token_data = [
                    {"query": i+1, "tokens": m["content"].get("tokens_used", 0)}
                    for i, m in enumerate(msgs)
                    if m["role"] == "assistant" and isinstance(m.get("content"), dict)
                ]
                if token_data:
                    tok_df = pd.DataFrame(token_data)
                    tok_df["cumulative"] = tok_df["tokens"].cumsum()
                    fig_tok = px.bar(tok_df, x="query", y="tokens",
                                     title="Tokens per Query",
                                     color="tokens", color_continuous_scale="Purples")
                    st.plotly_chart(_dark(fig_tok), use_container_width=True)
                    fig_cum = px.line(tok_df, x="query", y="cumulative",
                                      title="Cumulative Token Usage",
                                      markers=True)
                    fig_cum.update_traces(line_color="#bc8cff")
                    st.plotly_chart(_dark(fig_cum), use_container_width=True)
                else:
                    st.info("Token data available after queries.")

            with t4:
                taxonomy = st.session_state.error_taxonomy
                if taxonomy:
                    err_df = pd.DataFrame(
                        [{"Error Type": k, "Count": v}
                         for k, v in taxonomy.items()]
                    ).sort_values("Count", ascending=False)
                    fig_err = px.bar(err_df, x="Error Type", y="Count",
                                     title="Execution Error Taxonomy",
                                     color="Count", color_continuous_scale="Reds")
                    st.plotly_chart(_dark(fig_err), use_container_width=True)
                    st.dataframe(err_df, use_container_width=True)
                else:
                    st.success("✅ No execution errors recorded this session.")

                # Query type distribution
                qt_counts = {"timeseries": st.session_state.ts_count,
                             "tabular": st.session_state.tab_count,
                             "other": max(0, total_q
                                          - st.session_state.ts_count
                                          - st.session_state.tab_count)}
                qt_df = pd.DataFrame([{"Type": k, "Count": v}
                                       for k, v in qt_counts.items() if v > 0])
                if not qt_df.empty:
                    fig_qt = px.pie(qt_df, names="Type", values="Count",
                                    title="Query Type Distribution",
                                    color_discrete_sequence=["#58a6ff","#d29922","#3fb950"])
                    st.plotly_chart(_dark(fig_qt), use_container_width=True)

            with t5:
                if rec_df.empty:
                    st.info("No analysis queries yet — raw records will appear here.")
                else:
                    show = [c for c in [
                        "query","fvr","her","cce","response_time_s",
                        "execution_failed","needed_correction","correction_succeeded",
                        "verified_claims","unverified_claims","total_claims",
                    ] if c in rec_df.columns]
                    st.dataframe(rec_df[show], use_container_width=True)

                    csv_bytes = rec_df[show].to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Session Records (CSV)",
                        csv_bytes,
                        file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

            with t6:
                # Build hypothesis log from session messages
                hyp_rows = []
                hyp_num  = 0
                for msg in st.session_state.get("messages", []):
                    if msg["role"] != "assistant":
                        continue
                    content = msg.get("content", {})
                    if not isinstance(content, dict):
                        continue

                    if content.get("_type") == "hypothesis":
                        hyp_num += 1
                        r  = content.get("result", {})
                        q  = content.get("_question", "—")

                        test_type = str(r.get("test_type", r.get("test", "—"))).upper()
                        stat_key  = "f_stat" if "f_stat" in r else ("r" if "r" in r else "statistic")
                        stat_val  = r.get(stat_key, r.get("statistic", None))
                        p_val     = r.get("p_value", None)
                        alpha_v   = r.get("alpha", 0.05)
                        n_obs     = r.get("_n_aligned", r.get("n", "—"))
                        rejected  = r.get("reject_h0", None)
                        join_m    = str(r.get("_join_method", "—"))
                        warnings  = r.get("_guard_warnings", [])
                        agg_sql   = r.get("_agg_sql", "")

                        def _fmt_stat(v):
                            if v is None: return "—"
                            if v == float("inf"): return "∞"
                            if isinstance(v, float) and v != v: return "NaN"
                            try: return f"{float(v):.4f}"
                            except: return str(v)

                        def _fmt_p(v):
                            if v is None: return "—"
                            try:
                                fv = float(v)
                                if fv == 0 or fv < 1e-15: return "< 1e-15"
                                if fv < 0.0001: return f"{fv:.2e}"
                                return f"{fv:.4f}"
                            except: return str(v)

                        verdict   = ("✅ Reject H₀"          if rejected is True  else
                                     "🟡 Fail to Reject H₀"  if rejected is False else "—")
                        flag      = ("🚨" if any("🚨" in w for w in warnings) else
                                     "⚠️" if any("⚠️" in w for w in warnings) else
                                     "ℹ️" if warnings else "")

                        hyp_rows.append({
                            "#":           hyp_num,
                            "Query":       q[:80] + ("…" if len(q) > 80 else ""),
                            "Test":        test_type,
                            "Statistic":   _fmt_stat(stat_val),
                            "p-value":     _fmt_p(p_val),
                            "α":           alpha_v,
                            "n":           n_obs,
                            "Verdict":     verdict,
                            "Significant": "Yes" if rejected is True else ("No" if rejected is False else "—"),
                            "Join / Mode": join_m[:55] + ("…" if len(join_m) > 55 else ""),
                            "Flags":       flag,
                        })

                    elif content.get("_type") == "hypothesis_error":
                        hyp_num += 1
                        err_txt = content.get("error", "")
                        hyp_rows.append({
                            "#":           hyp_num,
                            "Query":       "—",
                            "Test":        "—",
                            "Statistic":   "—",
                            "p-value":     "—",
                            "α":           "—",
                            "n":           "—",
                            "Verdict":     "❌ Error",
                            "Significant": "—",
                            "Join / Mode": "—",
                            "Flags":       f"🚨 {err_txt[:55]}",
                        })

                if not hyp_rows:
                    st.info("No hypothesis tests run this session yet.")
                else:
                    hyp_df = pd.DataFrame(hyp_rows)

                    # Summary counters
                    n_reject  = sum(1 for r in hyp_rows if r["Significant"] == "Yes")
                    n_fail    = sum(1 for r in hyp_rows if r["Significant"] == "No")
                    n_error   = sum(1 for r in hyp_rows if r["Verdict"] == "❌ Error")
                    n_flagged = sum(1 for r in hyp_rows if r["Flags"] not in ("", "ℹ️"))

                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Total Tests",      len(hyp_rows))
                    mc2.metric("✅ Reject H₀",     n_reject)
                    mc3.metric("🟡 Fail to Reject", n_fail)
                    mc4.metric("❌ Errors",         n_error)
                    mc5.metric("⚠️ Flagged",        n_flagged)

                    st.markdown("#### All Hypothesis Tests")

                    def _style_verdict(val):
                        if "Reject H₀" in str(val) and "Fail" not in str(val):
                            return "color: #3fb950; font-weight: 600"
                        if "Fail" in str(val):
                            return "color: #d29922; font-weight: 600"
                        if "Error" in str(val):
                            return "color: #f85149; font-weight: 600"
                        return ""

                    styled_hyp = hyp_df.style.applymap(_style_verdict, subset=["Verdict"])
                    st.dataframe(styled_hyp, use_container_width=True, hide_index=True)

                    csv_hyp = hyp_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Hypothesis Log (CSV)",
                        csv_hyp,
                        file_name=f"hypothesis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                    # Per-test detail expanders
                    st.markdown("#### Per-Test Details")
                    for msg in st.session_state.get("messages", []):
                        if msg["role"] != "assistant":
                            continue
                        content = msg.get("content", {})
                        if not isinstance(content, dict):
                            continue
                        if content.get("_type") != "hypothesis":
                            continue
                        r        = content.get("result", {})
                        q        = content.get("_question", "—")
                        warnings = r.get("_guard_warnings", [])
                        agg_sql  = r.get("_agg_sql", "")
                        label    = q[:70] + ("…" if len(q) > 70 else "")
                        with st.expander(f"🔬 {label}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Test:** `{str(r.get('test_type', r.get('test','—'))).upper()}`")
                                st.markdown(f"**n:** {r.get('_n_aligned', r.get('n','—'))}")
                                st.markdown(f"**Join/Mode:** {r.get('_join_method','—')}")
                            with col2:
                                sk = "f_stat" if "f_stat" in r else ("r" if "r" in r else "statistic")
                                st.markdown(f"**Statistic:** `{r.get(sk, r.get('statistic','—'))}`")
                                st.markdown(f"**p-value:** `{r.get('p_value','—')}`")
                                st.markdown(f"**Conclusion:** {r.get('conclusion','—')}")
                            if warnings:
                                for w in warnings:
                                    if "🚨" in w:   st.error(w)
                                    elif "⚠️" in w: st.warning(w)
                                    else:            st.info(w)
                            if agg_sql:
                                st.code(agg_sql, language="sql")

    st.divider()

    # ── Section B: Cross-dataset registry ─────────────────────────────────────
    st.subheader("🗄️ Cross-Dataset Benchmark Registry")
    st.caption(
        "Save a dataset's benchmark via the sidebar button, then compare "
        "any two or more datasets across sessions here."
    )
    all_bm = load_all_benchmarks()

    if not all_bm:
        st.info("No benchmarks saved yet. Analyse a dataset and click 'Save Benchmark'.")
    else:
        reg_df = pd.DataFrame([{
            "Dataset":     e["dataset_name"],
            "Type":        e.get("file_type","?").upper(),
            "Rows":        e.get("row_count", 0),
            "Cols":        e.get("col_count", 0),
            "Queries":     e.get("total_queries", 0),
            "Avg FVR":     e.get("avg_fvr", 0.0),
            "Avg HER":     e.get("avg_her", 0.0),
            "Avg CCE":     e.get("avg_cce", 0.0),
            "EER":         e.get("eer", 0.0),
            "SCR":         e.get("scr", 0.0),
            "Avg QRT(s)":  e.get("avg_qrt_s", 0.0),
            "TS Queries":  e.get("ts_queries", 0),
            "Tab Queries": e.get("tabular_queries", 0),
            "Recorded":    e.get("recorded_at","")[:10],
        } for e in all_bm])

        st.dataframe(reg_df, use_container_width=True)

        names    = reg_df["Dataset"].tolist()
        selected = st.multiselect("Compare datasets:", names,
                                  default=names[:min(len(names), 4)])

        if len(selected) >= 2:
            cmp = reg_df[reg_df["Dataset"].isin(selected)].copy()

            ov, qual, spd, radar, export = st.tabs([
                "📊 Overview", "🎯 Quality", "⚡ Speed & Recovery",
                "🕸 Radar", "📥 Export",
            ])

            with ov:
                melted = cmp.melt(
                    id_vars="Dataset",
                    value_vars=["Avg FVR","Avg HER","EER","SCR"],
                )
                fig = px.bar(melted, x="Dataset", y="value", color="variable",
                             barmode="group", title="Quality & Reliability",
                             color_discrete_sequence=["#58a6ff","#f85149","#ffa657","#3fb950"])
                st.plotly_chart(_dark(fig), use_container_width=True)

            with qual:
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(_dark(px.bar(cmp, x="Dataset", y="Avg FVR",
                        title="Avg FVR ↑", color="Avg FVR",
                        color_continuous_scale="Greens")), use_container_width=True)
                with cb:
                    st.plotly_chart(_dark(px.bar(cmp, x="Dataset", y="Avg HER",
                        title="Avg HER ↓", color="Avg HER",
                        color_continuous_scale="Reds")), use_container_width=True)
                st.plotly_chart(_dark(px.bar(cmp, x="Dataset", y="Avg CCE",
                    title="Avg CCE ↓", color="Avg CCE",
                    color_continuous_scale="Oranges")), use_container_width=True)

            with spd:
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(_dark(px.bar(cmp, x="Dataset", y="Avg QRT(s)",
                        title="Avg QRT ↓", color="Avg QRT(s)",
                        color_continuous_scale="Blues")), use_container_width=True)
                with cb:
                    st.plotly_chart(_dark(px.bar(cmp, x="Dataset", y="SCR",
                        title="SCR ↑", color="SCR",
                        color_continuous_scale="Greens")), use_container_width=True)

            with radar:
                cats   = ["Avg FVR","SCR","1-EER","1-Avg HER","1-Avg CCE"]
                r_fig  = go.Figure()
                colors = px.colors.qualitative.Set2
                for idx, (_, row) in enumerate(cmp.iterrows()):
                    vals = [
                        row["Avg FVR"],
                        row["SCR"],
                        1.0 - row["EER"],
                        1.0 - row["Avg HER"],
                        max(0.0, 1.0 - row["Avg CCE"]),
                    ]
                    vals += [vals[0]]
                    r_fig.add_trace(go.Scatterpolar(
                        r=vals, theta=cats + [cats[0]],
                        name=row["Dataset"], fill="toself", opacity=0.55,
                        line=dict(color=colors[idx % len(colors)]),
                    ))
                r_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,1], color="#8b949e"),
                        bgcolor="#0d1117",
                    ),
                    title="Agent Quality Radar (all axes: ↑ = better)",
                )
                st.plotly_chart(_dark(r_fig), use_container_width=True)

            with export:
                st.markdown("Download the comparison table as CSV for your paper.")
                st.dataframe(cmp, use_container_width=True)
                st.download_button(
                    "⬇️ Download Comparison CSV",
                    cmp.to_csv(index=False).encode(),
                    file_name=f"benchmark_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

        st.divider()
        del_name = st.selectbox("Delete entry:", ["—"] + names)
        if del_name != "—":
            if st.button(f"Delete '{del_name}'", type="secondary"):
                delete_benchmark(del_name)
                st.success(f"Deleted '{del_name}'")
                st.rerun()

    st.divider()

    # ── Section C: Batch pipeline ──────────────────────────────────────────────
    st.subheader("🚀 Batch Pipeline")
    if st.button("▶ Run research_pipeline.py", type="primary"):
        with st.spinner("Running…"):
            import subprocess
            try:
                res = subprocess.run(
                    ["python", "research_pipeline.py"],
                    capture_output=True, text=True
                )
                if res.returncode == 0:
                    st.success("✅ Done!")
                    st.rerun()
                else:
                    st.error(res.stderr)
            except Exception as e:
                st.error(str(e))
