"""
code_executor.py
================
Executes AI-generated pandas/SQL code in a controlled namespace.

Fixes over original
--------------------
1. NAMESPACE BUG — exec(code, {}, local_env) split globals/locals so any
   function defined inside generated code couldn't see df, pd, etc.
   Fixed: single dict passed as globals so name resolution works correctly.

2. TIMEOUT — 30-second thread-based limit stops runaway queries from
   hanging the Streamlit app.

3. numpy pre-loaded — AI-generated code uses np constantly; it's now
   available by default without requiring an explicit import.

Security note
-------------
exec() is inherently unsafe. For production, replace with a subprocess
sandbox (RestrictedPython, Docker container, etc.). Acceptable for research.
"""

import re
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import sqlite3

_TIMEOUT_S  = 30
_EXECUTOR   = ThreadPoolExecutor(max_workers=4, thread_name_prefix="code_exec")
_FENCE_RE   = re.compile(r'```(?:python)?(.*?)```', re.DOTALL)


def _strip_fences(code: str) -> str:
    m = _FENCE_RE.search(code)
    return m.group(1).strip() if m else code.strip()


def execute_pandas_code(
    code_string: str,
    df:          pd.DataFrame,
    db_path:     str | None = None,
) -> dict:
    """
    Execute AI-generated code and return the result.

    The executed code MUST assign its final answer to ``RESULT``.

    Returns
    -------
    {
        "status":        "success" | "error",
        "output":        result value  |  traceback string,
        "code_executed": str,
    }
    """
    clean_code = _strip_fences(code_string)

    # Single namespace dict passed as *globals* so that functions defined
    # inside the generated code resolve df, pd, np, etc. without NameError.
    namespace: dict = {
        "df":      df,
        "pd":      pd,
        "np":      np,
        "sqlite3": sqlite3,
        "DB_PATH": db_path,
        "RESULT":  None,
    }

    thread_exc: dict = {}

    def _target():
        try:
            exec(clean_code, namespace)  # noqa: S102
        except Exception:
            thread_exc["tb"] = traceback.format_exc()

    fut = _EXECUTOR.submit(_target)
    try:
        fut.result(timeout=_TIMEOUT_S)
    except FuturesTimeoutError:
        return {
            "status": "error",
            "output": (
                f"Execution timed out after {_TIMEOUT_S}s. "
                "Simplify the query or reduce the data size."
            ),
            "code_executed": clean_code,
        }

    if thread_exc:
        return {
            "status": "error",
            "output": thread_exc["tb"],
            "code_executed": clean_code,
        }

    result = namespace.get("RESULT")
    if result is None:
        return {
            "status": "error",
            "output": (
                "Code executed successfully but `RESULT` was never assigned. "
                "Store your final answer in the variable `RESULT`."
            ),
            "code_executed": clean_code,
        }

    return {
        "status": "success",
        "output": result,
        "code_executed": clean_code,
    }
