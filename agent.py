import pandas as pd
import sqlite3
import time
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL
from stats_engine import compute_dataset_statistics
from evaluation_metrics import metrics_tracker
from code_executor import execute_pandas_code

client = Groq(api_key=GROQ_API_KEY)

def get_sqlite_schema(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_str = ""
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            col_details = [f"{col[1]} ({col[2]})" for col in columns]
            schema_str += f"Table: {table_name}\nColumns: {', '.join(col_details)}\n\n"
        conn.close()
        return schema_str
    except Exception as e:
        return f"Could not retrieve schema: {str(e)}"

def generate_code_prompt(question: str, df: pd.DataFrame, db_path: str = None) -> str:
    if db_path:
        schema = get_sqlite_schema(db_path)
        context_block = f"""SQLITE DATABASE CONTEXT:
You have access to a SQLite database. The file path is stored in the Python variable `DB_PATH`.
Here is the complete relational schema for all tables in the database:
{schema}

INSTRUCTIONS FOR SQLITE:
1. To query the database, use `import sqlite3`, create a connection `conn = sqlite3.connect(DB_PATH)`, and use `pd.read_sql_query(sql, conn)` to execute SQL queries.
2. ENVIRONMENT RULE: The variable `DB_PATH` is pre-loaded in your environment. You must use `DB_PATH` directly. NEVER initialize, define, or overwrite this variable in your code.
3. Write raw SQL queries inside your python code using JOINs to combine tables to answer the user's question.
4. You MUST assign your final calculated answer to a Python variable named `RESULT`."""
    else:
        schema = df.dtypes.to_string()
        sample_data = df.head(3).to_string()
        context_block = f"""DATASET CONTEXT (Variable name is `df`):
Columns and Data Types:
{schema}

Sample Rows:
{sample_data}

INSTRUCTIONS FOR CSV:
1. The dataframe is already loaded as a variable named `df`. Do not load the CSV.
2. You MUST assign your final calculated answer to a Python variable named `RESULT`."""

    return f"""You are an elite Data Scientist. Your task is to write Python pandas/SQL code to answer the user's query.

USER QUERY: "{question}"

{context_block}

GENERAL INSTRUCTIONS:
1. Write ONLY valid, executable Python code.
2. CHART EXPORT RULE: If asked for a BAR CHART, HISTOGRAM, or PIE CHART, group/aggregate the data, limit it to the top 20 rows, and assign the resulting Pandas DataFrame directly to `RESULT`. DO NOT use `.describe()`.
3. DEFENSIVE PROGRAMMING: When converting columns to numeric, ALWAYS use `pd.to_numeric(..., errors='coerce')` and drop NaNs before math operations.
4. NO PLOTTING LIBRARIES: DO NOT use matplotlib, seaborn, plotly, or any visual libraries. The frontend handles visualizations via JSON.
5. CURRENCY RULE: DO NOT use dollar signs ($) in your text output. Always use the word "dollars" or "USD" instead to prevent UI formatting errors.
"""

# FIX: Added correction_context parameter to handle manual self-correction
def analyze(question: str, df: pd.DataFrame, db_path: str = None, correction_context: dict = None) -> dict:
    start_time = time.time()
    stats = compute_dataset_statistics(df)
    
    code_prompt = generate_code_prompt(question, df, db_path)
    
    messages = [
        {"role": "system", "content": "You output only python code."},
        {"role": "user", "content": code_prompt}
    ]
    
    # If the user clicked the "Self-Correct" button, inject the previous broken code and error!
    if correction_context:
        messages.append({"role": "assistant", "content": correction_context["code"]})
        error_message = f"Execution failed with error:\n{correction_context['error']}\n\nFix the code. Ensure exact schema matching. Output ONLY valid python code."
        messages.append({"role": "user", "content": error_message})
        current_attempt = correction_context.get("attempt", 1) + 1
    else:
        current_attempt = 1

    # ONE-SHOT EXECUTION (No while loop!)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1
        )
        generated_code = response.choices[0].message.content
        
        execution = execute_pandas_code(generated_code, df, db_path)
        
        correction_log = [{
            "attempt": current_attempt,
            "code": generated_code,
            "status": execution["status"],
            "error_message": execution["output"] if execution["status"] == "error" else None
        }]
        
        # IF IT FAILS: Return immediately and tell Streamlit to show the button
        if execution["status"] == "error":
            insights = f"⚠️ **Execution Failed on Attempt {current_attempt}**\n\nThe AI wrote code that crashed. You can click the button below to feed the error back to the AI so it can fix it."
            return {
                "insights": insights,
                "metrics": {"hr": 0.0, "ngs": 0.0, "acs": 0.0, "cce": 0.0, "assigned_confidence": 0.0, "actual_accuracy": 0.0},
                "processing_time": round(time.time() - start_time, 2),
                "correction_log": correction_log,
                "executed_code": generated_code,
                "needs_correction": True, # The trigger for our UI button!
                "error_trace": execution['output'],
                "attempt": current_attempt
            }
        
    except Exception as e:
        return {
            "insights": f"Critical AI API Error: {str(e)}",
            "metrics": {"hr": 0.0, "ngs": 0.0, "acs": 0.0, "cce": 0.0, "assigned_confidence": 0.0, "actual_accuracy": 0.0},
            "processing_time": round(time.time() - start_time, 2),
            "correction_log": [],
            "executed_code": "API Failure",
            "needs_correction": False
        }
        
    # IF IT SUCCEEDS: Run the Synthesis
    try:
        raw_output = str(execution['output'])
        if len(raw_output) > 2000:
            raw_output = raw_output[:2000] + "\n...[OUTPUT TRUNCATED TO PREVENT API OVERLOAD]..."

        bt = "```"
        synthesis_prompt = f"""The user asked: "{question}"
The backend executed python code and returned this exact result: {raw_output}

Formulate a concise, professional answer to the user's question using ONLY the result provided. Do not add outside information.

CURRENCY RULE: DO NOT use dollar signs ($) in your text output. Always use the word "dollars" or "USD" instead to prevent frontend formatting errors.

CRITICAL VISUALIZATION RULE:
If the user asks to "plot", "graph", or "visualize" the data, you MUST append a JSON block at the very end of your response. If you do not include this JSON, the frontend will fail to render the chart.
Format exactly like this:
{bt}json
[
  {{"type": "bar", "x": "ColumnName1", "y": "ColumnName2"}}
]
{bt}
Supported types: bar, scatter, histogram, box. 
The "x" and "y" values MUST exactly match the column names printed in the result above (e.g., "BillingCountry" and "TotalRevenue")."""

        synth_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "You are a professional data analyst."},
                      {"role": "user", "content": synthesis_prompt}],
            temperature=0.1
        )
        insights = synth_response.choices[0].message.content
        
        exec_numbers = re.findall(r'-?\d+\.?\d*', str(execution['output']))
        query_numbers = re.findall(r'-?\d+\.?\d*', question)
        all_valid_numbers = exec_numbers + query_numbers
        
        if all_valid_numbers:
            stats.setdefault("numeric_summary", {})["dynamic_result"] = {}
            for i, num in enumerate(all_valid_numbers):
                try:
                    stats["numeric_summary"]["dynamic_result"][f"val_{i}"] = float(num)
                except ValueError:
                    pass
        
        hr_data = metrics_tracker.calculate_hallucination_rate(insights, stats)
        ngs_data = metrics_tracker.calculate_ngs(insights, stats)
        acs_data = metrics_tracker.calculate_acs(insights, stats)
        
        base_confidence = max(1.0 - ((current_attempt - 1) * 0.15), 0.0) 
        actual_accuracy = ngs_data["ngs"]
        
        cce_data = metrics_tracker.calculate_cce(base_confidence, actual_accuracy)

        return {
            "insights": insights,
            "metrics": {
                "hr": round(hr_data["hr"], 4),
                "ngs": round(ngs_data["ngs"], 4),
                "acs": round(acs_data["acs"], 4),
                "cce": round(cce_data["cce"], 4),
                "assigned_confidence": round(base_confidence, 4),
                "actual_accuracy": round(actual_accuracy, 4)
            },
            "processing_time": round(time.time() - start_time, 2),
            "correction_log": correction_log,
            "executed_code": generated_code,
            "needs_correction": False # Tells UI not to show the button
        }
        
    except Exception as e:
        return {
            "insights": f"Code executed successfully, but synthesis failed: {str(e)}",
            "metrics": {"hr": 0.0, "ngs": 0.0, "acs": 0.0, "cce": 0.0, "assigned_confidence": 0.0, "actual_accuracy": 0.0},
            "processing_time": round(time.time() - start_time, 2),
            "correction_log": correction_log,
            "executed_code": generated_code,
            "needs_correction": False
        }