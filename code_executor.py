import pandas as pd
import traceback
import re

def execute_pandas_code(code_string: str, df: pd.DataFrame, db_path: str = None) -> dict:
    """
    Securely executes AI-generated pandas code and captures the result.
    Requires the AI to save its final answer to a variable named 'RESULT'.
    """
    # Extract raw code from markdown blocks if present
    match = re.search(r'```(?:python)?(.*?)```', code_string, re.DOTALL)
    clean_code = match.group(1).strip() if match else code_string.strip()

    # Setup the execution environment
    # We pass 'df' and 'DB_PATH' directly into the environment so the code can interact with them
    local_env = {
        "df": df, 
        "pd": pd, 
        "RESULT": None,  # The AI must populate this variable
        "DB_PATH": db_path # Passes the SQLite DB path natively
    }
    
    # Execute the code
    try:
        exec(clean_code, {}, local_env)
        
        # Validate that the AI actually stored the result where we asked it to
        if local_env.get("RESULT") is None:
             return {
                "status": "error", 
                "output": "Code executed successfully, but 'RESULT' variable was not defined. You must assign your final answer to 'RESULT'."
            }
             
        return {
            "status": "success", 
            "output": local_env["RESULT"],
            "code_executed": clean_code
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "output": traceback.format_exc(),
            "code_executed": clean_code
        }