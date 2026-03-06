import pandas as pd
import sqlite3
import time
import os
from datetime import datetime
from agent import analyze

# ==========================================
# 1. DEFINE YOUR BENCHMARK DATASETS & QUERIES
# ==========================================
BENCHMARKS = [
    {
        "dataset_name": "Titanic (CSV)",
        "file_path": "titanic.csv", 
        "type": "csv",
        "queries": [
            "What is the overall survival rate of passengers?",
            "Plot a bar chart showing the survival rate broken down by passenger class.",
            "Plot a histogram of the passenger ages."
        ]
    },
    {
        "dataset_name": "Chinook (SQLite)",
        "file_path": "Chinook_Sqlite.sqlite", 
        "type": "sqlite",
        "queries": [
            "Total revenue generated per country.",
            "Plot a bar chart of the average track price by genre.",
            "Show me the top 3 customers by total spending."
        ]
    }
]

def run_pipeline():
    print("Starting Batch Research Pipeline...")
    results = []

    for suite in BENCHMARKS:
        name = suite["dataset_name"]
        file_path = suite["file_path"]
        
        if not os.path.exists(file_path):
            print(f"\nWarning: File '{file_path}' not found. Skipping {name} benchmarks.")
            continue

        print(f"\nLoading Dataset: {name}")
        
        # ------------------------------------------
        # 2. LOAD THE DATA
        # ------------------------------------------
        df = None
        db_path = None
        
        if suite["type"] == "csv":
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading CSV {file_path}: {e}")
                continue
        elif suite["type"] == "sqlite":
            db_path = file_path
            try:
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if tables:
                    df = pd.read_sql_query(f"SELECT * FROM {tables[0][0]}", conn)
                else:
                    df = pd.DataFrame()
                conn.close()
            except Exception as e:
                print(f"Error loading SQLite {file_path}: {e}")
                continue

        # ------------------------------------------
        # 3. RUN THE QUERIES & COLLECT METRICS
        # ------------------------------------------
        total_hr, total_ngs, total_acs, total_cce, total_time = 0.0, 0.0, 0.0, 0.0, 0.0
        success_count = 0
        num_queries = len(suite["queries"])

        for i, query in enumerate(suite["queries"], 1):
            print(f"  [{i}/{num_queries}] Running query: '{query}'")
            
            # Initial Run
            result = analyze(question=query, df=df, db_path=db_path)
            
            # Simulated Human-in-the-Loop: If it fails, automatically trigger the correction
            if result.get("needs_correction"):
                print("    -> Attempt 1 failed. Triggering automated self-correction...")
                correction_data = {
                    "query": query,
                    "code": result.get("executed_code", ""),
                    "error": result.get("error_trace", ""),
                    "attempt": result.get("attempt", 1)
                }
                # Run again with correction context
                result = analyze(question=query, df=df, db_path=db_path, correction_context=correction_data)

            # Extract metrics
            metrics = result.get("metrics", {})
            hr = metrics.get("hr", 0.0)
            ngs = metrics.get("ngs", 0.0)
            acs = metrics.get("acs", 0.0)
            cce = metrics.get("cce", 0.0)
            p_time = result.get("processing_time", 0.0)

            total_hr += hr
            total_ngs += ngs
            total_acs += acs
            total_cce += cce
            total_time += p_time
            
            # Check if it was ultimately successful
            if not result.get("needs_correction"): 
                success_count += 1
            
            print(f"    -> HR: {hr:.2%} | NGS: {ngs:.2%} | ACS: {acs:.2%} | Time: {p_time:.2f}s")

        # ------------------------------------------
        # 4. CALCULATE AVERAGES
        # ------------------------------------------
        avg_hr = total_hr / num_queries if num_queries > 0 else 0
        avg_ngs = total_ngs / num_queries if num_queries > 0 else 0
        avg_acs = total_acs / num_queries if num_queries > 0 else 0
        avg_cce = total_cce / num_queries if num_queries > 0 else 0
        stability = (success_count / num_queries) * 100 if num_queries > 0 else 0
        avg_time = total_time / num_queries if num_queries > 0 else 0

        results.append({
            "dataset": name,
            "total_queries": num_queries,
            "hr": avg_hr,
            "ngs": avg_ngs,
            "acs": avg_acs,
            "cce": avg_cce,
            "stability": stability,
            "avg_processing_time": avg_time
        })
        print(f"  Completed! Stability: {stability:.1f}%")

    # ------------------------------------------
    # 5. EXPORT TO CSV FOR STREAMLIT
    # ------------------------------------------
    if results:
        summary_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_summary_{timestamp}.csv"
        summary_df.to_csv(filename, index=False)
        print(f"\nBatch processing complete! Results saved to '{filename}'")
    else:
        print("\nNo datasets processed. Please check your file paths.")

if __name__ == "__main__":
    run_pipeline()