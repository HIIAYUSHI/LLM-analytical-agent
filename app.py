import streamlit as st
import pandas as pd
import glob
import os
import json
import re
import plotly.express as px
import sqlite3
import tempfile
import hashlib
from agent import analyze
from rag import ingest_dataframe_info

st.set_page_config(page_title="Research Data Analyst", layout="wide")
st.title("🤖 Enterprise AI Data Analyst - Research Edition")

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "needs_correction" not in st.session_state:
    st.session_state.needs_correction = False
if "correction_data" not in st.session_state:
    st.session_state.correction_data = {}
# ------------------------

def render_insights_and_charts(insights_text, df, db_path=None, executed_code=""):
    json_pattern = r'```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```'
    match = re.search(json_pattern, insights_text, re.DOTALL)
    
    if not match:
        json_pattern_raw = r'(\[\s*\{.*?\}\s*\])'
        match = re.search(json_pattern_raw, insights_text, re.DOTALL)

    if match:
        json_str = match.group(1)
        clean_text = insights_text.replace(match.group(0), "").strip()
        st.write(clean_text)
        
        try:
            charts = json.loads(json_str)
            if charts:
                st.write("### 📈 Visualizations")
                
                plot_df = df.copy() if df is not None and not df.empty else pd.DataFrame()
                
                if executed_code:
                    env = {"df": plot_df, "DB_PATH": db_path, "pd": pd, "sqlite3": sqlite3}
                    try:
                        clean_code = executed_code
                        code_match = re.search(r'```(?:python)?(.*?)```', clean_code, re.DOTALL)
                        if code_match:
                            clean_code = code_match.group(1).strip()
                            
                        exec(clean_code, env)
                        
                        if "RESULT" in env:
                            res_obj = env["RESULT"]
                            if isinstance(res_obj, pd.DataFrame):
                                plot_df = res_obj.copy()
                            elif isinstance(res_obj, pd.Series):
                                plot_df = res_obj.reset_index()
                                
                            # FIX: Catch JSON serialization errors by converting Pandas Intervals/Categories to strings
                            for col in plot_df.columns:
                                if 'interval' in str(plot_df[col].dtype).lower() or 'category' in str(plot_df[col].dtype).lower():
                                    plot_df[col] = plot_df[col].astype(str)

                    except Exception as e:
                        st.error(f"⚠️ Chart Data Loading Error: {e}")

                cols = st.columns(min(len(charts), 2)) 
                
                for i, chart in enumerate(charts):
                    chart_type = chart.get("type", "").lower()
                    x_col = chart.get("x")
                    y_col = chart.get("y")
                    
                    if x_col and x_col not in plot_df.columns:
                        st.warning(f"⚠️ Could not plot: Column '{x_col}' not found in the generated data. Available columns: {list(plot_df.columns)}")
                        continue
                        
                    fig = None
                    col_idx = i % 2
                    
                    with cols[col_idx]:
                        if chart_type in ["bar_chart", "bar"]:
                            if y_col and y_col in plot_df.columns:
                                if len(plot_df) <= 100:
                                    plot_df = plot_df.sort_values(by=y_col, ascending=False)
                                    fig = px.bar(plot_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                else:
                                    agg_df = plot_df.groupby(x_col)[y_col].mean().reset_index()
                                    agg_df = agg_df.sort_values(by=y_col, ascending=False).head(20)
                                    fig = px.bar(agg_df, x=x_col, y=y_col, title=f"Top Average {y_col} by {x_col}")
                            else:
                                val_counts = plot_df[x_col].value_counts().reset_index()
                                if len(val_counts.columns) == 2:
                                    val_counts.columns = [x_col, "count"]
                                val_counts = val_counts.sort_values(by="count", ascending=False).head(20)
                                fig = px.bar(val_counts, x=x_col, y="count", title=f"Top Distribution of {x_col}")
                                
                        elif chart_type in ["scatter_plot", "scatter"]:
                            if y_col and y_col in plot_df.columns:
                                fig = px.scatter(plot_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                
                        elif chart_type == "histogram":
                            if y_col and y_col in plot_df.columns:
                                # If the AI explicitly provided a Y column, use it
                                fig = px.histogram(plot_df, x=x_col, y=y_col, title=f"Histogram of {x_col}")
                            elif len(plot_df.columns) == 2 and x_col in plot_df.columns:
                                # If the AI pre-aggregated it into 2 columns (Bins and Counts), 
                                # grab the count column and plot it as a bar chart
                                y_auto = [c for c in plot_df.columns if c != x_col][0]
                                fig = px.bar(plot_df, x=x_col, y=y_auto, title=f"Histogram of {x_col}")
                            else:
                                # If it's raw, unaggregated data, let Plotly count natively
                                fig = px.histogram(plot_df, x=x_col, title=f"Histogram of {x_col}")
                            
                        elif chart_type in ["box_plot", "box"]:
                            if y_col and y_col in plot_df.columns:
                                fig = px.box(plot_df, x=x_col, y=y_col, title=f"Box Plot: {y_col} by {x_col}")
                            else:
                                fig = px.box(plot_df, y=x_col, title=f"Box Plot of {x_col}")
                        
                        if fig:
                            safe_key = hashlib.md5(f"{insights_text}_{i}".encode()).hexdigest()
                            st.plotly_chart(fig, use_container_width=True, key=safe_key)
                            
        except json.JSONDecodeError:
            st.error("Failed to parse the chart specifications from the AI.")
    else:
        st.write(insights_text)

# ==========================================
# SIDEBAR NAVIGATION & DATA UPLOAD
# ==========================================
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to:", ["💬 Interactive Chat", "📊 Research Metrics Dashboard"])
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset", type=["csv", "db", "sqlite"])

# ==========================================
# PAGE 1: INTERACTIVE CHAT
# ==========================================
if app_mode == "💬 Interactive Chat":
    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
            st.session_state.needs_correction = False 
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                st.session_state.db_path = None 
            elif file_extension in ['db', 'sqlite']:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                st.session_state.db_path = tmp_path
                conn = sqlite3.connect(tmp_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if tables:
                    first_table = tables[0][0]
                    df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                else:
                    df = pd.DataFrame()
                conn.close()
            
            st.session_state.df = df
            with st.spinner("Indexing dataset for RAG..."):
                ingest_dataframe_info(df)
        else:
            df = st.session_state.df
            
        with st.expander("👀 View Dataset Header", expanded=False):
            st.dataframe(df.head())

        # 1. RENDER PAST CHAT HISTORY
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                elif msg["role"] == "assistant":
                    res = msg["content"]
                    st.write("### 💡 Insights")
                    render_insights_and_charts(res["insights"], df, st.session_state.db_path, res.get("executed_code", ""))
                    
                    with st.expander("🔍 View Execution Log"):
                        log = res.get("correction_log", [])
                        for entry in log:
                            st.markdown(f"**Attempt {entry['attempt']}**")
                            st.code(entry['code'], language="python")
                            if entry['status'] == 'error':
                                st.error(f"Error:\n{entry['error_message']}")
                            else:
                                st.success("Success!")
                    
                    if not res.get("needs_correction", False):
                        cols = st.columns(4)
                        cols[0].metric("Hallucination Rate (HR)", f"{res['metrics']['hr']:.2%}")
                        cols[1].metric("Numerical Grounding (NGS)", f"{res['metrics']['ngs']:.2%}")
                        cols[2].metric("Analytical Consistency (ACS)", f"{res['metrics']['acs']:.2%}")
                        cols[3].metric("Confidence Error (CCE)", f"{res['metrics']['cce']:.4f}")

        # ==========================================
        # THE MANUAL SELF-CORRECTION BUTTON UI
        # ==========================================
        if st.session_state.needs_correction:
            st.warning("The AI encountered an error while writing the code. Would you like it to read the error and try fixing it?")
            if st.button("🔧 Trigger Self-Correction", type="primary"):
                with st.chat_message("assistant"):
                    with st.spinner(f"🧠 Analyzing error from Attempt {st.session_state.correction_data['attempt']} and rewriting code..."):
                        
                        result = analyze(
                            question=st.session_state.correction_data["query"], 
                            df=df, 
                            db_path=st.session_state.db_path,
                            correction_context=st.session_state.correction_data
                        )
                        
                        st.session_state.needs_correction = result.get("needs_correction", False)
                        if st.session_state.needs_correction:
                            st.session_state.correction_data["code"] = result["executed_code"]
                            st.session_state.correction_data["error"] = result["error_trace"]
                            st.session_state.correction_data["attempt"] = result["attempt"]
                        
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        st.rerun() 

        # ==========================================
        # 2. HANDLE NEW QUERIES
        # ==========================================
        if query := st.chat_input("Ask a question about the data"):
            st.session_state.needs_correction = False 
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(f"**You:** {query}")

            with st.chat_message("assistant"):
                with st.spinner("🧠 Writing and Executing Code..."):
                    result = analyze(query, df, st.session_state.db_path) 
                    
                    st.session_state.needs_correction = result.get("needs_correction", False)
                    if st.session_state.needs_correction:
                        st.session_state.correction_data = {
                            "query": query,
                            "code": result["executed_code"],
                            "error": result["error_trace"],
                            "attempt": result["attempt"]
                        }

                    st.write("### 💡 Insights")
                    render_insights_and_charts(result["insights"], df, st.session_state.db_path, result.get("executed_code", ""))
                    
                    with st.expander("🔍 View Execution Log"):
                        log = result.get("correction_log", [])
                        for entry in log:
                            st.markdown(f"**Attempt {entry['attempt']}**")
                            st.code(entry['code'], language="python")
                            if entry['status'] == 'error':
                                st.error(f"Error:\n{entry['error_message']}")
                            else:
                                st.success("Success!")
                    
                    if not st.session_state.needs_correction:
                        cols = st.columns(4)
                        cols[0].metric("Hallucination Rate (HR)", f"{result['metrics']['hr']:.2%}")
                        cols[1].metric("Numerical Grounding (NGS)", f"{result['metrics']['ngs']:.2%}")
                        cols[2].metric("Analytical Consistency (ACS)", f"{result['metrics']['acs']:.2%}")
                        cols[3].metric("Confidence Error (CCE)", f"{result['metrics']['cce']:.4f}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                    if st.session_state.needs_correction:
                        st.rerun()

    else:
        st.info("👈 Please upload a CSV or SQLite dataset in the sidebar to begin.")

# ==========================================
# PAGE 2: RESEARCH METRICS DASHBOARD
# ==========================================
elif app_mode == "📊 Research Metrics Dashboard":
    st.header("Batch Research Results")
    
    # 1. THE TRIGGER BUTTON
    st.markdown("Click the button below to automatically run all benchmark queries and generate a new summary CSV.")
    if st.button("🚀 Run Batch Research Pipeline", type="primary"):
        with st.spinner("Running automated benchmarks... This may take a few minutes depending on API limits and dataset size."):
            import subprocess
            try:
                # Run the external pipeline script just like you would in the terminal
                result = subprocess.run(["python", "research_pipeline.py"], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("✅ Batch pipeline completed successfully!")
                    st.rerun() # Force the page to refresh and load the new CSV
                else:
                    st.error(f"❌ Pipeline failed with error:\n{result.stderr}")
            except FileNotFoundError:
                st.error("❌ Could not find 'research_pipeline.py'. Make sure the file is in the same folder as app.py.")
            except Exception as e:
                st.error(f"❌ Failed to execute script: {str(e)}")
                
    st.divider()

    # 2. LOAD AND DISPLAY THE RESULTS
    summary_files = glob.glob("research_summary_*.csv")
    if summary_files:
        # Grab the most recently created CSV
        latest_file = max(summary_files, key=os.path.getctime)
        st.success(f"Loaded latest research data: {latest_file}")
        
        summary_df = pd.read_csv(latest_file)
        st.dataframe(summary_df, use_container_width=True)
        
        # Draw the comparison chart
        fig = px.bar(summary_df, x="dataset", y=["hr", "ngs", "acs", "stability"], barmode="group",
                     title="System Performance Metrics by Dataset")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No research summary files found. Click the button above to generate your first benchmark!")