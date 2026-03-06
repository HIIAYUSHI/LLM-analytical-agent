def build_rag_prompt(question: str, data_summary: dict, rag_context: str) -> str:
    prompt = f"USER QUESTION:\n{question}\n\n"
    prompt += f"DATASET: {data_summary['row_count']} rows, {data_summary['column_count']} columns.\n\n"
    
    if data_summary.get('numeric_summary'):
        prompt += "NUMERIC STATS:\n"
        for col, stats in list(data_summary['numeric_summary'].items())[:10]:
            prompt += f"- {col}: range [{stats['min']} - {stats['max']}], mean={stats['mean']}\n"
            
    prompt += f"\nCONTEXT:\n{rag_context}\n\n"
    prompt += (
        "INSTRUCTIONS:\n"
        "1. Answer the user's question directly, concisely, and strictly based on the provided stats.\n"
        "2. DO NOT suggest visualizations, DO NOT output JSON blocks, and DO NOT add extra commentary unless the user explicitly asks for a chart, plot, or visualization."
    )
    return prompt