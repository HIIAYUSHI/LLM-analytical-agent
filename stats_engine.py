import pandas as pd
import numpy as np

def compute_dataset_statistics(df: pd.DataFrame) -> dict:
    stats = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_summary": {},
        "categorical_summary": {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        clean_data = df[col].dropna()
        if len(clean_data) > 0:
            stats["numeric_summary"][col] = {
                "min": float(clean_data.min()),
                "max": float(clean_data.max()),
                "mean": float(clean_data.mean()),
                "median": float(clean_data.median())
            }

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        clean_data = df[col].dropna()
        if len(clean_data) > 0:
            stats["categorical_summary"][col] = clean_data.value_counts().head(5).to_dict()

    return stats