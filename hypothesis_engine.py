import pandas as pd
import numpy as np
from scipy import stats

def calculate_p_values(df: pd.DataFrame, target_col: str, feature_col: str) -> dict:
    if target_col not in df.columns or feature_col not in df.columns:
        return {"error": "Columns not found"}

    clean_df = df[[target_col, feature_col]].dropna()
    
    if pd.api.types.is_numeric_dtype(clean_df[target_col]) and not pd.api.types.is_numeric_dtype(clean_df[feature_col]):
        groups = [group[target_col].values for name, group in clean_df.groupby(feature_col)]
        
        if len(groups) == 2:
            stat, p_value = stats.ttest_ind(groups[0], groups[1])
            test_name = "T-Test"
        elif len(groups) > 2:
            stat, p_value = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            return {"error": "Not enough groups"}
            
        return {
            "test": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < 0.05)
        }
        
    elif pd.api.types.is_numeric_dtype(clean_df[target_col]) and pd.api.types.is_numeric_dtype(clean_df[feature_col]):
        stat, p_value = stats.pearsonr(clean_df[target_col], clean_df[feature_col])
        return {
            "test": "Pearson Correlation",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < 0.05)
        }
        
    return {"error": "Unsupported data types for automated testing"}