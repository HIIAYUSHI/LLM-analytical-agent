"""
hypothesis_engine.py
====================
Statistical hypothesis testing with explicit H0 / H1 framing.

Supported tests (auto-selected or caller-requested):
  T-Test, ANOVA, Pearson, Spearman, Mann-Whitney U, Chi-Square

Every result dict includes h0, h1, interpretation.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency


def _hypotheses(test_name, target_col, feature_col):
    if test_name == "T-Test":
        return (
            f"The mean of '{target_col}' is equal across both groups of '{feature_col}' (mu1 = mu2)",
            f"The mean of '{target_col}' differs significantly between the two groups of '{feature_col}' (mu1 != mu2)",
        )
    if test_name == "ANOVA":
        return (
            f"The mean of '{target_col}' is equal across all groups of '{feature_col}' (mu1 = mu2 = ... = muk)",
            f"At least one group mean of '{target_col}' differs significantly from the others",
        )
    if test_name == "Pearson Correlation":
        return (
            f"There is no linear correlation between '{target_col}' and '{feature_col}' (rho = 0)",
            f"There is a statistically significant linear correlation between '{target_col}' and '{feature_col}' (rho != 0)",
        )
    if test_name == "Spearman Correlation":
        return (
            f"There is no monotonic relationship between '{target_col}' and '{feature_col}' (rho_s = 0)",
            f"There is a statistically significant monotonic relationship between '{target_col}' and '{feature_col}' (rho_s != 0)",
        )
    if test_name == "Mann-Whitney U":
        return (
            f"The distribution of '{target_col}' is the same across both groups of '{feature_col}'",
            f"The distribution of '{target_col}' differs significantly between the two groups of '{feature_col}'",
        )
    if test_name == "Chi-Square":
        return (
            f"'{target_col}' and '{feature_col}' are independent (no association)",
            f"'{target_col}' and '{feature_col}' are NOT independent — there is a significant association",
        )
    return ("No specific null hypothesis", "No specific alternative hypothesis")


def _interpretation(test_name, stat, p_value, significant, target_col, feature_col, alpha):
    verdict = "IS" if significant else "is NOT"
    if test_name in ("Pearson Correlation", "Spearman Correlation"):
        direction = "positive" if stat >= 0 else "negative"
        strength  = "strong" if abs(stat) >= 0.7 else "moderate" if abs(stat) >= 0.4 else "weak"
        kind      = "linear" if test_name == "Pearson Correlation" else "monotonic"
        return (
            f"There {verdict} a statistically significant {strength} {direction} {kind} "
            f"relationship between '{target_col}' and '{feature_col}' "
            f"(r = {stat:.4f}, p = {p_value:.4f}, alpha = {alpha})."
        )
    if test_name in ("T-Test", "Mann-Whitney U"):
        return (
            f"The difference in '{target_col}' between the two groups of '{feature_col}' "
            f"{verdict} statistically significant "
            f"(stat = {stat:.4f}, p = {p_value:.4f}, alpha = {alpha})."
        )
    if test_name == "ANOVA":
        return (
            f"The difference in '{target_col}' across the groups of '{feature_col}' "
            f"{verdict} statistically significant "
            f"(F = {stat:.4f}, p = {p_value:.4f}, alpha = {alpha})."
        )
    if test_name == "Chi-Square":
        return (
            f"The association between '{target_col}' and '{feature_col}' "
            f"{verdict} statistically significant "
            f"(chi2 = {stat:.4f}, p = {p_value:.4f}, alpha = {alpha})."
        )
    return f"p = {p_value:.4f} (alpha = {alpha}) — result {verdict} significant."


def _build(test_name, h0, h1, stat, p, alpha, n, groups, target_col, feature_col):
    sig = bool(p < alpha)
    return {
        "test":           test_name,
        "h0":             h0,
        "h1":             h1,
        "statistic":      round(float(stat), 4),
        "p_value":        round(float(p),    4),
        "significant":    sig,
        "alpha":          alpha,
        "n":              n,
        "groups":         groups,
        "interpretation": _interpretation(test_name, float(stat), float(p),
                                          sig, target_col, feature_col, alpha),
    }


def calculate_p_values(
    df,
    target_col,
    feature_col,
    force_test="auto",
    alpha=0.05,
):
    """
    Run a statistical test and return a full result dict with H0, H1,
    test statistic, p-value, and plain-English interpretation.

    force_test options: "auto" | "ttest" | "anova" | "pearson" |
                        "spearman" | "mannwhitney" | "chi2"
    """
    if target_col not in df.columns or feature_col not in df.columns:
        missing = [c for c in [target_col, feature_col] if c not in df.columns]
        return {"error": f"Column(s) not found in DataFrame: {missing}"}

    clean = df[[target_col, feature_col]].dropna()
    n     = len(clean)
    if n < 3:
        return {"error": f"Only {n} complete rows after dropping NaNs — need at least 3."}

    target_numeric  = pd.api.types.is_numeric_dtype(clean[target_col])
    feature_numeric = pd.api.types.is_numeric_dtype(clean[feature_col])

    ft = force_test.lower().replace("-", "").replace("_", "").replace(" ", "")

    try:
        # Chi-Square: both categorical
        if ft == "chi2" or (ft == "auto" and not target_numeric and not feature_numeric):
            ct       = pd.crosstab(clean[target_col], clean[feature_col])
            stat, p, dof, _ = chi2_contingency(ct)
            test_name = "Chi-Square"
            h0, h1   = _hypotheses(test_name, target_col, feature_col)
            return _build(test_name, h0, h1, stat, p, alpha, n, None, target_col, feature_col)

        # Both numeric
        if target_numeric and feature_numeric:
            if ft == "spearman":
                stat, p   = sp_stats.spearmanr(clean[target_col], clean[feature_col])
                test_name = "Spearman Correlation"
            else:
                stat, p   = sp_stats.pearsonr(clean[target_col], clean[feature_col])
                test_name = "Pearson Correlation"
            h0, h1 = _hypotheses(test_name, target_col, feature_col)
            return _build(test_name, h0, h1, stat, p, alpha, n, None, target_col, feature_col)

        # Numeric target vs categorical feature
        if target_numeric and not feature_numeric:
            groups   = [g[target_col].values for _, g in clean.groupby(feature_col)]
            n_groups = len(groups)
            if n_groups < 2:
                return {"error": "Need at least 2 groups in the feature column."}

            if ft == "mannwhitney":
                if n_groups != 2:
                    return {"error": "Mann-Whitney U requires exactly 2 groups."}
                stat, p   = sp_stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                test_name = "Mann-Whitney U"
            elif ft == "anova":
                stat, p   = sp_stats.f_oneway(*groups)
                test_name = "ANOVA"
            elif ft == "ttest":
                if n_groups != 2:
                    return {"error": f"T-Test requires exactly 2 groups — found {n_groups}."}
                stat, p   = sp_stats.ttest_ind(groups[0], groups[1])
                test_name = "T-Test"
            else:  # auto
                if n_groups == 2:
                    stat, p   = sp_stats.ttest_ind(groups[0], groups[1])
                    test_name = "T-Test"
                else:
                    stat, p   = sp_stats.f_oneway(*groups)
                    test_name = "ANOVA"

            h0, h1 = _hypotheses(test_name, target_col, feature_col)
            return _build(test_name, h0, h1, stat, p, alpha, n, n_groups, target_col, feature_col)

        # Categorical target vs numeric feature — swap and rerun
        if not target_numeric and feature_numeric:
            return calculate_p_values(
                df, feature_col, target_col, force_test=force_test, alpha=alpha
            )

    except Exception as e:
        return {"error": f"Test execution failed: {e}"}

    return {"error": "Unsupported column type combination for automated testing."}
