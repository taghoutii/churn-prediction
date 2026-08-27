"""
Section 6 — Categorical association: Cramér's V between is_churn and each
demographic/tenure categorical variable, as a cross-check against the
churn-rate-by-group tables in Section 2 (target_analysis.py). churn_rate
tables already show the direction/magnitude per category; this adds a
single summary number per variable so they can be ranked against each
other directly.
"""
import pandas as pd
from scipy.stats import chi2_contingency

CATEGORICAL_COLS = ["gender", "marital_status", "customer_language", "classe_anciennete"]


def cramers_v(df: pd.DataFrame, col: str, target_col: str = "is_churn") -> float:
    """Bias-uncorrected Cramér's V from a chi-square test of independence
    between `col` and `target_col`. 0 = no association, 1 = perfect
    association. Missing values in `col` are treated as their own category
    (not dropped), consistent with target_analysis.churn_rate_by_demographic's
    dropna=False tables -- missingness itself has already been shown to
    carry churn signal for some of these columns, so silently dropping it
    here would understate the true association."""
    series = df[col].astype("object").fillna("Missing")
    contingency = pd.crosstab(series, df[target_col])
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return (chi2 / (n * min_dim)) ** 0.5


def compute_cramers_v_table(
    df: pd.DataFrame, cols: list[str] = CATEGORICAL_COLS, target_col: str = "is_churn"
) -> pd.DataFrame:
    rows = [{"feature": col, "cramers_v": cramers_v(df, col, target_col)} for col in cols]
    out = pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)
    print(f"\n--- Cramér's V vs {target_col} ---")
    print(out.round(4))
    return out
