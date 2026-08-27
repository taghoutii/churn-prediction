"""
Section 1 — Data overview: dimensions, variable types, missingness,
key-variable distributions, and a duplicate-check re-confirmation on the
merged interim tables.
"""
import numpy as np
import pandas as pd

KEY_NUMERIC_VARS = {
    "monthly_usage": ["recharge_amount", "mou_onnet", "data_trafic_volume"],
    "subscriber_static": ["age"],
}


def summarize_shape_and_types(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name}: shape={df.shape} ===")
    print(df.dtypes)


def summarize_missingness(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Returns a tidy missingness table (count + %), sorted worst-first."""
    n = len(df)
    missing = df.isna().sum()
    pct = (missing / n * 100).round(2)
    out = pd.DataFrame({"n_missing": missing, "pct_missing": pct})
    out = out[out["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    print(f"\n--- {name}: missingness ---")
    print(out if not out.empty else "  no missing values")
    return out


def summarize_key_distributions(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """Describe() on the key numeric variables — surfaces skew/outliers that
    matter later for scaling (step 8) and SMOTE behavior (step 7)."""
    present = [c for c in cols if c in df.columns]
    print(f"\n--- {name}: key numeric variable distributions ---")
    desc = df[present].describe(percentiles=[.01, .25, .5, .75, .99]).T
    print(desc)
    return desc


def compare_log1p_skew(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Skewness before/after a log1p transform, illustrative only -- the
    real transform is applied later at the aggregated snapshot-feature level
    (see churn.features.snapshot), not here. log1p is fixed/parameter-free,
    so it's safe to compute directly on the full raw table without any
    train-only fitting concern."""
    rows = [{
        "feature": col,
        "skew_before": df[col].skew(),
        "skew_after": np.log1p(df[col]).skew(),
    } for col in cols]
    out = pd.DataFrame(rows).set_index("feature")
    print("\n--- Skewness before/after log1p (illustrative) ---")
    print(out.round(2))
    return out


def confirm_no_duplicate_keys(df: pd.DataFrame, key_cols: list[str], name: str) -> int:
    """Quick re-confirmation only — full duplicate audit already done in step 3
    on raw sources. This just checks the merge didn't introduce new dupes."""
    n_dup = df.duplicated(subset=key_cols).sum()
    print(f"\n--- {name}: duplicate check on key={key_cols} ---")
    print(f"  duplicate rows: {n_dup} (expected 0)")
    return n_dup


def run_overview(subscriber_static: pd.DataFrame, monthly_usage: pd.DataFrame) -> None:
    summarize_shape_and_types(subscriber_static, "subscriber_static")
    summarize_shape_and_types(monthly_usage, "monthly_usage")

    summarize_missingness(subscriber_static, "subscriber_static")
    summarize_missingness(monthly_usage, "monthly_usage")

    summarize_key_distributions(subscriber_static, KEY_NUMERIC_VARS["subscriber_static"], "subscriber_static")
    summarize_key_distributions(monthly_usage, KEY_NUMERIC_VARS["monthly_usage"], "monthly_usage")

    confirm_no_duplicate_keys(subscriber_static, ["subscriber_id"], "subscriber_static")
    confirm_no_duplicate_keys(monthly_usage, ["subscriber_id", "period_date"], "monthly_usage")
    