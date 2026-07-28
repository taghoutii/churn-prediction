"""
Section 2 — Target analysis: overall churn rate, churn by demographics,
churn by tenure. Pure summary tables here — visual styling stays in the
notebook, this module just computes what needs plotting/reporting.
"""
import pandas as pd

DEMOGRAPHIC_COLS = ["gender", "marital_status", "customer_language", "contact_city"]


def overall_churn_rate(df: pd.DataFrame) -> pd.Series:
    counts = df["is_churn"].value_counts()
    rate = df["is_churn"].mean()
    print(f"\n--- Overall churn rate ---")
    print(counts)
    print(f"churn rate: {rate:.2%}")
    return pd.Series({"n_churn": counts.get(True, 0), "n_no_churn": counts.get(False, 0), "churn_rate": rate})


def churn_rate_by_demographic(df: pd.DataFrame, col: str, min_group_size: int = 100) -> pd.DataFrame:
    """
    Churn rate per category of a demographic column. Groups smaller than
    min_group_size are flagged rather than silently trusted — a churn rate
    from 8 subscribers is noise, not signal, and city fields in particular
    can have a long tail of tiny groups.
    """
    grouped = df.groupby(col, dropna=False)["is_churn"].agg(["mean", "count"])
    grouped.columns = ["churn_rate", "n_subscribers"]
    grouped = grouped.sort_values("churn_rate", ascending=False)

    small_groups = grouped[grouped["n_subscribers"] < min_group_size]
    print(f"\n--- Churn rate by {col} ---")
    print(grouped)
    if not small_groups.empty:
        print(f"  (Note: {len(small_groups)} group(s) below {min_group_size} subscribers — "
              f"treat their churn rates as unreliable, not a real pattern.)")
    return grouped


def churn_rate_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """classe_anciennete has a natural order; enforce it so the notebook plot
    reads left-to-right as increasing tenure, not alphabetical."""
    tenure_order = ["0 to 3 Months", "3 to 6 Months", "6 to 12 Months", "12 to 36 Months", "More than 36 Months"]
    grouped = df.groupby("classe_anciennete", dropna=False)["is_churn"].agg(["mean", "count"])
    grouped.columns = ["churn_rate", "n_subscribers"]
    grouped = grouped.reindex(tenure_order)
    print("\n--- Churn rate by tenure (classe_anciennete) ---")
    print(grouped)
    return grouped


def churn_rate_by_age_bucket(df: pd.DataFrame, bin_width: int = 10) -> pd.DataFrame:
    """Age is continuous — bucket it for a readable churn-rate-by-age view.
    Rows with null age (post-cleaning) are excluded and reported separately."""
    valid = df.dropna(subset=["age"]).copy()
    n_excluded = len(df) - len(valid)

    max_age = int(valid["age"].max())
    bins = list(range(0, max_age + bin_width, bin_width))
    valid["age_bucket"] = pd.cut(valid["age"], bins=bins, right=False)

    grouped = valid.groupby("age_bucket", observed=True)["is_churn"].agg(["mean", "count"])
    grouped.columns = ["churn_rate", "n_subscribers"]
    print(f"\n--- Churn rate by age bucket (excluding {n_excluded} null-age rows) ---")
    print(grouped)
    return grouped


def run_target_analysis(subscriber_static: pd.DataFrame) -> None:
    overall_churn_rate(subscriber_static)

    for col in DEMOGRAPHIC_COLS:
        churn_rate_by_demographic(subscriber_static, col)

    churn_rate_by_age_bucket(subscriber_static)
    churn_rate_by_tenure(subscriber_static)