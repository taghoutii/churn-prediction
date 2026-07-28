"""
Section 3 — Temporal exploratory analysis: timeline coverage, tenure
distribution, days since last activity, churn rate over time.

Note: "days since last activity" here mirrors the EXPLORATORY inactivity
analysis from inspect_raw.py — a single global reference date, for trend-
spotting only. The real per-snapshot feature is built in step 4.
"""
import pandas as pd


def summarize_timeline_coverage(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> None:
    print("\n--- Timeline coverage ---")
    print(f"monthly_usage period_date range: {monthly_usage['period_date'].min()} -> "
          f"{monthly_usage['period_date'].max()}")
    print(f"unique periods: {monthly_usage['period_date'].nunique()}")
    print(f"date_activation range: {subscriber_static['date_activation'].min()} -> "
          f"{subscriber_static['date_activation'].max()}")
    print(f"churn_date range (churners only): "
          f"{subscriber_static.loc[subscriber_static['is_churn'], 'churn_date'].min()} -> "
          f"{subscriber_static.loc[subscriber_static['is_churn'], 'churn_date'].max()}")


def summarize_tenure_distribution(subscriber_static: pd.DataFrame) -> pd.Series:
    """classe_anciennete is already bucketed; this just gives clean counts/%
    in natural order for a bar plot in the notebook."""
    tenure_order = ["0 to 3 Months", "3 to 6 Months", "6 to 12 Months", "12 to 36 Months", "More than 36 Months"]
    counts = subscriber_static["classe_anciennete"].value_counts().reindex(tenure_order)
    pct = (counts / counts.sum() * 100).round(2)
    print("\n--- Tenure distribution ---")
    print(pd.DataFrame({"n_subscribers": counts, "pct": pct}))
    return counts


def explore_inactivity_vs_reference(subscriber_static: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    EXPLORATORY ONLY — same caveat as inspect_raw.explore_inactivity_periods.
    Uses one global reference date to compare churners vs non-churners on
    days-since-last-call. This is NOT the days_since_last_call feature; the
    real per-snapshot version is built in step 4.
    """
    df = subscriber_static.copy()
    df["inactivity_days"] = (reference_date - df["last_call_date"]).dt.days

    print(f"\n--- Exploratory inactivity vs reference_date={reference_date} ---")
    print("churn=True:")
    print(df.loc[df["is_churn"], "inactivity_days"].describe(percentiles=[.1, .25, .5, .75, .9]))
    print("\nchurn=False:")
    print(df.loc[~df["is_churn"], "inactivity_days"].describe(percentiles=[.1, .25, .5, .75, .9]))
    return df[["subscriber_id", "is_churn", "inactivity_days"]]


def churn_count_over_time(subscriber_static: pd.DataFrame) -> pd.Series:
    """
    Monthly churn rate using churn_date's month (churners only contribute a
    month; this shows WHEN churns happened, not a survival curve — that's
    step 3 survival analysis's job).
    """
    churners = subscriber_static[subscriber_static["is_churn"]].copy()
    churners["churn_month"] = churners["churn_date"].dt.to_period("M")
    monthly_counts = churners.groupby("churn_month").size()

    print("\n--- Churn events by month (count of churn_date falling in that month) ---")
    print(monthly_counts)
    return monthly_counts

def check_churn_date_concentration(subscriber_static: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Checks whether churn_date clusters on a small number of exact dates
    (pointing to a batch administrative process) vs. being spread smoothly
    across many dates (pointing to continuous individual churn behavior).
    """
    churners = subscriber_static[subscriber_static["is_churn"]]
    n_total = len(churners)
    n_distinct = churners["churn_date"].nunique()

    counts = churners["churn_date"].value_counts().sort_values(ascending=False)
    top = counts.head(top_n)
    top_share = top.sum() / n_total

    print(f"\n--- churn_date concentration ---")
    print(f"total churners: {n_total}")
    print(f"distinct churn_date values: {n_distinct}")
    print(f"\ntop {top_n} exact dates by churner count:")
    print(top)
    print(f"\ntop {top_n} dates account for {top_share:.1%} of all churners")

    return top.to_frame("n_churners")


def run_temporal_analysis(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> None:
    summarize_timeline_coverage(monthly_usage, subscriber_static)
    summarize_tenure_distribution(subscriber_static)

    reference_date = monthly_usage["period_date"].max()
    explore_inactivity_vs_reference(subscriber_static, reference_date)

    churn_count_over_time(subscriber_static)
    check_churn_date_concentration(subscriber_static)