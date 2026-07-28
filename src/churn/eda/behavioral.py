"""
Section 4 — Behavioral analysis before churn: usage/recharge trends and
activity decline as a function of days-before-churn, for churners, plus a
non-churner baseline for comparison. This is the analysis that empirically
justifies the prediction window used in step 4.

IMPORTANT: 'days_before_churn' here is EXPLORATORY, computed against each
subscriber's own churn_date for trend-spotting only. The real snapshot-based
feature engineering (per-snapshot-date, no leakage) happens in step 4 — this
module must not be imported by anything outside EDA.
"""
import pandas as pd

TREND_METRICS = ["recharge_amount", "number_of_recharges", "mou_onnet", "data_trafic_volume"]


def compute_days_before_churn(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> pd.DataFrame:
    """
    Joins monthly_usage to each churner's churn_date and computes how many
    days before churn each usage period occurred. Non-churners are excluded
    here — they have no churn_date to anchor against, handled separately in
    compare_churner_vs_nonchurner_baseline via calendar time instead.
    """
    churners = subscriber_static.loc[subscriber_static["is_churn"], ["subscriber_id", "churn_date"]]
    merged = monthly_usage.merge(churners, on="subscriber_id", how="inner")
    merged["days_before_churn"] = (merged["churn_date"] - merged["period_date"]).dt.days
    return merged


def summarize_usage_trend_before_churn(
    churner_usage: pd.DataFrame, metrics: list[str] = TREND_METRICS, bin_width: int = 15
) -> pd.DataFrame:
    """
    Bins days_before_churn into bin_width-day buckets and averages each
    metric per bucket. Only keeps buckets in [0, ~210] days — beyond that,
    sample size thins out fast (usage data only spans 7 months) and isn't
    reliable for the window decision.
    """
    df = churner_usage[(churner_usage["days_before_churn"] >= 0) & (churner_usage["days_before_churn"] <= 210)].copy()
    df["bucket"] = (df["days_before_churn"] // bin_width) * bin_width

    agg = df.groupby("bucket")[metrics].mean()
    agg["n_observations"] = df.groupby("bucket").size()
    agg = agg.sort_index()

    print(f"\n--- Usage trend vs. days-before-churn (churners only, {bin_width}-day buckets) ---")
    print(agg)
    return agg


def compare_churner_vs_nonchurner_baseline(
    monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame, metrics: list[str] = TREND_METRICS
) -> pd.DataFrame | pd.Series:
    """
    Average usage per calendar month, split by eventual churn status. This is
    the control comparison: if churners' usage is already lower/declining
    relative to non-churners well before their churn window, that's a sign
    the behavioral signal starts earlier than a naive reading of the churn-
    only trend would suggest.
    """
    merged = monthly_usage.merge(subscriber_static[["subscriber_id", "is_churn"]], on="subscriber_id", how="left")
    agg = merged.groupby(["period_date", "is_churn"])[metrics].mean().unstack("is_churn")
    print("\n--- Usage by calendar month: churners vs non-churners ---")
    print(agg)
    return agg


def summarize_activity_decline(churner_usage: pd.DataFrame, bin_width: int = 15) -> pd.DataFrame:
    """
    Same idea as the usage trend, but for a binary 'any activity this period'
    signal (recharge or call), to see when subscribers go fully dormant vs.
    just spend/talk less.
    """
    df = churner_usage[(churner_usage["days_before_churn"] >= 0) & (churner_usage["days_before_churn"] <= 210)].copy()
    df["bucket"] = (df["days_before_churn"] // bin_width) * bin_width
    df["had_activity"] = (df["number_of_recharges"] > 0) | (df["mou_onnet"] > 0)

    agg = df.groupby("bucket")["had_activity"].mean().to_frame("pct_active")
    print(f"\n--- % of churners with any activity, by days-before-churn bucket ---")
    print(agg)
    return agg


def run_behavioral_analysis(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> dict:
    churner_usage = compute_days_before_churn(monthly_usage, subscriber_static)

    trend = summarize_usage_trend_before_churn(churner_usage)
    baseline = compare_churner_vs_nonchurner_baseline(monthly_usage, subscriber_static)
    activity_decline = summarize_activity_decline(churner_usage)

    return {"trend": trend, "baseline": baseline, "activity_decline": activity_decline}