"""
Section 5 — Feature relationships: correlation heatmap for intuition only.
Formal multicollinearity/redundancy handling (VIF, RF/permutation importance-
based pruning) is explicitly DEFERRED to step 6, fit on training data only.
Nothing here should be used to drop features — it's for building intuition
about which variables move together before that formal step.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "recharge_amount", "number_of_recharges",
    "number_of_sms_onnet", "number_of_sms_offnet", "number_of_sms_international",
    "mou_onnet", "mou_offnet_mobile", "mou_offnet_fix", "mou_international",
    "consumption_revenue", "arpu_out_without_bonus", "incoming_revenue",
    "data_trafic_volume", "total_voice_revenu_amount", "total_data_revenu_amount",
    "revenu_furfait_data_dinar",
]


def compute_correlation_matrix(monthly_usage: pd.DataFrame, features: list[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    present = [c for c in features if c in monthly_usage.columns]
    corr = monthly_usage[present].corr(method="pearson")
    print(f"\n--- Correlation matrix ({len(present)} numeric usage features) ---")
    print(corr.round(2))
    return corr


def top_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Flags pairs above |threshold| for the notebook's attention — informational
    only. Actual redundancy decisions (what to drop) happen in step 6 with
    proper train-only fitting, not here.
    """
    pairs = (
        corr.where(~pd.DataFrame(
            [[i == j for j in corr.columns] for i in corr.index],
            index=corr.index, columns=corr.columns
        ))
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs = pairs[pairs["feature_1"] < pairs["feature_2"]]  # dedupe symmetric pairs
    pairs = pairs[pairs["correlation"].abs() >= threshold].sort_values(
        "correlation", key=lambda s: s.abs(), ascending=False
    )
    print(f"\n--- Feature pairs with |correlation| >= {threshold} (informational only) ---")
    print(pairs.to_string(index=False))
    return pairs


def run_correlation_analysis(monthly_usage: pd.DataFrame) -> dict:
    corr = compute_correlation_matrix(monthly_usage)
    pairs = top_correlated_pairs(corr)
    return {"correlation_matrix": corr, "high_corr_pairs": pairs}