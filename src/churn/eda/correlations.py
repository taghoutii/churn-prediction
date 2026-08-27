"""
Section 5 — Feature relationships: correlation heatmap for intuition only.
Formal multicollinearity/redundancy handling (VIF, RF/permutation importance-
based pruning) is explicitly DEFERRED to step 6, fit on training data only.
Nothing here should be used to drop features — it's for building intuition
about which variables move together before that formal step.
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def compute_pca_projection(
    monthly_usage: pd.DataFrame, features: list[str] = NUMERIC_FEATURES
) -> tuple[pd.DataFrame, list[float]]:
    """2D PCA projection of the same numeric usage features used in the
    correlation heatmap above, for visual intuition only. Aggregated to ONE
    ROW PER SUBSCRIBER first (mean of each feature across their observed
    months) -- fitting PCA directly on the subscriber-month panel would let
    long-tenured subscribers' repeated months dominate/distort the
    projection. Fit on the FULL dataset (not train-only): this is a
    descriptive EDA visualization, not a modeling feature, consistent with
    how the rest of this module already treats the full pre-split dataset."""
    present = [c for c in features if c in monthly_usage.columns]
    per_subscriber = monthly_usage.groupby("subscriber_id")[present].mean()
    valid = per_subscriber.dropna(subset=present)
    n_excluded = len(per_subscriber) - len(valid)

    X_scaled = StandardScaler().fit_transform(valid[present])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    explained = (pca.explained_variance_ratio_ * 100).tolist()

    out = pd.DataFrame({"subscriber_id": valid.index})
    out["PC1"] = components[:, 0]
    out["PC2"] = components[:, 1]

    print(f"\n--- PCA on {len(present)} numeric usage features, aggregated to "
          f"{len(valid)} subscribers (excluding {n_excluded} with missing values) ---")
    print(f"Variance explained: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%, "
          f"total={sum(explained):.1f}%")
    return out, explained


def run_correlation_analysis(monthly_usage: pd.DataFrame) -> dict:
    corr = compute_correlation_matrix(monthly_usage)
    pairs = top_correlated_pairs(corr)
    return {"correlation_matrix": corr, "high_corr_pairs": pairs}