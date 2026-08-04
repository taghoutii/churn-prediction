"""
Correlation pruning + RF/permutation importance.
Both fit on TRAIN only. Test is never used to decide which features to keep.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from churn.selection.prepare import ID_AND_META_COLS

CORRELATION_THRESHOLD = 0.85


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_AND_META_COLS]


def drop_correlated_features(
    train: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[str]:
    """
    Drops one feature from each pair whose absolute correlation exceeds
    the specified threshold.
    """

    feature_cols = get_feature_cols(train)

    # Only numeric features participate in correlation pruning
    numeric_cols = list(
        train[feature_cols]
        .select_dtypes(include="number")
        .columns
    )

    corr = train[numeric_cols].corr().abs()
    corr_values = corr.to_numpy()

    to_drop: set[str] = set()

    for i, col_i in enumerate(numeric_cols):
        for j in range(i + 1, len(numeric_cols)):
            col_j = numeric_cols[j]

            if col_i in to_drop or col_j in to_drop:
                continue

            value = float(corr_values[i, j])

            if value >= threshold:
                print(
                    f"  dropping '{col_j}' "
                    f"(correlated {value:.2f} with '{col_i}')"
                )
                to_drop.add(col_j)

    kept = [c for c in feature_cols if c not in to_drop]

    print(
        f"Correlation pruning: dropped {len(to_drop)}, "
        f"kept {len(kept)} features"
    )

    return kept


def fit_rf_importance(train: pd.DataFrame, feature_cols: list[str], random_state: int = 42) -> pd.DataFrame:
    """Fits a RandomForest on TRAIN only, returns both built-in and
    permutation importance (permutation is more reliable — it isn't biased
    toward high-cardinality/continuous features the way built-in importance
    can be)."""
    X = train[feature_cols]
    y = train["label"]

    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)

    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=random_state, n_jobs=-1)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf.feature_importances_,
        "permutation_importance": perm.importances_mean, # type: ignore
    }).sort_values("permutation_importance", ascending=False)

    return importance_df

DROPPED_LOW_VALUE_FEATURES = ["n_periods_observed"]
def run_feature_selection(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kept_features = drop_correlated_features(train)

    kept_features = [f for f in kept_features if f not in DROPPED_LOW_VALUE_FEATURES]
    print(f"Dropped low-value features (negative permutation importance): {DROPPED_LOW_VALUE_FEATURES}")
    print(f"Final feature count: {len(kept_features)}")

    importance_df = fit_rf_importance(train, kept_features)

    print("\n--- Feature importance ranking ---")
    print(importance_df.to_string(index=False))

    final_cols = kept_features + ["subscriber_id", "snapshot_date", "label"]
    return train[final_cols], test[final_cols], importance_df