"""
Correlation pruning + RF/permutation importance.
Both fit on TRAIN only. Test is never used to decide which features to keep.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from churn.selection.prepare import ID_AND_META_COLS

# Lowered from 0.85: total_data_revenu_amount_mean and
# revenu_furfait_data_dinar_mean sat at 0.823 -- just under the old cutoff,
# both effectively saying "how much they spend on data" and splitting
# explanatory credit between two near-duplicate features. Chose a lower
# flat threshold over an importance-based tie-break (keep whichever of a
# pair has higher importance) because that would need importance computed
# on the FULL feature set before pruning, then again after -- more
# restructuring for a small number of borderline pairs; simple threshold
# change was the lower-risk option here.
CORRELATION_THRESHOLD = 0.80


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_AND_META_COLS]


def drop_correlated_features(
    train: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[str]:
    """
    Drops one feature from each pair whose absolute correlation exceeds
    the specified threshold.

    Categorical (dtype 'category') columns are intentionally SKIPPED here,
    not pruned: Pearson correlation isn't a meaningful similarity measure
    for nominal categories, and with only 3 low-cardinality categoricals in
    this feature set (gender, marital_status, customer_language --
    conceptually distinct dimensions), a categorical-association measure
    (e.g. Cramer's V) isn't worth the added complexity here. They pass
    through unpruned.
    """

    feature_cols = get_feature_cols(train)

    # Only numeric features participate in correlation pruning
    numeric_cols = list(
        train[feature_cols]
        .select_dtypes(include="number")
        .columns
    )
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    if categorical_cols:
        print(f"  skipping correlation pruning for categorical columns: {categorical_cols}")

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
    can be).

    RandomForestClassifier has no native categorical support (unlike the
    XGBoost/LightGBM models this feature set is actually trained for), so
    categorical columns are one-hot-encoded LOCALLY here, purely to make
    this importance ranking possible -- feature_cols / the returned kept
    columns / the exported dataset are untouched and keep native category
    dtype."""
    X = train[feature_cols]
    y = train["label"]

    categorical_cols = list(X.select_dtypes(include="category").columns)
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)

    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=random_state, n_jobs=-1)

    importance_df = pd.DataFrame({
        "feature": list(X.columns),
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
