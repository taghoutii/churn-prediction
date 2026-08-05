"""
Class balancing: compare SMOTE vs ADASYN, then apply the chosen
method to the full training set. Test is never touched.

IMPORTANT: To fairly compare SMOTE vs ADASYN, resampling must happen INSIDE
each cross-validation fold, not once before CV — otherwise synthetic samples
derived from what should be held-out validation data could influence training
folds, which is its own form of leakage. imblearn's Pipeline handles this
correctly (resampling is refit per-fold, only on that fold's training split).

"""
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

ID_META_COLS = ["subscriber_id", "snapshot_date"] #synthetic data won't have real subscriber_id or snapshot_date
RANDOM_STATE = 42


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=ID_META_COLS + ["label"])
    y = df["label"]
    return X, y


def compare_balancers(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    X, y = get_xy(train)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["recall", "precision", "f1", "roc_auc"] #accuracy is misleading on an ~8.9% imbalanced target

    results = {}
    samplers = {
        "none": None,
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
    }

    for name, sampler in samplers.items():
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        if sampler is None:
            pipeline = ImbPipeline([("clf", clf)])
        else:
            pipeline = ImbPipeline([("sampler", sampler), ("clf", clf)])

        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[name].items()))

    return pd.DataFrame(results).T


def apply_balancing(train: pd.DataFrame, method: str = "smote") -> pd.DataFrame:
    """Applies the chosen sampler ONCE to the full training set (fit on
    train only) — this is the actual resampled data step 8 will train on."""
    X, y = get_xy(train)

    sampler = SMOTE(random_state=RANDOM_STATE) if method == "smote" else ADASYN(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X, y) # type: ignore

    balanced = X_resampled.copy()
    balanced["label"] = y_resampled
    print(f"Balanced train ({method}): {len(balanced)} rows, "
          f"churn rate = {balanced['label'].mean():.2%} "
          f"(original: {len(train)} rows, {train['label'].mean():.2%})")
    return balanced # type: ignore