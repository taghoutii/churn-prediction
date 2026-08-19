"""
Class balancing: compare no-balancing vs SMOTENC, then apply the chosen
method to the full training set. Test is never touched.

SMOTENC (not plain SMOTE) is used because the feature set now includes
native pandas 'category' dtype columns (gender, marital_status,
customer_language) for XGBoost/LightGBM's native
categorical handling -- plain SMOTE requires all-numeric input and has no
way to interpolate a categorical value. SMOTENC handles mixed
numeric+categorical data: numeric features are still interpolated the
usual way, and each synthetic sample's categorical values are taken from
its nearest real neighbor instead.

ADASYN has no categorical-aware variant in imbalanced-learn, so it's
dropped from this comparison. That isn't a meaningful loss: the original
SMOTE-vs-ADASYN comparison (balancing_comparison.csv, one-hot feature set)
already found SMOTE slightly ahead on every metric, and SMOTE/SMOTENC
remains the chosen method either way.

IMPORTANT: To fairly evaluate balancing, resampling must happen INSIDE
each cross-validation fold, not once before CV -- otherwise synthetic
samples derived from what should be held-out validation data could
influence training folds, which is its own form of leakage. imblearn's
Pipeline handles this correctly (resampling is refit per-fold, only on
that fold's training split).
"""
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_META_COLS = ["subscriber_id", "snapshot_date"] #synthetic data won't have real subscriber_id or snapshot_date
RANDOM_STATE = 42


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=ID_META_COLS + ["label"])
    y = df["label"]
    return X, y


def get_categorical_indices(X: pd.DataFrame) -> list[int]:
    """Positional indices of the category-dtype columns, as SMOTENC's
    categorical_features param requires (this imblearn version has no
    'auto' dtype-detection option)."""
    return [i for i, col in enumerate(X.columns) if str(X[col].dtype) == "category"]


def _build_probe_pipeline(sampler) -> ImbPipeline:
    """Logistic-Regression probe used only to COMPARE balancing methods --
    needs its own one-hot+scale preprocessing since LR has no native
    categorical support, same as churn.modeling.train.build_logistic_regression."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_exclude="category")),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), make_column_selector(dtype_include="category")),
    ])
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    steps = ([("sampler", sampler)] if sampler is not None else []) + [
        ("preprocess", preprocessor), ("clf", clf),
    ]
    return ImbPipeline(steps)


def compare_balancers(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    X, y = get_xy(train)
    cat_idx = get_categorical_indices(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["recall", "precision", "f1", "roc_auc"] #accuracy is misleading on an ~8.9% imbalanced target

    results = {}
    samplers = {
        "none": None,
        "smotenc": SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE),
    }

    for name, sampler in samplers.items():
        pipeline = _build_probe_pipeline(sampler)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[name].items()))

    return pd.DataFrame(results).T


def apply_balancing(train: pd.DataFrame, method: str = "smotenc") -> pd.DataFrame:
    """Applies SMOTENC ONCE to the full training set (fit on train only) --
    this is the actual resampled data step 8 will train on."""
    X, y = get_xy(train)
    cat_idx = get_categorical_indices(X)

    sampler = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X, y) # type: ignore

    balanced = X_resampled.copy()
    balanced["label"] = y_resampled
    print(f"Balanced train ({method}): {len(balanced)} rows, "
          f"churn rate = {balanced['label'].mean():.2%} "
          f"(original: {len(train)} rows, {train['label'].mean():.2%})")
    return balanced # type: ignore
