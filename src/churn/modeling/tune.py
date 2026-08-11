"""
Hyperparameter tuning via RandomizedSearchCV.

IMPORTANT: tuning uses train_selected.parquet (real ~8.88% distribution),
NOT train_balanced.parquet. SMOTE is applied INSIDE an imblearn Pipeline so
resampling happens fresh per fold during cross-validation -- running CV on
already-resampled data would let synthetic samples leak across folds, the
same risk avoided in step 7.

Scoring uses average_precision (= PR-AUC), a better fit than F1/accuracy for
this imbalanced problem, and consistent with the metric already used to
compare models.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from churn.modeling.train import get_xy

RANDOM_STATE = 42

XGB_PARAM_GRID = {
    "clf__n_estimators": [200, 300, 500, 700],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}

LGBM_PARAM_GRID = {
    "clf__n_estimators": [200, 300, 500, 700],
    "clf__max_depth": [3, 4, 6, 8, -1],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__num_leaves": [15, 31, 63, 127],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}


def build_search(estimator_name: str, n_iter: int = 30, cv_folds: int = 5) -> RandomizedSearchCV:
    if estimator_name == "xgboost":
        clf = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
        param_grid = XGB_PARAM_GRID
    elif estimator_name == "lightgbm":
        clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        param_grid = LGBM_PARAM_GRID
    else:
        raise ValueError(estimator_name)

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def tune_model(estimator_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    search = build_search(estimator_name)
    search.fit(X_train, y_train)

    print(f"\n--- {estimator_name} tuning results ---")
    print(f"Best CV average_precision: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search