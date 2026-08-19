"""
Model training: Logistic Regression baseline, XGBoost, LightGBM,
soft voting ensemble. MLflow tracks params/metrics/model artifacts for every run.

- Training data: train_balanced.parquet (SMOTENC-balanced).
- Evaluation data: test_selected.parquet (untouched, real-world ~8.88% churn
  rate) — evaluating on balanced data would be misleading; only the real
  distribution tells us how the model performs in practice.
- Categorical features (gender, marital_status, customer_language)
  arrive as native pandas 'category' dtype columns (see
  churn.selection.prepare). XGBoost and LightGBM consume these directly
  (native categorical splits); Logistic Regression has no native
  categorical support, so it one-hot-encodes + scales inside its own
  pipeline below -- no other model needs a one-hot representation.
- Threshold: fixed at 0.5 here.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, average_precision_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ID_META_COLS = ["subscriber_id", "snapshot_date"]
RANDOM_STATE = 42


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols_to_drop = [c for c in ID_META_COLS if c in df.columns] + ["label"]
    X = df.drop(columns=cols_to_drop)
    y = df["label"]
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def build_logistic_regression() -> Pipeline:
    """The one model here with no native categorical support: one-hot
    encodes the category-dtype columns (dropping the first level per
    variable) and scales the rest, fit fresh inside this pipeline -- no
    other model in this module needs a one-hot representation."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_exclude="category")),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), make_column_selector(dtype_include="category")),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_xgboost() -> XGBClassifier:
    """Tuned via RandomizedSearchCV, optimized for average_precision
    (PR-AUC) with SMOTENC applied inside CV folds. See tuning_results.json.
    enable_categorical + tree_method="hist" let XGBoost split natively on
    the category-dtype columns instead of requiring one-hot input."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.7,
        min_child_weight=1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        n_jobs=-1,
    )


def build_lightgbm() -> LGBMClassifier:
    """Tuned via RandomizedSearchCV, optimized for average_precision
    (PR-AUC) with SMOTENC applied inside CV folds. See tuning_results.json.
    LightGBM auto-detects pandas 'category' dtype columns and splits on
    them natively -- no extra constructor param needed."""
    return LGBMClassifier(
        n_estimators=700,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def build_voting_ensemble() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("lr", build_logistic_regression()),
            ("xgb", build_xgboost()),
            ("lgbm", build_lightgbm()),
        ],
        voting="soft",
    )


import mlflow.xgboost
import mlflow.lightgbm

# build_logistic_regression()'s ColumnTransformer uses make_column_selector to
# pick columns by dtype at fit time -- mlflow's default skops serializer
# doesn't trust that type out of the box, so it must be listed explicitly for
# any flavor that (directly or, via the voting ensemble, indirectly) includes
# the Logistic Regression pipeline.
TRUSTED_LR_TYPES = ["sklearn.compose._column_transformer.make_column_selector"]

TRUSTED_ENSEMBLE_TYPES = [
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
    "collections.OrderedDict", "sklearn.utils._bunch.Bunch",
    *TRUSTED_LR_TYPES,
]


def train_and_log(
    name: str, builder, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, params: dict, flavor: str = "sklearn",
):
    with mlflow.start_run(run_name=name):
        model = builder()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, "model")
        elif flavor == "sklearn_ensemble":
            mlflow.sklearn.log_model(model, "model", skops_trusted_types=TRUSTED_ENSEMBLE_TYPES)
        else:
            mlflow.sklearn.log_model(model, "model", skops_trusted_types=TRUSTED_LR_TYPES)

        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return model, metrics