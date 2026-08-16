"""
Model training: Logistic Regression baseline, XGBoost, LightGBM,
soft voting ensemble. MLflow tracks params/metrics/model artifacts for every run.

- Training data: train_balanced.parquet (SMOTE-balanced).
- Evaluation data: test_selected.parquet (untouched, real-world ~8.88% churn
  rate) — evaluating on balanced data would be misleading; only the real
  distribution tells us how the model performs in practice.
- Scaling: only inside the Logistic Regression pipeline (StandardScaler,
  fit on train only). Tree-based models (XGBoost, LightGBM) skip scaling,
  per original project spec.
- Threshold: fixed at 0.5 here.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_xgboost() -> XGBClassifier:
    """Tuned via RandomizedSearchCV, optimized for average_precision
    (PR-AUC) with SMOTE applied inside CV folds. See tuning_results.json."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.7,
        min_child_weight=1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_lightgbm() -> LGBMClassifier:
    """Tuned via RandomizedSearchCV, optimized for average_precision
    (PR-AUC) with SMOTE applied inside CV folds. See tuning_results.json."""
    return LGBMClassifier(
        n_estimators=700,
        max_depth=-1,
        num_leaves=127,
        learning_rate=0.01,
        subsample=1.0,
        colsample_bytree=0.7,
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

TRUSTED_ENSEMBLE_TYPES = [
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier",
    "collections.OrderedDict", "sklearn.utils._bunch.Bunch",
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
            mlflow.sklearn.log_model(model, "model")

        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return model, metrics