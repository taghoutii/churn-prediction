"""
loads balanced train + untouched test, trains all four
models, logs each to MLflow, and prints a comparison summary.
"""
import pandas as pd
import mlflow
from churn.config import PROCESSED_DIR
from churn.modeling.train import (
    get_xy, train_and_log,
    build_logistic_regression, build_xgboost, build_lightgbm, build_voting_ensemble,
)

mlflow.set_experiment("churn_prediction")


def run() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train_balanced.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_selected.parquet")

    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)

    assert list(X_train.columns) == list(X_test.columns), \
        "Train/test feature columns must match exactly before modeling"

    results = {}

    _, results["logistic_regression"] = train_and_log(
        "logistic_regression", build_logistic_regression, X_train, y_train, X_test, y_test,
        params={"max_iter": 2000, "scaled": True},
        flavor="sklearn",
    )
    _, results["xgboost"] = train_and_log(
        "xgboost", build_xgboost, X_train, y_train, X_test, y_test,
        params={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
        flavor="xgboost",
    )
    _, results["lightgbm"] = train_and_log(
        "lightgbm", build_lightgbm, X_train, y_train, X_test, y_test,
        params={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
        flavor="lightgbm",
    )
    _, results["voting_ensemble"] = train_and_log(
        "voting_ensemble", build_voting_ensemble, X_train, y_train, X_test, y_test,
        params={"voting": "soft", "base_models": "lr+xgb+lgbm"},
        flavor="sklearn_ensemble",
    )

    summary = pd.DataFrame(results).T
    print("\n--- Model comparison (evaluated on untouched test set) ---")
    print(summary)
    summary.to_csv(PROCESSED_DIR / "model_comparison.csv")


if __name__ == "__main__":
    run()