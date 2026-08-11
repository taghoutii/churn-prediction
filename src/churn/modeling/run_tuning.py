import pandas as pd
import json
from churn.config import PROCESSED_DIR
from churn.modeling.train import get_xy, evaluate
from churn.modeling.tune import tune_model


def run() -> None:
    # Tuning uses train_selected (real distribution) -- NOT train_balanced.
    train = pd.read_parquet(PROCESSED_DIR / "train_selected.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_selected.parquet")

    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)

    results = {}
    for name in ["xgboost", "lightgbm"]:
        search = tune_model(name, X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        print(f"{name} (tuned) test metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        results[name] = {"best_params": search.best_params_, "test_metrics": metrics}

    with open(PROCESSED_DIR / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote tuning_results.json to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()