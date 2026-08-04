"""
Step 6 — Orchestrates missing-value handling, encoding, correlation pruning,
and importance ranking. Everything fit here uses TRAIN only.
"""
import pandas as pd
from churn.config import PROCESSED_DIR
from churn.selection.prepare import prepare_features
from churn.selection.select_features import run_feature_selection


def run() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    train_prepped, test_prepped = prepare_features(train, test)
    train_final, test_final, importance_df = run_feature_selection(train_prepped, test_prepped)

    train_final.to_parquet(PROCESSED_DIR / "train_selected.parquet", index=False)
    test_final.to_parquet(PROCESSED_DIR / "test_selected.parquet", index=False)
    importance_df.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    print(f"\nFinal train shape: {train_final.shape}")
    print(f"Final test shape:  {test_final.shape}")
    print(f"Wrote train_selected.parquet, test_selected.parquet, feature_importance.csv to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()