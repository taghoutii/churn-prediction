import pandas as pd
from churn.config import PROCESSED_DIR
from churn.balancing.balance import compare_balancers, apply_balancing

CHOSEN_METHOD = "smote"  # updated after reviewing comparison output


def run() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train_selected.parquet")

    print("--- Cross-validated comparison (resampling done per-fold) ---")
    comparison = compare_balancers(train)
    print("\n", comparison)
    comparison.to_csv(PROCESSED_DIR / "balancing_comparison.csv")

    print(f"\n--- Applying chosen method: {CHOSEN_METHOD} ---")
    train_balanced = apply_balancing(train, method=CHOSEN_METHOD)
    train_balanced.to_parquet(PROCESSED_DIR / "train_balanced.parquet", index=False)
    print(f"Wrote train_balanced.parquet to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()