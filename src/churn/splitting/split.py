import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.30
RANDOM_STATE = 42


def split_train_test(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        snapshot,
        test_size=TEST_SIZE,
        stratify=snapshot["label"],
        random_state=RANDOM_STATE,
    )

    print(f"train: {len(train)} rows, churn rate = {train['label'].mean():.2%}")
    print(f"test:  {len(test)} rows, churn rate = {test['label'].mean():.2%}")

    return train, test


def run_split() -> None:
    from churn.config import PROCESSED_DIR

    snapshot = pd.read_parquet(PROCESSED_DIR / "snapshot.parquet")
    train, test = split_train_test(snapshot)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Wrote train.parquet and test.parquet to {PROCESSED_DIR}")


if __name__ == "__main__":
    run_split()