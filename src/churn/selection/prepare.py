"""
Missing value handling + categorical encoding.

- age: median-imputed using TRAIN median only; adds age_was_missing flag so
  the MNAR signal found in EDA isn't erased, just made explicit.
- gender/marital_status/customer_language: filled with 'Missing' as their own
  category (not imputed) — EDA/Cox both showed missingness itself correlates
  with churn, so treating it as a real category preserves that signal.
- One-hot encoding is FIT on train's categories only; test is aligned to the
  same columns (reindexed, filling 0 for any category absent in test/train).
"""
import pandas as pd

CATEGORICAL_COLS = ["gender", "marital_status", "customer_language", "classe_anciennete"]
ID_AND_META_COLS = ["subscriber_id", "snapshot_date", "label"]


def fit_age_imputer(train: pd.DataFrame) -> float:
    return train["age"].median()


def apply_age_imputation(df: pd.DataFrame, median_value: float) -> pd.DataFrame:
    df = df.copy()
    df["age_was_missing"] = df["age"].isna().astype(int)
    df["age"] = df["age"].fillna(median_value)
    return df


def fill_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(object).where(df[col].notna(), "Missing")
    return df


def fit_encode_categoricals(train: pd.DataFrame) -> list[str]:
    """Returns the full list of one-hot column names learned from train,
    so test can be aligned to exactly these columns."""
    dummies = pd.get_dummies(train[CATEGORICAL_COLS], columns=CATEGORICAL_COLS, drop_first=True)
    return list(dummies.columns)


def apply_encode_categoricals(df: pd.DataFrame, fitted_columns: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], columns=CATEGORICAL_COLS, drop_first=True)
    dummies = dummies.reindex(columns=fitted_columns, fill_value=0)
    return pd.concat([df.drop(columns=CATEGORICAL_COLS), dummies], axis=1)


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # return clean train / test
    age_median = fit_age_imputer(train)
    print(f"age median (from train): {age_median}")

    train = apply_age_imputation(train, age_median)
    test = apply_age_imputation(test, age_median)

    train = fill_categorical_missing(train)
    test = fill_categorical_missing(test)

    fitted_cols = fit_encode_categoricals(train)
    print(f"one-hot columns learned from train: {len(fitted_cols)}")

    train = apply_encode_categoricals(train, fitted_cols)
    test = apply_encode_categoricals(test, fitted_cols)

    return train, test