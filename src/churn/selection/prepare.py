"""
Missing value handling + categorical encoding.

- age: median-imputed using TRAIN median only; adds age_was_missing flag so
  the MNAR signal found in EDA isn't erased, just made explicit.
- gender/marital_status: mode-imputed using TRAIN mode only, same
  fit-on-train-only pattern as age (fit_categorical_imputer /
  apply_categorical_imputation) -- simple, low-cardinality demographic
  fields with only a small amount of missingness, so filling with the
  training set's most frequent value is preferred over carrying a
  dedicated 'Missing' category/one-hot column for each.
- customer_language/classe_anciennete: filled with 'Missing' as their own
  category (not imputed). customer_language specifically is NOT
  mode-imputed: its mode ('Anglais') is also the highest-churn language
  group per EDA, so filling missing values with it would distort that
  signal rather than making a neutral guess -- keeping 'Missing' as an
  explicit category preserves it instead.
- One-hot encoding is FIT on train's categories only; test is aligned to the
  same columns (reindexed, filling 0 for any category absent in test/train).
"""
import pandas as pd

MODE_IMPUTE_COLS = ["gender", "marital_status"]
MISSING_CATEGORY_COLS = ["customer_language", "classe_anciennete"]
CATEGORICAL_COLS = MODE_IMPUTE_COLS + MISSING_CATEGORY_COLS
ID_AND_META_COLS = ["subscriber_id", "snapshot_date", "label"]


def fit_age_imputer(train: pd.DataFrame) -> float:
    return train["age"].median()


def apply_age_imputation(df: pd.DataFrame, median_value: float) -> pd.DataFrame:
    df = df.copy()
    df["age_was_missing"] = df["age"].isna().astype(int)
    df["age"] = df["age"].fillna(median_value)
    return df


def fit_categorical_imputer(train: pd.DataFrame) -> dict[str, str]:
    """Mode of each of MODE_IMPUTE_COLS, computed from TRAIN only."""
    return {col: train[col].mode(dropna=True).iloc[0] for col in MODE_IMPUTE_COLS}


def apply_categorical_imputation(df: pd.DataFrame, modes: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, mode_value in modes.items():
        df[col] = df[col].fillna(mode_value)
    return df


def fill_missing_as_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MISSING_CATEGORY_COLS:
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

    categorical_modes = fit_categorical_imputer(train)
    print(f"categorical modes (from train): {categorical_modes}")

    train = apply_categorical_imputation(train, categorical_modes)
    test = apply_categorical_imputation(test, categorical_modes)

    train = fill_missing_as_category(train)
    test = fill_missing_as_category(test)

    fitted_cols = fit_encode_categoricals(train)
    print(f"one-hot columns learned from train: {len(fitted_cols)}")

    train = apply_encode_categoricals(train, fitted_cols)
    test = apply_encode_categoricals(test, fitted_cols)

    return train, test
