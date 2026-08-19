"""
Missing value handling + categorical dtype prep.

- age: median-imputed using TRAIN median only; adds age_was_missing flag so
  the MNAR signal found in EDA isn't erased, just made explicit.
- gender/marital_status: mode-imputed using TRAIN mode only, same
  fit-on-train-only pattern as age (fit_categorical_imputer /
  apply_categorical_imputation).
- customer_language: filled with 'Missing' as its own category (not
  imputed). It is NOT mode-imputed: its mode ('Anglais') is also the
  highest-churn language group per EDA, so filling missing values with it
  would distort that signal rather than making a neutral guess -- keeping
  'Missing' as an explicit category preserves it instead.
  (classe_anciennete used to be handled the same way here, but was found
  to be a near-perfect bucketing of tenure_days -- see
  churn.features.snapshot's DEMOGRAPHIC_COLS comment -- and is excluded
  from the modeling feature set entirely, so it never reaches this module.)
- Categorical columns (gender, marital_status, customer_language) are
  converted to pandas 'category' dtype and left NATIVE -- not one-hot
  encoded -- because the two tree-based models
  trained on this shared feature set (XGBoost, LightGBM) both support
  categorical splits directly. This also avoids the SHAP-fragmentation
  problem where a one-hot dummy is ranked independently of its parent
  variable, and the "hidden baseline" problem of drop_first encoding.
  Logistic Regression is the one model here without native categorical
  support; it one-hot-encodes these same category-dtype columns itself,
  inside its own sklearn Pipeline (see
  churn.modeling.train.build_logistic_regression) -- prepare.py doesn't
  need to produce a separate one-hot representation for it.
- The set of allowed categories for each categorical column is FIT on
  train only, then applied to test (any category present in test but not
  train becomes NaN there) -- same fit-on-train-only discipline as
  everything else in this module.
"""
import pandas as pd

MODE_IMPUTE_COLS = ["gender", "marital_status"]
MISSING_CATEGORY_COLS = ["customer_language"]
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


def fit_categorical_dtypes(train: pd.DataFrame) -> dict[str, list]:
    """The set of category values for each categorical column, learned
    from TRAIN only, so test is aligned to exactly these categories."""
    return {col: sorted(train[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}


def apply_categorical_dtypes(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


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

    categories = fit_categorical_dtypes(train)
    print(f"categorical dtype categories (from train): { {k: len(v) for k, v in categories.items()} }")

    train = apply_categorical_dtypes(train, categories)
    test = apply_categorical_dtypes(test, categories)

    return train, test
