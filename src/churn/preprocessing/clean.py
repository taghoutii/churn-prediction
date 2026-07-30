"""
Per-source cleaning functions. Each takes a raw dataframe and returns a
cleaned one. (dtypes, the churn_date nulling rule)
"""
import numpy as np
import pandas as pd

SAS_DATETIME_FMT = "%d%b%Y:%H:%M:%S.%f"
MIN_PLAUSIBLE_AGE = 0
MAX_PLAUSIBLE_AGE = 100


def _parse_sas_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=SAS_DATETIME_FMT, errors="coerce")


def _parse_period(series: pd.Series) -> pd.Series:
    """Handles 'YYYY/MM' strings and int/str YYYYMM -> first-of-month Timestamp."""
    s = series.astype(str).str.replace("/", "", regex=False)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")


def clean_sociodemo(df: pd.DataFrame) -> pd.DataFrame:
    #drop rows with missing id
    #convert subscriber_id to int64 
    #check if age is within plausible range, if not set to NaN
    df = df.copy()
    df = df.dropna(subset=["subscriber_id"])
    df["subscriber_id"] = df["subscriber_id"].astype("int64")

    out_of_range = (df["age"] < MIN_PLAUSIBLE_AGE) | (df["age"] > MAX_PLAUSIBLE_AGE)
    n_invalid = out_of_range.sum()
    if n_invalid:
        print(f"clean_sociodemo: nulling {n_invalid} out-of-range age value(s) "
              f"(outside {MIN_PLAUSIBLE_AGE}-{MAX_PLAUSIBLE_AGE})")
    df.loc[out_of_range, "age"] = np.nan
    n_variant = (df["customer_language"] == "Francais").sum()
    if n_variant:
        print(f"clean_sociodemo: merging {n_variant} 'Francais' (no cedilla) into 'Français'")
    df["customer_language"] = df["customer_language"].replace({"Francais": "Français"})

    return df


def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    #drop rows with missing id
    #convert subscriber_id to int64
    #convert is_churn col to boolean 
    #if churn = no, set churn_date to NaT
    #drop SelectionProb and SamplingWeight columns
    df = df.copy()
    df = df.dropna(subset=["subscriber_id"])
    df["subscriber_id"] = df["subscriber_id"].astype("int64")

    df["is_churn"] = df["churn"].astype(str).str.strip().str.lower() == "yes"

    date_cols = [
        "date_activation", "churn_date", "first_call_date", "last_call_date",
        "first_recharge_date", "last_international_call_date",
        "dat_attrib_kridi", "dat_remb_kridi_net", "ported_in_request_date",
    ]
    for col in date_cols:
        df[col] = _parse_sas_datetime(df[col])
    df.loc[~df["is_churn"], "churn_date"] = pd.NaT

    df = df.drop(columns=["SelectionProb", "SamplingWeight"])  # confirmed drop, doesn't affect balance
    return df


def clean_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    #convert period_id to period_date timestamp
    #fill missing vals in activity/rev cols with 0.0 (no activity that month)
    df = df.copy()
    df["period_date"] = _parse_period(df["period_id"])
    zero_fill_cols = ["data_trafic_volume", "total_data_revenu_amount", "total_voice_revenu_amount"]
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0.0)

    return df


def clean_data_bundle(df: pd.DataFrame) -> pd.DataFrame:
    #convert period_id to period_date timestamp
    df = df.copy()
    df["period_date"] = _parse_period(df["periode"])
    return df