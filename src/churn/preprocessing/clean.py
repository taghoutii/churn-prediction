"""
Per-source cleaning functions. Each takes a raw dataframe and returns a
cleaned one. Keeping these
separate means each cleaner can be unit-tested against a small synthetic
fixture without needing the other 3 sources.
"""
import pandas as pd

SAS_DATETIME_FMT = "%d%b%Y:%H:%M:%S.%f"


def _parse_sas_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=SAS_DATETIME_FMT, errors="coerce")


def _parse_period(series: pd.Series) -> pd.Series:
    """Handles 'YYYY/MM' strings and int/str YYYYMM -> first-of-month Timestamp."""
    s = series.astype(str).str.replace("/", "", regex=False)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")


def clean_sociodemo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # The single all-null row found in inspect_raw audit — not a real subscriber.
    df = df.dropna(subset=["subscriber_id"])
    df["subscriber_id"] = df["subscriber_id"].astype("int64")
    return df


def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops the one all-null row (same anomaly as sociodemo).
    - Casts subscriber_id to int64 to match monthly_agg/data_bundle dtype
      (required for a clean merge key later).
    - Parses all SAS-format date columns into real datetime dtype.
    - Applies the confirmed business rule: churn_date is only meaningful when
      churn == 'Yes'. For churn == 'No', a populated churn_date reflects a
      PAST churn event for a since-reactivated customer, not current risk —
      so it's nulled out to prevent it being misread as a live cutoff date
      anywhere downstream.
    """
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

    # churn=No -> ignore churn_date.
    df.loc[~df["is_churn"], "churn_date"] = pd.NaT

    df = df.drop(columns=["SelectionProb", "SamplingWeight"])  # confirmed drop, doesn't affect balance
    return df


def clean_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_date"] = _parse_period(df["period_id"])
    return df


def clean_data_bundle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_date"] = _parse_period(df["periode"])
    return df