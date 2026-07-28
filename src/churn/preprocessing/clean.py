"""
Per-source cleaning functions. Each takes a raw dataframe and returns a
cleaned one. No merging happens here — that's merge.py's job. Keeping these
separate means each cleaner can be unit-tested against a small synthetic
fixture without needing the other 3 sources.
"""
import numpy as np
import pandas as pd

SAS_DATETIME_FMT = "%d%b%Y:%H:%M:%S.%f"

# Plausible human age range for a mobile subscriber. Values outside this are
# data-entry errors (e.g. the age=1833 found in EDA Section 1), not real ages.
MIN_PLAUSIBLE_AGE = 0
MAX_PLAUSIBLE_AGE = 100


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

    # Data-entry errors (e.g. age=1833 found in EDA Section 1) — nulled rather
    # than dropped, since the row itself may still have useful demographic
    # data elsewhere. Null is preferred over capping/clipping here: we don't
    # know the true intended value, and inventing one (e.g. clipping 1833 to
    # 100) would silently fabricate data rather than honestly flag it missing.
    out_of_range = (df["age"] < MIN_PLAUSIBLE_AGE) | (df["age"] > MAX_PLAUSIBLE_AGE)
    n_invalid = out_of_range.sum()
    if n_invalid:
        print(f"clean_sociodemo: nulling {n_invalid} out-of-range age value(s) "
              f"(outside {MIN_PLAUSIBLE_AGE}-{MAX_PLAUSIBLE_AGE})")
    df.loc[out_of_range, "age"] = np.nan

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
      anywhere downstream (e.g. snapshot construction in step 4).
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

    # Business rule confirmed with supervisor: churn=No -> ignore churn_date.
    df.loc[~df["is_churn"], "churn_date"] = pd.NaT

    df = df.drop(columns=["SelectionProb", "SamplingWeight"])  # confirmed drop, doesn't affect balance
    return df


def clean_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_date"] = _parse_period(df["period_id"])

    # These three fields are null when a subscriber had no data/voice activity
    # that period, not genuinely unknown — same reasoning as revenu_furfait_
    # data_dinar in build_monthly_usage. Filling with 0 here (in clean.py)
    # rather than in merge.py keeps this monthly_agg-specific business rule
    # colocated with the rest of monthly_agg's cleaning, since it doesn't
    # depend on data_bundle at all.
    zero_fill_cols = ["data_trafic_volume", "total_data_revenu_amount", "total_voice_revenu_amount"]
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0.0)

    return df


def clean_data_bundle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_date"] = _parse_period(df["periode"])
    return df