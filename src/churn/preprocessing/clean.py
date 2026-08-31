"""
Per-source cleaning functions. Each takes a raw dataframe and returns a
cleaned one. (dtypes, the churn_date nulling rule)
"""
import numpy as np
import pandas as pd

SAS_DATETIME_FMT = "%d%b%Y:%H:%M:%S.%f"
MIN_PLAUSIBLE_AGE = 0
MAX_PLAUSIBLE_AGE = 100

# Physical ceiling: a 31-day month contains 31*24*60 = 44,640 minutes total, so no
# single call-minutes column can exceed that even with continuous back-to-back calls
# 24/7 for the whole month. Confirmed via inspect_raw-style investigation: mou_onnet
# alone had ~54k rows above this (max 868,281 min = ~603 days of talk-time in one
# month), so these are impossible values, not extreme-but-real heavy users.
MOU_COLS = ["mou_onnet", "mou_offnet_mobile", "mou_offnet_fix", "mou_international"]
MAX_PLAUSIBLE_MOU_MINUTES = 31 * 24 * 60

# data_trafic_volume's unit was confirmed to be bytes by cross-checking it against
# revenu_furfait_data_dinar / total_data_revenu_amount for rows with real data spend:
# the implied rate is ~250MB per dinar, which matches typical Tunisia mobile-data
# pricing -- so the unit reading is correct, not the problem. 100GB in a single month
# is already far beyond a realistic fair-use ceiling for one subscriber line. And
# empirically, above that bound the values stop looking like a smooth heavy-user tail:
# 86%+ of rows over 100GB report ZERO recharge_amount, ZERO total_data_revenu_amount,
# AND ZERO revenu_furfait_data_dinar in that same row -- i.e. claiming massive data
# usage with no purchase trigger of any kind, which is internally inconsistent and
# points to a data/aggregation error (below 100GB, most rows DO carry a matching
# revenue/recharge signal, consistent with genuine usage).
MAX_PLAUSIBLE_DATA_TRAFIC_BYTES = 100_000_000_000


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
    #null out physically-impossible mou values and implausible data_trafic_volume spikes
    df = df.copy()
    df["period_date"] = _parse_period(df["period_id"])
    zero_fill_cols = ["data_trafic_volume", "total_data_revenu_amount", "total_voice_revenu_amount"]
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0.0)

    for col in MOU_COLS:
        out_of_range = df[col] > MAX_PLAUSIBLE_MOU_MINUTES
        n_invalid = out_of_range.sum()
        if n_invalid:
            print(f"clean_monthly_agg: nulling {n_invalid} physically-impossible {col} value(s) "
                  f"(> {MAX_PLAUSIBLE_MOU_MINUTES} minutes in a 31-day month)")
        df.loc[out_of_range, col] = np.nan

    out_of_range = df["data_trafic_volume"] > MAX_PLAUSIBLE_DATA_TRAFIC_BYTES
    n_invalid = out_of_range.sum()
    if n_invalid:
        print(f"clean_monthly_agg: nulling {n_invalid} implausible data_trafic_volume value(s) "
              f"(> {MAX_PLAUSIBLE_DATA_TRAFIC_BYTES:,} bytes, ~100GB/month)")
    df.loc[out_of_range, "data_trafic_volume"] = np.nan

    return df


def clean_data_bundle(df: pd.DataFrame) -> pd.DataFrame:
    #convert period_id to period_date timestamp
    df = df.copy()
    df["period_date"] = _parse_period(df["periode"])
    return df