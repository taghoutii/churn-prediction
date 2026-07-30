"""
  - subscriber_static: one row per subscriber_id (sociodemo + churn label/dates)
  - monthly_usage:     one row per (subscriber_id, period_date) (usage + bundle spend)
"""
import pandas as pd

from churn.config import RAW_FILES, INTERIM_DIR
from churn.preprocessing.clean import (
    clean_sociodemo, clean_churn, clean_monthly_agg, clean_data_bundle,
)


def load_and_clean_all() -> dict[str, pd.DataFrame]:
    sociodemo = clean_sociodemo(pd.read_csv(RAW_FILES["sociodemo"], low_memory=False))
    churn = clean_churn(pd.read_csv(RAW_FILES["churn"], low_memory=False))
    monthly_agg = clean_monthly_agg(pd.read_csv(RAW_FILES["monthly_agg"], low_memory=False))
    data_bundle = clean_data_bundle(pd.read_csv(RAW_FILES["data_bundle"], low_memory=False))
    return {
        "sociodemo": sociodemo,
        "churn": churn,
        "monthly_agg": monthly_agg,
        "data_bundle": data_bundle,
    }


def build_subscriber_static(sociodemo: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    sociodemo_no_msisdn = sociodemo.drop(columns=["msisdn"])

    static = churn.merge(sociodemo_no_msisdn, on="subscriber_id", how="left", validate="one_to_one")
    return static


def build_monthly_usage(monthly_agg: pd.DataFrame, data_bundle: pd.DataFrame) -> pd.DataFrame:
    usage = monthly_agg.merge(
        data_bundle.drop(columns=["periode"]),
        on=["subscriber_id", "period_date"],
        how="left",
        validate="one_to_one",
    )
    usage["revenu_furfait_data_dinar"] = usage["revenu_furfait_data_dinar"].fillna(0.0)
    return usage


def run_merge() -> None:
    sources = load_and_clean_all()

    static = build_subscriber_static(sources["sociodemo"], sources["churn"])
    usage = build_monthly_usage(sources["monthly_agg"], sources["data_bundle"])

    print(f"subscriber_static: shape={static.shape}, unique subscribers={static['subscriber_id'].nunique()}")
    print(f"monthly_usage:     shape={usage.shape}, unique subscribers={usage['subscriber_id'].nunique()}")

    # Sanity checks before writing to disk
    assert static["subscriber_id"].is_unique, "subscriber_static must have one row per subscriber"
    assert not usage.duplicated(subset=["subscriber_id", "period_date"]).any(), \
        "monthly_usage must have one row per subscriber-period"
    assert static["is_churn"].isin([True, False]).all(), "is_churn must be fully boolean, no nulls"

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    static.to_parquet(INTERIM_DIR / "subscriber_static.parquet", index=False)
    usage.to_parquet(INTERIM_DIR / "monthly_usage.parquet", index=False)
    print(f"\nWrote subscriber_static.parquet and monthly_usage.parquet to {INTERIM_DIR}")


if __name__ == "__main__":
    run_merge()