#One snapshot → Feature engineering → Random stratified train/test split

import numpy as np
import pandas as pd

SNAPSHOT_DATE = "2024-12-01"
PREDICTION_WINDOW_DAYS = 90
OBSERVATION_WINDOW_DAYS = 90

LOG1P_FEATURES = ["recharge_amount", "mou_onnet", "data_trafic_volume",
                   "total_data_revenu_amount", "total_voice_revenu_amount",
                   "revenu_furfait_data_dinar"]


def get_active_population(subscriber_static: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Subscribers who are live customers as of snapshot_date: activated
    before it, and not yet churned as of it."""
    activated = subscriber_static["date_activation"] <= snapshot_date
    not_yet_churned = subscriber_static["churn_date"].isna() | (subscriber_static["churn_date"] > snapshot_date)
    return subscriber_static.loc[activated & not_yet_churned].copy()


def compute_observation_features(
    monthly_usage: pd.DataFrame, population: pd.DataFrame, snapshot_date: pd.Timestamp,
    window_days: int = OBSERVATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Aggregates usage over the trailing window_days before snapshot_date."""
    window_start = snapshot_date - pd.Timedelta(days=window_days)
    window_usage = monthly_usage[
        (monthly_usage["period_date"] >= window_start) & (monthly_usage["period_date"] < snapshot_date)
    ]
    window_usage = window_usage[window_usage["subscriber_id"].isin(population["subscriber_id"])]

    agg = window_usage.groupby("subscriber_id").agg(
        recharge_amount_mean=("recharge_amount", "mean"),
        recharge_amount_sum=("recharge_amount", "sum"),
        number_of_recharges_mean=("number_of_recharges", "mean"),
        mou_onnet_mean=("mou_onnet", "mean"),
        data_trafic_volume_mean=("data_trafic_volume", "mean"),
        total_data_revenu_amount_mean=("total_data_revenu_amount", "mean"),
        total_voice_revenu_amount_mean=("total_voice_revenu_amount", "mean"),
        revenu_furfait_data_dinar_mean=("revenu_furfait_data_dinar", "mean"),
        n_periods_observed=("period_date", "nunique"),
    ).reset_index()

    features = population[["subscriber_id"]].merge(agg, on="subscriber_id", how="left")
    fill_cols = [c for c in features.columns if c != "subscriber_id"]
    features[fill_cols] = features[fill_cols].fillna(0.0)
    return features


def add_recency_and_tenure_features(
    features: pd.DataFrame, population: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """days_since_last_call computed strictly from history before snapshot_date."""
    df = features.merge(
        population[["subscriber_id", "last_call_date", "date_activation"]],
        on="subscriber_id", how="left",
    )
    last_call_before_snapshot = df["last_call_date"].where(df["last_call_date"] < snapshot_date)
    df["days_since_last_call"] = (snapshot_date - last_call_before_snapshot).dt.days
    df["days_since_last_call"] = df["days_since_last_call"].fillna(OBSERVATION_WINDOW_DAYS + 1)
    df["tenure_days"] = (snapshot_date - df["date_activation"]).dt.days
    return df.drop(columns=["last_call_date", "date_activation"])


def apply_log1p(features: pd.DataFrame) -> pd.DataFrame:
    """Fixed, parameter-free transform — safe to apply before train/test split."""
    df = features.copy()
    target_cols = [c for c in df.columns if any(c.startswith(f) for f in LOG1P_FEATURES)]
    for col in target_cols:
        df[col] = np.log1p(df[col].clip(lower=0))
    return df


def compute_label(
    # Target definition:
    # Predict whether an active subscriber at the snapshot date
    # will churn within the next 90 days.    
    subscriber_static: pd.DataFrame, population: pd.DataFrame, snapshot_date: pd.Timestamp,
    window_days: int = PREDICTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """1 if churn_date falls within (snapshot_date, snapshot_date + window_days]."""
    window_end = snapshot_date + pd.Timedelta(days=window_days)
    static_subset = subscriber_static.loc[
        subscriber_static["subscriber_id"].isin(population["subscriber_id"]),
        ["subscriber_id", "churn_date"],
    ].copy()
    in_window = (static_subset["churn_date"] > snapshot_date) & (static_subset["churn_date"] <= window_end)
    static_subset["label"] = in_window.astype(int)
    return static_subset[["subscriber_id", "label"]]


def build_snapshot(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = pd.Timestamp(SNAPSHOT_DATE)
    population = get_active_population(subscriber_static, snapshot_date)

    features = compute_observation_features(monthly_usage, population, snapshot_date)
    features = add_recency_and_tenure_features(features, population, snapshot_date)
    features = apply_log1p(features)

    label = compute_label(subscriber_static, population, snapshot_date)

    snapshot = features.merge(label, on="subscriber_id", how="inner")
    snapshot["snapshot_date"] = snapshot_date

    print(f"snapshot {snapshot_date.date()}: {len(snapshot)} subscribers, "
          f"churn rate in window = {snapshot['label'].mean():.2%}")
    return snapshot

def run_build_snapshot() -> None:
    from churn.config import INTERIM_DIR, PROCESSED_DIR

    monthly_usage = pd.read_parquet(INTERIM_DIR / "monthly_usage.parquet")
    subscriber_static = pd.read_parquet(INTERIM_DIR / "subscriber_static.parquet")

    snapshot = build_snapshot(monthly_usage, subscriber_static)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(PROCESSED_DIR / "snapshot.parquet", index=False)
    print(f"Wrote snapshot.parquet to {PROCESSED_DIR}")


if __name__ == "__main__":
    run_build_snapshot()