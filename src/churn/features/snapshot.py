#One snapshot → Feature engineering → Random stratified train/test split

import numpy as np
import pandas as pd

SNAPSHOT_DATE = "2024-12-01"
PREDICTION_WINDOW_DAYS = 90
OBSERVATION_WINDOW_DAYS = 90

LOG1P_FEATURES = ["recharge_amount", "mou_onnet", "data_trafic_volume",
                   "total_data_revenu_amount", "total_voice_revenu_amount",
                   "revenu_furfait_data_dinar"]

DELTA_SOURCE_COLS = ["recharge_amount", "mou_onnet", "data_trafic_volume",
                      "total_data_revenu_amount", "revenu_furfait_data_dinar"]
RECENCY_TENURE_RATIO_CAP = 10.0


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
    df = features.merge(
        population[["subscriber_id", "last_call_date", "date_activation"]],
        on="subscriber_id", how="left",
    )
    last_call_before_snapshot = df["last_call_date"].where(df["last_call_date"] < snapshot_date)
    df["days_since_last_call"] = (snapshot_date - last_call_before_snapshot).dt.days
    df["days_since_last_call"] = df["days_since_last_call"].fillna(OBSERVATION_WINDOW_DAYS + 1)
    df["days_since_last_call"] = df["days_since_last_call"].clip(upper=OBSERVATION_WINDOW_DAYS + 1)
    df["tenure_days"] = (snapshot_date - df["date_activation"]).dt.days

    # Inactivity relative to the customer's OWN history, not absolute days --
    # 30 days silent means something different for a 2-month-old account than
    # a 3-year one. tenure_days is clipped to >=1 to avoid divide-by-zero for
    # subscribers activated on the snapshot date itself, and the ratio is
    # capped since a brand-new account with a long silence can otherwise
    # produce an extreme outlier (days_since_last_call maxes at 91, so a
    # 1-day-old account could otherwise score a ratio of 91).
    df["recency_tenure_ratio"] = (
        df["days_since_last_call"] / df["tenure_days"].clip(lower=1)
    ).clip(upper=RECENCY_TENURE_RATIO_CAP)

    return df.drop(columns=["last_call_date", "date_activation"])


def compute_delta_features(
    monthly_usage: pd.DataFrame, population: pd.DataFrame, snapshot_date: pd.Timestamp,
    window_days: int = OBSERVATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Month-over-month CHANGE within the observation window, on the
    log1p-transformed scale: later period minus earlier period, per
    DELTA_SOURCE_COLS.

    IMPORTANT: with a 90-day window and monthly-granularity source data,
    the window currently contains exactly TWO periods (Oct and Nov 2024,
    for the current 2024-12-01 snapshot). This is a single month-over-month
    DELTA between two points, not a fitted trend/slope -- do not describe
    it as a "trend" anywhere. If a subscriber has usage in fewer than 2
    periods within the window (e.g. a very new or newly-reactivated
    account), or in neither, their delta is 0.0 -- there's no second point
    to compare against, same "no signal" convention used elsewhere in this
    module (e.g. compute_observation_features' zero-fill).

    Deltas are computed and merged in AFTER apply_log1p() has already run
    on the rest of the snapshot -- these columns start with the same
    prefixes as LOG1P_FEATURES (e.g. 'recharge_amount_delta' starts with
    'recharge_amount'), so if they existed before that step ran, they'd get
    log1p-transformed a SECOND time, which would clip every negative
    (declining) delta to zero and silently destroy the signal this feature
    exists to capture.
    """
    window_start = snapshot_date - pd.Timedelta(days=window_days)
    window_usage = monthly_usage[
        (monthly_usage["period_date"] >= window_start) & (monthly_usage["period_date"] < snapshot_date)
    ]
    window_usage = window_usage[window_usage["subscriber_id"].isin(population["subscriber_id"])]

    delta_features = population[["subscriber_id"]].copy()
    delta_cols = [f"{col}_delta" for col in DELTA_SOURCE_COLS]

    periods_present = sorted(window_usage["period_date"].unique())
    if len(periods_present) < 2:
        for col in delta_cols:
            delta_features[col] = 0.0
        return delta_features

    # The two most recent periods actually present in the window (today:
    # Oct and Nov 2024) -- not hardcoded, so this keeps working if the
    # snapshot date or window length ever changes.
    earlier_period, later_period = periods_present[-2], periods_present[-1]

    log1p_usage = window_usage.loc[
        window_usage["period_date"].isin([earlier_period, later_period]),
        ["subscriber_id", "period_date"] + DELTA_SOURCE_COLS,
    ].copy()
    for col in DELTA_SOURCE_COLS:
        log1p_usage[col] = np.log1p(log1p_usage[col].clip(lower=0))

    earlier = log1p_usage.loc[log1p_usage["period_date"] == earlier_period].set_index("subscriber_id")
    later = log1p_usage.loc[log1p_usage["period_date"] == later_period].set_index("subscriber_id")

    for col in DELTA_SOURCE_COLS:
        earlier_vals = earlier[col].reindex(delta_features["subscriber_id"]).fillna(0.0).to_numpy()
        later_vals = later[col].reindex(delta_features["subscriber_id"]).fillna(0.0).to_numpy()
        delta_features[f"{col}_delta"] = later_vals - earlier_vals

    return delta_features


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

DEMOGRAPHIC_COLS = ["age", "gender", "marital_status", "customer_language"]
# classe_anciennete is deliberately EXCLUDED from the modeling feature set:
# verified it's a non-overlapping bucketing of tenure (0 overlap between
# adjacent buckets at cutoffs 73/256/986 days, 95/393205 rank inversions --
# noise-level), i.e. essentially the same information as tenure_days with
# information lost to discretization. tenure_days is kept instead (continuous,
# lets tree models find their own split points). classe_anciennete remains
# available in subscriber_static.parquet for EDA/reporting -- see
# eda/temporal.py, eda/target_analysis.py, survival/cox_model.py, which all
# read subscriber_static directly and are unaffected by this exclusion.


def add_demographic_features(features: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls demographic fields straight from subscriber_static — no imputation,
    no fill. Nulls stay as NaN/None on purpose: any missing-value treatment
    (impute vs. keep 'Missing' as its own category) is a later decision,
    fit on training data only. contact_city excluded
    still unresolved (high-cardinality, inconsistent text) from EDA.
    """
    demo = population[["subscriber_id"] + DEMOGRAPHIC_COLS]
    return features.merge(demo, on="subscriber_id", how="left")


def build_snapshot(monthly_usage: pd.DataFrame, subscriber_static: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = pd.Timestamp(SNAPSHOT_DATE)
    population = get_active_population(subscriber_static, snapshot_date)

    features = compute_observation_features(monthly_usage, population, snapshot_date)
    features = add_recency_and_tenure_features(features, population, snapshot_date)
    features = add_demographic_features(features, population)   # <-- new
    features = apply_log1p(features)

    # Merged in AFTER apply_log1p -- see compute_delta_features' docstring
    # for why (these columns must not go through apply_log1p a second time).
    delta_features = compute_delta_features(monthly_usage, population, snapshot_date)
    features = features.merge(delta_features, on="subscriber_id", how="left")

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