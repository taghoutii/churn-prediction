"""
section 1 : survival data preparation: duration, event, censoring.
"""

import pandas as pd

#find the latest date in dataset, either churn_date or last_call_date, to use as censoring reference date
def compute_censoring_reference_date(subscriber_static: pd.DataFrame) -> pd.Timestamp:
    return max(
        subscriber_static["churn_date"].max(),
        subscriber_static["last_call_date"].max(),
    )


def prepare_survival_data(subscriber_static: pd.DataFrame) -> pd.DataFrame:
    df = subscriber_static.copy()
    reference_date = compute_censoring_reference_date(df)
#event_date is churn_date if churned, else reference_date (censoring)
#duration is days between activation and event_date
#event is 1 if churned, else 0 (censored)
    event_date = df["churn_date"].where(df["is_churn"], reference_date)
    df["duration"] = (event_date - df["date_activation"]).dt.days
    df["event"] = df["is_churn"].astype(int)

    n_before = len(df)
    df = df[df["duration"] >= 0]
    n_dropped = n_before - len(df) #drop negative durations

    print(f"Survival data prepared: {len(df)} subscribers "
          f"({n_dropped} dropped for negative duration — activation after event/reference date)")
    print(f"Censoring reference date: {reference_date}")
    print(f"Event rate: {df['event'].mean():.2%}")
    print(df[["duration", "event"]].describe())

    return df