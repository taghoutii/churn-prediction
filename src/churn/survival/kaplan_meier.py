"""
Section 2 : Kaplan-Meier overall survival curve.
"""
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from typing import Optional, List, Dict
from lifelines.statistics import multivariate_logrank_test


def fit_overall_km(survival_df) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df["duration"], event_observed=survival_df["event"], label="Overall")
    return kmf


def plot_overall_km(kmf: KaplanMeierFitter):
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Tenure (days since activation)")
    ax.set_ylabel("Survival probability (not yet churned)")
    ax.set_title("Kaplan-Meier — Overall Survival")
    plt.tight_layout()
    return fig

AGE_BUCKET_ORDER = None 
TENURE_ORDER = ["0 to 3 Months", "3 to 6 Months", "6 to 12 Months", "12 to 36 Months", "More than 36 Months"]


def add_age_bucket(survival_df, bin_width: int = 20):  
    df = survival_df.copy()
    valid = df.dropna(subset=["age"]).copy()
    max_age = int(valid["age"].max())
    bins = list(range(0, max_age + bin_width, bin_width))
    valid["age_bucket"] = pd.cut(valid["age"], bins=bins, right=False)
    return valid


def fit_km_by_group(survival_df, group_col: str, order: Optional[List[str]] = None) -> dict[str, KaplanMeierFitter]:
    """Fits one KaplanMeierFitter per group; returns {group_value: fitted_kmf}."""
    fitters = {}
    groups = order if order else sorted(survival_df[group_col].dropna().unique(), key=str)
    for g in groups:
        subset = survival_df[survival_df[group_col] == g]
        if len(subset) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(durations=subset["duration"], event_observed=subset["event"], label=str(g))
        fitters[g] = kmf
    return fitters


def plot_km_by_group(fitters: dict, title: str, ci_show: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5))
    for kmf in fitters.values():
        kmf.plot_survival_function(ax=ax, ci_show=ci_show)
    ax.set_xlabel("Tenure (days since activation)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def run_logrank_test(survival_df, group_col: str) -> None:
    df = survival_df.dropna(subset=[group_col])
    result = multivariate_logrank_test(
        df["duration"], df[group_col], df["event"]
    )
    print(f"\n--- Log-rank test: {group_col} ---")
    print(f"test statistic: {result.test_statistic:.2f}, p-value: {result.p_value:.4g}")