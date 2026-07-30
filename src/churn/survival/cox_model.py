"""
Section 4 : Cox Proportional Hazards model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

COX_COVARIATES = ["age", "gender", "marital_status", "customer_language", "classe_anciennete"]


def prepare_cox_data(survival_df: pd.DataFrame) -> pd.DataFrame:
    df = survival_df[["duration", "event"] + COX_COVARIATES].copy()

    n_before = len(df)
    df = df.dropna(subset=["age"])
    print(f"Dropped {n_before - len(df)} rows with missing age (Cox requires complete covariates)")

    categorical_cols = ["gender", "marital_status", "customer_language", "classe_anciennete"]
    for col in categorical_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), "Missing")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def fit_cox_model(cox_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event")
    return cph


def get_top_hazard_ratios(cph: CoxPHFitter, top_n: int = 12) -> pd.DataFrame:
    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])
    summary["abs_log_effect"] = summary["coef"].abs()
    top = summary.sort_values("abs_log_effect", ascending=False).head(top_n)
    return top[["hazard_ratio", "coef", "p", "coef lower 95%", "coef upper 95%"]]


def plot_hazard_ratios(top_hr: pd.DataFrame):
    """Forest plot: HR on log scale, with 95% CI (converted from log-coef CI)."""
    df = top_hr.copy()
    df["hr_lower"] = np.exp(df["coef lower 95%"])
    df["hr_upper"] = np.exp(df["coef upper 95%"])
    df = df.sort_values("hazard_ratio")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df))))
    y_pos = range(len(df))

    ax.errorbar(
        df["hazard_ratio"], y_pos,
        xerr=[df["hazard_ratio"] - df["hr_lower"], df["hr_upper"] - df["hazard_ratio"]],
        fmt="o", color="darkred", ecolor="gray", capsize=3,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title("Cox Model — Top Hazard Ratios (95% CI)")
    plt.tight_layout()
    return fig