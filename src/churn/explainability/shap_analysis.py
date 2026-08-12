"""
SHAP explainability: global feature importance (summary plot) and
per-customer explanation (waterfall/force plot) for the leading model.

Using LightGBM (best F1/Brier score from steps 8-9) as the primary model for
explainability.
"""
import shap
import pandas as pd

from churn.modeling.train import get_xy, build_lightgbm


def fit_explainer(model, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """TreeExplainer is exact and fast for tree-based models like LightGBM —
    no approximation needed, unlike KernelExplainer for arbitrary models."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame):
    return explainer(X)


def get_top_customers_by_risk(model, X_test: pd.DataFrame, threshold: float, n: int = 3) -> pd.DataFrame:
    """Selects example customers at/above the working threshold (5:1 cost
    ratio, threshold=0.38 for LightGBM per step 9) for per-customer SHAP demo."""
    proba = model.predict_proba(X_test)[:, 1]
    flagged = X_test.copy()
    flagged["predicted_proba"] = proba
    flagged = flagged[flagged["predicted_proba"] >= threshold]
    return flagged.sort_values("predicted_proba", ascending=False).head(n)


def explain_customer(explainer: shap.TreeExplainer, customer_row: pd.DataFrame) -> pd.DataFrame:
    """Returns a tidy DataFrame of feature -> SHAP value for one customer,
    sorted by absolute impact — the input to the GenAI explanation layer."""
    shap_values = explainer(customer_row)
    df = pd.DataFrame({
        "feature": customer_row.columns,
        "value": customer_row.iloc[0].values,
        "shap_value": shap_values.values[0],
    })
    df["abs_impact"] = df["shap_value"].abs()
    return df.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")