"""
Exports a static JSON snapshot comparing all FOUR models in the project
(Logistic Regression, XGBoost, LightGBM, Voting Ensemble) for the standalone
engineer-facing React app (model-lab/). This is a separate, honest
side-by-side comparison -- it does not assume LightGBM (the model used by
dashboard/) is "the" model.

Reuses shap_analysis.py's TreeExplainer pattern for the two tree models, and
export_dashboard_data.py's FEATURE_META / describe_factor / grouped-categorical
machinery so every model's SHAP factors are described in the same
plain-feature vocabulary and land on the same original (pre-encoding) columns
-- which is also what makes it possible to compare/average SHAP contributions
*across* models that don't share a native feature representation (see
LR_SHAP and ENSEMBLE_SHAP notes below).

Threshold policy:
- The "metrics" table (recall/precision/f1/ROC-AUC/PR-AUC) uses each model's
  plain .predict() (0.5 cutoff) -- identical to churn.modeling.train.evaluate,
  so it reproduces data/processed/model_comparison.csv exactly.
- Confusion matrices and each customer's "predicted_class" instead use each
  model's OWN 5:1 (FN:FP) cost-optimal threshold, computed fresh here via
  churn.evaluation.threshold for all four models -- including Logistic
  Regression and the Voting Ensemble, which have no prior established
  threshold anywhere else in the project. 5:1 is used (not 10:1/2:1) because
  it's the ratio already adopted as the project's working assumption for
  LightGBM (see notebooks/06_threshold_calibration.ipynb); using the same
  ratio for all four keeps the comparison apples-to-apples.

SHAP approach per model:
- XGBoost, LightGBM: shap.TreeExplainer, exact, on the native (categorical
  dtype) feature space -- same as shap_analysis.py.
- Logistic Regression (LR_SHAP): TreeExplainer doesn't apply to a linear
  model, so this uses shap.LinearExplainer on the fitted clf step with a
  background sample of the (one-hot + scaled) preprocessed training data.
  LinearExplainer's output lands in the ENCODED feature space (one dummy
  column per category, scaled numerics) -- group_columns_by_parent (reused
  from export_dashboard_data.py) sums each categorical's dummy contributions
  back onto its single parent column, so the result is directly comparable
  to the tree models' native-column SHAP output.
- Voting Ensemble (ENSEMBLE_SHAP): a soft VotingClassifier's predicted
  probability *is* the equal-weighted mean of its three sub-estimators'
  predicted probabilities. There's no single native explainer for a
  heterogeneous ensemble, and KernelExplainer over 150 customers x 3
  underlying model types would be slow and only approximate anyway. Instead:
  pull the ensemble's own fitted sub-estimators from
  voting_model.named_estimators_, compute each one's grouped SHAP values in
  the shared original-feature space described above, and average the three
  elementwise (equal weight, matching soft voting). This exactly reproduces
  the ensemble's predicted probability's additive decomposition to first
  order (SHAP's efficiency property is preserved under this average because
  it's an average of three already-efficient decompositions) while being
  cheap enough to run on all 150 sampled customers.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import sparse as sp
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from churn.config import PROCESSED_DIR, PROJECT_ROOT
from churn.evaluation.threshold import run_threshold_analysis
from churn.explainability.export_dashboard_data import (
    compute_numeric_reference,
    describe_factor,
    group_columns_by_parent,
)
from churn.modeling.train import (
    build_lightgbm,
    build_logistic_regression,
    build_voting_ensemble,
    build_xgboost,
    evaluate,
    get_xy,
)
from churn.selection.prepare import CATEGORICAL_COLS

OUTPUT_PATH = PROJECT_ROOT / "model-lab" / "src" / "data" / "modelComparisonData.json"

SAMPLE_SIZE = 150
SAMPLE_RANDOM_STATE = 42
TOP_N_FACTORS = 6
CURVE_POINTS = 150
COST_RATIO_KEY = "5:1"
LR_SHAP_BACKGROUND_SIZE = 200

MODEL_ORDER = ["logistic_regression", "xgboost", "lightgbm", "voting_ensemble"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "voting_ensemble": "Voting Ensemble",
}
BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "voting_ensemble": build_voting_ensemble,
}


def thin_curve(x: np.ndarray, y: np.ndarray, n_points: int = CURVE_POINTS) -> list[dict]:
    """Evenly-spaced downsample (always keeps both endpoints) so ROC/PR curves
    with tens of thousands of raw threshold points stay a reasonable JSON
    size without visibly changing the plotted shape."""
    if len(x) <= n_points:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, n_points).astype(int))
    return list(zip(x[idx].tolist(), y[idx].tolist()))


def to_dense_df(transformed, columns: list[str], index) -> pd.DataFrame:
    if sp.issparse(transformed):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, columns=columns, index=index)


def strip_transformer_prefix(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def grouped_shap_rows(shap_values: np.ndarray, raw_columns: list[str]) -> list[dict[str, float]]:
    """Sums a (n_rows, n_encoded_cols) SHAP array onto original feature names
    -- one-hot dummy columns collapse onto their parent categorical, every
    other column maps 1:1 onto itself. Returns one {feature: shap_value}
    dict per row, in the ORIGINAL (pre-encoding) feature space so it lines
    up with X_test.columns for the tree models."""
    groups = group_columns_by_parent(raw_columns, CATEGORICAL_COLS)
    col_index = {c: i for i, c in enumerate(raw_columns)}
    rows = []
    for r in range(shap_values.shape[0]):
        row = {}
        for parent, cols in groups.items():
            row[parent] = float(sum(shap_values[r, col_index[c]] for c in cols))
        rows.append(row)
    return rows


def tree_shap_rows(model, X_rows: pd.DataFrame) -> list[dict[str, float]]:
    explainer = shap.TreeExplainer(model)
    values = explainer(X_rows).values
    return [dict(zip(X_rows.columns, values[r])) for r in range(values.shape[0])]


def lr_shap_rows(lr_pipeline, X_train: pd.DataFrame, X_rows: pd.DataFrame) -> list[dict[str, float]]:
    preprocessor = lr_pipeline.named_steps["preprocess"]
    clf = lr_pipeline.named_steps["clf"]
    raw_columns = [strip_transformer_prefix(n) for n in preprocessor.get_feature_names_out()]

    background = to_dense_df(
        preprocessor.transform(X_train.sample(n=LR_SHAP_BACKGROUND_SIZE, random_state=SAMPLE_RANDOM_STATE)),
        raw_columns,
        None,
    )
    rows_transformed = to_dense_df(preprocessor.transform(X_rows), raw_columns, X_rows.index)

    explainer = shap.LinearExplainer(clf, background)
    values = explainer(rows_transformed).values
    return grouped_shap_rows(values, raw_columns)


def average_shap_rows(*row_lists: list[dict[str, float]]) -> list[dict[str, float]]:
    n = len(row_lists[0])
    averaged = []
    for i in range(n):
        keys = row_lists[0][i].keys()
        averaged.append({k: sum(rows[i][k] for rows in row_lists) / len(row_lists) for k in keys})
    return averaged


def top_factors_for_row(shap_row: dict[str, float], value_row: pd.Series, reference: dict[str, float]) -> list[dict]:
    ranked = sorted(shap_row.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N_FACTORS]
    return [describe_factor(feature, value_row[feature], shap_value, reference) for feature, shap_value in ranked]


def main() -> None:
    test = pd.read_parquet(PROCESSED_DIR / "test_selected.parquet")
    train_balanced = pd.read_parquet(PROCESSED_DIR / "train_balanced.parquet")

    X_train, y_train = get_xy(train_balanced)
    X_test, y_test = get_xy(test)

    print("Fitting all four models on train_balanced.parquet ...")
    fitted = {name: BUILDERS[name]().fit(X_train, y_train) for name in MODEL_ORDER}
    print("Done fitting.\n")

    # ---- metrics table: plain .predict() (0.5 cutoff), matches model_comparison.csv ----
    metrics = {}
    probas = {}
    for name in MODEL_ORDER:
        model = fitted[name]
        metrics[name] = evaluate(model, X_test, y_test)
        probas[name] = model.predict_proba(X_test)[:, 1]
        print(f"[metrics @0.5] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics[name].items()))
    print()

    # ---- each model's own fresh 5:1 cost-optimal threshold ----
    thresholds = {}
    for name in MODEL_ORDER:
        analysis = run_threshold_analysis(y_test.values, probas[name], name)
        thresholds[name] = analysis[COST_RATIO_KEY]["optimal_threshold"]
    print()
    print("5:1 cost-optimal thresholds:", {k: round(v, 4) for k, v in thresholds.items()})

    # ---- ROC / PR curves + confusion matrices at each model's own threshold ----
    roc_curves = {}
    pr_curves = {}
    confusion_matrices = {}
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        roc_curves[name] = [{"fpr": f, "tpr": t} for f, t in thin_curve(fpr, tpr)]

        precision, recall, _ = precision_recall_curve(y_test, probas[name])
        # precision_recall_curve returns points in descending-threshold order;
        # reverse so recall is ascending, matching the ROC curve's convention.
        pr_curves[name] = [{"recall": r, "precision": p} for r, p in thin_curve(recall[::-1], precision[::-1])]

        y_pred = (probas[name] >= thresholds[name]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_matrices[name] = {
            "threshold": float(thresholds[name]),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    print("\nConfusion matrices (at each model's own 5:1 threshold):")
    print(json.dumps(confusion_matrices, indent=2))

    # ---- fixed 150-customer sample (same index-based draw as export_dashboard_data.py) ----
    sample_index = X_test.sample(n=SAMPLE_SIZE, random_state=SAMPLE_RANDOM_STATE).index
    X_sample = X_test.loc[sample_index]
    reference = compute_numeric_reference(X_test)

    lr_pipeline = fitted["logistic_regression"]
    xgb_model = fitted["xgboost"]
    lgbm_model = fitted["lightgbm"]
    voting_model = fitted["voting_ensemble"]

    print("\nComputing SHAP factors for the 150-customer sample across all four models ...")
    shap_rows = {
        "logistic_regression": lr_shap_rows(lr_pipeline, X_train, X_sample),
        "xgboost": tree_shap_rows(xgb_model, X_sample),
        "lightgbm": tree_shap_rows(lgbm_model, X_sample),
    }
    # Voting Ensemble: average of its OWN fitted sub-estimators (see module
    # docstring, ENSEMBLE_SHAP note) -- not a fresh average of the standalone
    # models above, since VotingClassifier fits its own internal clones.
    ens_lr = voting_model.named_estimators_["lr"]
    ens_xgb = voting_model.named_estimators_["xgb"]
    ens_lgbm = voting_model.named_estimators_["lgbm"]
    shap_rows["voting_ensemble"] = average_shap_rows(
        lr_shap_rows(ens_lr, X_train, X_sample),
        tree_shap_rows(ens_xgb, X_sample),
        tree_shap_rows(ens_lgbm, X_sample),
    )
    print("Done.\n")

    # ---- per-customer predictions + top factors from every model ----
    customers = []
    for i, idx in enumerate(sample_index):
        value_row = X_sample.loc[idx]
        predictions = {}
        classes = {}
        for name in MODEL_ORDER:
            proba_series = pd.Series(probas[name], index=X_test.index)
            p = float(proba_series.loc[idx])
            predicted_class = int(p >= thresholds[name])
            classes[name] = predicted_class
            predictions[name] = {
                "predicted_proba": p,
                "predicted_class": predicted_class,
                "threshold": float(thresholds[name]),
                "top_factors": top_factors_for_row(shap_rows[name][i], value_row, reference),
            }

        votes_churn = sum(classes.values())
        customers.append({
            "subscriber_id": int(test.loc[idx, "subscriber_id"]),
            "predictions": predictions,
            "votes_churn": votes_churn,
            "all_agree": votes_churn in (0, 4),
        })

    customers.sort(key=lambda c: c["predictions"]["lightgbm"]["predicted_proba"], reverse=True)

    # ---- agreement summary across the 150 ----
    agreement_summary = {"all_4_agree": 0, "3_1_split": 0, "2_2_split": 0}
    for c in customers:
        v = c["votes_churn"]
        if v in (0, 4):
            agreement_summary["all_4_agree"] += 1
        elif v in (1, 3):
            agreement_summary["3_1_split"] += 1
        else:
            agreement_summary["2_2_split"] += 1
    print("Agreement summary (150 sampled customers):", agreement_summary)

    # ---- XGBoost vs LightGBM disagreements, gap-sorted ----
    xgb_lgbm_disagreements = []
    for c in customers:
        xgb_p = c["predictions"]["xgboost"]
        lgbm_p = c["predictions"]["lightgbm"]
        if xgb_p["predicted_class"] != lgbm_p["predicted_class"]:
            xgb_lgbm_disagreements.append({
                "subscriber_id": c["subscriber_id"],
                "xgboost_proba": xgb_p["predicted_proba"],
                "xgboost_class": xgb_p["predicted_class"],
                "lightgbm_proba": lgbm_p["predicted_proba"],
                "lightgbm_class": lgbm_p["predicted_class"],
                "gap": abs(xgb_p["predicted_proba"] - lgbm_p["predicted_proba"]),
            })
    xgb_lgbm_disagreements.sort(key=lambda d: d["gap"], reverse=True)
    print(f"XGBoost vs LightGBM disagreements: {len(xgb_lgbm_disagreements)} of {SAMPLE_SIZE}")

    output = {
        "meta": {
            "model_order": MODEL_ORDER,
            "model_labels": MODEL_LABELS,
            "cost_ratio": COST_RATIO_KEY,
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "sample_size": SAMPLE_SIZE,
            "sample_random_state": SAMPLE_RANDOM_STATE,
            "test_set_size": int(len(X_test)),
        },
        "metrics": metrics,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "confusion_matrices": confusion_matrices,
        "customers": customers,
        "agreement_summary": agreement_summary,
        "xgb_lgbm_disagreements": xgb_lgbm_disagreements,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(customers)} sampled customers x 4 models to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
