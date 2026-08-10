"""
Calibration check + cost-based threshold optimization.

"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

COST_RATIOS = [2, 5, 10]  # FN : FP 
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

#check whether model's predicted probabilities are trustworthy (calibrated) or not
def get_calibration_data(y_true, y_proba, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_proba)
    return prob_true, prob_pred, brier

#for one specific threshold, compute the confusion matrix and total cost based on FN and FP costs
def compute_cost_at_threshold(y_true, y_proba, threshold: float, fn_cost: float, fp_cost: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return {"threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_cost": total_cost}

#find one optimal threshold for a given FN:FP cost ratio
def find_optimal_threshold(y_true, y_proba, fn_cost: float, fp_cost: float) -> pd.DataFrame:
    results = [compute_cost_at_threshold(y_true, y_proba, t, fn_cost, fp_cost) for t in THRESHOLD_GRID]
    return pd.DataFrame(results)


def run_threshold_analysis(y_true, y_proba, model_name: str) -> dict:
    """Run the full grid search across all HYPOTHETICAL cost ratios (FP cost fixed at 1)."""
    all_results = {}
    for ratio in COST_RATIOS:
        df = find_optimal_threshold(y_true, y_proba, fn_cost=ratio, fp_cost=1)
        best_row = df.loc[df["total_cost"].idxmin()].to_dict()
        all_results[f"{ratio}:1"] = {
            "optimal_threshold": best_row["threshold"],
            "total_cost": best_row["total_cost"],
            "fn": int(best_row["fn"]),
            "fp": int(best_row["fp"]),
            "tp": int(best_row["tp"]),
        }
        print(f"[{model_name}] cost ratio {ratio}:1 -> optimal threshold = {best_row['threshold']:.2f}, "
              f"FN={int(best_row['fn'])}, FP={int(best_row['fp'])}, cost={best_row['total_cost']:.0f}")
    return all_results