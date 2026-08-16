# Churn Prediction

A customer churn prediction project for Ooredoo Tunisia, built as a telecom data
science internship project. It predicts, for each active subscriber, whether they
will churn (cancel their subscription) within the next 90 days, using sociodemographic
data and monthly usage/billing behavior. The project covers the full workflow from raw
data through EDA, survival analysis, feature engineering, model training/tuning,
threshold calibration, and SHAP-based explainability.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

Raw data files (`data/raw/*.csv`) contain real subscriber PII and are not committed —
place them locally per `src/churn/config.py`'s `RAW_FILES` mapping.

Copy `.env` and set:

```
ANTHROPIC_API_KEY=   # optional — only needed for GenAI plain-language SHAP explanations
```

## Project structure

`src/churn/` is organized as one subpackage per pipeline step:

| Subpackage | Purpose |
|---|---|
| `preprocessing/` | Raw data auditing (`inspect_raw.py`) and per-source cleaning/dtype fixes (`clean.py`, `merge.py`) |
| `eda/` | Exploratory analysis: overview, temporal, target/correlation, behavioral views |
| `survival/` | Kaplan-Meier and Cox Proportional Hazards survival analysis |
| `features/` | Snapshot-based feature engineering (`snapshot.py`) |
| `splitting/` | Train/test split |
| `selection/` | Missing-value handling, encoding, correlation pruning, importance ranking |
| `balancing/` | SMOTE vs. ADASYN comparison and class balancing |
| `modeling/` | Model training (`train.py`), hyperparameter tuning (`tune.py`), and their `run_*.py` orchestrators |
| `evaluation/` | Calibration check and cost-based threshold optimization |
| `explainability/` | SHAP values and optional Claude-generated plain-language explanations |

Each stage has pure, testable functions plus a thin `run_*.py` (or `__main__`) entry
point that reads/writes `data/{raw,interim,processed}/`. Numbered notebooks under
`notebooks/` (`01_eda` → `07_explainability`) call into this same code and save
figures to `reports/figures/`.

## Running the pipeline end to end

```bash
uv run python -m churn.preprocessing.inspect_raw   # raw data audit (optional, prints only)
uv run python -m churn.preprocessing.merge          # -> data/interim/{subscriber_static,monthly_usage}.parquet
uv run python -m churn.features.snapshot            # -> data/processed/snapshot.parquet
uv run python -m churn.splitting.split              # -> data/processed/{train,test}.parquet
uv run python -m churn.selection.run_selection      # -> data/processed/{train,test}_selected.parquet
uv run python -m churn.balancing.run_balance        # -> data/processed/train_balanced.parquet
uv run python -m churn.modeling.run_tuning          # -> data/processed/tuning_results.json
uv run python -m churn.modeling.run_training         # -> data/processed/model_comparison.csv, logs to MLflow
```

Threshold calibration and SHAP explainability are run interactively — see
`notebooks/06_threshold_calibration.ipynb` and `notebooks/07_explainability.ipynb`.

## Notebooks and MLflow

Open notebooks with:

```bash
uv run jupyter lab
```

View tracked training runs (params, metrics, logged models) with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Key design decisions

- **Single snapshot date** (2024-12-01): features are computed as of one fixed point
  in time rather than multiple rolling snapshots, to keep the leakage story simple for
  this stage of the project.
- **90-day prediction window**: the label is whether a subscriber active at the
  snapshot churns within the following 90 days.
- **90-day observation window**: usage/billing features are aggregated over the 90
  days *before* the snapshot date.
- **Random, label-stratified train/test split** (not chronological): since there is
  only one snapshot date, there is no future period to hold out chronologically.
- **Leakage discipline**: anything learned from data (imputation values, encoders,
  correlation-pruning decisions, resamplers) is fit on the training set only, then
  applied unchanged to test.
