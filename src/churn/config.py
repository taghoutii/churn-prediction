from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILES = {
    "sociodemo": RAW_DIR / "socio-demographic_sample_churn.csv",
    "monthly_agg": RAW_DIR / "monthly_aggregation_sample.csv",
    "data_bundle": RAW_DIR / "data_bundle_purchase_july2024_jan2025.csv",
    "churn": RAW_DIR / "sample_churn.csv",
}