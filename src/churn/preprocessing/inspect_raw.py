import pandas as pd
from churn.config import RAW_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

SAS_DATETIME_FMT = "%d%b%Y:%H:%M:%S.%f"
PERIOD_LIKE_NAMES = {"period_id", "periode"}
TENURE_NAME_PATTERNS = ["tenure", "anciennete", "ancien", "age_compte", "jours", "days", "delai", "duration"]


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_FILES[name], low_memory=False)


def profile_dataframe(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*70}\n{name}  —  shape={df.shape}\n{'='*70}")
    print(df.dtypes)
    print("\n--- null counts ---")
    print(df.isna().sum())

#DDATE HANDLING
#there are 2 diff formats of dates in the raw data: "01JAN2020:00:00:00.000" and 202001. The following functions parse these formats into pandas datetime objects.
def parse_sas_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=SAS_DATETIME_FMT, errors="coerce")


def parse_period(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("/", "", regex=False)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")

#scan every col to find date cols
def find_candidate_date_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    period_cols = [c for c in df.columns if c.lower() in PERIOD_LIKE_NAMES]
    sas_cols = []
    for c in df.columns:
        if df[c].dtype != object or c in period_cols:
            continue
        sample = df[c].dropna().astype(str).head(50)
        if sample.empty:
            continue
        if parse_sas_datetime(sample).notna().mean() > 0.8:
            sas_cols.append(c)
    return sas_cols, period_cols

#which % of cols parsed successfully as dates, and what is the range of those dates
def profile_date_column(df: pd.DataFrame, col: str, parser) -> None:
    parsed = parser(df[col])
    n_total = len(df)
    n_valid = parsed.notna().sum()
    print(f"\n  [{col}]")
    print(f"    non-null parsed as date: {n_valid}/{n_total} ({n_valid/n_total:.1%})")
    if n_valid > 0:
        print(f"    range: {parsed.min()}  ->  {parsed.max()}")

#CHURN LABEL ANALYSIS
#check whether churn label is consistent with the dates in the churn table. For example, if a subscriber has a churn date, then the churn label should be "yes". If a subscriber has no churn date, then the churn label should be "no".
def audit_churn_relationship(df_churn: pd.DataFrame, sas_cols: list[str], label_col: str) -> pd.Series:
    print(f"  churn label unique values: {df_churn[label_col].unique()}")
    is_churn = df_churn[label_col].astype(str).str.strip().str.lower() == "yes"
    for col in sas_cols:
        parsed = parse_sas_datetime(df_churn[col])
        pop_1 = parsed[is_churn].notna().mean()
        pop_0 = parsed[~is_churn].notna().mean()
        print(f"  [{col}] populated when churn=Yes: {pop_1:.1%} | populated when churn=No: {pop_0:.1%}")
    return is_churn

#TENURE ANALYSIS
#find any cols that look like they might be tenure-related and splits by churn / no churn to see if tenure is meaningful
def find_tenure_like_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(p in c.lower() for p in TENURE_NAME_PATTERNS)]


def profile_tenure_column(df: pd.DataFrame, col: str, is_churn: pd.Series | None = None) -> None:
    print(f"\n  [{col}]  dtype={df[col].dtype}")
    if pd.api.types.is_numeric_dtype(df[col]):
        print(df[col].describe())
        if is_churn is not None:
            print("  -- split by churn --")
            print("  churn=Yes:\n", df.loc[is_churn, col].describe())
            print("  churn=No:\n", df.loc[~is_churn, col].describe())
    else:
        print(df[col].value_counts(dropna=False))
        if is_churn is not None:
            print("  -- crosstab vs churn --")
            print(pd.crosstab(df[col], is_churn, normalize="index"))


#DATA QUALITY CHECKS

def check_chronological_order(df: pd.DataFrame, earlier_col: str, later_col: str) -> None:
    if earlier_col not in df.columns or later_col not in df.columns:
        return
    e = parse_sas_datetime(df[earlier_col])
    l = parse_sas_datetime(df[later_col])
    both_present = e.notna() & l.notna()
    n_both = both_present.sum()
    if n_both == 0:
        print(f"  [{earlier_col} <= {later_col}]  no rows with both fields populated")
        return
    violations = (e > l) & both_present
    print(f"  [{earlier_col} <= {later_col}]  violations: {violations.sum()}/{n_both} "
          f"({violations.sum()/n_both:.2%} of rows with both dates present)")

#INACTIVITY CHECKS
#how long since last activity using a single global reference date (the max period in the monthly usage data)
def explore_inactivity_periods(df_churn: pd.DataFrame, is_churn: pd.Series, reference_date: pd.Timestamp) -> None:
    print(f"\n  Using reference_date = {reference_date} (max period observed in monthly usage data, exploratory only)")

    last_call = parse_sas_datetime(df_churn["last_call_date"])
    inactivity_days = (reference_date - last_call).dt.days
    print("\n  inactivity_days (reference_date - last_call_date) — churn=Yes:")
    print(inactivity_days[is_churn].describe(percentiles=[.1, .25, .5, .75, .9]))
    print("\n  inactivity_days (reference_date - last_call_date) — churn=No:")
    print(inactivity_days[~is_churn].describe(percentiles=[.1, .25, .5, .75, .9]))

    churn_date = parse_sas_datetime(df_churn["churn_date"])
    activation_to_churn_days = (churn_date - parse_sas_datetime(df_churn["date_activation"])).dt.days
    print("\n  activation_to_churn_days (churners only — lifespan before churn):")
    print(activation_to_churn_days[is_churn].describe(percentiles=[.1, .25, .5, .75, .9]))

    churn_to_reference_days = (reference_date - churn_date).dt.days
    print("\n  churn_to_reference_days (churners only — how far in the 'future' relative "
          "to the churn event our reference_date sits; tells us how much lead time exists "
          "for choosing a prediction window without leaking past the churn event):")
    print(churn_to_reference_days[is_churn].describe(percentiles=[.1, .25, .5, .75, .9]))


#DUPLICATE-KEY CHECKS to avoid merge issues later
def check_duplicate_keys(df: pd.DataFrame, name: str, key_cols: list[str]) -> None:
    present = [c for c in key_cols if c in df.columns]
    if not present:
        return
    dup_mask = df.duplicated(subset=present, keep=False)
    n_dup_rows = dup_mask.sum()
    n_dup_keys = df.loc[dup_mask, present].drop_duplicates().shape[0]
    print(f"\n  [{name}] key={present}: {n_dup_rows} rows across {n_dup_keys} duplicated key values "
          f"({n_dup_rows/len(df):.2%} of rows)")
    if n_dup_rows > 0:
        sample_key = df.loc[dup_mask, present].drop_duplicates().iloc[0]
        mask = (df[present] == sample_key.values).all(axis=1)
        print("  -- sample duplicated group --")
        print(df.loc[mask].head(5))
        exact_full_dupes = df.duplicated(keep=False).sum()
        print(f"  fully-identical duplicate rows (all columns, not just key): {exact_full_dupes}")


def run_full_audit():
    dfs = {name: load_raw(name) for name in RAW_FILES}

    for name, df in dfs.items():
        profile_dataframe(df, name)
        sas_cols, period_cols = find_candidate_date_columns(df)
        print(f"\n--- SAS-datetime columns: {sas_cols} ---")
        for col in sas_cols:
            profile_date_column(df, col, parse_sas_datetime)
        print(f"\n--- period columns: {period_cols} ---")
        for col in period_cols:
            profile_date_column(df, col, parse_period)

    print(f"\n{'='*70}\nCHURN vs DATE-FIELD RELATIONSHIP (sample_churn)\n{'='*70}")
    churn_df = dfs["churn"]
    sas_cols, _ = find_candidate_date_columns(churn_df)
    is_churn = audit_churn_relationship(churn_df, sas_cols, label_col="churn")

    print(f"\n{'='*70}\nTENURE-LIKE FIELDS\n{'='*70}")
    for name, df in dfs.items():
        tenure_cols = find_tenure_like_columns(df)
        if not tenure_cols:
            continue
        print(f"\n-- {name}: {tenure_cols} --")
        for col in tenure_cols:
            profile_tenure_column(df, col, is_churn=is_churn if name == "churn" else None)

    print(f"\n{'='*70}\nCHRONOLOGICAL CONSISTENCY CHECKS (sample_churn)\n{'='*70}")
    for earlier, later in [
        ("date_activation", "first_call_date"),
        ("first_call_date", "last_call_date"),
        ("date_activation", "first_recharge_date"),
        ("date_activation", "churn_date"),
        ("last_call_date", "churn_date"),
    ]:
        check_chronological_order(churn_df, earlier, later)

    print(f"\n{'='*70}\nEXPLORATORY INACTIVITY-PERIOD ANALYSIS\n{'='*70}")
    reference_date = parse_period(dfs["monthly_agg"]["period_id"]).max()
    explore_inactivity_periods(churn_df, is_churn, reference_date)

    print(f"\n{'='*70}\nSUBSCRIBER_ID OVERLAP ACROSS SOURCES\n{'='*70}")
    ids = {name: set(df["subscriber_id"].dropna()) for name, df in dfs.items()}
    base = ids["churn"]
    for name, s in ids.items():
        overlap = len(base & s) / len(base) if base else 0
        print(f"  {name}: {len(s)} unique ids | overlap with churn table: {overlap:.1%}")

    print(f"\n{'='*70}\nDUPLICATE-KEY CHECKS\n{'='*70}")
    check_duplicate_keys(dfs["sociodemo"], "sociodemo", ["subscriber_id"])
    check_duplicate_keys(dfs["churn"], "churn", ["subscriber_id"])
    check_duplicate_keys(dfs["monthly_agg"], "monthly_agg", ["subscriber_id", "period_id"])
    check_duplicate_keys(dfs["data_bundle"], "data_bundle", ["subscriber_id", "periode"])

    print(f"\n{'='*70}\nSUSPICIOUS ROWS (null subscriber_id)\n{'='*70}")
    suspicious = churn_df[churn_df["subscriber_id"].isna()]
    print(f"  rows with null subscriber_id: {len(suspicious)}")
    print(suspicious.head())


if __name__ == "__main__":
    run_full_audit()