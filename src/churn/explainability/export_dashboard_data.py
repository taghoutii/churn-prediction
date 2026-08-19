"""
Exports a static JSON snapshot of the tuned LightGBM model's predictions +
SHAP explanations for the standalone React dashboard (dashboard/).

Why pre-generated, not live: the dashboard displays a plain-language "AI-generated
summary" for each sampled customer. genai_explain.py is the project's reference
implementation for a LIVE version of that feature (a real Anthropic API call, with
fresh generation on every click) -- it is left untouched and still demonstrates that
design. This script instead WRITES the explanation text itself, once, at export
time, using the exact same constraints as genai_explain.py's SYSTEM_PROMPT (no
jargon, top 3-4 factors only, one concrete suggestion, grounded only in the data,
no causal claims). No Anthropic API key or network call is required to run this
script or to use the resulting dashboard, and there is zero ongoing generation cost.
If a billed API key becomes available later, genai_explain.py shows how to swap in
live, per-request generation instead.

Two different notions of "risk" are exported, deliberately kept separate:
- risk_tier (low/medium/high, cut at 0.20/0.50): a general-purpose visual banding
  used for the signal-bar indicator and the overview distribution chart.
- decision_threshold (0.16): the actual retention working threshold. Whether a
  given customer is "flagged" is NOT precomputed here -- the dashboard computes it
  client-side from predicted_proba vs. decision_threshold, so the UI can make the
  point explicit that probability and decision are two different things.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import PROCESSED_DIR, PROJECT_ROOT
from churn.modeling.train import get_xy, build_lightgbm
from churn.explainability.shap_analysis import fit_explainer, explain_customer
from churn.selection.prepare import CATEGORICAL_COLS

OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "src" / "data" / "dashboardData.json"

MODEL_VERSION = "lightgbm_v1"
SNAPSHOT_DATE = "2024-12-01"
DECISION_THRESHOLD = 0.16

RISK_TIER_LOW_MAX = 0.20
RISK_TIER_MEDIUM_MAX = 0.50

SAMPLE_SIZE = 150
SAMPLE_RANDOM_STATE = 42
TOP_N_FACTORS = 6
TOP_N_GLOBAL_DRIVERS = 10


# ---------------------------------------------------------------------------
# Plain-language feature knowledge -- the only place technical column names
# are translated into business language. Used both for the "why this
# customer is at risk" factor labels and for composing each customer's
# explanation paragraph below.
# ---------------------------------------------------------------------------
FEATURE_META = {
    "recharge_amount_mean": {
        "short_label": "Recharge amount",
        "category": "recharge",
        "kind": "numeric",
        "high": "they typically recharge with larger amounts than most customers",
        "low": "they typically recharge with smaller amounts than most customers",
    },
    "number_of_recharges_mean": {
        "short_label": "Recharge frequency",
        "category": "recharge",
        "kind": "numeric",
        "high": "they recharge their account often",
        "low": "they recharge their account less often than most customers",
    },
    "mou_onnet_mean": {
        "short_label": "On-net calling minutes",
        "category": "voice",
        "kind": "numeric",
        "high": "they spend a lot of time calling other Ooredoo customers",
        "low": "they rarely call other Ooredoo customers",
    },
    "data_trafic_volume_mean": {
        "short_label": "Data usage volume",
        "category": "data",
        "kind": "numeric",
        "high": "they use a large amount of mobile data",
        "low": "their mobile data use has been unusually low",
    },
    "total_data_revenu_amount_mean": {
        "short_label": "Data spend",
        "category": "data",
        "kind": "numeric",
        "high": "they spend a good amount on data",
        "low": "they spend very little on data",
    },
    "revenu_furfait_data_dinar_mean": {
        "short_label": "Data package spend",
        "category": "data",
        "kind": "numeric",
        "high": "they spend more than most on their data package",
        "low": "they spend very little on their data package",
    },
    "days_since_last_call": {
        "short_label": "Time since last activity",
        "category": "recency",
        "kind": "numeric",
        "high": "it's been a while since they were last active",
        "low": "they've been active quite recently",
    },
    "tenure_days": {
        "short_label": "Account tenure",
        "category": "tenure",
        "kind": "numeric",
        "high": "they've been a customer with us for a long time",
        "low": "they're still relatively new to us, without much history yet",
    },
    "age": {
        "short_label": "Age",
        "category": "demographic",
        "kind": "numeric",
        "high": "they're on the older side of our customer base",
        "low": "they're on the younger side of our customer base",
    },
    "age_was_missing": {
        "short_label": "Age on file",
        "category": "demographic",
        "kind": "flag",
        "present": "we don't have their age on file",
    },

    # Phase 4: month-over-month deltas (later minus earlier of the two
    # periods actually present in the observation window -- a single
    # change, not a fitted trend) and inactivity-relative-to-tenure.
    "recharge_amount_delta": {
        "short_label": "Recharge amount, month over month",
        "category": "recharge",
        "kind": "numeric",
        "high": "how much they recharge has been going up recently",
        "low": "how much they recharge has been dropping recently",
    },
    "mou_onnet_delta": {
        "short_label": "On-net minutes, month over month",
        "category": "voice",
        "kind": "numeric",
        "high": "their calling to other Ooredoo customers has been picking up recently",
        "low": "their calling to other Ooredoo customers has been dropping off recently",
    },
    "data_trafic_volume_delta": {
        "short_label": "Data usage, month over month",
        "category": "data",
        "kind": "numeric",
        "high": "their data use has been picking up recently",
        "low": "their data use has been dropping off recently",
    },
    "total_data_revenu_amount_delta": {
        "short_label": "Data spend, month over month",
        "category": "data",
        "kind": "numeric",
        "high": "their data spend has been going up recently",
        "low": "their data spend has been dropping recently",
    },
    "revenu_furfait_data_dinar_delta": {
        "short_label": "Data package spend, month over month",
        "category": "data",
        "kind": "numeric",
        "high": "their data package spend has been going up recently",
        "low": "their data package spend has been dropping recently",
    },
    "recency_tenure_ratio": {
        "short_label": "Inactivity relative to tenure",
        "category": "recency",
        "kind": "numeric",
        "high": "they've gone quiet for a stretch that's long relative to how long they've been a customer",
        "low": "they've stayed active relative to how long they've been a customer",
    },

    # Phase 5: native categoricals -- one column per variable (no one-hot
    # dummies, so every actual category gets its own clause here, including
    # what used to be an invisible dropped baseline category).
    "gender": {
        "short_label": "Registered gender",
        "category": "demographic",
        "kind": "categorical",
        "values": {
            "M": "they're registered as male",
            "F": "they're registered as female",
            "U": "their registered gender is listed as unspecified",
        },
    },
    "marital_status": {
        "short_label": "Marital status",
        "category": "demographic",
        "kind": "categorical",
        "values": {
            "Celibataire": "they're registered as single",
            "Marie": "they're registered as married",
            "Veuf": "they're registered as widowed",
            "Divorce": "they're registered as divorced",
        },
    },
    "customer_language": {
        "short_label": "Preferred language",
        "category": "demographic",
        "kind": "categorical",
        "values": {
            "Anglais": "their preferred language is English",
            "Arabe": "their preferred language is Arabic",
            "Français": "their preferred language is French",
            "Italien": "their preferred language is Italian",
            "Missing": "we don't have a preferred language on file for them",
        },
    },
    # classe_anciennete used to have an entry here, but was found to be a
    # near-perfect bucketing of tenure_days (0 overlap between adjacent
    # buckets; 95/393205 rank inversions -- noise) and is now excluded from
    # the modeling feature set entirely (see churn.features.snapshot). It
    # never reaches this module, so there's nothing to describe here.
}

# Short label for a whole categorical variable, used when its one-hot
# dummies are grouped back together for the global drivers ranking (Phase 2
# fix -- see group_columns_by_parent below). Keys must match
# churn.selection.prepare.CATEGORICAL_COLS.
PARENT_SHORT_LABELS = {
    "gender": "Registered gender",
    "marital_status": "Marital status",
    "customer_language": "Preferred language",
}

# Per-driver caveats surfaced in the dashboard tooltip for features whose
# average effect is concentrated in a small subgroup rather than spread
# broadly -- age_was_missing was found to have ~5% prevalence but a ~50x
# larger effect when triggered than when not (see investigation notes).
DRIVER_NOTES = {
    "age_was_missing": (
        "This effect is concentrated in the small share of customers with no "
        "age on file, not spread broadly across the customer base."
    ),
}

SUGGESTION_BY_CATEGORY = {
    "recharge": "A proactive recharge reminder or a small top-up bonus could help re-engage them before they lapse.",
    "data": "Offering a data bundle that better matches how they actually use their phone could help keep them engaged.",
    "voice": "A promotional on-net calling offer could give them a reason to stay engaged.",
    "recency": "A personal check-in call or message could help re-engage them before they go quiet for good.",
    "tenure_short": "A welcome-stage loyalty perk could help build stickiness while they're still new.",
    "tenure_long": "A loyalty reward recognizing their long history with us could help reinforce their reason to stay.",
    "demographic": "A more personalized retention offer based on their profile could help improve engagement.",
    "none": "Their overall profile looks broadly stable right now; standard engagement touchpoints should be enough, with no special retention action needed at this time.",
}


def compute_numeric_reference(X: pd.DataFrame) -> dict[str, float]:
    """Population medians (full test set) used only to decide 'higher/lower
    than usual' phrasing -- not shown to the user directly. FEATURE_META is
    a superset of every feature this module knows how to describe; the
    actual selected feature set changes as selection/correlation-pruning
    is retuned (e.g. revenu_furfait_data_dinar_mean was dropped once
    CORRELATION_THRESHOLD was lowered), so this only computes a reference
    for columns that are actually present in X."""
    numeric_cols = [f for f, meta in FEATURE_META.items() if meta["kind"] == "numeric" and f in X.columns]
    return {col: float(X[col].median()) for col in numeric_cols}


def describe_factor(feature: str, value, shap_value: float, reference: dict[str, float]) -> dict:
    meta = FEATURE_META[feature]
    increases_risk = shap_value > 0

    if meta["kind"] == "numeric":
        is_high = float(value) >= reference[feature]
        clause = meta["high"] if is_high else meta["low"]
    elif meta["kind"] == "flag":
        is_present = float(value) >= 0.5
        clause = meta["present"] if is_present else meta.get("absent", f"the opposite of \"{meta['present']}\" applies")
    elif meta["kind"] == "categorical":
        clause = meta["values"].get(str(value), meta["short_label"])
    else:
        clause = meta["short_label"]

    # Categorical values are strings (e.g. "Marie"), not something that
    # can be cast to float -- only numeric/flag features get a numeric
    # "value" in the exported JSON.
    json_value = str(value) if meta["kind"] == "categorical" else float(value)

    return {
        "feature": feature,
        "short_label": meta["short_label"],
        "value": json_value,
        "shap_value": float(shap_value),
        "direction": "increases_risk" if increases_risk else "decreases_risk",
        "clause": clause,
        "category": meta["category"],
    }


def join_clauses(clauses: list[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


TENURE_LONG_DAYS = 365 * 3  # matches the old "More than 36 Months" cutoff


def pick_suggestion(increasing_factors: list[dict]) -> str:
    if not increasing_factors:
        return SUGGESTION_BY_CATEGORY["none"]

    top = increasing_factors[0]
    category = top["category"]
    if category == "tenure":
        # tenure_days is the only "tenure" feature now that classe_anciennete
        # has been dropped (near-perfectly redundant with it -- see
        # investigation notes). Note: this branch previously compared
        # top["feature"] against the string "classe_anciennete_More than 36
        # Months" -- a pre-Phase-5 one-hot dummy name that the native
        # categorical column (just "classe_anciennete") never actually
        # matched, so "tenure_long" could never fire. Fixed here by
        # thresholding tenure_days' own value directly instead.
        category = "tenure_long" if float(top["value"]) >= TENURE_LONG_DAYS else "tenure_short"
    return SUGGESTION_BY_CATEGORY.get(category, SUGGESTION_BY_CATEGORY["demographic"])


def opening_qualifier(proba: float) -> str:
    if proba >= 0.7:
        return "very high"
    if proba >= 0.5:
        return "high"
    if proba >= DECISION_THRESHOLD:
        return "elevated"
    if proba >= 0.15:
        return "moderate"
    return "low"


INTRO_TEMPLATES = [
    "This customer's estimated churn risk is {qualifier}.",
    "Right now, this customer's estimated risk of churning comes out {qualifier}.",
]
MAIN_LEAD_INS = [
    "The main reasons are that ",
    "That's mostly explained by the fact that ",
]
OFFSET_LEAD_INS_WITH_RISK = [
    "On the positive side, ",
    "Working in the other direction, ",
]
OFFSET_LEAD_INS_NO_RISK = [
    "The clearest pattern here is that ",
    "What stands out most is that ",
]
CAVEAT_SENTENCES = [
    "These are patterns associated with churn, not proven causes.",
    "Note that these are patterns linked to churn, not confirmed reasons on their own.",
]


def build_explanation(factors: list[dict], predicted_proba: float, seed: int) -> str:
    """Writes the plain-language explanation for one customer. Same
    constraints as genai_explain.py's SYSTEM_PROMPT: no jargon, top 3-4
    factors only, one concrete suggestion, grounded only in the factors
    passed in, no causal claims -- just composed with fixed templates
    instead of a live model call.

    `seed` (the subscriber_id) only ever picks between equivalent phrasing
    variants, deterministically -- it never changes which facts are stated,
    only avoids identical wording for customers who happen to share the
    same top drivers."""
    top4 = factors[:4]
    increasing = [f for f in top4 if f["direction"] == "increases_risk"]
    decreasing = [f for f in top4 if f["direction"] == "decreases_risk"]

    v = seed % 2
    sentences = [INTRO_TEMPLATES[v].format(qualifier=opening_qualifier(predicted_proba))]

    if increasing:
        sentences.append(
            MAIN_LEAD_INS[v] + join_clauses([f["clause"] for f in increasing[:3]]) + "."
        )
    if decreasing:
        offset_clause = decreasing[0]["clause"]
        lead_in = (OFFSET_LEAD_INS_WITH_RISK if increasing else OFFSET_LEAD_INS_NO_RISK)[v]
        tail = ", which helps offset some of that risk." if increasing else "."
        sentences.append(f"{lead_in}{offset_clause}{tail}")

    sentences.append(pick_suggestion(increasing))
    sentences.append(CAVEAT_SENTENCES[v])

    return " ".join(sentences)


def group_columns_by_parent(columns: list[str], categorical_names: list[str]) -> dict[str, list[str]]:
    """Maps each column name to the group it should be scored under for the
    global drivers ranking.

    A one-hot dummy column (e.g. 'gender_M', from CATEGORICAL_COLS='gender')
    maps to its parent variable ('gender'), so all of that variable's
    dummies are scored together instead of competing as separate fragments.
    A column that already equals a categorical name maps to itself (the
    case once Phase 5 switches XGBoost/LightGBM to native categorical
    dtype, with no one-hot dummies at all -- this function stays correct
    either way). Everything else (continuous features, flags like
    age_was_missing) maps to itself, unchanged.
    """
    groups: dict[str, list[str]] = {}
    for col in columns:
        parent = col
        for cat in categorical_names:
            if col == cat or col.startswith(f"{cat}_"):
                parent = cat
                break
        groups.setdefault(parent, []).append(col)
    return groups


def compute_global_drivers(shap_values_full: np.ndarray, columns: pd.Index, top_n: int) -> list[dict]:
    """Global driver importance, aggregated so a one-hot-encoded categorical
    competes as ONE line (summing its dummies' SHAP contribution per row,
    before taking the mean of the absolute value) rather than as N
    independent, individually-ranked fragments. Continuous features and
    standalone flags (age_was_missing) are unaffected -- they're already a
    single column, i.e. a 'group' of one."""
    shap_df = pd.DataFrame(shap_values_full, columns=columns)
    groups = group_columns_by_parent(list(columns), CATEGORICAL_COLS)

    scores = {}
    is_group = {}
    for parent, cols in groups.items():
        row_sum = shap_df[cols].sum(axis=1)
        scores[parent] = float(row_sum.abs().mean())
        is_group[parent] = len(cols) > 1

    driver_series = pd.Series(scores).sort_values(ascending=False)

    global_drivers = []
    for feature, score in driver_series.head(top_n).items():
        short_label = PARENT_SHORT_LABELS.get(feature) or FEATURE_META.get(feature, {}).get("short_label", feature)
        entry = {
            "feature": feature,
            "short_label": short_label,
            "mean_abs_shap": score,
            "is_categorical_group": is_group[feature],
        }
        if feature in DRIVER_NOTES:
            entry["note"] = DRIVER_NOTES[feature]
        global_drivers.append(entry)
    return global_drivers


def risk_tier(proba: float) -> str:
    if proba < RISK_TIER_LOW_MAX:
        return "low"
    if proba < RISK_TIER_MEDIUM_MAX:
        return "medium"
    return "high"


def main() -> None:
    test = pd.read_parquet(PROCESSED_DIR / "test_selected.parquet")
    train_balanced = pd.read_parquet(PROCESSED_DIR / "train_balanced.parquet")

    X_train, y_train = get_xy(train_balanced)
    X_test, y_test = get_xy(test)

    model = build_lightgbm().fit(X_train, y_train)
    explainer = fit_explainer(model, X_train)

    proba = model.predict_proba(X_test)[:, 1]
    proba_series = pd.Series(proba, index=X_test.index)

    # ---- full-population overview stats (real, not sampled) ----
    tiers = proba_series.apply(risk_tier)
    tier_counts = tiers.value_counts()
    overview = {
        "total_customers": int(len(X_test)),
        "avg_probability": float(proba_series.mean()),
        "risk_tiers": {
            "low": int(tier_counts.get("low", 0)),
            "medium": int(tier_counts.get("medium", 0)),
            "high": int(tier_counts.get("high", 0)),
        },
    }
    print("Overview KPIs (full test set, n=%d):" % overview["total_customers"])
    print(json.dumps(overview, indent=2))

    # ---- full-population global driver importance (real, not sampled) ----
    # Categoricals are scored as ONE group (see compute_global_drivers) so a
    # one-hot dummy fragment can no longer out- or under-rank its own
    # parent variable's true combined effect.
    shap_values_full = explainer(X_test)
    global_drivers = compute_global_drivers(shap_values_full.values, X_test.columns, TOP_N_GLOBAL_DRIVERS)
    print("\nTop global drivers (mean |impact| across full test set, categoricals grouped):")
    for d in global_drivers:
        flag = " [grouped categorical]" if d["is_categorical_group"] else ""
        print(f"  {d['short_label']:<30} {d['mean_abs_shap']:.4f}{flag}")

    # ---- properly random customer sample (fixed seed) ----
    reference = compute_numeric_reference(X_test)
    sample_index = proba_series.sample(n=SAMPLE_SIZE, random_state=SAMPLE_RANDOM_STATE).index

    customers = []
    for idx in sample_index:
        row = X_test.loc[[idx]]
        shap_df = explain_customer(explainer, row)
        top_rows = shap_df.head(TOP_N_FACTORS)

        factors = [
            describe_factor(r.feature, r.value, r.shap_value, reference)
            for r in top_rows.itertuples()
        ]
        p = float(proba_series.loc[idx])
        customers.append({
            "subscriber_id": int(test.loc[idx, "subscriber_id"]),
            "predicted_proba": p,
            "risk_tier": risk_tier(p),
            "top_factors": factors,
            "explanation": build_explanation(factors, p, seed=int(test.loc[idx, "subscriber_id"])),
        })

    customers.sort(key=lambda c: c["predicted_proba"], reverse=True)

    output = {
        "meta": {
            "model_used": "Tuned LightGBM",
            "model_version": MODEL_VERSION,
            "snapshot_date": SNAPSHOT_DATE,
            "decision_threshold": DECISION_THRESHOLD,
            "risk_tier_bounds": {"low_max": RISK_TIER_LOW_MAX, "medium_max": RISK_TIER_MEDIUM_MAX},
            "sample_size": SAMPLE_SIZE,
            "sample_random_state": SAMPLE_RANDOM_STATE,
        },
        "overview": overview,
        "global_drivers": global_drivers,
        "customers": customers,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(customers)} sampled customers to {OUTPUT_PATH}")
    print("\n--- Sample explanations (for review) ---")
    for c in customers[:2] + customers[-1:]:
        print(f"\nsubscriber_id={c['subscriber_id']}  proba={c['predicted_proba']:.3f}  tier={c['risk_tier']}")
        print(c["explanation"])


if __name__ == "__main__":
    main()
