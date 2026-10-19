"""
phase1_investigation.py

Phase 1 Feasibility Investigation.

This script does not modify source code or input data. It writes versioned
feasibility reports under results/corrected_v1 and refuses to overwrite them.
It reads the existing processed data and produces findings that inform
methodological decisions described in the dissertation gap-closure plan.

Outputs:
    results/corrected_v1/feasibility_findings_v4.json
    results/corrected_v1/feasibility_summary_v4.txt

Usage:
    python src/phase1_investigation.py
"""

import json
import sys
import platform
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED = Path("data/processed")
OUTPUT = Path("results/corrected_v1")
FINDINGS_PATH = OUTPUT / "feasibility_findings_v4.json"
SUMMARY_PATH = OUTPUT / "feasibility_summary_v4.txt"

FINDINGS = {}
WARNINGS = []
ERRORS = []


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def log(msg):
    print(msg)


def safe(fn, name):
    """Run a check; capture exceptions rather than crashing the whole run."""
    try:
        fn()
    except Exception as e:
        ERRORS.append({"check": name, "error": str(e)})
        print(f"  [!] {name} failed: {e}")


# ---------------------------------------------------------------------------
# Load base tables once
# ---------------------------------------------------------------------------
def load_base():
    log("Loading base tables...")
    loan = pd.read_csv(PROCESSED / "loan.csv")
    trans = pd.read_csv(PROCESSED / "trans.csv")
    account = pd.read_csv(PROCESSED / "account.csv")

    loan["date"] = pd.to_datetime(loan["date"], format="%y%m%d", errors="coerce")
    trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d", errors="coerce")
    if "date" in account.columns:
        account["date"] = pd.to_datetime(account["date"], format="%y%m%d", errors="coerce")

    log(f"  Loan records: {len(loan)}")
    log(f"  Transaction records: {len(trans)}")
    log(f"  Account records: {len(account)}")

    return {"loan": loan, "trans": trans, "account": account}


# ---------------------------------------------------------------------------
# Investigation 1: Same-day transactions
# ---------------------------------------------------------------------------
def check_same_day(t):
    log("\n[1] Same-day transactions (Decision 1)")
    loan = t["loan"]
    trans = t["trans"]

    same_day_count = 0
    same_day_trans_rows = 0
    same_day_per_loan = []

    for _, l in loan.iterrows():
        same_day = trans[(trans["account_id"] == l["account_id"])
                         & (trans["date"] == l["date"])]
        n = len(same_day)
        if n > 0:
            same_day_count += 1
            same_day_trans_rows += n
            same_day_per_loan.append(n)

    FINDINGS["same_day_transactions"] = {
        "total_loans": len(loan),
        "loans_with_same_day_transactions": same_day_count,
        "percent_loans_affected": round(100 * same_day_count / len(loan), 2),
        "total_same_day_transaction_rows": same_day_trans_rows,
        "avg_same_day_rows_per_affected_loan": round(
            float(np.mean(same_day_per_loan)), 2) if same_day_per_loan else 0.0,
        "max_same_day_rows_in_one_loan": int(max(same_day_per_loan)) if same_day_per_loan else 0,
    }
    log(f"  Loans with same-day transactions: {same_day_count} "
        f"({100 * same_day_count / len(loan):.1f}%)")
    log(f"  Total same-day transaction rows: {same_day_trans_rows}")


# ---------------------------------------------------------------------------
# Investigation 2: Prior-history availability under strict cutoff
# ---------------------------------------------------------------------------
def check_prior_history(t):
    log("\n[2] Prior-history availability under strict cutoff (Decision 2)")
    loan = t["loan"]
    trans = t["trans"]

    no_history_loans = []
    history_depths = []

    for _, l in loan.iterrows():
        prior = trans[(trans["account_id"] == l["account_id"])
                      & (trans["date"] < l["date"])]
        history_depths.append(len(prior))
        if len(prior) == 0:
            no_history_loans.append(int(l["loan_id"]))

    history_depths = np.array(history_depths)

    FINDINGS["prior_history_availability"] = {
        "total_loans": len(loan),
        "cutoff": "transaction_date < loan_date",
        "feature_definition": (
            "Availability of transaction rows for the intended corrected "
            "history features; this check does not recompute feature values."
        ),
        "current_feature_pipeline_cutoff": "transaction_date <= loan_date",
        "corrected_features_regenerated": False,
        "loans_with_no_prior_history": len(no_history_loans),
        "percent_with_no_prior_history": round(
            100 * len(no_history_loans) / len(loan), 2),
        "loan_ids_with_no_prior_history": no_history_loans,
        "min_transactions_per_loan": int(history_depths.min()),
        "median_transactions_per_loan": float(np.median(history_depths)),
        "mean_transactions_per_loan": round(float(history_depths.mean()), 2),
        "max_transactions_per_loan": int(history_depths.max()),
        "loans_with_1_to_5_prior_transactions": int(((history_depths >= 1)
                                                     & (history_depths <= 5)).sum()),
    }
    FINDINGS["prior_history_availability"]["cutoff"] = "transaction_date < loan_date"
    FINDINGS["prior_history_availability"]["feature_definition"] = (
        "Availability of transaction rows for the intended corrected "
        "history features; this check does not recompute feature values."
    )
    FINDINGS["prior_history_availability"]["current_feature_pipeline_cutoff"] = "transaction_date <= loan_date"
    FINDINGS["prior_history_availability"]["corrected_features_regenerated"] = False
    log(f"  Loans with zero prior-day transactions: {len(no_history_loans)} "
        f"({100 * len(no_history_loans) / len(loan):.1f}%)")
    log(f"  Median prior transactions per loan: {np.median(history_depths):.0f}")


# ---------------------------------------------------------------------------
# Investigation 3: Per-loan history depth distribution
# ---------------------------------------------------------------------------
def check_history_distribution(t):
    log("\n[3] Distribution of prior-history depth (Decision 2 impact)")
    loan = t["loan"]
    trans = t["trans"]

    depths = []
    for _, l in loan.iterrows():
        prior = trans[(trans["account_id"] == l["account_id"])
                      & (trans["date"] < l["date"])]
        depths.append(len(prior))

    depths = np.array(depths)
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    dist = {f"p{p}": float(np.percentile(depths, p)) for p in percentiles}

    buckets = {
        "0": int((depths == 0).sum()),
        "1-10": int(((depths >= 1) & (depths <= 10)).sum()),
        "11-50": int(((depths >= 11) & (depths <= 50)).sum()),
        "51-100": int(((depths >= 51) & (depths <= 100)).sum()),
        "101-500": int(((depths >= 101) & (depths <= 500)).sum()),
        "500+": int((depths > 500).sum()),
    }

    FINDINGS["history_depth_distribution"] = {
        "percentiles": dist,
        "buckets": buckets,
    }
    log(f"  Buckets: {buckets}")


# ---------------------------------------------------------------------------
# Investigation 4: Temporal distribution of defaults
# ---------------------------------------------------------------------------
def check_temporal_default_distribution(t):
    log("\n[4] Temporal distribution of defaults (Decision 4)")
    loan = t["loan"].copy()

    mapping = {"A": 0, "B": 1, "C": 0, "D": 1}
    loan["target"] = loan["status"].map(mapping)
    loan["year"] = loan["date"].dt.year

    by_year = loan.groupby("year")["target"].agg(["count", "sum"])
    by_year.columns = ["total", "defaults"]
    by_year["default_rate_pct"] = (100 * by_year["defaults"]
                                    / by_year["total"]).round(2)

    FINDINGS["defaults_by_year"] = {
        str(year): {
            "total": int(row["total"]),
            "defaults": int(row["defaults"]),
            "default_rate_pct": float(row["default_rate_pct"]),
        }
        for year, row in by_year.iterrows()
    }
    log(f"  Yearly breakdown:\n{by_year.to_string()}")

    # Also: distribution across chronological thirds
    loan_sorted = loan.sort_values("date").reset_index(drop=True)
    n = len(loan_sorted)
    thirds = {
        "first_third": loan_sorted.iloc[: n // 3],
        "middle_third": loan_sorted.iloc[n // 3 : 2 * n // 3],
        "last_third": loan_sorted.iloc[2 * n // 3 :],
    }
    thirds_summary = {}
    for name, chunk in thirds.items():
        thirds_summary[name] = {
            "n": int(len(chunk)),
            "defaults": int(chunk["target"].sum()),
            "date_range": [
                chunk["date"].min().strftime("%Y-%m-%d"),
                chunk["date"].max().strftime("%Y-%m-%d"),
            ],
        }
    FINDINGS["defaults_by_thirds"] = thirds_summary
    log(f"  Thirds: {thirds_summary}")


# ---------------------------------------------------------------------------
# Investigation 5: Forward-chaining fold feasibility
# ---------------------------------------------------------------------------
def check_fold_feasibility(t):
    log("\n[5] Forward-chaining fold feasibility (Decision 5)")
    train_path = PROCESSED / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Saved training partition not found: {train_path}")

    train = pd.read_csv(train_path)
    date_col = "loan_date" if "loan_date" in train.columns else "date"
    required = {"loan_id", "target", date_col}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Saved training partition is missing columns: {sorted(missing)}")
    if train["loan_id"].duplicated().any():
        raise ValueError("Saved training partition contains duplicate loan IDs")
    train[date_col] = pd.to_datetime(train[date_col], errors="raise")
    train = train.sort_values([date_col, "loan_id"], kind="mergesort").reset_index(drop=True)

    n_folds = 4
    initial_train_fraction = 0.40
    practical_min_val_defaults = 5
    initial_train_size = int(len(train) * initial_train_fraction)
    if initial_train_size < 1 or len(train) - initial_train_size < n_folds:
        raise ValueError("Not enough saved training rows for four validation windows")

    # Four sequential validation windows follow the initial 40% training window.
    validation_positions = np.array_split(
        np.arange(initial_train_size, len(train)), n_folds
    )
    fold_info = []
    for fold_number, positions in enumerate(validation_positions, start=1):
        val_start = int(positions[0])
        val_end = int(positions[-1]) + 1
        val = train.iloc[val_start:val_end]
        train_fold = train.iloc[:val_start]
        train_counts = {
            "0": int((train_fold["target"] == 0).sum()),
            "1": int((train_fold["target"] == 1).sum()),
        }
        val_counts = {
            "0": int((val["target"] == 0).sum()),
            "1": int((val["target"] == 1).sum()),
        }
        shared_dates = sorted(
            set(train_fold[date_col].dropna()) & set(val[date_col].dropna())
        )
        fold_info.append({
            "fold": fold_number,
            "train_n": int(len(train_fold)),
            "train_class_counts": train_counts,
            "train_date_range": [
                train_fold[date_col].min().strftime("%Y-%m-%d"),
                train_fold[date_col].max().strftime("%Y-%m-%d"),
            ],
            "train_loan_ids": [int(value) for value in train_fold["loan_id"]],
            "validation_n": int(len(val)),
            "validation_class_counts": val_counts,
            "validation_date_range": [
                val[date_col].min().strftime("%Y-%m-%d"),
                val[date_col].max().strftime("%Y-%m-%d"),
            ],
            "validation_loan_ids": [int(value) for value in val["loan_id"]],
            "shared_train_validation_boundary_dates": [
                value.strftime("%Y-%m-%d") for value in shared_dates
            ],
            "has_both_classes_in_train": all(count > 0 for count in train_counts.values()),
            "has_both_classes_in_validation": all(count > 0 for count in val_counts.values()),
            "meets_practical_min_validation_defaults": (
                val_counts["1"] >= practical_min_val_defaults
            ),
        })

    min_val_defaults = min(fold["validation_class_counts"]["1"] for fold in fold_info)
    viable = all(
        fold["has_both_classes_in_train"]
        and fold["has_both_classes_in_validation"]
        and fold["meets_practical_min_validation_defaults"]
        for fold in fold_info
    )
    FINDINGS["forward_chaining_feasibility"] = {
        "source": "data/processed/train.csv loan IDs and saved targets",
        "saved_training_rows": int(len(train)),
        "saved_training_class_counts": {
            "0": int((train["target"] == 0).sum()),
            "1": int((train["target"] == 1).sum()),
        },
        "n_validation_folds": n_folds,
        "initial_train_fraction": initial_train_fraction,
        "initial_train_n": initial_train_size,
        "ordering": f"{date_col} ascending, then loan_id ascending for deterministic ties",
        "minimum_validation_defaults_rule": {
            "count": practical_min_val_defaults,
            "interpretation": "Practical feasibility threshold chosen for this investigation, not a universal statistical guarantee.",
        },
        "min_validation_defaults": int(min_val_defaults),
        "all_training_windows_have_both_classes": all(
            fold["has_both_classes_in_train"] for fold in fold_info
        ),
        "all_validation_windows_have_both_classes": all(
            fold["has_both_classes_in_validation"] for fold in fold_info
        ),
        "viable": viable,
        "folds": fold_info,
    }

    for fold in fold_info:
        log(f"  Fold {fold['fold']}: train={fold['train_n']} "
            f"(defaults={fold['train_class_counts']['1']}), "
            f"validation={fold['validation_n']} "
            f"(defaults={fold['validation_class_counts']['1']}), "
            f"shared dates={fold['shared_train_validation_boundary_dates']}")
    log(f"  Four validation windows viable={viable}; practical minimum defaults/window="
        f"{practical_min_val_defaults} (minimum observed={min_val_defaults})")


# ---------------------------------------------------------------------------
# Investigation 6: Source semantics
# ---------------------------------------------------------------------------
def check_source_semantics(t):
    log("\n[6] Source semantics (Decision 1 justification)")
    trans = t["trans"]
    loan = t["loan"]

    FINDINGS["source_semantics"] = {
        "trans_columns": list(trans.columns),
        "loan_columns": list(loan.columns),
        "has_time_of_day_in_trans": any(
            ("time" in str(c).lower()) or ("hour" in str(c).lower())
            for c in trans.columns),
        "has_trans_id_that_might_imply_order": "trans_id" in trans.columns
            or "id" in [str(c).lower() for c in trans.columns],
        "date_format_documented": "yymmdd (day-level resolution only)",
        "note": ("PKDD '99 (Berka 1999) documents dates at day resolution. "
                 "No time-of-day field exists in the documented schema. "
                 "Transaction IDs are sequential within an account but do "
                 "not carry published intraday semantics."),
    }
    log(f"  Transaction columns: {list(trans.columns)}")
    log(f"  Has time-of-day field: {FINDINGS['source_semantics']['has_time_of_day_in_trans']}")


# ---------------------------------------------------------------------------
# Investigation 7: Target status distribution by date
# ---------------------------------------------------------------------------
def check_target_semantics(t):
    log("\n[7] Target semantics by status and date (Decision 3)")
    loan = t["loan"].copy()
    loan["year"] = loan["date"].dt.year

    by_status = loan.groupby(["status"]).agg(
        count=("loan_id", "count"),
        min_year=("year", "min"),
        max_year=("year", "max"),
    ).reset_index()

    FINDINGS["target_status_distribution"] = {
        str(row["status"]): {
            "count": int(row["count"]),
            "min_year": int(row["min_year"]),
            "max_year": int(row["max_year"]),
            "interpretation": {
                "A": "finished / paid — Non-default",
                "B": "finished / not paid — Default",
                "C": "running / paid — Non-default (snapshot)",
                "D": "running / in debt — Default (snapshot)",
            }.get(row["status"], "unknown"),
        }
        for _, row in by_status.iterrows()
    }
    FINDINGS["target_definition_note"] = {
        "mapping": {"A": 0, "B": 1, "C": 0, "D": 1},
        "recommended_name": "status-derived binary target in the PKDD '99 snapshot",
        "interpretation_limit": (
            "Combining finished (A/B) and running (C/D) contracts does not "
            "establish prediction of eventual default over a common future horizon."
        ),
        "horizon_established_by_this_investigation": False,
    }
    log(f"  Status distribution:\n{by_status.to_string()}")


# ---------------------------------------------------------------------------
# Investigation 8: SHAP/LIME compatibility
# ---------------------------------------------------------------------------
def check_xai_compatibility():
    log("\n[8] SHAP/LIME compatibility (Decision 8)")
    result = {}
    result["scope_note"] = (
        "Package availability only; this check does not establish fitted-model "
        "API compatibility, successful explanation generation, local fidelity, "
        "or explanation quality."
    )

    try:
        import shap
        result["shap_version"] = shap.__version__
        result["shap_available"] = True
    except ImportError as e:
        result["shap_available"] = False
        result["shap_error"] = str(e)

    try:
        import lime
        result["lime_available"] = True
        result["lime_version"] = getattr(lime, "__version__", "unknown")
    except ImportError as e:
        result["lime_available"] = False
        result["lime_error"] = str(e)

    try:
        import xgboost
        result["xgboost_version"] = xgboost.__version__
    except ImportError:
        result["xgboost_version"] = None

    FINDINGS["xai_compatibility"] = result
    log(f"  SHAP: {result.get('shap_version')}")
    log(f"  LIME available: {result.get('lime_available')}")


# ---------------------------------------------------------------------------
# Investigation 9: Stratified CV stability estimate
# ---------------------------------------------------------------------------
def check_stratified_cv_estimate(t):
    log("\n[9] Exploratory stratified baseline resampling (Decision 4 secondary)")
    from sklearn.base import clone
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    analytic_path = PROCESSED / "analytical_dataset.csv"
    if not analytic_path.exists():
        FINDINGS["stratified_cv_estimate"] = {
            "skipped": True,
            "reason": "analytical_dataset.csv not found",
        }
        log("  [skipped] analytical_dataset.csv not present")
        return

    df = pd.read_csv(analytic_path)
    feature_cols = [c for c in df.columns
                    if c not in ("loan_id", "account_id", "target",
                                 "loan_date", "status")]
    X = df[feature_cols].fillna(0)
    y = df["target"]

    models = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        C=0.1, random_state=42)),
        ]),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=20,
                                      min_samples_split=5, class_weight="balanced",
                                      random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                              random_state=42,
                              eval_metric="logloss", n_jobs=-1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "accuracy": accuracy_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }
    results = {
        "scope": {
            "purpose": "Exploratory supplementary resampling; not an independent validation of the fully tuned procedure.",
            "dataset": "data/processed/analytical_dataset.csv",
            "feature_cutoff": "Existing analytical dataset from the current inclusive date <= loan_date feature pipeline.",
            "fold_design": "5-fold shuffled stratified, non-temporal; specified hyperparameters are not retuned within folds.",
            "class_weight": "XGBoost scale_pos_weight is calculated separately from each fold's training labels.",
            "metric_name": "average_precision (AP), not necessarily trapezoidal PR-AUC",
        }
    }
    for name, model in models.items():
        try:
            fold_scores = {metric: [] for metric in scoring}
            for train_indices, val_indices in skf.split(X, y):
                fold_model = clone(model)
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
                if name == "XGB":
                    positives = int((y_train == 1).sum())
                    negatives = int((y_train == 0).sum())
                    fold_model.set_params(
                        scale_pos_weight=negatives / max(positives, 1)
                    )
                fold_model.fit(X.iloc[train_indices], y_train)
                probabilities = fold_model.predict_proba(X.iloc[val_indices])[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
                for metric, scorer in scoring.items():
                    if metric in ("f1", "precision", "recall"):
                        score = scorer(y_val, predictions, zero_division=0)
                    elif metric == "accuracy":
                        score = scorer(y_val, predictions)
                    else:
                        score = scorer(y_val, probabilities)
                    fold_scores[metric].append(float(score))
            results[name] = {
                metric: {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values, ddof=0)), 4),
                }
                for metric, values in fold_scores.items()
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    FINDINGS["stratified_cv_estimate"] = results
    for name, r in results.items():
        if name != "scope" and isinstance(r, dict) and "error" not in r:
            log(f"  {name}: ROC-AUC = {r['roc_auc']['mean']:.3f} "
                f"± {r['roc_auc']['std']:.3f} | AP = "
                f"{r['average_precision']['mean']:.3f}")


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------
def record_environment():
    FINDINGS["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "investigation_timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------
def write_summary():
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 1 FEASIBILITY INVESTIGATION SUMMARY")
    lines.append(f"Generated: {FINDINGS['environment']['investigation_timestamp']}")
    lines.append("=" * 72)
    lines.append("")

    # Decision 1
    lines.append("DECISION 1 — Feature cutoff")
    sd = FINDINGS.get("same_day_transactions", {})
    lines.append(f"  Loans with same-day transactions: "
                 f"{sd.get('loans_with_same_day_transactions', 'N/A')} / "
                 f"{sd.get('total_loans', 'N/A')} "
                 f"({sd.get('percent_loans_affected', 'N/A')}%)")
    lines.append(f"  Time-of-day field available: "
                 f"{FINDINGS.get('source_semantics', {}).get('has_time_of_day_in_trans', 'N/A')}")
    lines.append("  Recommendation: strict cutoff (transaction_date < loan_date) "
                 "unless time-of-day is available.")
    lines.append("")

    # Decision 2
    lines.append("DECISION 2 — No-history policy")
    ph = FINDINGS.get("prior_history_availability", {})
    lines.append(f"  Loans with NO prior-day history: "
                 f"{ph.get('loans_with_no_prior_history', 'N/A')}")
    lines.append(f"  Loans with 1–5 prior transactions: "
                 f"{ph.get('loans_with_1_to_5_prior_transactions', 'N/A')}")
    lines.append(f"  Median prior transactions per loan: "
                 f"{ph.get('median_transactions_per_loan', 'N/A')}")
    lines.append(f"  Cutoff checked: {ph.get('cutoff', 'N/A')}; current feature pipeline uses "
                 f"{ph.get('current_feature_pipeline_cutoff', 'unknown')}.")
    lines.append(f"  Corrected feature values regenerated: "
                 f"{ph.get('corrected_features_regenerated', False)}")
    lines.append("")

    # Decision 3
    lines.append("DECISION 3 — Target semantics")
    for status, info in FINDINGS.get("target_status_distribution", {}).items():
        lines.append(f"  Status {status}: {info['count']} loans "
                     f"({info['min_year']}–{info['max_year']})")
    target_note = FINDINGS.get("target_definition_note", {})
    if target_note:
        lines.append(f"  Target: {target_note['recommended_name']} "
                     f"({target_note['mapping']}).")
        lines.append(f"  Limitation: {target_note['interpretation_limit']}")
    lines.append("")

    # Decision 4
    lines.append("DECISION 4 — Test set")
    lines.append("  Temporal distribution of defaults:")
    for year, info in FINDINGS.get("defaults_by_year", {}).items():
        lines.append(f"    {year}: {info['defaults']}/{info['total']} "
                     f"({info['default_rate_pct']}%)")
    lines.append("")

    # Decision 5
    lines.append("DECISION 5 — Inner CV")
    fc = FINDINGS.get("forward_chaining_feasibility", {})
    if fc:
        lines.append(f"  Saved train rows: {fc.get('saved_training_rows', 'N/A')}; "
                     f"initial training window: {fc.get('initial_train_n', 'N/A')}")
        lines.append(f"  Validation windows: {fc.get('n_validation_folds', 'N/A')}; "
                     f"minimum defaults: {fc.get('min_validation_defaults', 'N/A')} "
                     f"(practical threshold: "
                     f"{fc.get('minimum_validation_defaults_rule', {}).get('count', 'N/A')})")
        lines.append(f"  Feasible under stated rule: {fc.get('viable', 'N/A')}")
        lines.append("  The five-default threshold is a practical investigation rule, "
                     "not a universal statistical guarantee.")
    lines.append("")

    # XAI
    lines.append("XAI COMPATIBILITY")
    xc = FINDINGS.get("xai_compatibility", {})
    lines.append(f"  SHAP: {xc.get('shap_version', 'N/A')}")
    lines.append(f"  LIME available: {xc.get('lime_available', 'N/A')}")
    lines.append(f"  Scope: {xc.get('scope_note', 'N/A')}")
    lines.append("")

    # Secondary CV
    lines.append("SECONDARY CV ESTIMATE (exploratory baseline resampling)")
    scv = FINDINGS.get("stratified_cv_estimate", {})
    for model, r in scv.items():
        if model == "scope":
            continue
        if isinstance(r, dict) and "error" not in r and "skipped" not in r:
            roc = r.get("roc_auc", {})
            ap = r.get("average_precision", {})
            lines.append(f"  {model}: ROC-AUC {roc.get('mean')} "
                         f"± {roc.get('std')} | AP {ap.get('mean')}")
        else:
            lines.append(f"  {model}: {r}")
    if scv.get("scope"):
        lines.append(f"  Scope: {scv['scope']['purpose']}")
        lines.append(f"  Dataset/cutoff: {scv['scope']['dataset']}; "
                     f"{scv['scope']['feature_cutoff']}")
        lines.append(f"  Folds/weights: {scv['scope']['fold_design']}; "
                     f"{scv['scope']['class_weight']}")
        lines.append("  AP is average precision, not necessarily trapezoidal PR-AUC.")
    lines.append("")

    if WARNINGS:
        lines.append("WARNINGS")
        for w in WARNINGS:
            lines.append(f"  - {w}")
        lines.append("")

    if ERRORS:
        lines.append("ERRORS")
        for e in ERRORS:
            lines.append(f"  - {e['check']}: {e['error']}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("NEXT STEPS")
    lines.append("1. Review the JSON findings at "
                 "results/corrected_v1/feasibility_findings_v4.json")
    lines.append("2. Treat this as feasibility evidence, not proof that temporal CV or XAI is valid.")
    lines.append("3. Seek supervisor approval for methodology changes before rerunning the main pipeline.")
    lines.append("=" * 72)

    summary_text = "\n".join(lines)
    SUMMARY_PATH.write_text(summary_text)
    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if FINDINGS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing feasibility report. "
            f"Choose new output paths; existing files: {FINDINGS_PATH}, {SUMMARY_PATH}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log("PHASE 1 FEASIBILITY INVESTIGATION")
    log("Source code and input data are read-only; versioned reports will be written.")
    log("=" * 72)

    record_environment()
    t = load_base()

    safe(lambda: check_same_day(t), "same_day")
    safe(lambda: check_prior_history(t), "prior_history")
    safe(lambda: check_history_distribution(t), "history_distribution")
    safe(lambda: check_temporal_default_distribution(t), "temporal_defaults")
    safe(lambda: check_fold_feasibility(t), "fold_feasibility")
    safe(lambda: check_source_semantics(t), "source_semantics")
    safe(lambda: check_target_semantics(t), "target_semantics")
    safe(check_xai_compatibility, "xai_compatibility")
    safe(lambda: check_stratified_cv_estimate(t), "stratified_cv")

    with open(FINDINGS_PATH, "w") as f:
        json.dump(FINDINGS, f, indent=2, default=str)

    write_summary()

    log(f"\nFull findings: {FINDINGS_PATH}")
    log(f"Human-readable summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
