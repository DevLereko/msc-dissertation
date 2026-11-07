"""
07_model_training.py
Hyperparameter tuning, model selection and threshold freezing (Section 3.7.4).

Protocol (corrected v2)
1. Forward-chaining CV feasibility is checked inside the TRAINING partition:
   training dates are cut into k+1 equal chronological segments (on unique
   issuance dates, so same-day loans stay together); fold i trains on segments
   1..i and validates on segment i+1. Fold date ranges and class counts are
   logged for k = 3..5.
2. If some k >= MIN_CV_FOLDS gives every validation fold >= MIN_POS_PER_CV_FOLD
   defaults (and every fitting window has a default), hyperparameters are tuned
   by forward-chaining CV (mean PR-AUC across folds).
   Otherwise temporal CV is DECLARED INFEASIBLE (never replaced by shuffled CV)
   and each hyperparameter combination is fitted on the training partition and
   scored on the validation partition (declared fallback).
3. Selection criterion (pre-declared): PR-AUC (average precision), ties broken
   by ROC-AUC. The final model is the tuned model with the best validation
   PR-AUC.
4. Threshold is frozen at 0.5 (reproducible baseline, Section 3.7.4 step 8).
5. Final models are fitted on the training partition only. The test partition
   is NOT read by this script.
"""

import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (FEATURE_COLS, MIN_CV_FOLDS, MIN_POS_PER_CV_FOLD, MODELS,
                    SPLITS)
from modeling import (DISPLAY, MODEL_NAMES, PARAM_GRIDS, NoPositivesError,
                      fit_model)

DECISION_THRESHOLD = 0.5


def forward_chaining_folds(train, k):
    dates = np.array(sorted(train["loan_date"].unique()))
    segments = np.array_split(dates, k + 1)
    folds = []
    for i in range(1, k + 1):
        fit_dates = np.concatenate(segments[:i])
        val_dates = segments[i]
        fit_idx = train.index[train["loan_date"].isin(fit_dates)]
        val_idx = train.index[train["loan_date"].isin(val_dates)]
        folds.append({
            "fold": i,
            "fit_idx": fit_idx, "val_idx": val_idx,
            "fit_range": [str(pd.Timestamp(fit_dates[0]).date()), str(pd.Timestamp(fit_dates[-1]).date())],
            "val_range": [str(pd.Timestamp(val_dates[0]).date()), str(pd.Timestamp(val_dates[-1]).date())],
            "fit_loans": int(len(fit_idx)), "fit_defaults": int(train.loc[fit_idx, "target"].sum()),
            "val_loans": int(len(val_idx)), "val_defaults": int(train.loc[val_idx, "target"].sum()),
        })
    return folds


def cv_feasibility(train):
    report, feasible_k = {}, None
    for k in range(MIN_CV_FOLDS, 6):
        folds = forward_chaining_folds(train, k)
        ok = all(f["val_defaults"] >= MIN_POS_PER_CV_FOLD and f["fit_defaults"] > 0 for f in folds)
        report[f"k={k}"] = {
            "feasible": ok,
            "folds": [{key: v for key, v in f.items() if not key.endswith("_idx")} for f in folds],
        }
        if ok and feasible_k is None:
            feasible_k = k
    return feasible_k, report


def score(y, p):
    return average_precision_score(y, p), roc_auc_score(y, p)


def grid(name):
    keys = list(PARAM_GRIDS[name])
    return [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRIDS[name].values())]


def main():
    train = pd.read_csv(SPLITS / "train.csv", parse_dates=["loan_date"])
    val = pd.read_csv(SPLITS / "val.csv", parse_dates=["loan_date"])
    assert train["loan_date"].max() < val["loan_date"].min()
    X_tr, y_tr = train[FEATURE_COLS], train["target"]
    X_va, y_va = val[FEATURE_COLS], val["target"]

    # ---- Step 1-2: temporal CV feasibility ---------------------------------
    feasible_k, feas_report = cv_feasibility(train)
    if feasible_k is None:
        tuning_mode = "validation_partition"
        print(f"Temporal CV INFEASIBLE: no k in {MIN_CV_FOLDS}..5 gives every validation "
              f"fold >= {MIN_POS_PER_CV_FOLD} defaults. Using declared fallback: "
              "tune on the validation partition.")
    else:
        tuning_mode = f"forward_chaining_cv_k{feasible_k}"
        print(f"Temporal CV feasible with k={feasible_k}.")
    for k, r in feas_report.items():
        print(f"  {k}: val-fold defaults {[f['val_defaults'] for f in r['folds']]} "
              f"-> {'feasible' if r['feasible'] else 'infeasible'}")

    folds = forward_chaining_folds(train, feasible_k) if feasible_k else None

    # ---- Step 3: hyperparameter search -------------------------------------
    log_rows, best = [], {}
    for name in MODEL_NAMES:
        for params in grid(name):
            row = {"model": name, "params": json.dumps(params)}
            if folds:
                aps, aucs = [], []
                for f in folds:
                    try:
                        m = fit_model(name, params, X_tr.loc[f["fit_idx"]], y_tr.loc[f["fit_idx"]])
                    except NoPositivesError:
                        row[f"fold{f['fold']}_rejected"] = "no positives in fitting window"
                        continue
                    p = m.predict_proba(X_tr.loc[f["val_idx"]])[:, 1]
                    ap, auc = score(y_tr.loc[f["val_idx"]], p)
                    aps.append(ap); aucs.append(auc)
                row.update(pr_auc=float(np.mean(aps)), pr_auc_sd=float(np.std(aps)),
                           roc_auc=float(np.mean(aucs)), roc_auc_sd=float(np.std(aucs)))
            else:
                m = fit_model(name, params, X_tr, y_tr)
                ap, auc = score(y_va, m.predict_proba(X_va)[:, 1])
                row.update(pr_auc=float(ap), roc_auc=float(auc))
            log_rows.append(row)
        rows = [r for r in log_rows if r["model"] == name]
        top = max(rows, key=lambda r: (r["pr_auc"], r["roc_auc"]))
        best[name] = json.loads(top["params"])
        print(f"{DISPLAY[name]:20s} best {best[name]}  "
              f"PR-AUC={top['pr_auc']:.3f} ROC-AUC={top['roc_auc']:.3f}")

    pd.DataFrame(log_rows).to_csv(MODELS / "tuning_log.csv", index=False)

    # ---- Step 4-5: fit tuned models on train, compare on validation --------
    comparison, times = {}, {}
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        model = fit_model(name, best[name], X_tr, y_tr)
        times[name] = time.perf_counter() - t0
        p = model.predict_proba(X_va)[:, 1]
        ap, auc = score(y_va, p)
        comparison[name] = {"val_pr_auc": round(float(ap), 4), "val_roc_auc": round(float(auc), 4)}
        joblib.dump(model, MODELS / f"{name}.pkl")

    final = max(MODEL_NAMES, key=lambda n: (comparison[n]["val_pr_auc"], comparison[n]["val_roc_auc"]))

    selection = {
        "tuning_mode": tuning_mode,
        "selection_criterion": "validation PR-AUC (average precision); tie-break validation ROC-AUC",
        "best_params": best,
        "validation_comparison": comparison,
        "final_model": final,
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_rule": "fixed 0.5 baseline (Section 3.7.4 step 8)",
        "fitted_on": "training partition only",
        "train_rows": int(len(train)), "train_defaults": int(y_tr.sum()),
        "val_rows": int(len(val)), "val_defaults": int(y_va.sum()),
        "caveat": ("When tuning_mode is validation_partition, the same validation partition "
                   "is used for hyperparameter tuning and model selection, so validation "
                   "scores are optimistic; only test scores are unbiased."),
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    with open(MODELS / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    with open(MODELS / "cv_feasibility.json", "w") as f:
        json.dump({"min_pos_per_fold": MIN_POS_PER_CV_FOLD, "min_folds": MIN_CV_FOLDS,
                   "chosen_k": feasible_k, "folds_by_k": feas_report}, f, indent=2)
    with open(MODELS / "training_times.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in times.items()}, f, indent=2)

    print("\nValidation comparison:")
    for n in MODEL_NAMES:
        print(f"  {DISPLAY[n]:20s} {comparison[n]}")
    print(f"\nFINAL MODEL (frozen): {DISPLAY[final]}, threshold {DECISION_THRESHOLD}")


if __name__ == "__main__":
    main()
