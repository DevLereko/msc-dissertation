"""
08_evaluation.py
One-time evaluation of the FROZEN models and FROZEN threshold on the untouched
test partition (Section 3.7.4 step 9).

- Models, hyperparameters, final-model choice and threshold all come from
  07_model_training.py (models/selection.json); nothing here is tuned.
- Every access to the test labels is appended to evaluation/test_access_log.jsonl
  with model file hashes, so any re-evaluation is visible in the audit trail.
- With very few test defaults, percentile bootstrap 95% CIs are reported.
"""

import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import EVALUATION, FEATURE_COLS, MODELS, RANDOM_SEED, SPLITS
from modeling import DISPLAY, MODEL_NAMES

N_BOOT = 2000


def bootstrap_ci(y, p, metric, rng):
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():   # resample lacks a class
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], len(vals)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    selection = json.loads((MODELS / "selection.json").read_text())
    thr = selection["decision_threshold"]

    test = pd.read_csv(SPLITS / "test.csv")
    X_te, y_te = test[FEATURE_COLS], test["target"].to_numpy()

    with open(EVALUATION / "test_access_log.jsonl", "a") as f:
        f.write(json.dumps({
            "time": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model_hashes": {n: sha256(MODELS / f"{n}.pkl") for n in MODEL_NAMES},
            "selection_frozen_at": selection["frozen_at"],
        }) + "\n")

    rng = np.random.default_rng(RANDOM_SEED)
    rows, proba = {}, {}
    for name in MODEL_NAMES:
        model = joblib.load(MODELS / f"{name}.pkl")
        p = model.predict_proba(X_te)[:, 1]
        yhat = (p >= thr).astype(int)
        proba[name] = p
        tn, fp, fn, tp = confusion_matrix(y_te, yhat, labels=[0, 1]).ravel()
        roc_ci, nb = bootstrap_ci(y_te, p, roc_auc_score, rng)
        pr_ci, _ = bootstrap_ci(y_te, p, average_precision_score, rng)
        rows[DISPLAY[name]] = {
            "final_model": name == selection["final_model"],
            "Accuracy": accuracy_score(y_te, yhat),
            "Precision": precision_score(y_te, yhat, zero_division=0),
            "Recall": recall_score(y_te, yhat, zero_division=0),
            "F1": f1_score(y_te, yhat, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, p), "ROC-AUC 95% CI": roc_ci,
            "PR-AUC": average_precision_score(y_te, p), "PR-AUC 95% CI": pr_ci,
            "Brier": brier_score_loss(y_te, p),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "bootstrap_valid_resamples": nb,
        }

    perf = pd.DataFrame(rows).T
    perf.to_csv(EVALUATION / "test_performance.csv")
    print(perf.to_string())

    meta = {"test_rows": int(len(y_te)), "test_defaults": int(y_te.sum()),
            "prevalence": round(float(y_te.mean()), 4), "threshold": thr,
            "final_model": selection["final_model"], "bootstrap_resamples": N_BOOT}
    (EVALUATION / "test_meta.json").write_text(json.dumps(meta, indent=2))

    pred = test[["loan_id", "loan_date", "target"]].copy()
    for name in MODEL_NAMES:
        pred[f"proba_{name}"] = proba[name]
    pred.to_csv(EVALUATION / "test_predictions.csv", index=False)

    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in MODEL_NAMES:
            if kind == "roc":
                x, y, _ = roc_curve(y_te, proba[name])
                lab = f"AUC={roc_auc_score(y_te, proba[name]):.3f}"
            else:
                y, x, _ = precision_recall_curve(y_te, proba[name])
                lab = f"AP={average_precision_score(y_te, proba[name]):.3f}"
            ax.plot(x, y, label=f"{DISPLAY[name]} ({lab})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--")
            ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        else:
            ax.axhline(y_te.mean(), color="k", ls="--", label=f"Prevalence ({y_te.mean():.3f})")
            ax.set(xlabel="Recall", ylabel="Precision")
        ax.set_title(f"{kind.upper()} curves — test partition (n={len(y_te)}, defaults={int(y_te.sum())})")
        ax.legend()
        fig.savefig(EVALUATION / f"fig_4_{kind}_curves.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"\nTest partition: {meta['test_rows']} loans, {meta['test_defaults']} defaults. "
          "Interpret positive-class metrics with the CIs above.")


if __name__ == "__main__":
    main()
