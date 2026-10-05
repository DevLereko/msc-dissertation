"""
08_evaluation.py
Evaluates all models on the held-out test set.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
ARTIFACTS = Path("results/artifacts")
FIGURES = Path("results/figures")
TABLES = Path("results/tables")
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "total_inflow", "total_outflow", "avg_monthly_inflow", "avg_monthly_outflow",
    "inflow_volatility", "outflow_volatility", "transaction_frequency",
    "avg_balance", "min_balance", "max_balance", "balance_volatility",
    "active_months", "account_tenure_days",
    "loan_amount", "loan_duration", "loan_payment",
]

def main():
    test = pd.read_csv(PROCESSED / "test.csv")
    X_test = test[FEATURE_COLS].fillna(0)
    y_test = test["target"].values

    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    X_test_s = scaler.transform(X_test)

    lr  = joblib.load(ARTIFACTS / "logreg.pkl")
    rf  = joblib.load(ARTIFACTS / "rf.pkl")
    xgb = joblib.load(ARTIFACTS / "xgb.pkl")

    results = {}
    preds = {}
    for name, model, X in [("Logistic Regression", lr, X_test_s),
                           ("Random Forest", rf, X_test),
                           ("XGBoost", xgb, X_test)]:
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1":        f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, y_proba),
            "PR-AUC":    average_precision_score(y_test, y_proba),
            "Brier":     brier_score_loss(y_test, y_proba),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        preds[name] = y_proba

    df = pd.DataFrame(results).T
    df.to_csv(TABLES / "model_performance.csv")
    print(df.to_string())

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_proba in [("LR", preds["Logistic Regression"]),
                           ("RF", preds["Random Forest"]),
                           ("XGB", preds["XGBoost"])]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.savefig(FIGURES / "fig_4_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_proba in [("LR", preds["Logistic Regression"]),
                           ("RF", preds["Random Forest"]),
                           ("XGB", preds["XGBoost"])]:
        p, r, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(r, p, label=f"{label} (AP={average_precision_score(y_test, y_proba):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.savefig(FIGURES / "fig_4_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save y_test and predictions for later phases
    np.save(ARTIFACTS / "y_test.npy", y_test)
    np.save(ARTIFACTS / "y_proba_logreg.npy", preds["Logistic Regression"])
    np.save(ARTIFACTS / "y_proba_rf.npy", preds["Random Forest"])
    np.save(ARTIFACTS / "y_proba_xgb.npy", preds["XGBoost"])

    print("\nPredictions and figures saved.")

if __name__ == "__main__":
    main()