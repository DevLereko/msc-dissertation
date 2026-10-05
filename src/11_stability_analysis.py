"""
11_stability_analysis.py
Cross-fold stability of SHAP and LIME feature importance.
Note: The pipeline computes SHAP-based stability. LIME is available as an
extension but is not run by default (see README).
"""

import numpy as np
import pandas as pd
import joblib
import shap
from scipy.stats import spearmanr
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

PROCESSED = Path("data/processed")
ARTIFACTS = Path("results/artifacts")
TABLES = Path("results/tables")
TABLES.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "total_inflow", "total_outflow", "avg_monthly_inflow", "avg_monthly_outflow",
    "inflow_volatility", "outflow_volatility", "transaction_frequency",
    "avg_balance", "min_balance", "max_balance", "balance_volatility",
    "active_months", "account_tenure_days",
    "loan_amount", "loan_duration", "loan_payment",
]

def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["target"]

    xgb = joblib.load(ARTIFACTS / "xgb.pkl").best_estimator_

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importance_per_fold = []
    explainer = shap.TreeExplainer(xgb)

    for fold, (tr_idx, _) in enumerate(skf.split(X_train, y_train)):
        X_fold = X_train.iloc[tr_idx]
        shap_vals = explainer.shap_values(X_fold)
        imp = np.abs(shap_vals).mean(axis=0)
        importance_per_fold.append(imp)

    # Cross-fold Spearman correlation between feature-importance rankings
    correlations = []
    for i in range(len(importance_per_fold)):
        for j in range(i + 1, len(importance_per_fold)):
            rho, _ = spearmanr(importance_per_fold[i], importance_per_fold[j])
            correlations.append(rho)

    mean_rho = np.mean(correlations)
    std_rho = np.std(correlations)
    print(f"SHAP mean cross-fold Spearman correlation: {mean_rho:.4f} (std {std_rho:.4f})")

    mean_importance = np.mean(importance_per_fold, axis=0)
    importance_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_importance})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(TABLES / "shap_stability_importance.csv", index=False)

    with open(TABLES / "stability_shap.txt", "w") as f:
        f.write(f"Mean Spearman rho (SHAP): {mean_rho:.4f}\n")
        f.write(f"SD Spearman rho (SHAP): {std_rho:.4f}\n")

    print("Stability report saved to results/tables/stability_shap.txt")

if __name__ == "__main__":
    main()