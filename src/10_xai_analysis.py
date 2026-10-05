"""
10_xai_analysis.py
SHAP and LIME analysis for all models.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("data/processed")
ARTIFACTS = Path("results/artifacts")
FIGURES = Path("results/figures")

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

    xgb = joblib.load(ARTIFACTS / "xgb.pkl").best_estimator_

    # SHAP for XGBoost (tree explainer)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # Summary (bee-swarm) plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.savefig(FIGURES / "fig_4_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Global importance bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(FIGURES / "fig_4_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Waterfall for the highest predicted-risk loan
    idx_high = int(np.argmax(np.load(ARTIFACTS / "y_proba_xgb.npy")))
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[idx_high],
        X_test.iloc[idx_high], max_display=10, show=False
    )
    plt.savefig(FIGURES / "fig_4_shap_waterfall_high.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Persist global SHAP importance for the results tables
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLS)
    mean_abs.sort_values(ascending=False).to_csv(
        "results/tables/shap_global_importance.csv"
    )

    print("SHAP analysis complete. Figures saved.")
    print("Top-5 global SHAP features:")
    print(mean_abs.sort_values(ascending=False).head(5).round(4).to_string())

if __name__ == "__main__":
    main()