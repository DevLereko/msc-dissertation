"""
09_calibration.py
Calibration curves and Brier score analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ARTIFACTS = Path("results/artifacts")
FIGURES = Path("results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

def main():
    y_test = np.load(ARTIFACTS / "y_test.npy")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["logreg", "rf", "xgb"]:
        y_proba = np.load(ARTIFACTS / f"y_proba_{name}.npy")
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        brier = brier_score_loss(y_test, y_proba)
        ax.plot(prob_pred, prob_true, marker="o", label=f"{name.upper()} (Brier={brier:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curves")
    ax.legend()
    fig.savefig(FIGURES / "fig_4_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Persist Brier scores for the report
    from sklearn.metrics import roc_auc_score, average_precision_score
    with open("results/tables/calibration.txt", "w") as f:
        for name in ["logreg", "rf", "xgb"]:
            y_proba = np.load(ARTIFACTS / f"y_proba_{name}.npy")
            f.write(f"{name.upper()} brier={brier_score_loss(y_test, y_proba):.4f}\n")
    print("Calibration figure saved. Brier scores written to results/tables/calibration.txt")

if __name__ == "__main__":
    main()