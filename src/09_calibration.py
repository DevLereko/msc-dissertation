"""
09_calibration.py
Calibration ASSESSMENT (Brier score, reliability curves).

No calibrator (Platt / isotonic) is fitted, so no labels are used to change
any model. Curves are shown for the validation partition (used during
selection) and the test partition (from 08's single evaluation).
Quantile bins are used because the partitions are small.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from config import EVALUATION, FEATURE_COLS, MODELS, SPLITS
from modeling import DISPLAY, MODEL_NAMES

N_BINS = 5


def main():
    val = pd.read_csv(SPLITS / "val.csv")
    pred = pd.read_csv(EVALUATION / "test_predictions.csv")  # produced by 08

    sets = {"validation": (val["target"].to_numpy(),
                           {n: joblib.load(MODELS / f"{n}.pkl").predict_proba(val[FEATURE_COLS])[:, 1]
                            for n in MODEL_NAMES}),
            "test": (pred["target"].to_numpy(),
                     {n: pred[f"proba_{n}"].to_numpy() for n in MODEL_NAMES})}

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (part, (y, probs)) in zip(axes, sets.items()):
        out[part] = {"rows": int(len(y)), "defaults": int(y.sum()), "brier": {}}
        for n in MODEL_NAMES:
            p = probs[n]
            out[part]["brier"][n] = round(float(brier_score_loss(y, p)), 4)
            pt, pp = calibration_curve(y, p, n_bins=N_BINS, strategy="quantile")
            ax.plot(pp, pt, marker="o", label=f"{DISPLAY[n]} (Brier={out[part]['brier'][n]:.3f})")
        # Reference: Brier of always predicting the partition prevalence
        out[part]["brier_prevalence_baseline"] = round(float(y.mean() * (1 - y.mean())), 4)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set(title=f"{part.capitalize()} (n={len(y)}, defaults={int(y.sum())})",
               xlabel="Mean predicted probability", ylabel="Observed default rate")
        ax.legend()
    fig.suptitle(f"Reliability curves ({N_BINS} quantile bins)")
    fig.savefig(EVALUATION / "fig_4_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    out["method"] = f"assessment only; no calibrator fitted; {N_BINS} quantile bins"
    (EVALUATION / "calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
