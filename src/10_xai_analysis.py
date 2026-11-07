"""
10_xai_analysis.py
SHAP (global + local) and LIME (local) explanations of the FROZEN final model.

- The model explained is the one selected in 07 (models/selection.json), not a
  hard-coded algorithm.
- Background / reference data: the TRAINING partition (documented in output).
- Explained instances: the test partition (features only; no labels used here).
- Global SHAP importance is also tabulated for all three tuned models.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from config import FEATURE_COLS, MODELS, RANDOM_SEED, SPLITS, XAI
from modeling import DISPLAY, MODEL_NAMES


def shap_explanation(name, model, X_background, X_explain):
    """Return a shap.Explanation for the positive class."""
    if name == "logreg":
        scaler, lr = model.named_steps["scaler"], model.named_steps["model"]
        bg = pd.DataFrame(scaler.transform(X_background), columns=FEATURE_COLS)
        ex = shap.LinearExplainer(lr, bg)
        e = ex(pd.DataFrame(scaler.transform(X_explain), columns=FEATURE_COLS))
        e.data = X_explain.to_numpy()          # show raw feature values in plots
        return e
    ex = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional",
                            model_output="probability")
    e = ex(X_explain, check_additivity=False)
    if e.values.ndim == 3:                      # RF: (n, features, classes)
        e = e[:, :, 1]
    return e


def main():
    selection = json.loads((MODELS / "selection.json").read_text())
    final = selection["final_model"]
    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    X_bg, X_te = train[FEATURE_COLS], test[FEATURE_COLS]

    importance = {}
    explanations = {}
    for name in MODEL_NAMES:
        model = joblib.load(MODELS / f"{name}.pkl")
        e = shap_explanation(name, model, X_bg, X_te)
        explanations[name] = (model, e)
        importance[DISPLAY[name]] = np.abs(e.values).mean(axis=0)
    imp = pd.DataFrame(importance, index=FEATURE_COLS)
    imp.to_csv(XAI / "shap_global_importance_all_models.csv")
    imp.rank(ascending=False).to_csv(XAI / "shap_rank_all_models.csv")

    model, e = explanations[final]
    plt.figure()
    shap.plots.beeswarm(e, max_display=16, show=False)
    plt.title(f"SHAP summary — {DISPLAY[final]} (test partition)")
    plt.savefig(XAI / "fig_4_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(e, max_display=16, show=False)
    plt.savefig(XAI / "fig_4_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    proba = model.predict_proba(X_te)[:, 1]
    cases = {"highest_risk": int(np.argmax(proba)), "lowest_risk": int(np.argmin(proba))}
    for label, i in cases.items():
        plt.figure()
        shap.plots.waterfall(e[i], max_display=10, show=False)
        plt.savefig(XAI / f"fig_4_shap_waterfall_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # LIME for the same two loans
    lime = LimeTabularExplainer(X_bg.to_numpy(), feature_names=FEATURE_COLS,
                                class_names=["non-default", "default"],
                                mode="classification", random_state=RANDOM_SEED)
    predict_fn = lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURE_COLS))
    lime_out = {}
    for label, i in cases.items():
        exp = lime.explain_instance(X_te.iloc[i].to_numpy(), predict_fn,
                                    num_features=10, num_samples=5000)
        lime_out[label] = {"loan_id": int(test.loc[i, "loan_id"]),
                           "predicted_proba": round(float(proba[i]), 4),
                           "weights": [(f, round(float(w), 5)) for f, w in exp.as_list()]}
        exp.as_pyplot_figure().savefig(XAI / f"fig_4_lime_{label}.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    meta = {
        "explained_model": final,
        "shap_explainer": "LinearExplainer" if final == "logreg" else
                          "TreeExplainer (interventional, probability output)",
        "background_data": f"training partition ({len(X_bg)} loans)",
        "explained_instances": f"test partition ({len(X_te)} loans)",
        "local_cases": lime_out,
        "top5_final_model": imp[DISPLAY[final]].sort_values(ascending=False).head(5).round(4).to_dict(),
    }
    (XAI / "xai_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
