"""
11_stability_analysis.py
Explanation stability and consistency (Section 3.7.5 B).

Corrected v2 — stability is measured across INDEPENDENTLY REFITTED models,
not repeated explanations of one fitted model:

1. Refit stability: the final model's frozen hyperparameters are refitted on
   N_REFITS bootstrap resamples of the training partition (XGBoost class weight
   recomputed per resample). Each refit is explained on the same reference set
   (validation partition) and pairwise Spearman rho of mean |SHAP| rankings is
   reported.
2. Cross-technique consistency: SHAP vs LIME global rankings for the frozen
   final model on the validation partition.

The test partition is not used.
"""

import json

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from config import FEATURE_COLS, MODELS, RANDOM_SEED, SPLITS, STABILITY
from modeling import DISPLAY, fit_model

import importlib
shap_explanation = importlib.import_module("10_xai_analysis").shap_explanation

N_REFITS = 20
N_LIME = 48            # all validation loans


def pairwise_spearman(mat):
    rhos = [spearmanr(mat[i], mat[j])[0]
            for i in range(len(mat)) for j in range(i + 1, len(mat))]
    return (float(np.mean(rhos)), float(np.std(rhos)), float(np.min(rhos)),
            float(np.max(rhos)), float(np.median(rhos)))


def main():
    sel = json.loads((MODELS / "selection.json").read_text())
    final, params = sel["final_model"], sel["best_params"][sel["final_model"]]
    train = pd.read_csv(SPLITS / "train.csv")
    val = pd.read_csv(SPLITS / "val.csv")
    X_tr, y_tr, X_va = train[FEATURE_COLS], train["target"], val[FEATURE_COLS]

    # ---- 1. refit stability ------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    imps, skipped = [], 0
    while len(imps) < N_REFITS:
        idx = rng.integers(0, len(X_tr), len(X_tr))
        if y_tr.iloc[idx].sum() == 0:
            skipped += 1
            continue
        m = fit_model(final, params, X_tr.iloc[idx], y_tr.iloc[idx])
        e = shap_explanation(final, m, X_tr.iloc[idx], X_va)
        imps.append(np.abs(e.values).mean(axis=0))
    imps = np.array(imps)
    mean_rho, sd_rho, min_rho, max_rho, median_rho = pairwise_spearman(imps)
    ranks = pd.DataFrame(imps, columns=FEATURE_COLS).rank(axis=1, ascending=False)
    per_feature = pd.DataFrame({
        "mean_abs_shap": imps.mean(axis=0),
        "sd_abs_shap": imps.std(axis=0),
        "mean_rank": ranks.mean().values,
        "rank_sd": ranks.std().values,
        "top5_frequency": (ranks <= 5).mean().values,
    }, index=FEATURE_COLS).sort_values("mean_rank")
    per_feature.to_csv(STABILITY / "refit_feature_stability.csv")

    # ---- 2. SHAP vs LIME on the frozen final model ---------------------------
    model = joblib.load(MODELS / f"{final}.pkl")
    sv_val = shap_explanation(final, model, X_tr, X_va).values
    shap_imp = np.abs(sv_val).mean(axis=0)
    lime = LimeTabularExplainer(X_tr.to_numpy(), feature_names=FEATURE_COLS,
                                mode="classification", random_state=RANDOM_SEED,
                                discretize_continuous=False)
    predict_fn = lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURE_COLS))
    lime_w = np.zeros(len(FEATURE_COLS))
    n = min(N_LIME, len(X_va))
    for i in range(n):
        exp = lime.explain_instance(X_va.iloc[i].to_numpy(), predict_fn,
                                    num_features=len(FEATURE_COLS), num_samples=2000)
        for fidx, w in exp.as_map()[1]:
            lime_w[fidx] += abs(w)
    lime_imp = lime_w / n
    rho_sl, p_sl = spearmanr(shap_imp, lime_imp)
    pd.DataFrame({"shap_mean_abs": shap_imp, "lime_mean_abs_weight": lime_imp},
                 index=FEATURE_COLS).to_csv(STABILITY / "shap_vs_lime.csv")

    # ---- 3. explanation variability for similar observations ---------------
    # Each validation loan is paired with its nearest neighbour (Euclidean
    # distance on features standardised with training-partition statistics)
    # and, as a reference, with a randomly drawn validation loan. Similarity of
    # the two SHAP vectors is measured by Spearman rho. Similar loans should
    # receive more similar explanations than random pairs.
    mu, sd = X_tr.mean(), X_tr.std(ddof=0).replace(0, 1)
    Z = ((X_va - mu) / sd).to_numpy()
    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(axis=1)
    rng_pairs = np.random.default_rng(RANDOM_SEED)
    rand = np.array([rng_pairs.choice([j for j in range(len(Z)) if j != i]) for i in range(len(Z))])
    nn_rho = np.array([spearmanr(sv_val[i], sv_val[nn[i]])[0] for i in range(len(Z))])
    rand_rho = np.array([spearmanr(sv_val[i], sv_val[rand[i]])[0] for i in range(len(Z))])
    pd.DataFrame({"loan_id": val["loan_id"], "nearest_neighbour_loan_id": val["loan_id"].iloc[nn].values,
                  "nn_distance": dist[np.arange(len(Z)), nn], "nn_shap_rho": nn_rho,
                  "random_pair_shap_rho": rand_rho}).to_csv(STABILITY / "similar_observation_consistency.csv",
                                                            index=False)
    from scipy.stats import wilcoxon
    w_stat, w_p = wilcoxon(nn_rho, rand_rho)

    report = {
        "explained_model": final,
        "similar_observation_consistency": {
            "method": "nearest neighbour vs random pair within validation partition; "
                      "Spearman rho between SHAP vectors; Wilcoxon signed-rank test",
            "pairs": int(len(Z)),
            "nn_median_rho": round(float(np.median(nn_rho)), 4),
            "random_median_rho": round(float(np.median(rand_rho)), 4),
            "nn_mean_rho": round(float(np.mean(nn_rho)), 4),
            "random_mean_rho": round(float(np.mean(rand_rho)), 4),
            "wilcoxon_p": round(float(w_p), 4),
        },
        "refit_stability": {
            "method": f"{N_REFITS} bootstrap refits of training partition, frozen hyperparameters; "
                      "SHAP on validation partition; pairwise Spearman of mean |SHAP|",
            "mean_rho": round(mean_rho, 4), "sd_rho": round(sd_rho, 4), "min_rho": round(min_rho, 4),
            "max_rho": round(max_rho, 4), "median_rho": round(median_rho, 4),
            "n_pairs": N_REFITS * (N_REFITS - 1) // 2,
            "resamples_skipped_no_positives": skipped,
        },
        "shap_vs_lime": {
            "instances": n, "spearman_rho": round(float(rho_sl), 4), "p_value": round(float(p_sl), 4),
            "lime_settings": "num_samples=2000, discretize_continuous=False, training-partition background",
        },
        "top5_by_mean_rank": per_feature.head(5).index.tolist(),
    }
    (STABILITY / "stability_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
