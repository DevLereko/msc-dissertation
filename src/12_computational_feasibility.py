"""
12_computational_feasibility.py
Computational feasibility of the corrected models (Section 3.7.5 C).

Measurement procedure (recorded in the output):
- training time: wall-clock fit of the tuned model on the training partition
  (from 07, single fit);
- inference time: median of REPEATS timed predict_proba calls on the
  validation partition, reported per loan;
- explanation time: SHAP for the full validation partition and LIME for
  LIME_N single loans, reported per loan;
- memory: peak resident set size of this process.
The test partition is not used.
"""

import json
import os
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import shap
import xgboost
from lime.lime_tabular import LimeTabularExplainer

from config import FEASIBILITY, FEATURE_COLS, MODELS, RANDOM_SEED, SPLITS
from modeling import MODEL_NAMES

import importlib
shap_explanation = importlib.import_module("10_xai_analysis").shap_explanation

REPEATS = 50
LIME_N = 5


def cpu_model():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def main():
    proc = psutil.Process(os.getpid())
    train = pd.read_csv(SPLITS / "train.csv")
    val = pd.read_csv(SPLITS / "val.csv")
    X_tr, X_va = train[FEATURE_COLS], val[FEATURE_COLS]
    training_times = json.loads((MODELS / "training_times.json").read_text())

    results = {}
    for name in MODEL_NAMES:
        model = joblib.load(MODELS / f"{name}.pkl")
        model.predict_proba(X_va)  # warm-up
        t = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            model.predict_proba(X_va)
            t.append(time.perf_counter() - t0)
        inf_ms = np.median(t) / len(X_va) * 1000

        t0 = time.perf_counter()
        shap_explanation(name, model, X_tr, X_va)
        shap_ms = (time.perf_counter() - t0) / len(X_va) * 1000

        lime = LimeTabularExplainer(X_tr.to_numpy(), feature_names=FEATURE_COLS,
                                    mode="classification", random_state=RANDOM_SEED)
        fn = lambda a, m=model: m.predict_proba(pd.DataFrame(a, columns=FEATURE_COLS))
        t0 = time.perf_counter()
        for i in range(LIME_N):
            lime.explain_instance(X_va.iloc[i].to_numpy(), fn, num_features=10, num_samples=5000)
        lime_ms = (time.perf_counter() - t0) / LIME_N * 1000

        results[name] = {
            "training_time_s": training_times[name],
            "inference_ms_per_loan": round(float(inf_ms), 4),
            "shap_ms_per_loan": round(float(shap_ms), 2),
            "lime_ms_per_loan": round(float(lime_ms), 1),
            "model_file_kb": round((MODELS / f"{name}.pkl").stat().st_size / 1024, 1),
        }

    mem = proc.memory_info()
    report = {
        "models": results,
        "peak_rss_mb": round(getattr(mem, "peak_wset", mem.rss) / 1024 ** 2, 1),
        "hardware": {
            "cpu_model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
            "platform": platform.platform(),
        },
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "shap": shap.__version__},
        "procedure": {"inference": f"median of {REPEATS} predict_proba calls on validation partition",
                      "shap": "one pass over validation partition, training-partition background",
                      "lime": f"mean over {LIME_N} loans, 5000 samples each",
                      "training": "single fit on training partition, from 07"},
    }
    (FEASIBILITY / "computational_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
