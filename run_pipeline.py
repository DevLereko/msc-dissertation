"""
run_pipeline.py
Runs the full framework end-to-end. Every number and figure in Chapter 4 can
be regenerated with a single command.

    ./venv/bin/python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("A-D1", "01_data_loading.py",            "Load and verify raw PKDD '99 tables"),
    ("D2",   "02_dataset_construction.py",    "Build loan-level analytical dataset"),
    ("D3",   "04_feature_engineering.py",     "Behavioural feature engineering"),
    ("E3",   "03_eda.py",                     "Exploratory data analysis (figures)"),
    ("D4",   "05_leakage_validation.py",      "Temporal leakage validation"),
    ("F",    "06_data_splitting.py",          "Chronological train/val/test split"),
    ("G",    "07_model_training.py",          "Train LR, RF, XGBoost + tuning"),
    ("H",    "08_evaluation.py",              "Test-set evaluation + ROC/PR curves"),
    ("I",    "09_calibration.py",             "Calibration analysis"),
    ("J",    "10_xai_analysis.py",            "SHAP analysis"),
    ("K",    "11_stability_analysis.py",      "SHAP stability across folds"),
    ("L",    "12_computational_feasibility.py","Computational feasibility report"),
]

def main():
    t_start = time.time()
    failed = []
    for tag, script, desc in STEPS:
        print(f"\n{'='*70}\n[{tag}] {desc}  ({script})")
        r = subprocess.run([PYTHON, str(ROOT / "src" / script)], cwd=ROOT)
        if r.returncode != 0:
            failed.append(script)
            print(f"[{tag}] FAILED: {script} (exit {r.returncode})")
        else:
            print(f"[{tag}] OK")
    elapsed = time.time() - t_start
    print(f"\n{'='*70}\nPipeline finished in {elapsed:.1f}s")
    if failed:
        print("FAILED steps:", failed)
        sys.exit(1)
    print("All steps passed.")

if __name__ == "__main__":
    main()