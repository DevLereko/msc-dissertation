"""
run_pipeline.py
Runs the corrected (v2) experiment end-to-end. All outputs are written to
results/corrected_v2/; results/baseline_v1/ (the original experiment) is never
modified.

    ./venv/bin/python run_pipeline.py

Note: step H is the ONE-TIME test evaluation (Section 3.7.4 step 9). Every run
appends to results/corrected_v2/evaluation/test_access_log.jsonl, so repeated
test evaluations are visible in the audit trail.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("A",  "01_data_loading.py",              "Load and verify raw PKDD '99 tables"),
    ("B",  "02_dataset_construction.py",      "Completed-loan cohort (A/B) and target"),
    ("C",  "04_feature_engineering.py",       "Behavioural features (strict < loan_date)"),
    ("D",  "05_leakage_validation.py",        "Leakage-prevention assertions"),
    ("E",  "03_eda.py",                       "Exploratory data analysis"),
    ("F",  "06_data_splitting.py",            "Date-grouped chronological split"),
    ("G",  "07_model_training.py",            "Tuning, model selection, frozen threshold"),
    ("H",  "08_evaluation.py",                "ONE-TIME test evaluation of frozen models"),
    ("I",  "09_calibration.py",               "Calibration assessment"),
    ("J",  "10_xai_analysis.py",              "SHAP + LIME for the frozen final model"),
    ("K",  "11_stability_analysis.py",        "Refit stability and SHAP-vs-LIME consistency"),
    ("L",  "12_computational_feasibility.py", "Computational feasibility"),
]


def main():
    t_start = time.time()
    for tag, script, desc in STEPS:
        print(f"\n{'=' * 70}\n[{tag}] {desc}  ({script})")
        r = subprocess.run([PYTHON, str(ROOT / "src" / script)], cwd=ROOT)
        if r.returncode != 0:
            # Stop immediately: later steps depend on earlier outputs
            print(f"[{tag}] FAILED: {script} (exit {r.returncode}). Pipeline stopped.")
            sys.exit(1)
        print(f"[{tag}] OK")
    print(f"\n{'=' * 70}\nPipeline finished in {time.time() - t_start:.1f}s. All steps passed.")


if __name__ == "__main__":
    main()
