# Explainable Loan Default Prediction Framework

Code and results for the MSc dissertation *An Explainable Machine Learning
Framework for Loan Default Prediction for Financial Institutions in Lesotho*
(Lereko Foloko, Botho University, supervised by Prof. Kishor Kumar Reddy).

The framework turns relational banking data (customers, accounts,
transactions, loans) into leakage-controlled behavioural features. It then
compares Logistic Regression, Random Forest and XGBoost under a time-aware
protocol, and explains the selected model with SHAP and LIME. The PKDD '99
Czech Banking Dataset (Berka, 1999) is used as the reference institutional
schema.

## Experiment at a glance (corrected v2)

| Item | Value |
|---|---|
| Cohort | Completed loans only: status A → 0 (203), status B → 1 (31); 234 loans |
| Features | 16 (13 behavioural + 3 loan attributes), defined in `src/config.py` |
| Observation window | Transactions dated **strictly before** the loan issuance date |
| Split | Chronological on unique issuance dates: train 141 (20 defaults) / validation 48 (8) / test 45 (3) |
| Tuning | Forward-chaining CV declared infeasible (fold defaults below 5); hyperparameters tuned on the validation partition |
| Selection | Validation PR-AUC (ties: ROC-AUC); final model **Random Forest**, threshold 0.5 |
| Test evaluation | Run once; every access logged in `results/corrected_v2/evaluation/test_access_log.jsonl` |
| Explanations | SHAP (global, dependence, waterfall), LIME (local), refit stability, SHAP-vs-LIME, similar-observation consistency |

Methodological decisions and their approval trail are recorded in
[docs/SCOPE_ADJUSTMENT_v1.md](docs/SCOPE_ADJUSTMENT_v1.md). The feature set is
reconciled against the dissertation in
[docs/FEATURE_RECONCILIATION_v1.md](docs/FEATURE_RECONCILIATION_v1.md).

## Repository structure

```
run_pipeline.py              Runs the full corrected (v2) experiment
requirements.txt             Pinned package versions
src/
  config.py                  Paths, feature list, protocol constants (single source of truth)
  modeling.py                Model construction shared by training and stability analysis
  01_data_loading.py         Load and verify the 8 raw PKDD '99 tables
  02_dataset_construction.py Completed-loan cohort and target
  03_eda.py                  Exploratory data analysis
  04_feature_engineering.py  16 features from the strict pre-issuance window
  05_leakage_validation.py   Leakage-prevention assertions (Table 5)
  06_data_splitting.py       Date-grouped chronological split
  07_model_training.py       CV feasibility gate, tuning, model selection, frozen threshold
  08_evaluation.py           One-time test evaluation with bootstrap CIs
  09_calibration.py          Calibration assessment (no recalibration)
  10_xai_analysis.py         SHAP and LIME for the selected model
  11_stability_analysis.py   Refit stability and explanation consistency
  12_computational_feasibility.py  Timing, memory and hardware report
  13_prior_loan_feasibility.py     Audit showing prior-loan history cannot be built (writes results/audit/)
  archive/                   One-off investigation script kept for the audit trail (not part of the pipeline)
data/
  raw/                       Original PKDD '99 .asc files (not versioned)
  processed/                 CSV copies of the raw tables written by step 01 (not versioned)
results/
  corrected_v2/              Results reported in the dissertation (see results/README.md)
  baseline_v1/               Snapshot of the original, superseded experiment (audit only)
  corrected_v1/              Pre-correction feasibility investigation outputs
  audit/                     Read-only audit of the baseline and prior-loan feasibility
docs/
  SCOPE_ADJUSTMENT_v1.md     Supervisor-approved scope decisions
  FEATURE_RECONCILIATION_v1.md  Table 7 ↔ code reconciliation
  drafts/                    Working drafts of Chapters 4–6
  history/                   Earlier audit report and gap-closure plan
```

## Setup

Tested with Python 3.11.17 on macOS (Apple M4).

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the eight PKDD '99 `.asc` files (`account`, `card`, `client`, `disp`,
`district`, `loan`, `order`, `trans`) in `data/raw/`. The original sources
(relational.fit.cvut.cz / sorry.vse.cz) were unreachable from the working
environment, so the files were retrieved from the public GitHub mirror
`jlacko/berka-dataset`, which reproduces the original dataset byte-for-byte.
Step 01 checks every table against its documented record count.

## Reproduce the results

From the repository root:

```bash
./venv/bin/python run_pipeline.py
```

This runs steps 01–12 in order, stops at the first failing step, and writes
everything to `results/corrected_v2/`. Every step asserts its own checks (for
example, the strict cutoff, disjoint partitions and no missing values), so a
completed run means all checks passed.

Note on the test partition: the dissertation's test results come from a
single evaluation, recorded as the first line of
`results/corrected_v2/evaluation/test_access_log.jsonl`. Re-running the
pipeline evaluates the test partition again and appends a new line. All
randomised steps use fixed seeds. A from-scratch run in a clean copy of the
repository (13 Nov 2026) reproduced every reported table and metric exactly.
Predicted probabilities differed only at floating-point precision (≤ 1e-16).
The log shows that such a re-run was a reproducibility check.

Individual steps can also be run on their own, for example
`./venv/bin/python src/07_model_training.py`. Each step reads the outputs of
earlier steps from `results/corrected_v2/`.

## Where each dissertation result comes from

| Dissertation section | File in `results/corrected_v2/` |
|---|---|
| 4.2.1 Data verification | `data/processed/verification.json` (repo root) |
| 4.2.2–4.2.3 Cohort and final dataset | `data/cohort_summary.json`, `data/feature_engineering_report.json` |
| 4.2.4 Exploratory analysis | `data/defaults_by_year.csv`, `data/eda_medians_by_class.csv`, `figures/` |
| 4.3.2 Leakage validation | `data/leakage_validation.json` |
| 4.3.3 Partitions | `splits/split_meta.json`, `splits/candidate_boundaries.csv` |
| 4.3.4 CV feasibility | `models/cv_feasibility.json` |
| 4.3.5 Tuning and selection | `models/selection.json`, `models/tuning_log.csv` |
| 4.4.1 Test performance | `evaluation/test_performance.csv`, `evaluation/fig_4_roc_curves.png`, `evaluation/fig_4_pr_curves.png` |
| 4.4.2 Calibration | `evaluation/calibration.json`, `evaluation/fig_4_calibration_curves.png` |
| 4.4.3–4.4.4 Explanations | `xai/` |
| 4.4.5 Stability and consistency | `stability/` |
| 4.4.6 Computational feasibility | `feasibility/computational_report.json` |

## Superseded experiment

`results/baseline_v1/` keeps the outputs of the original pipeline: all 682
loans with statuses A/C → 0 and B/D → 1, an inclusive `date <= loan_date`
cutoff, a positional split, and shuffled cross-validation. It is kept so that
differences from the corrected experiment can be explained. Its numbers are
not comparable with, and are not reported as, the dissertation's results.
