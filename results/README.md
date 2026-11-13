# Results

| Folder | Status | Contents |
|---|---|---|
| `corrected_v2/` | **Current — reported in the dissertation** | Outputs of `run_pipeline.py` (steps 01–12) |
| `baseline_v1/` | Superseded, audit only | Snapshot of the original experiment (682 loans, inclusive cutoff, shuffled CV) |
| `corrected_v1/` | Historical | Feasibility investigation run on the baseline data before the corrections |
| `audit/` | Historical / supporting | Read-only audit of the baseline, plus the prior-loan feasibility audit (`13_prior_loan_feasibility.py`) |

## `corrected_v2/` layout

| Subfolder | Written by | Contents |
|---|---|---|
| `data/` | 02, 03, 04, 05 | Cohort, analytical dataset (234 × 21), feature-engineering report, EDA tables, leakage checks |
| `figures/` | 03 | EDA figures |
| `splits/` | 06 | Train / validation / test CSVs, split metadata, candidate boundaries |
| `models/` | 07 | Tuned models (`*.pkl`, not versioned), selection record, tuning log, CV feasibility, training times |
| `evaluation/` | 08, 09 | Test metrics with bootstrap CIs, predictions, ROC/PR/calibration figures, test access log |
| `xai/` | 10 | SHAP summary, bar, dependence and waterfall plots; LIME plots; global importance for all models |
| `stability/` | 11 | Refit stability, SHAP-vs-LIME comparison, similar-observation consistency |
| `feasibility/` | 12 | Timing, memory, hardware and software versions |

Model files (`*.pkl`) and arrays (`*.npy`) are excluded from git; re-run
`run_pipeline.py` to regenerate them.
