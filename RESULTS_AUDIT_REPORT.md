# Results Audit Report

Audit scope: read-only Phase 1 and Phase 2. Source code and original data/results were not changed. The report distinguishes observations from interpretation. Existing model pickles were loaded for a read-only prediction comparison and latency measurement; no fitting or pipeline step was rerun.

## A. Executive Summary and Overall Readiness

The repository contains a functioning, reproducible computational prototype using the public PKDD '99 Czech Banking data. The reported source counts, loan population, chronological split counts, best parameters, test metrics, confusion matrices, SHAP top-five ranking and rounded SHAP stability statistic are supported by the available files. Saved probability arrays and labels match persisted-model inference and the processed test labels.

Readiness is **not yet sufficient for a defensible final empirical claim of leakage-controlled pre-issuance prediction or robust generalisation**. The most consequential unresolved point is that transaction windows include dates equal to loan issuance, and 122 loans have at least one such transaction; dates have day resolution, so temporal availability cannot be established. The model-selection procedure also uses non-temporal stratified CV, and logistic-regression scaling is fitted before those folds. The validation set is not used. The test set has only three defaults, so its positive-class estimates are extremely fragile. The stability statistic describes one full-training XGBoost model over overlapping subsets, not model/explanation stability over refitted folds. The code and outputs do not demonstrate Lesotho validity, fairness, decision-maker comprehension or credit-loss reduction.

No methodology change is made here. The required detailed gap-closure plan is deferred because this request explicitly limits work to Phases 1 and 2; ranked issues and immediate resolution directions are listed in section P.

## B. Repository Structure and Exact Reproduction Command

The entry point is [run_pipeline.py](run_pipeline.py). It executes `src/01_data_loading.py`, `02_dataset_construction.py`, `04_feature_engineering.py`, `03_eda.py`, `05_leakage_validation.py`, `06_data_splitting.py`, `07_model_training.py`, `08_evaluation.py`, `09_calibration.py`, `10_xai_analysis.py`, `11_stability_analysis.py` and `12_computational_feasibility.py` in sequence. The README gives the same sequence and the pinned direct dependencies.

From the repository root, the exact reproduction command is:

```sh
./venv/bin/python run_pipeline.py
```

This command overwrites multiple processed CSVs, result tables, figures and serialized artifacts. It was **not** run during this audit. All eight raw `.asc` files are currently present. See [results/audit/reproduction_environment.txt](results/audit/reproduction_environment.txt) for the observed environment and timing methods.

## C. Dataset Verification and Analytical Population

The processed CSVs were recounted directly. Each agrees with `data/processed/verification.json` and the expected count embedded in `src/01_data_loading.py`:

| Table       |      Rows |
| ----------- | --------: |
| client      |     5,369 |
| account     |     4,500 |
| disposition |     5,369 |
| transaction | 1,056,320 |
| order       |     6,471 |
| loan        |       682 |
| card        |       892 |
| district    |        77 |

The analytical data contain 682 rows, 682 unique loan IDs, 682 unique account IDs, no duplicate rows, no duplicate loan IDs and no missing cells. The observed target counts are 606 non-defaults and 76 defaults. The source status counts and mapped targets are A: 203 -> 0, B: 31 -> 1, C: 403 -> 0, D: 45 -> 1, matching [src/02_dataset_construction.py](src/02_dataset_construction.py) and `loan_summary.json`.

The population is one loan per account in the analytical dataset. No loan ID or account ID overlaps across train, validation and test. This does not establish general entity independence beyond those identifiers or independence for future institution data.

## D. Analytical Column Inventory

The 20 columns reconcile as follows. The model uses 16 predictor columns; identifiers and date are not passed to the classifiers.

| Group                         | Columns                                                                                                                                                                                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identifiers (2)               | `loan_id`, `account_id`                                                                                                                                                                                                                                     |
| Date (1)                      | `loan_date`                                                                                                                                                                                                                                                 |
| Behavioural predictors (13)   | `total_inflow`, `total_outflow`, `avg_monthly_inflow`, `avg_monthly_outflow`, `inflow_volatility`, `outflow_volatility`, `transaction_frequency`, `avg_balance`, `min_balance`, `max_balance`, `balance_volatility`, `active_months`, `account_tenure_days` |
| Loan-attribute predictors (3) | `loan_amount`, `loan_duration`, `loan_payment`                                                                                                                                                                                                              |
| Target (1)                    | `target`                                                                                                                                                                                                                                                    |
| Other metadata (0)            | None                                                                                                                                                                                                                                                        |

Thus, 13 behavioural + 3 loan attributes + target is 17 columns, not 20. The additional three are two identifiers and the loan date. The 20-column count is correct; the earlier feature-count description omitted these non-predictor columns.

## E. Feature Definitions and Leakage/Time-Window Audit

[src/04_feature_engineering.py](src/04_feature_engineering.py) filters each account's transactions using `date <= loan_date`. Inflows/outflows sum transaction amounts by `PRIJEM`/`VYDAJ`; average monthly amounts divide the corresponding total by the number of months with any transaction; volatility is the sample standard deviation of observed monthly transaction sums; balance mean/min/max/std summarize transaction balance observations; frequency is transaction-row count; active months count months with a transaction; tenure is days from the earliest included transaction to the loan date.

The code does not create a complete calendar-month series: months with no activity are absent from monthly group summaries and the monthly denominator is the count of months with any transaction. `account_tenure_days` is based on earliest observed transaction, not an independently verified account-opening date. These definitions should be reported as implemented, not described more broadly as account-level income or account age.

The status/target fields and IDs/date are not in the explicit 16-column model feature list. `loan_amount`, `loan_duration` and `loan_payment` are included. Their availability is valid only if the prediction point is after these proposed contract terms are known; the repository does not define that operational decision point precisely.

The critical temporal concern is the inclusive same-day boundary. There are 122 loans with one or more transactions dated on the loan date (166 such transaction rows). Since the available dates have day resolution, the repository cannot prove that these transactions occurred before issuance. The existing leakage check in [src/05_leakage_validation.py](src/05_leakage_validation.py) independently recounts transaction frequency for a random sample of 50 accounts (seed 42), but it repeats the same inclusive date comparison and does not independently reconstruct all features for all rows. It therefore verifies code consistency, not strict within-day pre-issuance availability.

No test resampling or class rebalancing is evidenced. There are no learned imputations or feature-selection steps. Constant `fillna(0)` is used; the actual analytical dataset has no missing cells. However, LR `StandardScaler` is fit on all training rows before CV, rather than within each CV fold. XGBoost `scale_pos_weight` is computed using all training labels before CV; this does not use validation/test labels, but inner-fold validation labels contribute to the fixed ratio.

## F. Chronological Split Dates, Row Counts and Class Distributions

[src/06_data_splitting.py](src/06_data_splitting.py) sorts by `loan_date` and takes positional 60%/20%/20% slices. The saved data and split metadata agree:

| Partition  | Rows | Dates inclusive          | Class 0 | Class 1 |
| ---------- | ---: | ------------------------ | ------: | ------: |
| Train      |  409 | 1993-07-05 to 1997-06-13 |     354 |      55 |
| Validation |  136 | 1997-06-13 to 1998-01-27 |     118 |      18 |
| Test       |  137 | 1998-01-27 to 1998-12-08 |     134 |       3 |

One row on 1997-06-13 is in both train and validation; one row on 1998-01-27 is in both validation and test. These are tied calendar dates, not evidence of a later date moving backward, but the exact order of same-day issuance is unavailable and the split boundary is not a clean date cutoff. Loan and account IDs are disjoint across partitions. The primary test distribution has not been resampled.

## G. Model Selection/Tuning Procedure and Best Parameters

[src/07_model_training.py](src/07_model_training.py) performs grid search with ROC-AUC scoring and `StratifiedKFold(5)` for all three models. The model searches and persisted `best_params.json` agree:

| Model               | Verified best parameters                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Logistic Regression | `C=0.1`; `class_weight="balanced"`; `max_iter=2000`                                       |
| Random Forest       | `max_depth=20`, `min_samples_split=5`, `n_estimators=100`; `class_weight="balanced"`      |
| XGBoost             | `learning_rate=0.1`, `max_depth=3`, `n_estimators=200`; `scale_pos_weight=354/55=6.43636` |

The grids are LR C `{0.01, 0.1, 1, 10, 100}`; RF estimators `{100, 200}`, depth `{10, 20, None}`, split `{2, 5}`; and XGBoost estimators `{100, 200}`, learning rate `{0.05, 0.1}`, depth `{3, 5, 7}`. `random_state=42` is set for estimators; RF/XGBoost use `n_jobs=-1`. Stratified folds are not forward-chaining temporal folds and can train on later training-period rows when evaluating earlier rows. The LR scaler is fitted before CV. Thus the available CV score selection is not a temporal generalisation estimate and has fold-level preprocessing contamination for LR.

The validation dataset is loaded and transformed, but it is not passed to any search, comparison or threshold selection. The best estimator from each grid search is refit on all training rows. No validation-based model-selection record is saved. Threshold decisions are the classifiers' default 0.5 decisions; there is no documented threshold search on validation. The test labels are not used for threshold tuning in the saved evaluation path.

## H. Verified Performance Table and Confusion Matrices

All saved `y_test.npy` labels and probability arrays have 137 elements. `y_test.npy` equals the target column of `data/processed/test.csv`. Persisted-model inference reproduces LR and XGBoost probabilities exactly; RF differs by at most `1.11e-16`, with identical 0.5 predictions. Metrics were recomputed with positive label 1, default threshold 0.5, precision/recall/F1 binary averaging, ROC-AUC from probabilities, average precision as PR-AUC, and standard `[[TN, FP], [FN, TP]]` order.

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC/AP |   Brier | Confusion matrix      |
| ------------------- | -------: | --------: | -----: | -----: | ------: | --------: | ------: | --------------------- |
| Logistic Regression |   0.8394 |    0.0870 | 0.6667 | 0.1538 |  0.9179 |    0.6944 | 0.12685 | `[[113, 21], [1, 2]]` |
| Random Forest       |   0.9781 |    0.5000 | 0.6667 | 0.5714 |  0.9950 |    0.8667 | 0.03200 | `[[132, 2], [1, 2]]`  |
| XGBoost             |   0.9270 |    0.2308 | 1.0000 | 0.3750 |  0.9950 |    0.8056 | 0.04188 | `[[124, 10], [0, 3]]` |

The values match `results/tables/model_performance.csv` within stored precision. The raw artifact matrices are standard sklearn ordering; the README summary's compressed notation is ambiguous/malformed for LR, but the underlying table and reconstructed matrices are consistent.

There are only three positive test labels. One case changes recall by 1/3; XGBoost's 3/3 recall is not robust evidence of general recall. Precision estimates are similarly unstable. High ranking scores are determined by ordering of only three positive cases among 134 negatives. Accuracy is dominated by the majority class. These results describe this one held-out slice only and should not be presented as reliable population estimates.

## I. Calibration Results and Interpretation

The saved test Brier scores are LR 0.12685, RF 0.03200 and XGBoost 0.04188. The Brier values in `calibration.txt` round correctly. `src/09_calibration.py` plots 10 uniform bins using the test labels, but there are only three positives, so bin-level observed frequencies are sparse and sensitive to individual observations. No uncertainty interval, calibration slope or calibration-in-the-large is provided.

As a reference, predicting the test prevalence (3/137) for every case yields Brier approximately 0.02142. All three model Brier scores are higher than this constant-prevalence reference. Brier score is a proper overall probabilistic score, not a pure calibration-only measure; these outputs do not support the summary's unqualified statement that tree ensembles are notably better calibrated. RF has the lowest model Brier score, but its small-test performance does not establish reliable calibration.

## J. Global SHAP, Local SHAP, LIME and Limitations

`src/10_xai_analysis.py` explains the persisted best XGBoost estimator using `TreeExplainer` on all 137 test feature rows. The stored mean absolute SHAP ranking is:

| Rank | Feature             | Mean absolute SHAP |
| ---: | ------------------- | -----------------: |
|    1 | `avg_balance`       |          1.1969458 |
|    2 | `inflow_volatility` |          1.1815703 |
|    3 | `loan_payment`      |          0.7414500 |
|    4 | `loan_amount`       |         0.67443484 |
|    5 | `min_balance`       |          0.6239102 |

The ranking matches the reported top five. Summary (beeswarm) and bar plots exist. A single waterfall is selected for the test loan with the highest saved XGBoost risk probability. No SHAP dependence plot, LIME output, multi-case local explanation table or explanation-method comparison exists. Although LIME is listed as a dependency and named in script comments, it is not implemented in the executed workflow.

Because global importance is calculated on test feature rows and the local example is chosen from test predictions, the test cohort influenced the explanation outputs. This does not appear to feed back into model fitting or performance metrics, but it conflicts with a strict rule that test data not influence explanation choices. SHAP values are model- and background-dependent attributions; they do not establish causality, fairness, user comprehension or decision validity.

## K. Explanation Stability Procedure, Values, Data and Interpretation

The reported statistic is reproducible to rounding. Recomputing the source procedure gives 10 pairwise Spearman correlations with mean `0.9970588`; population SD (NumPy default `ddof=0`) is `0.0018602`, reported as `0.9971 (SD 0.0019)`. The five subsets are `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` training-index subsets of the 409 training rows. They are not independent subsamples: their training subsets overlap substantially.

Crucially, `src/11_stability_analysis.py` loads one XGBoost estimator already fit on the full training set, creates one explainer, then explains each subset. It does not refit a model within each fold. The high correlation therefore indicates similar global feature-importance rankings for overlapping subsets explained by the same fitted model; it does not quantify training/model instability across folds or across explanation techniques. No local explanation variability or similar-case analysis is implemented. Do not characterize 0.9971 as broad explanation stability.

## L. Computational Feasibility

The persisted report records Python 3.11.17, macOS 26.6 arm64, Apple M4, 10 physical/logical cores, and training times LR 2.1677 s, RF 1.4926 s and XGBoost 0.4275 s. These times surround each `GridSearchCV.fit`, including candidate fits and final refit. They are single-run wall-clock values. The summary's approximately 18 s full-pipeline time is not backed by a saved run log.

The original `memory_usage_mb=17.02` is a one-time RSS reading by the short final reporting process in `src/12_computational_feasibility.py`. It is not a peak training or pipeline memory measurement. In a read-only audit benchmark process that loaded the persisted estimators and ran SHAP, the process high-water RSS was 251.16 MiB; this includes interpreter/dependency/model memory and is not a measured full-pipeline peak.

Additional audit-session latency measurements, not present in the original pipeline: median single-row `predict_proba` was 0.0181 ms LR, 14.0504 ms RF and 0.4991 ms XGBoost; means were 0.0193, 13.7780 and 0.5706 ms respectively. One 137-row batch call averaged 0.000407, 0.09251 and 0.004605 ms per prediction, respectively. The large RF single-row/batch discrepancy makes batch mode and thread overhead material; the two measures must not be conflated. XGBoost SHAP one-row median was 0.6145 ms; one 137-row SHAP calculation took 3.1180 ms. These are single-session exploratory measurements, not repeated-run distributions. See `results/audit/reproduction_environment.txt` for exact methods and package versions.

Loading the XGBoost artifact emitted a warning that its binary serialization format is deprecated and JSON/UBJ is recommended. The artifact loaded and predictions were consistent, but future compatibility should be monitored.

## M. Framework Architecture and Institution-Specific Adaptation Workflow

The code is a sequential Python pipeline with separate modules for loading, target construction, feature engineering, splitting, training, evaluation, SHAP, stability and feasibility. That modular decomposition is a useful prototype architecture. It is not yet a configurable institution-agnostic framework: raw table schemas, feature names, label mapping, paths, model families and split logic are hard-coded to the Berka dataset. No adapter/interface, institution-specific data contract, governance process, local recalibration/validation guide or operational monitoring implementation is evidenced.

The empirical results are from Czech banking data only. Local institutions would need to map their own data and outcome definitions, establish permissible feature availability at the exact decision time, validate temporal performance/calibration and explanations locally, and establish governance and monitoring. This audit found no evidence that Czech-data performance generalizes to Lesotho.

## N. Methodology Coverage Matrix

The machine-readable commitment-by-commitment classifications are in [results/audit/methodology_coverage.csv](results/audit/methodology_coverage.csv). They distinguish code/output evidence from documentation commitments that cannot be established by code. Summary: core data processing, model fitting, test metrics and basic XGBoost SHAP outputs exist; strict pre-issuance timing, temporal CV, validation-based selection, threshold analysis, comprehensive local/global XAI, refitted-fold stability and complete computational feasibility are partial or missing.

## O. Research Objective and Research-Question Traceability

| Objective/question                                                                           | Repository evidence                                                                                              | Coverage and limits                                                                                                                                                    |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Objective 1 / RQ1: existing ML/XAI approaches for loan default                               | LR, RF, XGBoost and SHAP are applied to one public dataset.                                                      | Partial empirical example only. No literature review or systematic comparison of approaches used in the field is in the repository; LIME is absent.                    |
| Objective 2 / RQ2: requirements for a Lesotho-suitable framework                             | No Lesotho data, stakeholder requirements or documented requirement set found.                                   | Missing in repository. Must not infer institutional requirements solely from Czech data.                                                                               |
| Objective 3 / RQ3: develop a framework from historical behaviour to interpretable prediction | A modular, loan-level Python workflow and selected features/models are implemented.                              | Partial: hard-coded PKDD schema, unresolved same-day availability, no institution adapter or documented governance/adaptation process.                                 |
| Objective 4 / RQ4: evaluate predictive performance and interpretability                      | Saved test probabilities/metrics, calibration plot, SHAP global plots/waterfall and a stability statistic exist. | Partial: only three test defaults, validation unused, CV is not temporal, strict feature timing not verified, XAI test-set use, no LIME/dependence/refitted stability. |
| Objective 5 / RQ5: adoption recommendations for Lesotho                                      | No adoption recommendations or Lesotho-specific implementation evidence found in repository.                     | Missing as a documented result; recommendations require local requirements, validation and governance evidence.                                                        |
| Main RQ: support transparent and reliable prediction for Lesotho institutions                | Prototype demonstrates a possible computational pipeline on Czech data.                                          | Partially addressed as a prototype; transparency is limited and reliability/generalisation to Lesotho is not evaluated.                                                |

## P. Critical Issues Ranked

| Rank        | Issue and evidence                                                                                                                                                                            | Recommended resolution direction                                                                                                                                                           |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **BLOCKER** | Strict pre-issuance feature availability is unproven: `src/04_feature_engineering.py` uses `<= loan_date`; 122 of 682 loans have same-day transaction rows and timestamps are day-resolution. | Establish within-day ordering from source data or use a defensible strictly prior cutoff and re-evaluate. Until then, do not claim all behavioural features were observed before issuance. |
| **HIGH**    | CV does not respect temporal order (`StratifiedKFold(5)`); LR scaling is fit on all training rows before CV. See `src/07_model_training.py`.                                                  | Use time-aware CV consistent with the approved procedure and fold-local preprocessing; preserve the primary chronological test partition.                                                  |
| **HIGH**    | Validation set is not used for model/threshold selection; estimator default 0.5 is used.                                                                                                      | Define selection and threshold policy using training/validation only; keep test untouched.                                                                                                 |
| **HIGH**    | Test contains only 3 positives (134 negatives); 3/3 recall for XGBoost is not robust.                                                                                                         | Retain this primary test result as-is; present uncertainty and limitations. Any supplementary analysis must be clearly labelled and must not replace this result.                          |
| **HIGH**    | Stability statistic uses a single model fit on all train rows and five overlapping subsets; it is not refitted-fold model/explanation stability.                                              | Recompute using the approved fold/subsample procedure with model refitting and disclose exact comparison scope.                                                                            |
| **HIGH**    | Target maps running contracts C/D at an extraction snapshot; no later outcome horizon or censoring treatment is evidenced.                                                                    | Clarify estimand and observation horizon; report possible label censoring and do not imply mature-loan default prediction without evidence.                                                |
| **MEDIUM**  | Train/validation and validation/test share one date each at split boundaries; timestamps cannot order same-day loans.                                                                         | Document date ties and choose/document a tie-aware cutoff policy without changing the approved test for metric improvement.                                                                |
| **MEDIUM**  | Global SHAP uses test rows and the one local case is selected from test predictions; no LIME or dependence plot.                                                                              | Separate explanation-development/selection from final-test explanation reporting as approved; complete only the approved explanation set and document limitations.                         |
| **MEDIUM**  | 17.02 MB is not peak pipeline RSS; inference/explanation latencies were absent from original artifacts.                                                                                       | Use an explicitly scoped repeatable benchmark and process high-water memory method; retain audit-session timings as exploratory only.                                                      |
| **MEDIUM**  | Framework is PKDD-specific in paths/schema/features and has no Lesotho validation/adaptation guide.                                                                                           | Document the generic architecture separately from institution-specific data mapping, governance and local validation.                                                                      |
| **LOW**     | XGBoost load emits deprecated binary-format warning; environment has pinned direct requirements but no complete lock of transitive dependencies.                                              | Preserve current evidence; address serialization/environment portability in an approved future reproducibility update.                                                                     |

## Q. Reproducibility Instructions and Unresolved Limitations

From the repository root, the recorded command is `./venv/bin/python run_pipeline.py`. It requires the raw `.asc` files and installed dependencies from `requirements.txt`. The command rewrites existing processed and result outputs; use a repository copy when preserving current evidence. This audit did not execute it.

Verified artifacts created in this audit are `results/audit/verified_dataset_summary.json`, `verified_split_summary.json`, `verified_metrics.csv`, `verified_confusion_matrices.json`, `methodology_coverage.csv` and `reproduction_environment.txt`. Original results remain in place. The audit report itself is `RESULTS_AUDIT_REPORT.md`.

Unresolved limitations include same-day feature timing, temporal CV/model-selection validity, unused validation, tiny positive test count, snapshot target censoring, test-based explanation summaries, limited SHAP/LIME scope, non-refitted stability analysis, non-peak original memory measurement and the lack of Lesotho institutional evidence. No causal, fairness, user-comprehension, credit-loss-reduction or Lesotho-generalisation claim is supported by these artifacts.
