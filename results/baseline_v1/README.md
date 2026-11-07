# baseline_v1 — snapshot of the original experiment (copied 2026-11-07)

Outputs produced by the pipeline *before* the methodology corrections:
- target mapped A/C→0, B/D→1 (all 682 loans)
- transactions `date <= loan_date` (inclusive cutoff)
- positional 60/20/20 split; shuffled StratifiedKFold tuning; scaler fit outside CV
- validation partition loaded but unused

Kept for audit only, so that differences from `results/corrected_v2/` can be explained.
Do not cite these numbers as the study's results.
