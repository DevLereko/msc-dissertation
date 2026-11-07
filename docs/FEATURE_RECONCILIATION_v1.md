# Feature Reconciliation

**Source of truth:**
- Documented: Chapter 3, Section 3.7.3, Table 7
- Implemented: `src/04_feature_engineering.py`, `FEATURE_COLS` in `src/07_model_training.py`

**Inspection (read-only, 2026-11-07):**
- `src/04_feature_engineering.py` → `compute_account_features()` (lines 39–53) builds 13 behavioural features. `main()` (lines 81–83) adds 3 loan attributes.
- `FEATURE_COLS` in `src/07_model_training.py` (lines 23–29) lists 16 features. The list is the same in `src/08_evaluation.py`, `src/10_xai_analysis.py` and `src/11_stability_analysis.py`.
- `data/processed/analytical_dataset.csv`: 682 rows × 20 columns (16 features + `loan_id`, `account_id`, `target`, `loan_date`). No missing values.
- The modelling pipeline (scripts 04–11) never reads the `client`, `disp` or `district` tables. `client` and `disp` are used only in `src/13_prior_loan_feasibility.py`.

Code column key: ✓ = computed **and** passed to the models; ✗ = not computed anywhere.

| Feature | Table 7 | Code | Final Decision |
|---------|---------|------|----------------|
| total_inflow | ✓ | ✓ | Include |
| total_outflow | ✓ | ✓ | Include |
| avg_monthly_inflow | ✓ | ✓ | Include |
| avg_monthly_outflow | ✓ | ✓ | Include |
| inflow_volatility | ✓ | ✓ | Include |
| outflow_volatility | ✓ | ✓ | Include |
| transaction_frequency | ✓ | ✓ | Include |
| avg_balance | ✓ | ✓ | Include |
| min_balance | ✓ | ✓ | Include |
| max_balance | (not listed) | ✓ | Include (add to Table 7) |
| balance_volatility | ✓ | ✓ | Include |
| low_balance_frequency | ✓ | ✗ | Exclude (remove from Table 7) |
| account_tenure_days | ✓ | ✓ | Include |
| active_months | ✓ | ✓ | Include |
| dormant_periods | ✓ | ✗ | Exclude (remove from Table 7) |
| transaction_regularity | ✓ | ✗ | Exclude (remove from Table 7) |
| loan_amount | ✓ | ✓ | Include |
| loan_duration | ✓ | ✓ | Include |
| loan_payment | ✓ | ✓ | Include |
| prior_loan_history | ✓ (removing) | ✗ | Exclude (not feasible; see `src/13_prior_loan_feasibility.py`) |
| client_age | (not listed) | ✗ | Exclude |
| client_gender | (not listed) | ✗ | Exclude |
| district features | (not listed) | ✗ | Exclude |

**Summary:** 16 features are included: 13 behavioural features and 3 loan attributes. Table 7 needs 1 feature added (`max_balance`) and 4 removed (`low_balance_frequency`, `dormant_periods`, `transaction_regularity`, `prior_loan_history`).

**Decision:** The final Table 7 will list exactly the features in the "Final Decision" = Include column.

## Implementation details to state in Section 3.7.3

The current code does the following, and Table 7 and its text should describe it this way:

1. **Observation window:** transactions dated `<= loan_date`. Transactions on the origination day are **included**.
2. **Inflow / outflow definitions:** inflow = `type == "PRIJEM"`; outflow = `type == "VYDAJ"`. Transactions with `type == "VYBER"` (16,666 rows, total amount ≈ 208.6M) count in neither inflows nor outflows. They **do** count in `transaction_frequency`, `active_months` and the balance statistics.
3. **Monthly averages:** `avg_monthly_inflow` and `avg_monthly_outflow` divide the total by the number of months with *any* transaction (`n_months`). This is not the number of months with inflows (or outflows) specifically.
4. **Volatility:** the standard deviation of monthly sums, taken only over months that had an inflow (or outflow). It is set to 0 when there is ≤1 such month.
5. **Balance features:** statistics of the post-transaction `balance` field. They are not computed on a daily balance series.
6. **`account_tenure_days`:** days from the first observed transaction to `loan_date`. This is not the account creation date.
7. **Missing values:** `fillna(0)` in scripts 07, 08, 10 and 11. No NaNs are present in the current dataset, so this has no effect.
