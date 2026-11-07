# Feature Reconciliation

**Source of truth:**
- Documented: Chapter 3, Section 3.7.3, Table 7 ("Final empirical feature set")
- Implemented: `FEATURE_COLS` in `src/config.py` (`BEHAVIOURAL_FEATURES` + `LOAN_FEATURES`). It is computed by `compute_account_features()` and `main()` in `src/04_feature_engineering.py`.

**Inspection of the corrected v2 code (2026-11-07):**
- `src/config.py` defines `FEATURE_COLS` once: 13 behavioural features plus 3 loan attributes. It asserts that the list has exactly 16 entries. Scripts 04–12 import it, so no script keeps its own copy of the list.
- `compute_account_features()` in `src/04_feature_engineering.py` builds the 13 behavioural features. `main()` adds the 3 loan attributes from the `loan` table (`amount`, `duration`, `payments`).
- `results/corrected_v2/data/analytical_dataset.csv` has 234 rows × 21 columns: 16 features + `loan_id`, `account_id`, `loan_date`, `transaction_cutoff` (an audit column) and `target`. It has no missing values, which `04_feature_engineering.py` asserts.
- The modelling pipeline never uses `client`, `disp` or `district` data as predictors. `02_dataset_construction.py` joins them for data understanding only. `src/13_prior_loan_feasibility.py` uses `client` and `disp` for the prior-loan feasibility audit.

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
| max_balance | ✓ | ✓ | Include |
| balance_volatility | ✓ | ✓ | Include |
| account_tenure_days | ✓ | ✓ | Include |
| active_months | ✓ | ✓ | Include |
| loan_amount | ✓ | ✓ | Include |
| loan_duration | ✓ | ✓ | Include |
| loan_payment | ✓ | ✓ | Include |
| low_balance_frequency | Excluded (note) | ✗ | Exclude — not implemented; future extension |
| dormant_periods | Excluded (note) | ✗ | Exclude — not implemented; future extension |
| transaction_regularity | Excluded (note) | ✗ | Exclude — not implemented; future extension |
| prior_loan_history | Excluded (note) | ✗ | Exclude — no reconstructable same-client prior loans (see `src/13_prior_loan_feasibility.py`) |
| client_age | Excluded (note) | ✗ | Exclude — outside the behavioural + loan-terms feature scope |
| client_gender | Excluded (note) | ✗ | Exclude — outside the behavioural + loan-terms feature scope |
| district features | Excluded (note) | ✗ | Exclude — outside the behavioural + loan-terms feature scope |

**Summary:** 16 features are included: 13 behavioural features and 3 loan attributes. Table 7 and the code now match exactly. `max_balance` has been added to Table 7. The four candidate features that were never computed are listed in the "Table 7 note on excluded features", not in the table itself.

**Decision:** Table 7 lists exactly the features whose Final Decision is Include, and so does `FEATURE_COLS`.

## Implementation details stated in Section 3.7.3

The corrected code does the following, and Table 7 and its notes describe it this way:

1. **Observation window:** only transactions dated **strictly before** `loan_date` (`date < loan_date`) are used. Same-day transactions are excluded because PKDD '99 has no time-of-day field. In the 234-loan cohort this removes 55 transactions, affecting 40 loans. The audit column `transaction_cutoff` (the latest transaction used) is asserted to be earlier than `loan_date` for every loan. `05_leakage_validation.py` recounts the window independently for all 234 loans.
2. **Inflow / outflow definitions:** inflow = `type == "PRIJEM"` and outflow = `type == "VYDAJ"`. Transactions with `type == "VYBER"` count in neither inflows nor outflows. They **do** count in `transaction_frequency`, `active_months` and the balance statistics.
3. **Monthly averages:** `avg_monthly_inflow` and `avg_monthly_outflow` divide the total by the number of months that had *any* transaction (`active_months`). This is not the number of months with inflows (or outflows) specifically.
4. **Volatility:** `inflow_volatility` and `outflow_volatility` are the sample standard deviation (n−1) of monthly sums, taken only over months that had an inflow (or outflow). Each is set to 0 when there are fewer than two such months. `balance_volatility` is the sample SD of post-transaction balances, set to 0 when the window has a single transaction.
5. **Balance features:** statistics of the post-transaction `balance` field. They are not computed on a daily balance series.
6. **`account_tenure_days`:** days from the first observed transaction to `loan_date`. This is not the account creation date.
7. **Missing values and exclusions:** nothing is zero-filled. A loan with no transactions before its issue date, or with missing loan attributes, is excluded and logged by reason in `results/corrected_v2/data/feature_engineering_report.json`. In the current cohort no loans were excluded: 234 completed loans went in and 234 rows came out.

## Revision history

- **2026-11-07 (initial):** reconciled Table 7 against the original code. That code used an inclusive cutoff (`date <= loan_date`), applied `fillna(0)`, covered all 682 loans (A/B/C/D), and repeated `FEATURE_COLS` in scripts 07, 08, 10 and 11. Those outputs are kept in `results/baseline_v1/`.
- **2026-11-07 (corrected v2):** updated for the corrected pipeline: cohort limited to completed loans (A/B), strict cutoff, no zero-filling, `FEATURE_COLS` centralised in `src/config.py`. The feature set itself (16 features) is unchanged.
