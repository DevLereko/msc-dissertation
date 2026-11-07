# Scope Adjustment Record

**Date:** 04th November 2026
**Approved by:** Prof. Kishor Kumar Reddy (supervisor)
**Reference:** Request for guidance and approval on dissertation target definition -  04th November 2026

## Decision

Following consultation with the supervisor, the study adopts **Approach 2** for
the target variable definition:

- **Target cohort:** Completed loans only
- **Included statuses:** A (finished/paid) and B (finished/not paid)
- **Excluded statuses:** C (running/paid) and D (running/in debt)
- **Target mapping:** A → 0 (non-default); B → 1 (default)
- **Expected analytical cohort:** 234 loans (203 non-default; 31 default)

## Rationale (as per supervisor correspondence)

Statuses A and B describe completed loan outcomes. Statuses C and D describe
loans that were still running at the dataset extraction point. Combining
them into a single binary target conflates completed non-payment with
ongoing indebtedness. Using completed loans only produces a target that
more closely represents a completed-loan outcome task, while explicitly
acknowledging the reduced sample size.

## Consequences for the study

- The empirical cohort changes from 682 to 234 loans.
- Chapter 3 (target definition, population, inclusion/exclusion criteria)
  and Chapter 4 (analytical dataset section) must be updated.
- Existing baseline results (682-loan run) remain preserved under
  `results/baseline_v1/` as a documented comparator.
- The framework itself is unchanged; the target change affects the
  empirical demonstration, not the architecture.

## Feature-timing correction (separate)

A separate methodological correction — strict pre-issuance cutoff
(`transaction_date < loan_date`) — remains required. This will be confirmed
with the supervisor separately if the university requires explicit approval.

## Approval status

Approved by supervisor.