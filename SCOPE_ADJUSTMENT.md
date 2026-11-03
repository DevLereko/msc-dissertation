# Scope Adjustment Record

**Status:** pending supervisor approval. This record preserves evidence and
documents a proposed scope adjustment. It does **not** change the analytical
dataset, target mapping, model artifacts, or any existing `results/` output.
Only documentation files are added/updated here, consistent with the controlled
sequence (record the scope adjustment now; change the pipeline later, only after
approval).

## 1. Change proposed

Mark the **prior-loan history** predictor as **unavailable in the current PKDD
'99 dataset**, rather than as a candidate feature pending further investigation.

- The prior-loan history feature is **not currently a predictor** in the model.
  The 16 predictors are the 13 behavioural + 3 loan-attribute columns listed in
  `RESULTS_AUDIT_REPORT.md` section D. No pipeline file regresses because of this
  record; it captures the decision to **stop treating prior-loan history as a
  viable feature** in this dataset.
- Because no same-client multi-loan relationships exist, there is nothing to
  engineer. Including this finding in the methodology means the feature is
  documented as **data-unavailable**, not as a modelling choice made on the
  current data.

## 2. Evidence (read-only audit)

Audit source and outputs (read only; pipeline and models untouched):

- Script: `src/13_prior_loan_feasibility.py`
- Findings: `results/audit/prior_loan_feasibility.json`
- Summary: `results/audit/prior_loan_feasibility_summary.txt`

The audit reads the raw `data/raw/*.asc` tables directly and reports:

| Question | Finding |
| --- | --- |
| Q1. Clients linked to multiple loan accounts | **0** (owner-only: 0/682; any disp link: 0/827). Every loan-bearing account is owned by a distinct, unique client. |
| Q2. Earlier-to-later same-client loan pairs | **0**. With no client holding two loans, no ordered pair exists. |
| Q3. When an earlier loan's outcome became known | Not establishable from the schema (no completion date, no snapshot extraction date); moot because no pairs exist. |
| Q4. Supportable prior-loan features | None usable: relationship prerequisite fails. Tier-1 count/date features are vacuous (always 0); Tier-2 outcome flags are unsupported; Tier-3 requires fields not in the schema. |

Decisive finding (`prior_loan_feasibility.json` → `conclusion`):
> ZERO clients are associated with more than one loan account in the PKDD '99
> snapshot ... there are ZERO earlier->later same-client loan pairs to
> reconstruct, so a previous-loan repayment feature cannot be built from this
> dataset at all.

## 3. Implication for the methodology / feature table

The methodology's feature table remains **as implemented**: 16 predictors
(13 behavioural + 3 loan attributes). Prior-loan history is recorded as:

- **Status: unavailable in this dataset** (a tested dataset limitation, replacing
  a previously unverified assumption about prior-loan history).
- **Reason:** the relational tables (`client` → `disp` → `account` → `loan`)
  contain no client associated with more than one loan account.
- **Consequence:** no prior-loan predictor is constructed; no leakage risk arises
  from it; no claim about prior-loan repayment behaviour is made.

## 4. Boundary of this scope adjustment

This record deliberately does **not**:

- change the target mapping (A/C = non-default, B/D = default) — retained pending
  a separate, explicit target review;
- change the feature pipeline or same-day transaction handling — deferred to the
  controlled pipeline-correction step;
- regenerate the analytical dataset, refit models, or overwrite any `results/`
  artifact;
- introduce the prior-loan feature into the model.

## 5. Approval

Draft for supervisor review. On approval, this record becomes the documented
basis for (a) keeping prior-loan history out of the model and (b) describing it
in dissertation Chapters 4/5 as a dataset limitation rather than a modelling
choice. Until approval, the existing results (baseline) remain the authoritative
reported outputs.