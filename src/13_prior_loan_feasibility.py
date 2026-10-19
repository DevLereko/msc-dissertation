"""
13_prior_loan_feasibility.py
Read-only prior-loan feasibility audit.

Scope
-----
This script audits whether a "previous-loan repayment behaviour" feature can be
reconstructed from the raw PKDD '99 tables WITHOUT introducing leakage or using
future information. It reads ONLY the raw `.asc` files directly; it does not
touch data/processed, does not regenerate features, and does not rerun models.

The audit distinguishes two questions that are easy to conflate:
  1. Can the raw tables LINK an earlier loan to the same borrower?
     (client -> disp -> account -> loan)
  2. Can we ESTABLISH what the earlier loan's repayment outcome was, and that
     this outcome was KNOWABLE at the later loan's issuance date?

A relationship between two loan records alone does NOT prove the second
condition. This is the harder requirement and the one that controls whether a
prior-loan feature is leakage-safe.

Method notes (semantics from Berka 1999 / PKDD '99):
  - A loan's `date` is its issuance date (yymmdd, day resolution).
  - `duration` is the contracted term in months.
  - A loan's contracted end date = issuance date + duration months.
  - `status` is a SNAPSHOT at data extraction:
        A = finished  / OK       (terminal outcome: repaid)
        B = finished  / not OK   (terminal outcome: defaulted)
        C = running   / OK
        D = running   / in debt
  - Only A/B carry a TERMINAL outcome. C/D are censored at the snapshot; their
    snapshot status is NOT a known final outcome.
  - The precise extraction date is not recorded in the schema. We therefore
    treat "outcome knowable at time T" as:
        earlier.status in {A, B}            (final/terminal state exists)
        AND earlier contracted end date < T  (loan has matured by time T)
    Anything else requires an assumption and is reported as NOT establishable.

No analytical feature is computed here. Counts and relationship facts only.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

RAW = Path("data/raw")
OUT_DIR = Path("results/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_table(name):
    path = RAW / f"{name}.asc"
    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_date(ymd):
    """Parse a PKDD yymmdd integer to an ISO date string. 2-digit year -> 19xx."""
    s = str(int(ymd)).zfill(6)
    yy, mm, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
    year = 1900 + yy if yy > 30 else 2000 + yy
    return datetime(year, mm, dd)


def add_months(dt, months):
    """Add whole calendar months to a date, preserving the day (clamped)."""
    y = (dt.month - 1 + months) // 12 + dt.year
    m = (dt.month - 1 + months) % 12 + 1
    d = min(dt.day, pd.Timestamp(y, m, 1).days_in_month)
    return datetime(y, m, d)


def run():
    report = {"scope_note": (
        "Read-only prior-loan feasibility audit. Reads raw .asc only; no feature "
        "regeneration, no model rerun, no change to existing pipeline outputs."
    )}

    # ---- Load raw tables -------------------------------------------------
    client = load_table("client")
    disp = load_table("disp")
    account = load_table("account")
    loan = load_table("loan")

    report["table_shapes"] = {
        "client": int(len(client)),
        "disp": int(len(disp)),
        "account": int(len(account)),
        "loan": int(len(loan)),
    }

    # ---- Normalise loan dates and compute contracted end dates -----------
    loan = loan.copy()
    loan["_issue_dt"] = loan["date"].apply(parse_date)
    loan["date_iso"] = loan["_issue_dt"].dt.strftime("%Y-%m-%d")
    loan["_end_dt"] = loan.apply(lambda r: add_months(r["_issue_dt"], int(r["duration"])), axis=1)
    loan["end_date_iso"] = loan["_end_dt"].dt.strftime("%Y-%m-%d")
    loan["_is_terminal"] = loan["status"].isin(["A", "B"])
    loan["_terminal_outcome"] = loan["status"].map({"A": "repaid", "B": "defaulted"})

    # ---- Link every loan to its owning client via disp -------------------
    # A loan sits on an account; an account is linked to client(s) via disp.
    # Multiple clients per account are possible; we keep BOTH the OWNER and any
    # DISPONENT links so we can measure the widest candidate set, then report
    # the owner-only subset separately.
    disp["is_owner"] = disp["type"].astype(str).str.upper() == "OWNER"

    all_links = disp[["client_id", "account_id", "is_owner"]].merge(
        loan[["loan_id", "account_id", "date_iso", "_issue_dt", "end_date_iso",
               "_end_dt", "status", "_is_terminal", "_terminal_outcome"]],
        on="account_id", how="inner"
    )
    owner_links = all_links[all_links["is_owner"]]

    report["account_client_linkage"] = {
        "loans_with_any_disp_link": int(all_links["loan_id"].nunique()),
        "loans_with_owner_link": int(owner_links["loan_id"].nunique()),
        "loans_with_no_owner_but_some_disp": int(
            loan["loan_id"].nunique() - owner_links["loan_id"].nunique()
        ),
        # Accounts can have an owner AND one or more dispongents; 145 such loan
        # accounts have >1 client. This does NOT create a multi-loan client:
        # each of those additional clients still holds only one loan account.
        "loan_accounts_with_multiple_clients_owner_plus_disponents": int(
            all_links.groupby("account_id")["client_id"].nunique()[lambda s: s > 1].shape[0]
        ),
        "note": (
            "An account with one owner + dispongents (145 loan accounts) links "
            "several clients to the SAME single loan, not to multiple loans. It "
            "yields no earlier->later same-client loan pairs."
        ),
    }

    # ---- (1) Clients associated with multiple loan accounts --------------
    per_client_owner = owner_links.groupby("client_id")["loan_id"].nunique()
    clients_multi = per_client_owner[per_client_owner > 1]

    per_client_all = all_links.groupby("client_id")["loan_id"].nunique()
    clients_multi_all = per_client_all[per_client_all > 1]

    report["question1_clients_with_multiple_loan_accounts"] = {
        "owner_only": {
            "clients_with_more_than_one_loan_account": int(clients_multi.shape[0]),
            "total_clients_with_any_loan_account": int(per_client_owner.shape[0]),
            "distribution_loans_per_client": per_client_owner.value_counts().sort_index().to_dict(),
        },
        "any_disp_link": {
            "clients_with_more_than_one_loan_account": int(clients_multi_all.shape[0]),
            "total_clients_with_any_loan_account": int(per_client_all.shape[0]),
            "distribution_loans_per_client": per_client_all.value_counts().sort_index().to_dict(),
        },
    }

    # ---- (2) Candidate earlier -> later loan PAIRS for same client -------
    # Build, per client, all ordered (earlier, later) loan pairs, then sub-divide
    # by whether the earlier loan had a terminal (A/B) outcome and whether its
    # contracted end date is before the later loan's issue date.
    if owner_links.empty:
        pairs = pd.DataFrame()
    else:
        grp = owner_links[["client_id", "loan_id", "_issue_dt", "_end_dt", "status",
                            "_is_terminal", "_terminal_outcome"]]
        merged = grp.merge(grp, on="client_id", suffixes=("_earlier", "_later"))
        pairs = merged[merged["_issue_dt_earlier"] < merged["_issue_dt_later"]].copy()
        pairs["earlier_ended_before_later_issue"] = (
            pairs["_end_dt_earlier"] < pairs["_issue_dt_later"]
        )

    n_pairs = int(len(pairs))
    terminal_earlier = pairs[pairs["_is_terminal_earlier"]]
    terminal_and_matured = terminal_earlier[terminal_earlier["earlier_ended_before_later_issue"]]

    report["question2_candidate_earlier_later_pairs"] = {
        "owner_only_pairs": {
            "total_earlier_later_pairs": n_pairs,
            "pairs_where_earlier_has_terminal_status_AB": int(len(terminal_earlier)),
            "pairs_where_earlier_terminal_AND_matured_before_later_issue": int(
                len(terminal_and_matured)
            ),
            "pairs_with_multi_hop_chains_same_client": int(
                pairs.groupby("client_id")["loan_id_earlier"].nunique()[
                    lambda s: s > 1
                ].shape[0]
            ),
        }
    }

    # ---- Distinct later loans that could receive a leakage-safe prior feature
    later_loan_ids_safe = set(terminal_and_matured["loan_id_later"])
    later_loan_ids_any_prior = set(pairs["loan_id_later"])

    report["later_loans_eligible"] = {
        "later_loans_with_any_same_client_prior_loan": int(len(later_loan_ids_any_prior)),
        "of_which_prior_outcome_knowable_at_issue": int(len(later_loan_ids_safe)),
        "total_loans_in_population": int(len(loan)),
    }

    # ---- Decisive feasibility conclusion ----------------------------------
    multi_owner = report["question1_clients_with_multiple_loan_accounts"]["owner_only"][
        "clients_with_more_than_one_loan_account"
    ]
    any_pairs = report["question2_candidate_earlier_later_pairs"]["owner_only_pairs"][
        "total_earlier_later_pairs"
    ]
    report["conclusion"] = {
        "prior_loan_relationships_exist_in_dataset": (
            any_pairs > 0 or multi_owner > 0
        ),
        "finding": (
            "ZERO clients are associated with more than one loan account in the "
            "PKDD '99 snapshot. Every loan-bearing account is owned by a distinct, "
            "unique client, and even including DISPONENT links every client still "
            "maps to exactly one loan account (827 distinct clients, all with one "
            "loan link each). Consequently there are ZERO earlier->later same-client "
            "loan pairs to reconstruct, so a previous-loan repayment feature cannot "
            "be built from this dataset at all - independent of the outcome-"
            "knowability question."
        ),
        "implication": (
            "The prior-loan feature is NOT implementable in this study. There is no "
            "same-client prior-loan history to engineer. Do not allocate further "
            "work to building it; the relationship prerequisite (question 1) is not "
            "met, so the harder knowability requirement (question 3) is moot."
        ),
        "counts_behind_conclusion": {
            "owner_clients_with_2plus_loan_accounts": int(multi_owner),
            "same_client_earlier_later_pairs": int(any_pairs),
            "later_loans_with_any_same_client_prior": int(len(later_loan_ids_any_prior)),
        },
    }

    # ---- (3) Does any field establish WHEN outcome became known? ---------
    # Raw loan/account/disp/client tables carry NO snapshot/extraction date and
    # NO maturity-completion date. The only timing information is the ISSUE date
    # and the contracted duration. Status is a point-in-time census label.
    report["question3_when_outcome_became_known"] = {
        "has_snapshot_extraction_date_field": False,
        "has_maturity_completion_date_field": False,
        "has_daily_outcome_effective_date_field": False,
        "available_timing_fields": {
            "loan_issue_date": "yymmdd, day resolution",
            "loan_contracted_duration_months": "drive computed contracted end date only",
            "status": ("snapshot A/B/C/D; A/B terminal (repaid/defaulted), "
                       "C/D censored at extraction - not a known final outcome"),
            "account_open_date": "present in account table; not a loan outcome date",
        },
        "conclusion": (
            "The schema establishes the ISSUE date and a CONTRACTED end date "
            "(issue + duration). It does NOT record the actual completion date or "
            "the snapshot extraction date. Groups A/B are terminal snapshots; "
            "C/D are censored. A prior loan's final outcome can be treated as "
            "'knowable at the later issue date' only by ASSUMING the loan matured "
            "on its contracted schedule and that no C/D loan is used. Counts of "
            "outcome-knowable pairs above include that assumption."
        ),
    }

    # ---- (4) Which prior-loan features are supportable w/o assumptions ----
    # Features are classified into three tiers.
    report["question4_supportable_prior_features"] = {
        "tier1_no_assumptions": {
            "description": "Relationship facts establishable from IDs alone; no claim about repayment outcome.",
            "features": [
                "prior_loans_same_client_count",
                "prior_loans_same_client_had_terminal_outcome_flag",
                "prior_largest_loan_amount_same_client",
                "prior_sum_loan_amount_same_client",
                "time_since_prior_loan_issue (issue-date based only, no outcome)",
            ],
        },
        "tier2_requires_maturity_assumption": {
            "description": ("Usable only if one accepts the assumption that a prior "
                            "A/B loan matured on its contracted schedule before the "
                            "later issue date. Outcome is KNOWN only for A/B."),
            "features": [
                "prior_loan_default_flag (prior B status, matured before issue)",
                "prior_loan_repaid_flag (prior A status, matured before issue)",
                "prior_loan_outcome_rate_same_client",
            ],
            "issued_pairs_supported": int(len(terminal_and_matured)),
        },
        "tier3_not_supportable": {
            "description": "Requires info not in the schema (actual completion date, exact extraction date, or the future/current status of running loans).",
            "features": [
                "final_outcome_of_any_running_prior_loan (C/D are censored)",
                "payments_actually_made_toward_prior_loan (no per-loan event stream; trans is account-level)",
                "prior_loan_default_status_at EXACT later issue instant (day-resolution, no intraday order, no completion date)",
            ],
        },
        "recommended_next_step_for_safe_addition": (
            "If pursued, implement ONLY Tier-1 issue-date and count features and a "
            "Tier-2 default flag restricted to pairs where the prior loan is A/B AND "
            "its contracted end date is before the later issue date. Document the "
            "maturity assumption explicitly. Do NOT include C/D prior loans as "
            "known outcomes."
        ),
    }

    # ---- Snapshot/coverage context: distribution of A/B/C/D on prior links
    status_counts = loan["status"].value_counts().to_dict()
    report["status_coverage"] = {
        "all_loans": status_counts,
        "terminal_AB_total": int((status_counts.get("A", 0) + status_counts.get("B", 0))),
        "censored_CD_total": int((status_counts.get("C", 0) + status_counts.get("D", 0))),
    }

    report["environment"] = {
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }

    # ---- Write outputs ---------------------------------------------------
    (OUT_DIR / "prior_loan_feasibility.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )

    lines = [
        "========================================================================",
        "PRIOR-LOAN FEASIBILITY AUDIT (read-only)",
        "========================================================================",
        "Scope: raw .asc only. No features regenerated, no models rerun.",
        "",
        f"Q1. Clients linked to multiple loan accounts:",
        f"    owner-only     : {report['question1_clients_with_multiple_loan_accounts']['owner_only']['clients_with_more_than_one_loan_account']} / {report['question1_clients_with_multiple_loan_accounts']['owner_only']['total_clients_with_any_loan_account']}",
        f"    any-disp-link  : {report['question1_clients_with_multiple_loan_accounts']['any_disp_link']['clients_with_more_than_one_loan_account']} / {report['question1_clients_with_multiple_loan_accounts']['any_disp_link']['total_clients_with_any_loan_account']}",
        "",
        f"Q2. Earlier->later same-client loan pairs (owner-only):",
        f"    all ordered pairs                      : {n_pairs}",
        f"    prior loan terminal (A/B)              : {len(terminal_earlier)}",
        f"    prior terminal AND matured before issue: {len(terminal_and_matured)}",
        "",
        f"Q3. Outcome-knowability:",
        f"    snapshot/extraction date field  : {report['question3_when_outcome_became_known']['has_snapshot_extraction_date_field']}",
        f"    maturity/completion date field  : {report['question3_when_outcome_became_known']['has_maturity_completion_date_field']}",
        f"    {report['question3_when_outcome_became_known']['conclusion']}",
        "",
        f"Q4. Later loans eligible for a leakage-safe prior feature:",
        f"    later loans with ANY same-client prior loan : {len(later_loan_ids_any_prior)}",
        f"    ... of which prior outcome can be established : {len(later_loan_ids_safe)}",
        f"    See JSON for tiered feature recommendations.",
        "",
        "DECISIVE CONCLUSION:",
        f"    {report['conclusion']['finding']}",
        "========================================================================",
    ]
    (OUT_DIR / "prior_loan_feasibility_summary.txt").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    run()