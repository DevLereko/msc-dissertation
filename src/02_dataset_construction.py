"""
02_dataset_construction.py
Constructs the loan-level analytical cohort (Section 3.4).

Corrected v2:
- Only completed loans (status A or B) enter the cohort (Section 3.4.3).
  Running loans (C, D) are excluded and counted, never relabelled.
- Source tables in data/processed are read, never modified.
- Client/district joins are kept for data understanding only; they are not
  predictors (Section 3.7.3).
"""

import json

import pandas as pd

from config import DATA, SOURCE, TARGET_MAP


def load_all():
    return {
        name: pd.read_csv(SOURCE / f"{name}.csv", low_memory=False)
        for name in ["client", "account", "disp", "loan", "district"]
    }


def build_loan_dataset(t):
    loan = t["loan"].merge(t["account"], on="account_id", how="left",
                           suffixes=("", "_acc"))

    # Owner client per account (data understanding only, not a predictor)
    disp_client = t["disp"].merge(t["client"], on="client_id", how="left")
    owners = (disp_client[disp_client["type"] == "OWNER"]
              [["account_id", "client_id", "district_id"]]
              .drop_duplicates(subset="account_id", keep="first")
              .rename(columns={"client_id": "owner_client_id",
                               "district_id": "owner_district_id"}))
    loan = loan.merge(owners, on="account_id", how="left")

    district = t["district"].rename(columns={"A1": "district_id"})
    loan = loan.merge(district, on="district_id", how="left", suffixes=("", "_dist"))
    return loan


def assign_target(loans: pd.DataFrame) -> pd.DataFrame:
    """A = completed and repaid -> 0; B = completed and not paid -> 1.
    C and D (running contracts) are excluded from the cohort."""
    loans = loans[loans["status"].isin(TARGET_MAP.keys())].copy()
    loans["target"] = loans["status"].map(TARGET_MAP).astype(int)
    return loans


def main():
    t = load_all()
    all_loans = build_loan_dataset(t)
    assert len(all_loans) == len(t["loan"]), "Join changed the loan row count"

    status_counts = all_loans["status"].value_counts().sort_index().to_dict()
    cohort = assign_target(all_loans)

    # Checks against Section 3.4.3 / Table 4.2
    assert cohort["loan_id"].is_unique
    assert set(cohort["status"]) <= {"A", "B"}
    assert cohort["target"].notna().all()

    summary = {
        "total_loan_records": int(len(all_loans)),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "excluded_status_C": int(status_counts.get("C", 0)),
        "excluded_status_D": int(status_counts.get("D", 0)),
        "completed_cohort": int(len(cohort)),
        "non_default_A": int((cohort["target"] == 0).sum()),
        "default_B": int((cohort["target"] == 1).sum()),
    }
    print(json.dumps(summary, indent=2))

    cohort.to_csv(DATA / "loan_cohort.csv", index=False)
    with open(DATA / "cohort_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {DATA / 'loan_cohort.csv'}")


if __name__ == "__main__":
    main()
