"""
02_dataset_construction.py
Constructs the loan-level analytical dataset with leakage-controlled
historical observation windows.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROCESSED = Path("data/processed")

def load_all():
    return {
        name: pd.read_csv(PROCESSED / f"{name}.csv", low_memory=False)
        for name in ["client", "account", "disp", "trans", "order", "loan", "card", "district"]
    }

def build_loan_dataset(t):
    loan = t["loan"].copy()
    account = t["account"].copy()
    disp = t["disp"].copy()
    client = t["client"].copy()
    district = t["district"].copy()

    # Attach account info to each loan
    loan = loan.merge(account, on="account_id", how="left", suffixes=("", "_acc"))

    # Attach clients linked to each account (via disp).
    # Multiple clients per account possible; take owner (type='OWNER') if available.
    disp_client = disp.merge(client, on="client_id", how="left")
    owners = disp_client[disp_client["type"] == "OWNER"][
        ["account_id", "client_id", "district_id"]
    ]
    owners = owners.drop_duplicates(subset="account_id", keep="first")
    # Rename to avoid collision with account.district_id
    owners = owners.rename(columns={
        "client_id": "owner_client_id",
        "district_id": "owner_district_id",
    })
    loan = loan.merge(owners, on="account_id", how="left")

    # Attach district details: primary key A1 == loan.district_id
    district_keyed = district.rename(columns={"A1": "district_id"})
    loan = loan.merge(district_keyed, on="district_id", how="left", suffixes=("", "_dist"))

    return loan

def assign_target(loan):
    """
    A: finished/paid       -> 0
    B: finished/not paid   -> 1
    C: running/paid        -> 0
    D: running/client debt -> 1
    """
    mapping = {"A": 0, "B": 1, "C": 0, "D": 1}
    loan["target"] = loan["status"].map(mapping)
    return loan

def main():
    print("Loading tables...")
    t = load_all()

    print("Building loan-level dataset...")
    loan = build_loan_dataset(t)
    loan = assign_target(loan)

    # Report
    print(f"\nLoan records: {len(loan)}")
    print(f"Target distribution:\n{loan['target'].value_counts()}")
    print(f"Missing targets: {loan['target'].isna().sum()}")

    # Save
    loan.to_csv(PROCESSED / "loan_with_target.csv", index=False)
    print("\nSaved: data/processed/loan_with_target.csv")

    # Save summary for Chapter 4
    summary = {
        "total_loans": len(loan),
        "non_default": int((loan["target"] == 0).sum()),
        "default": int((loan["target"] == 1).sum()),
        "missing_target": int(loan["target"].isna().sum()),
    }
    with open(PROCESSED / "loan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()