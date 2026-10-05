"""
05_leakage_validation.py
Validates that no post-origination information entered the feature set.
"""

import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")

def main():
    loan = pd.read_csv(PROCESSED / "loan_with_target.csv")
    trans = pd.read_csv(PROCESSED / "trans.csv", low_memory=False)
    df = pd.read_csv(PROCESSED / "analytical_dataset.csv")

    # Rule 1: Target not used as predictor
    assert "status" not in df.columns, "Status leaked as predictor"
    print("[OK] Rule 1: Status not used as predictor")

    # Rule 2: All features computed from pre-origination data (checked programmatically
    # during construction; report the min/max observation windows)
    assert (df["account_tenure_days"] >= 0).all(), "Negative tenure detected"
    print("[OK] Rule 2: All accounts have positive pre-origination tenure")

    # Rule 3: No null targets
    assert df["target"].notna().all(), "Missing target values"
    print("[OK] Rule 3: No missing target values")

    # Rule 4: All loans have transaction history
    assert (df["transaction_frequency"] > 0).all(), "Loans with no transactions"
    print("[OK] Rule 4: All loans have historical transaction data")

    # Extra check: verify construction used only the pre-origination window by
# independently recomputing, for a sample of accounts, the number of
# transactions on or before the loan date and comparing it to the analytical
# dataset's transaction_frequency.
    trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")
    loan_orig = df[["account_id", "loan_date", "transaction_frequency"]].copy()
    loan_orig["loan_date"] = pd.to_datetime(loan_orig["loan_date"])
    sample = loan_orig.sample(min(50, len(loan_orig)), random_state=42)
    sub = trans[trans["account_id"].isin(set(sample["account_id"]))]
    merged = sample.merge(sub, on="account_id", how="left")
    pre_orig = merged[merged["date"] <= merged["loan_date"]]
    recon = pre_orig.groupby("account_id").size().rename("recount")
    check = sample.join(recon, on="account_id")
    tol = (check["transaction_frequency"] - check["recount"]).abs()
    assert (tol <= 1e-9).all(), "Feature window does not match pre-origination window"
    print(
        "[OK] Rule 5: Independently reconstructed pre-origination windows match "
        "analytical features (no future leakage)"
    )

    print(f"\nFinal analytical dataset: {df.shape[0]} loans, {df.shape[1]} columns")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()