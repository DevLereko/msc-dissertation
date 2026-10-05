"""
04_feature_engineering.py
Constructs behavioural features from historical transactions
using only data available BEFORE each loan's prediction point.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")

def compute_account_features(account_id, loan_date, txns):
    """
    For a single account, compute behavioural features using only
    transactions BEFORE (<=) loan_date.
    """
    if txns is None or len(txns) == 0:
        return None  # Insufficient history

    # Filter to observation window (pre-origination)
    txns = txns[txns["date"] <= loan_date]
    if len(txns) == 0:
        return None

    # Type: 'PRIJEM' (credit/inflow), 'VYDAJ' (debit/outflow)
    inflows  = txns[txns["type"] == "PRIJEM"]["amount"]
    outflows = txns[txns["type"] == "VYDAJ"]["amount"]

    # Group by month for volatility
    txns = txns.copy()
    txns["year_month"] = txns["date"].dt.to_period("M")

    monthly_inflow  = txns[txns["type"] == "PRIJEM"].groupby("year_month")["amount"].sum()
    monthly_outflow = txns[txns["type"] == "VYDAJ"].groupby("year_month")["amount"].sum()

    n_months = max(len(txns["year_month"].unique()), 1)

    feats = {
        "total_inflow":             float(inflows.sum()),
        "total_outflow":            float(outflows.sum()),
        "avg_monthly_inflow":       float(inflows.sum() / n_months),
        "avg_monthly_outflow":      float(outflows.sum() / n_months),
        "inflow_volatility":        float(monthly_inflow.std()) if len(monthly_inflow) > 1 else 0.0,
        "outflow_volatility":       float(monthly_outflow.std()) if len(monthly_outflow) > 1 else 0.0,
        "transaction_frequency":    float(len(txns)),
        "avg_balance":              float(txns["balance"].mean()),
        "min_balance":              float(txns["balance"].min()),
        "max_balance":              float(txns["balance"].max()),
        "balance_volatility":       float(txns["balance"].std()),
        "active_months":            int(txns["year_month"].nunique()),
        "account_tenure_days":      int((loan_date - txns["date"].min()).days),
    }
    return feats

def main():
    loan = pd.read_csv(PROCESSED / "loan_with_target.csv")
    trans = pd.read_csv(PROCESSED / "trans.csv", low_memory=False)

    # Parse dates
    loan["date"] = pd.to_datetime(loan["date"], format="%y%m%d")
    trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")

    # Pre-index transactions per account for performance
    print("Indexing transactions by account...")
    trans_by_account = {
        acct: grp.reset_index(drop=True)
        for acct, grp in trans.groupby("account_id")
    }

    print(f"Processing {len(loan)} loans...")
    rows = []
    for _, row in loan.iterrows():
        txns = trans_by_account.get(row["account_id"])
        feats = compute_account_features(row["account_id"], row["date"], txns)
        if feats is None:
            continue
        feats["loan_id"] = row["loan_id"]
        feats["account_id"] = row["account_id"]
        feats["target"] = row["target"]
        feats["loan_amount"] = row["amount"]
        feats["loan_duration"] = row["duration"]
        feats["loan_payment"] = row["payments"]
        feats["loan_date"] = row["date"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"Final analytical dataset: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    df.to_csv(PROCESSED / "analytical_dataset.csv", index=False)
    print("Saved: data/processed/analytical_dataset.csv")

if __name__ == "__main__":
    main()