"""
04_feature_engineering.py
Behavioural feature engineering (Section 3.7.3, Table 7).

Corrected v2:
- Observation window is STRICTLY before issuance: transaction date < loan_date.
  Same-day transactions are excluded (no time-of-day field in PKDD '99).
- Loans with no eligible pre-issuance transactions are excluded and logged,
  never filled with zeros.
- Loans with missing loan attributes are excluded and logged.
- An audit column (transaction_cutoff = latest transaction used) is kept and
  asserted to be earlier than loan_date.
"""

import json

import pandas as pd

from config import DATA, FEATURE_COLS, SOURCE


def compute_account_features(loan_date, txns):
    """Table 7 features from transactions already restricted to date < loan_date."""
    inflows = txns.loc[txns["type"] == "PRIJEM", "amount"]
    outflows = txns.loc[txns["type"] == "VYDAJ", "amount"]

    ym = txns["date"].dt.to_period("M")
    monthly_inflow = inflows.groupby(ym[inflows.index]).sum()
    monthly_outflow = outflows.groupby(ym[outflows.index]).sum()
    n_months = ym.nunique()  # months containing at least one transaction (>=1 here)

    return {
        "total_inflow": float(inflows.sum()),
        "total_outflow": float(outflows.sum()),
        "avg_monthly_inflow": float(inflows.sum() / n_months),
        "avg_monthly_outflow": float(outflows.sum() / n_months),
        "inflow_volatility": float(monthly_inflow.std()) if len(monthly_inflow) > 1 else 0.0,
        "outflow_volatility": float(monthly_outflow.std()) if len(monthly_outflow) > 1 else 0.0,
        "transaction_frequency": float(len(txns)),
        "avg_balance": float(txns["balance"].mean()),
        "min_balance": float(txns["balance"].min()),
        "max_balance": float(txns["balance"].max()),
        # sample SD; a single transaction has no spread
        "balance_volatility": float(txns["balance"].std()) if len(txns) > 1 else 0.0,
        "account_tenure_days": int((loan_date - txns["date"].min()).days),
        "active_months": int(n_months),
    }


def main():
    cohort = pd.read_csv(DATA / "loan_cohort.csv")
    trans = pd.read_csv(SOURCE / "trans.csv", low_memory=False,
                        usecols=["account_id", "date", "type", "amount", "balance"])

    cohort["loan_date"] = pd.to_datetime(cohort["date"], format="%y%m%d")
    trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")
    by_account = {a: g for a, g in trans.groupby("account_id")}

    rows, excluded = [], []
    same_day_dropped = 0
    for _, loan in cohort.iterrows():
        loan_date = loan["loan_date"]
        acct = by_account.get(loan["account_id"])

        if loan[["amount", "duration", "payments"]].isna().any():
            excluded.append({"loan_id": int(loan["loan_id"]), "reason": "missing_loan_attributes"})
            continue

        if acct is None:
            excluded.append({"loan_id": int(loan["loan_id"]), "reason": "no_transactions_for_account"})
            continue

        same_day_dropped += int((acct["date"] == loan_date).sum())
        window = acct[acct["date"] < loan_date]          # strict cutoff
        if window.empty:
            excluded.append({"loan_id": int(loan["loan_id"]),
                             "reason": "no_transactions_before_issue_date"})
            continue

        feats = compute_account_features(loan_date, window)
        feats.update({
            "loan_id": int(loan["loan_id"]),
            "account_id": int(loan["account_id"]),
            "loan_date": loan_date,
            "transaction_cutoff": window["date"].max(),
            "loan_amount": float(loan["amount"]),
            "loan_duration": int(loan["duration"]),
            "loan_payment": float(loan["payments"]),
            "target": int(loan["target"]),
        })
        rows.append(feats)

    df = pd.DataFrame(rows)

    # ---- Validation -------------------------------------------------------
    assert (df["transaction_cutoff"] < df["loan_date"]).all(), "Window reaches issue date"
    assert df["loan_id"].is_unique
    assert df[FEATURE_COLS].notna().all().all(), "NaN in features"
    assert (df["transaction_frequency"] > 0).all()
    assert (df["account_tenure_days"] > 0).all()
    assert "status" not in df.columns
    assert len(df) + len(excluded) == len(cohort)

    cols = ["loan_id", "account_id", "loan_date", "transaction_cutoff"] + FEATURE_COLS + ["target"]
    df = df[cols].sort_values(["loan_date", "loan_id"]).reset_index(drop=True)
    df.to_csv(DATA / "analytical_dataset.csv", index=False)

    report = {
        "cohort_in": int(len(cohort)),
        "final_rows": int(len(df)),
        "excluded": len(excluded),
        "excluded_by_reason": pd.Series([e["reason"] for e in excluded],
                                        dtype=object).value_counts().to_dict(),
        "excluded_loans": excluded,
        "target_distribution": {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()},
        "same_day_transactions_excluded": same_day_dropped,
        "cutoff_rule": "transaction date < loan_date",
        "n_features": len(FEATURE_COLS),
        "loan_date_range": [str(df["loan_date"].min().date()), str(df["loan_date"].max().date())],
    }
    with open(DATA / "feature_engineering_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_loans"}, indent=2))


if __name__ == "__main__":
    main()
