"""
05_leakage_validation.py
Validates the leakage-prevention rules in Table 5 against the corrected (v2)
analytical dataset. Every rule is an assertion; the script fails loudly.
"""

import json

import pandas as pd

from config import DATA, FEATURE_COLS, SOURCE


def main():
    df = pd.read_csv(DATA / "analytical_dataset.csv",
                     parse_dates=["loan_date", "transaction_cutoff"])
    loan = pd.read_csv(SOURCE / "loan.csv")
    trans = pd.read_csv(SOURCE / "trans.csv", low_memory=False,
                        usecols=["account_id", "date"])
    trans["date"] = pd.to_datetime(trans["date"], format="%y%m%d")
    results = {}

    # Rule 1 — status only builds the target, never a predictor
    assert "status" not in df.columns and "status" not in FEATURE_COLS
    assert "target" not in FEATURE_COLS
    results["status_not_predictor"] = True

    # Rule 2 — cohort is completed loans only (no C/D)
    status = loan.set_index("loan_id").loc[df["loan_id"], "status"]
    assert set(status) <= {"A", "B"}, "Running loans (C/D) in cohort"
    assert ((status == "B").astype(int).values == df["target"].values).all()
    results["completed_loans_only"] = True

    # Rule 3 — strict cutoff: latest transaction used is before issue date
    assert (df["transaction_cutoff"] < df["loan_date"]).all()
    results["cutoff_strictly_before_issue"] = True

    # Rule 4 — full independent recount of the strict window for EVERY loan
    m = df[["loan_id", "account_id", "loan_date"]].merge(trans, on="account_id")
    strict = m[m["date"] < m["loan_date"]].groupby("loan_id").size()
    inclusive = m[m["date"] <= m["loan_date"]].groupby("loan_id").size()
    recount = strict.reindex(df["loan_id"]).fillna(0).values
    assert (recount == df["transaction_frequency"].values).all(), \
        "transaction_frequency does not match strict pre-issue window"
    n_same_day = int((inclusive.reindex(df["loan_id"]).fillna(0).values - recount).sum())
    results["window_recount_matches_all_loans"] = True
    results["same_day_transactions_excluded"] = n_same_day

    # Rule 5 — no missing targets/features, every loan has history
    assert df["target"].notna().all()
    assert df[FEATURE_COLS].notna().all().all()
    assert (df["transaction_frequency"] > 0).all()
    results["no_missing_values"] = True

    # Rule 6 — one row per loan
    assert df["loan_id"].is_unique
    results["one_row_per_loan"] = True

    results["rows"] = int(len(df))
    results["target_distribution"] = {str(k): int(v) for k, v in
                                      df["target"].value_counts().sort_index().items()}
    with open(DATA / "leakage_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("All leakage rules passed.")


if __name__ == "__main__":
    main()
