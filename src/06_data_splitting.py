"""
06_data_splitting.py
Date-grouped chronological train / validation / test split (Section 3.7.4).

Corrected v2:
- Boundaries are chosen on UNIQUE issuance dates, so all loans issued on the
  same day fall in the same partition.
- The pre-declared fractions (config.TRAIN_FRAC / VAL_FRAC) decide the split.
  A table of candidate boundaries is written for transparency (Step 2), but it
  is NOT used to pick boundaries, and no test metric is ever consulted.
"""

import json

import pandas as pd

from config import DATA, SPLITS, TRAIN_FRAC, VAL_FRAC


def split_by_dates(df, train_frac, val_frac):
    dates = sorted(df["loan_date"].unique())
    train_end = dates[int(len(dates) * train_frac) - 1]
    val_end = dates[int(len(dates) * (train_frac + val_frac)) - 1]
    train = df[df["loan_date"] <= train_end]
    val = df[(df["loan_date"] > train_end) & (df["loan_date"] <= val_end)]
    test = df[df["loan_date"] > val_end]
    return train, val, test


def describe(part):
    return {
        "loans": int(len(part)),
        "defaults": int(part["target"].sum()),
        "non_defaults": int((part["target"] == 0).sum()),
        "default_rate": round(float(part["target"].mean()), 4),
        "start_date": str(part["loan_date"].min().date()),
        "end_date": str(part["loan_date"].max().date()),
        "unique_dates": int(part["loan_date"].nunique()),
    }


def main():
    df = pd.read_csv(DATA / "analytical_dataset.csv",
                     parse_dates=["loan_date", "transaction_cutoff"])
    df = df.sort_values(["loan_date", "loan_id"]).reset_index(drop=True)

    # Step 2 — inspect candidate boundaries (reported, not optimised)
    cands = []
    for tf, vf in [(0.50, 0.25), (0.60, 0.20), (0.60, 0.15), (0.70, 0.15)]:
        tr, va, te = split_by_dates(df, tf, vf)
        row = {"train_frac": tf, "val_frac": vf}
        for n, p in [("train", tr), ("val", va), ("test", te)]:
            row[f"{n}_loans"], row[f"{n}_defaults"] = len(p), int(p["target"].sum())
            row[f"{n}_dates"] = f"{p['loan_date'].min().date()}..{p['loan_date'].max().date()}"
        cands.append(row)
    pd.DataFrame(cands).to_csv(SPLITS / "candidate_boundaries.csv", index=False)

    # Step 3 — assign with the pre-declared fractions
    train, val, test = split_by_dates(df, TRAIN_FRAC, VAL_FRAC)

    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.loan_id).isdisjoint(val.loan_id) and set(val.loan_id).isdisjoint(test.loan_id)
    assert train["loan_date"].max() < val["loan_date"].min()
    assert val["loan_date"].max() < test["loan_date"].min()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        assert part["target"].sum() > 0, f"{name} partition has no defaults"

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(SPLITS / f"{name}.csv", index=False)

    meta = {
        "rule": "boundaries on unique issuance dates; same-date loans kept together",
        "train_frac_of_dates": TRAIN_FRAC,
        "val_frac_of_dates": VAL_FRAC,
        "train_end_date": str(train["loan_date"].max().date()),
        "val_end_date": str(val["loan_date"].max().date()),
        "partitions": {n: describe(p) for n, p in
                       [("train", train), ("val", val), ("test", test)]},
    }
    with open(SPLITS / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    if meta["partitions"]["test"]["defaults"] < 5:
        print(f"\nWARNING: test partition has only {meta['partitions']['test']['defaults']} "
              "defaults; positive-class test metrics will be highly uncertain. "
              "Report this as a limitation (Section 3.7.4) — do not rebalance.")


if __name__ == "__main__":
    main()
