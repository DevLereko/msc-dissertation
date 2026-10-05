"""
06_data_splitting.py
Chronological train/validation/test split preserving temporal ordering.
"""

import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
RANDOM_SEED = 42

def main():
    df = pd.read_csv(PROCESSED / "analytical_dataset.csv")
    df["loan_date"] = pd.to_datetime(df["loan_date"])
    df = df.sort_values("loan_date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Train: {len(train)} loans ({train['loan_date'].min().date()} to {train['loan_date'].max().date()})")
    print(f"Val:   {len(val)} loans ({val['loan_date'].min().date()} to {val['loan_date'].max().date()})")
    print(f"Test:  {len(test)} loans ({test['loan_date'].min().date()} to {test['loan_date'].max().date()})")

    print(f"\nClass distribution:")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name}: {split['target'].value_counts().to_dict()}")

    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED / "val.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    # Save split metadata for Chapter 4
    import json
    meta = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        meta[name] = {
            "count": len(split),
            "start_date": str(split["loan_date"].min().date()),
            "end_date": str(split["loan_date"].max().date()),
            "class_distribution": split["target"].value_counts().to_dict(),
        }
    with open(PROCESSED / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()