"""
01_data_loading.py
Loads all PKDD '99 tables and verifies record counts against documented values.
"""

import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# Documented expected counts from Berka (1999)
EXPECTED_COUNTS = {
    "client":     5369,
    "account":    4500,
    "disp":       5369,
    "trans":      1056320,
    "order":      6471,
    "loan":       682,
    "card":       892,
    "district":   77,
}

def load_table(name):
    path = RAW / f"{name}.asc"
    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def verify(name, df, expected):
    actual = len(df)
    match = actual == expected
    status = "OK" if match else "MISMATCH"
    print(f"{name:10s} expected={expected:>8d}  actual={actual:>8d}  [{status}]")
    return {"expected": expected, "actual": actual, "match": match}

def main():
    tables = {}
    verification = {}
    for name in EXPECTED_COUNTS:
        df = load_table(name)
        tables[name] = df
        verification[name] = verify(name, df, EXPECTED_COUNTS[name])
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
    with open(PROCESSED / "verification.json", "w") as f:
        json.dump(verification, f, indent=2)
    print("\nAll tables loaded and saved to data/processed/")

if __name__ == "__main__":
    main()