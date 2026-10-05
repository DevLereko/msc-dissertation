"""
03_eda.py
Generates exploratory data analysis figures and summary statistics
for Chapter 4, Section 4.3.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED = Path("data/processed")
FIGURES = Path("results/figures")
TABLES = Path("results/tables")
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

def main():
    df = pd.read_csv(PROCESSED / "analytical_dataset.csv")

    # Figure 4.1: Loan amount distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["loan_amount"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Loan Amounts")
    ax.set_xlabel("Loan Amount (CZK)")
    fig.savefig(FIGURES / "fig_4_1_loan_amount.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4.2: Loan duration
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["loan_duration"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Loan Durations")
    ax.set_xlabel("Loan Duration (months)")
    fig.savefig(FIGURES / "fig_4_2_loan_duration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4.3: Target distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    df["target"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Target Class Distribution")
    ax.set_xticklabels(["Non-default (0)", "Default (1)"], rotation=0)
    fig.savefig(FIGURES / "fig_4_3_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary statistics table
    summary = df.describe().T
    summary.to_csv(TABLES / "eda_summary.csv")

    print("EDA figures saved to results/figures/")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    main()