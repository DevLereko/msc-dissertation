"""
03_eda.py
Exploratory data analysis on the corrected (v2) completed-loan cohort
(Chapter 4, Section 4.3).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA, EXPERIMENT, FEATURE_COLS

FIGURES = EXPERIMENT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")


def main():
    df = pd.read_csv(DATA / "analytical_dataset.csv", parse_dates=["loan_date"])
    assert set(df["target"].unique()) <= {0, 1}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["loan_amount"], bins=30, kde=True, ax=ax)
    ax.set(title="Distribution of Loan Amounts (completed loans)", xlabel="Loan Amount (CZK)")
    fig.savefig(FIGURES / "fig_4_1_loan_amount.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["loan_duration"], bins=20, ax=ax)
    ax.set(title="Distribution of Loan Durations (completed loans)", xlabel="Loan Duration (months)")
    fig.savefig(FIGURES / "fig_4_2_loan_duration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Non-default (A)", "Default (B)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Target Class Distribution")
    fig.savefig(FIGURES / "fig_4_3_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Defaults by issuance year — matters for the temporal split
    by_year = (df.groupby(df["loan_date"].dt.year)["target"]
               .agg(loans="count", defaults="sum"))
    by_year["default_rate"] = (by_year["defaults"] / by_year["loans"]).round(4)
    by_year.to_csv(DATA / "defaults_by_year.csv")

    df[FEATURE_COLS].describe().T.to_csv(DATA / "eda_summary.csv")
    df.groupby("target")[FEATURE_COLS].median().T.to_csv(DATA / "eda_medians_by_class.csv")

    print(f"Dataset: {df.shape[0]} loans; target {counts.to_dict()}")
    print(by_year.to_string())


if __name__ == "__main__":
    main()
