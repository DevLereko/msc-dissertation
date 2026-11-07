"""
config.py
Single source of truth for the corrected (v2) experiment.

Every pipeline script imports paths, the feature list and protocol constants
from here, so Table 7 and the code cannot drift apart again.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw tables converted by 01_data_loading.py (read-only for the experiment)
SOURCE = ROOT / "data" / "processed"

# All outputs of the corrected experiment live here; baseline_v1 is never touched
EXPERIMENT = ROOT / "results" / "corrected_v2"
DATA = EXPERIMENT / "data"
SPLITS = EXPERIMENT / "splits"
MODELS = EXPERIMENT / "models"
EVALUATION = EXPERIMENT / "evaluation"
XAI = EXPERIMENT / "xai"
STABILITY = EXPERIMENT / "stability"
FEASIBILITY = EXPERIMENT / "feasibility"

for _d in (DATA, SPLITS, MODELS, EVALUATION, XAI, STABILITY, FEASIBILITY):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Section 3.4.3 — completed loans only
TARGET_MAP = {"A": 0, "B": 1}

# Table 7 — final empirical feature set (13 behavioural + 3 loan attributes)
BEHAVIOURAL_FEATURES = [
    "total_inflow", "total_outflow", "avg_monthly_inflow", "avg_monthly_outflow",
    "inflow_volatility", "outflow_volatility", "transaction_frequency",
    "avg_balance", "min_balance", "max_balance", "balance_volatility",
    "account_tenure_days", "active_months",
]
LOAN_FEATURES = ["loan_amount", "loan_duration", "loan_payment"]
FEATURE_COLS = BEHAVIOURAL_FEATURES + LOAN_FEATURES
assert len(FEATURE_COLS) == 16

# Section 3.7.4 — partition and CV protocol
TRAIN_FRAC = 0.60          # fraction of unique issuance dates in training
VAL_FRAC = 0.20            # fraction of unique issuance dates in validation
MIN_POS_PER_CV_FOLD = 5    # pre-declared minimum defaults per validation fold
MIN_CV_FOLDS = 3           # lowest viable fold count
