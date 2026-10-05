"""
07_model_training.py
Trains LR, RF, XGBoost with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
ARTIFACTS = Path("results/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURE_COLS = [
    "total_inflow", "total_outflow", "avg_monthly_inflow", "avg_monthly_outflow",
    "inflow_volatility", "outflow_volatility", "transaction_frequency",
    "avg_balance", "min_balance", "max_balance", "balance_volatility",
    "active_months", "account_tenure_days",
    "loan_amount", "loan_duration", "loan_payment",
]

def load_splits():
    train = pd.read_csv(PROCESSED / "train.csv")
    val   = pd.read_csv(PROCESSED / "val.csv")
    return train, val

def prepare(train, val):
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["target"]
    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    return X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled, scaler

def train_logreg(X_train, y_train):
    print("Training Logistic Regression...")
    params = {"C": [0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        params, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best: {model.best_params_}  Time: {elapsed:.2f}s")
    return model, elapsed

def train_rf(X_train, y_train):
    print("Training Random Forest...")
    params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    model = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        params, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best: {model.best_params_}  Time: {elapsed:.2f}s")
    return model, elapsed

def train_xgb(X_train, y_train):
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
    }
    model = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1
        ),
        params, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best: {model.best_params_}  Time: {elapsed:.2f}s")
    return model, elapsed

def main():
    train, val = load_splits()
    X_train, y_train, X_val, y_val, X_train_s, X_val_s, scaler = prepare(train, val)

    lr, t_lr = train_logreg(X_train_s, y_train)
    rf, t_rf = train_rf(X_train, y_train)
    xgb, t_xgb = train_xgb(X_train, y_train)

    joblib.dump(lr, ARTIFACTS / "logreg.pkl")
    joblib.dump(rf, ARTIFACTS / "rf.pkl")
    joblib.dump(xgb, ARTIFACTS / "xgb.pkl")
    joblib.dump(scaler, ARTIFACTS / "scaler.pkl")

    best_params = {
        "logreg": lr.best_params_,
        "rf": rf.best_params_,
        "xgb": xgb.best_params_,
        "xgb_scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
    }
    times = {"logreg": t_lr, "rf": t_rf, "xgb": t_xgb}
    with open(ARTIFACTS / "training_times.json", "w") as f:
        json.dump(times, f, indent=2)
    with open(ARTIFACTS / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Training times: {times}")

if __name__ == "__main__":
    main()