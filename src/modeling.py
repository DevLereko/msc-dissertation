"""
modeling.py
Model construction shared by training (07) and stability analysis (11).

- Logistic Regression is a Pipeline(StandardScaler -> LR), so scaling is
  fitted only on whatever rows the model is fitted on (no preprocessing leakage).
- XGBoost scale_pos_weight is computed from the labels of the rows the model
  is fitted on, never from validation or test labels.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import RANDOM_SEED

MODEL_NAMES = ["logreg", "rf", "xgb"]
DISPLAY = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}

# Same grids as baseline_v1 so that only the protocol differs between experiments
PARAM_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1, 10, 100]},
    "rf": {"n_estimators": [100, 200], "max_depth": [10, 20, None],
           "min_samples_split": [2, 5]},
    "xgb": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 7]},
}


class NoPositivesError(ValueError):
    """Raised when a fitting set has no defaults (fold must be rejected)."""


def scale_pos_weight(y_fit):
    y_fit = np.asarray(y_fit)
    n_pos, n_neg = int((y_fit == 1).sum()), int((y_fit == 0).sum())
    if n_pos == 0:
        raise NoPositivesError("fitting set contains no positive cases")
    return n_neg / n_pos


def build_model(name, params, y_fit):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=RANDOM_SEED, **params)),
        ])
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED,
                                      n_jobs=-1, **params)
    if name == "xgb":
        return XGBClassifier(scale_pos_weight=scale_pos_weight(y_fit),
                             random_state=RANDOM_SEED, eval_metric="logloss",
                             n_jobs=-1, **params)
    raise KeyError(name)


def fit_model(name, params, X, y):
    if int(np.asarray(y).sum()) == 0:
        raise NoPositivesError("fitting set contains no positive cases")
    model = build_model(name, params, y)
    model.fit(X, y)
    return model
