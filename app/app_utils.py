"""Shared helpers for the Titanic learning app.

Every page imports from here. The whole point of the app is that it runs the
project's REAL pipeline, so this module adds ../src to the path and imports
`build_features` from src/pipeline.py rather than reimplementing any feature
engineering.

Caching strategy (see APP_README.md "Architecture notes"):
  - @st.cache_data for CSV loading + feature building (returns DataFrames).
  - @st.cache_data for CV scores (returns plain floats), keyed on the
    hyperparameters + feature flags so a slider nudge only retrains when an
    input that matters actually changes.
  - @st.cache_resource for fitted model objects.

Windows joblib gotcha (CLAUDE.md lesson 6): never set n_jobs=-1 on BOTH the
outer cross_val_score and the inner estimator — nested process pools hang on
Windows. We let the estimator parallelise internally (n_jobs=-1) and keep the
outer CV loop serial (n_jobs=1).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- make src/pipeline.py importable from inside app/ -----------------------
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline import build_features  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Match the iterations exactly so in-app CV numbers line up with README /
# outputs/cv_scores_*.txt.
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
RANDOM_STATE = 42
TRAIN_RATE = 0.384  # project-wide reference survival rate
HONEST_CEILING = 0.82
GAP_DANGER = 0.06  # CLAUDE.md lesson 3

# The project's best RF hyperparameters (Optuna-discovered in iter 5, used for
# iters 6 and 9). class_weight is deliberately None (iter 4's lesson).
BEST_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "min_samples_split": 3,
    "max_features": 0.5,
    "max_samples": 0.6448931749101726,
    "criterion": "entropy",
    "class_weight": None,
}

# Seeds used for iter 9's 5-seed averaging.
SEEDS = [42, 0, 1, 7, 13]


# ---------------------------------------------------------------------------
# Data + features
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw train + test frames, for screens that show the unprocessed data."""
    train = pd.read_csv(os.path.join(_DATA, "train.csv"))
    test = pd.read_csv(os.path.join(_DATA, "test.csv"))
    return train, test


@st.cache_data(show_spinner=False)
def get_features(include_family_survival: bool, include_ticket_group: bool):
    """Engineered (X_train, y_train, X_test, test_ids) from the REAL pipeline.

    NOTE: pipeline.build_features reads the CSVs itself and takes only the two
    opt-in flags — it does NOT accept train/test frames, and it returns a
    4-tuple (the spec's stub was wrong about both; verified against
    src/pipeline.py).
    """
    X_train, y_train, X_test, test_ids = build_features(
        include_family_survival=include_family_survival,
        include_ticket_group=include_ticket_group,
    )
    return X_train, y_train, X_test, test_ids


# ---------------------------------------------------------------------------
# Model fitting + scoring (cached on hyperparameters)
# ---------------------------------------------------------------------------
def _params_from_items(param_items: tuple) -> dict:
    """Cache keys must be hashable, so callers pass sorted (key, value) tuples.
    This turns them back into a kwargs dict for the estimator."""
    return dict(param_items)


@st.cache_data(show_spinner=False)
def cv_accuracy_rf(include_family_survival: bool, include_ticket_group: bool, param_items: tuple):
    """10-fold stratified CV accuracy (mean, std) for an RF with the given
    params on the given feature set. Cached on every argument.

    param_items: tuple(sorted(params.items())) — a hashable view of the RF
    hyperparameters. Use rf_param_items(...) to build it.
    """
    params = _params_from_items(param_items)
    X_train, y_train, _, _ = get_features(include_family_survival, include_ticket_group)
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std())


@st.cache_resource(show_spinner=False)
def fit_rf(include_family_survival: bool, include_ticket_group: bool, param_items: tuple, seed: int = RANDOM_STATE):
    """Fit an RF on the full training set and return the fitted model.
    Cached as a resource (model object) keyed on flags + params + seed."""
    params = _params_from_items(param_items)
    X_train, y_train, _, _ = get_features(include_family_survival, include_ticket_group)
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    return model


def rf_param_items(**overrides) -> tuple:
    """Build the hashable param tuple for cv_accuracy_rf / fit_rf, starting
    from BEST_RF_PARAMS and applying any overrides (e.g. max_depth=8)."""
    params = {**BEST_RF_PARAMS, **overrides}
    return tuple(sorted(params.items()))


def predicted_rate(model, X_test) -> float:
    """Fraction of test rows the model predicts as survived (class 1).
    The project's calibration canary — compare against TRAIN_RATE (0.384)."""
    return float(model.predict(X_test).mean())
