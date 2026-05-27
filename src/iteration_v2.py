"""
Iteration 2: Soft-voting ensemble of RF + XGBoost + LightGBM + GradientBoosting.

Why an ensemble over a single tuned model? Each model family makes
different kinds of mistakes:
  - Random Forest votes by averaging deep-ish trees on bagged samples
  - Gradient boosting fits residuals sequentially (XGB / LGB / sklearn GB)
  - The three boosters use different splitting rules (XGB exact / hist,
    LGB leaf-wise, sklearn GB level-wise), so they disagree on edge cases

Averaging their probabilities (soft voting) tends to beat any single
member by 1-2 points because their errors are partially uncorrelated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import lightgbm as lgb
import xgboost as xgb

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def build_ensemble() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=4,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    return VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
        ],
        voting="soft",
        n_jobs=-1,
    )


def main() -> None:
    print("[1/4] Loading + engineering features ...")
    X_train, y_train, X_test, test_ids = build_features()
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print("\n[2/4] Building soft-voting ensemble (RF + GB + XGB + LGB) ...")
    model = build_ensemble()

    print("\n[2b/4] 10-fold Stratified CV on ensemble + each member ...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Also report per-model CV so we know which member is carrying the team.
    per_model_means = {}
    for name, est in model.estimators:
        s = cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        per_model_means[name] = (s.mean(), s.std())
        print(f"      {name:5s}: {s.mean():.4f}  (std {s.std():.4f})")

    ensemble_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print("\n      Ensemble per-fold accuracy:")
    for i, s in enumerate(ensemble_scores, 1):
        print(f"        fold {i:2d}: {s:.4f}")
    print(f"      Ensemble Mean:  {ensemble_scores.mean():.4f}")
    print(f"      Ensemble Std:   {ensemble_scores.std():.4f}")

    print("\n[3/4] Refitting ensemble on full training set ...")
    model.fit(X_train, y_train)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v2.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 2 -- Soft-voting ensemble (RF + GB + XGB + LGB)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-model 10-fold CV means:\n")
        for name, (mean, std) in per_model_means.items():
            f.write(f"  {name:5s}: {mean:.4f}  (std {std:.4f})\n")
        f.write("\nEnsemble per-fold accuracy:\n")
        for i, s in enumerate(ensemble_scores, 1):
            f.write(f"  fold {i:2d}: {s:.4f}\n")
        f.write(f"\nEnsemble Mean : {ensemble_scores.mean():.4f}\n")
        f.write(f"Ensemble Std  : {ensemble_scores.std():.4f}\n")
        f.write(f"Ensemble Min  : {ensemble_scores.min():.4f}\n")
        f.write(f"Ensemble Max  : {ensemble_scores.max():.4f}\n")
    print(f"      Wrote {report_path}")

    print("\n[4/4] Predicting on test set + writing submission ...")
    preds = model.predict(X_test).astype(int)
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")
    print(f"      Predicted survival rate: {preds.mean():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
