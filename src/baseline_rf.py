"""
Random Forest baseline for Titanic survival prediction.

LEARNING NOTES
==============
This script does three things:

  1. Loads engineered features via pipeline.build_features().
  2. Evaluates a Random Forest with 10-fold Stratified Cross-Validation.
  3. Refits on ALL training data and writes data/submission.csv.

Why these three steps, and why in this order?

  EVALUATE BEFORE PREDICTING. We need to know how good the model is
  BEFORE we use it to predict on test. The cross-validation score is
  our honest estimate of leaderboard performance. If we skipped this
  and just trained-and-predicted, we'd have a submission but no idea
  whether it's any good.

  STRATIFIED K-FOLD specifically: this splits training data into 10
  chunks ("folds") and, for each fold, trains on the other 9 and scores
  on the held-out 1. The "stratified" part means each fold preserves
  the 38.4% survival rate of the full dataset -- without stratification,
  random folds can swing 30% to 45% survival, which makes scores
  noisier and harder to compare. Final score = mean of 10 fold scores
  (with std telling us how stable the model is).

  REFIT ON FULL DATA. CV is for *estimating* performance. Once we
  trust the estimate, we throw away the CV models and train one final
  model on ALL 891 rows -- more data, better model. That's the one
  whose predictions go in submission.csv.

About RANDOM FORESTS specifically:
  A Decision Tree asks a series of yes/no questions ("Is Sex_male=1?
  Yes -> ask about Age; No -> ask about Pclass...") and ends at a
  leaf that predicts a class. A single tree overfits like crazy on
  small data -- it can memorize the training set with deep enough
  branches.
  A Random Forest fixes this by training MANY trees (we use 500),
  each on a random subset of rows AND columns, and averaging their
  votes. Individual trees overfit in different directions; averaging
  cancels out their noise and leaves the real signal.
  Key hyperparameters we use:
    n_estimators=500   -- more trees = more stable vote (diminishing returns past a few hundred)
    max_depth=6        -- caps tree depth to prevent memorization (regularization)
    min_samples_leaf=2 -- a leaf needs >=2 samples, prevents leaves that fit a single passenger
    random_state=42    -- makes the run reproducible
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    print("[1/4] Loading + engineering features ...")
    X_train, y_train, X_test, test_ids = build_features()
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"      Class balance (survival rate): {y_train.mean():.3f}")
    print(f"      Feature count: {X_train.shape[1]}")

    # The model. See top-of-file notes for why these hyperparameters.
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        max_features="sqrt",  # at each split, consider sqrt(n_features) -- decorrelates trees
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )

    # Stratified 10-fold CV. shuffle=True randomizes row order before
    # splitting (the data isn't sorted by survival here, but shuffling
    # is good practice and guards against future surprises).
    print("\n[2/4] Running 10-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print("      Per-fold accuracy:")
    for i, s in enumerate(scores, 1):
        print(f"        fold {i:2d}: {s:.4f}")
    print(f"      Mean:  {scores.mean():.4f}")
    print(f"      Std:   {scores.std():.4f}")
    print(f"      Range: {scores.min():.4f} -- {scores.max():.4f}")

    # Refit on ALL training data and grab feature importances.
    # importances_ is the fraction of total impurity reduction
    # contributed by each feature, averaged across all 500 trees.
    # It's a rough "how much did the model lean on this column" score.
    print("\n[3/4] Refitting on full training set ...")
    model.fit(X_train, y_train)
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print("      Top 20 features:")
    for name, val in importances.head(20).items():
        print(f"        {val:.4f}  {name}")

    # Persist CV report + importances to a text file so future-us can
    # compare runs without re-running the script.
    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Random Forest baseline -- 10-fold Stratified CV\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-fold accuracy:\n")
        for i, s in enumerate(scores, 1):
            f.write(f"  fold {i:2d}: {s:.4f}\n")
        f.write(f"\nMean : {scores.mean():.4f}\n")
        f.write(f"Std  : {scores.std():.4f}\n")
        f.write(f"Min  : {scores.min():.4f}\n")
        f.write(f"Max  : {scores.max():.4f}\n\n")
        f.write("Top 20 feature importances:\n")
        for name, val in importances.head(20).items():
            f.write(f"  {val:.4f}  {name}\n")
    print(f"      Wrote {report_path}")

    # Generate predictions on the test set and save submission.csv.
    # Kaggle wants integers (0/1), not floats (0.0/1.0).
    print("\n[4/4] Predicting on test set + writing submission ...")
    preds = model.predict(X_test).astype(int)
    submission = pd.DataFrame(
        {"PassengerId": test_ids, "Survived": preds}
    )
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")
    print(f"      Predicted survival rate: {preds.mean():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
