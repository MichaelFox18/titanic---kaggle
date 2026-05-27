"""
Iteration 9: Multi-seed RF averaging (variance reduction).

Strategy: train iter 6's tuned RF five times with five different
random_state values, then average their P(survived) on test rows.
Threshold at 0.5.

Why this is safe:
- Same model family that's already proven on LB (RF won iters 3, 5, 6).
- Same features (no FE risk).
- Same hyperparameters (no tuning CV-mirage risk).
- Only the random seed varies, which controls bagging/feature-subset
  draws. Averaging multiple RFs cancels seed-dependent noise that a
  single RF cannot.

Expected impact: +0.1-0.3pt LB. Downside is essentially zero -- in the
worst case it ties iter 6.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PRIMARY_SEED = 42
SEEDS = [42, 0, 1, 7, 13]

# iter-6 best params (Optuna-discovered in iter 5)
ITER6_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "min_samples_split": 3,
    "max_features": 0.5,
    "max_samples": 0.6448931749101726,
    "criterion": "entropy",
    "class_weight": None,
    "n_jobs": -1,
}


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**ITER6_RF_PARAMS, random_state=seed)


def main() -> None:
    print("[1/4] Loading features (FamilySurvival + ticket-group) ...")
    X_train, y_train, X_test, test_ids = build_features(
        include_family_survival=True, include_ticket_group=True
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print(f"\n[2/4] Training {len(SEEDS)} RFs with seeds {SEEDS} and computing 10-fold CV ...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=PRIMARY_SEED)

    # First: per-seed individual CV scores, for a sanity baseline.
    seed_scores = []
    for s in SEEDS:
        sc = cross_val_score(make_rf(s), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        print(f"      seed {s:>2}: CV mean {sc.mean():.4f}  (std {sc.std():.4f})")
        seed_scores.append(sc.mean())
    print(f"      Mean of per-seed CV means: {np.mean(seed_scores):.4f}")

    # Then: averaged-probability OOF CV.
    # For each fold, train all 5 RFs on the train portion, average
    # their predicted probabilities on the holdout, threshold at 0.5,
    # and score against true labels.
    print("\n[3/4] 10-fold CV of the AVERAGED ensemble (avg probs across all 5 seeds) ...")
    oof_probs = np.zeros(len(y_train), dtype=float)
    for fold_idx, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        fold_probs = np.zeros(len(va), dtype=float)
        for s in SEEDS:
            m = make_rf(s)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            fold_probs += m.predict_proba(X_train.iloc[va])[:, 1]
        fold_probs /= len(SEEDS)
        oof_probs[va] = fold_probs
    oof_preds = (oof_probs >= 0.5).astype(int)
    ens_cv = (oof_preds == y_train.values).mean()
    print(f"      Averaged-ensemble OOF accuracy: {ens_cv:.4f}")
    print(f"      Single-seed mean CV:            {np.mean(seed_scores):.4f}")
    print(f"      Delta:                          {ens_cv - np.mean(seed_scores):+.4f}")

    # Final: refit each RF on full train, average test probs.
    print("\n[4/4] Refit on full train + writing submission ...")
    test_probs = np.zeros(len(X_test), dtype=float)
    for s in SEEDS:
        m = make_rf(s)
        m.fit(X_train, y_train)
        test_probs += m.predict_proba(X_test)[:, 1]
    test_probs /= len(SEEDS)
    preds = (test_probs >= 0.5).astype(int)

    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    pred_rate = preds.mean()
    train_rate = y_train.mean()
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")
    print(f"      Predicted survival rate: {pred_rate:.3f}  (train rate: {train_rate:.3f}, drift: {pred_rate - train_rate:+.3f})")
    if abs(pred_rate - train_rate) > 0.025:
        print(f"      WARNING: predicted rate drifted >0.025 from train rate")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v9.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 9 -- Multi-seed RF averaging\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Seeds used: {SEEDS}\n\n")
        f.write("Per-seed CV (10-fold):\n")
        for s, sc in zip(SEEDS, seed_scores):
            f.write(f"  seed {s:>2}: {sc:.4f}\n")
        f.write(f"\nMean of per-seed CVs:           {np.mean(seed_scores):.4f}\n")
        f.write(f"Averaged-ensemble OOF accuracy: {ens_cv:.4f}\n")
        f.write(f"Delta:                          {ens_cv - np.mean(seed_scores):+.4f}\n\n")
        f.write(f"Test predicted survival rate:   {pred_rate:.4f}\n")
        f.write(f"Train survival rate:            {train_rate:.4f}\n")
    print(f"      Wrote {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
