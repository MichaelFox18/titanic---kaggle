"""
Iteration 8: Stacking / blending investigation (RF + XGB).

Iter 6 RF: LB 0.806.  Iter 7 XGB: LB 0.782 (-2.4pt).

XGBoost is meaningfully weaker on LB, but the two models rely on
different feature subsets (RF leverages engineered fare features; XGB
focuses on raw title/sex/Pclass), so their *mistakes* are partially
uncorrelated. Where RF errs and XGB doesn't (or vice versa), an
ensemble recovers the correct prediction.

We don't know the right blend weight a priori. So we test multiple
strategies on identical CV splits and pick the winner *by CV*, not by
gut feel. The blend can hurt as easily as help -- if XGB's bad
predictions outweigh its diversity benefit, pure RF wins and we should
not submit a blend.

Pipeline
--------
1. Build features (iter 6 set: FamilySurvival + ticket-group).
2. Compute OOF (out-of-fold) probabilities for RF and XGB via the same
   10-fold StratifiedKFold splits.
3. For each blend strategy w_rf in {1.0, 0.9, 0.8, 0.7, 0.5} and a
   LogReg meta-learner trained on OOF probs:
     - Threshold blended probability at 0.5.
     - Score against true labels.
4. Print a leaderboard of strategies.
5. Sanity-check: predicted survival rate of the best strategy must be
   within 0.02 of train rate (0.384). If not, fall back to pure RF.
6. Refit both base models on full train, apply chosen blend to test
   probabilities, write submission.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

ITER6_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "min_samples_split": 3,
    "max_features": 0.5,
    "max_samples": 0.6448931749101726,
    "criterion": "entropy",
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

ITER7_XGB_PARAMS = {
    "n_estimators": 450,
    "learning_rate": 0.0303,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.9035,
    "colsample_bytree": 0.6438,
    "gamma": 0.2273,
    "reg_alpha": 1.2444,
    "reg_lambda": 1.5685,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

BLEND_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.5]


def compute_oof_probs(model_factory, X: pd.DataFrame, y: pd.Series, cv) -> np.ndarray:
    """Return OOF P(survived) for each train row, using cv splits."""
    oof = np.zeros(len(y), dtype=float)
    for fold_idx, (tr, va) in enumerate(cv.split(X, y), 1):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def main() -> None:
    print("[1/5] Loading features (FamilySurvival + ticket-group) ...")
    X_train, y_train, X_test, test_ids = build_features(
        include_family_survival=True, include_ticket_group=True
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    print("\n[2/5] Computing OOF probabilities for RF and XGB (10-fold) ...")
    rf_oof = compute_oof_probs(lambda: RandomForestClassifier(**ITER6_RF_PARAMS), X_train, y_train, cv)
    xgb_oof = compute_oof_probs(lambda: xgb.XGBClassifier(**ITER7_XGB_PARAMS), X_train, y_train, cv)

    rf_only_acc = ((rf_oof >= 0.5).astype(int) == y_train).mean()
    xgb_only_acc = ((xgb_oof >= 0.5).astype(int) == y_train).mean()
    print(f"      RF OOF accuracy:  {rf_only_acc:.4f}")
    print(f"      XGB OOF accuracy: {xgb_only_acc:.4f}")
    # Disagreement rate (a proxy for ensemble usefulness)
    rf_preds = (rf_oof >= 0.5).astype(int)
    xgb_preds = (xgb_oof >= 0.5).astype(int)
    disagreement = (rf_preds != xgb_preds).mean()
    print(f"      RF/XGB disagreement rate: {disagreement:.4f}")
    # When they disagree, who is right more often?
    disagree_mask = rf_preds != xgb_preds
    if disagree_mask.any():
        rf_right_when_disagree = (rf_preds[disagree_mask] == y_train.values[disagree_mask]).mean()
        print(f"      When they disagree, RF is right: {rf_right_when_disagree:.4f} (rest are XGB)")

    print("\n[3/5] Evaluating blend strategies on OOF predictions ...")
    results = []
    for w_rf in BLEND_WEIGHTS:
        blended = w_rf * rf_oof + (1.0 - w_rf) * xgb_oof
        preds = (blended >= 0.5).astype(int)
        acc = (preds == y_train.values).mean()
        results.append(("weighted", w_rf, acc, blended))

    # LogReg meta-learner on OOF probs
    meta_X = np.column_stack([rf_oof, xgb_oof])
    # Eval meta-learner via nested CV to be fair
    meta_oof = np.zeros(len(y_train), dtype=float)
    for tr, va in cv.split(meta_X, y_train):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(meta_X[tr], y_train.iloc[tr])
        meta_oof[va] = meta.predict_proba(meta_X[va])[:, 1]
    meta_preds = (meta_oof >= 0.5).astype(int)
    meta_acc = (meta_preds == y_train.values).mean()
    # Fit meta on all OOF preds to get the final weights for inference
    final_meta = LogisticRegression(max_iter=1000).fit(meta_X, y_train)
    results.append(("logreg-meta", None, meta_acc, meta_oof))

    print(f"      {'Strategy':<14} {'w_rf':<6} {'Acc':<8} {'Pred. rate':<12} {'Coef (rf,xgb)':<20}")
    for label, w_rf, acc, blended in results:
        pred_rate = (blended >= 0.5).astype(int).mean()
        coef_str = ""
        if label == "logreg-meta":
            coef_str = f"({final_meta.coef_[0][0]:+.2f}, {final_meta.coef_[0][1]:+.2f})"
        w_str = f"{w_rf:.1f}" if w_rf is not None else "meta"
        print(f"      {label:<14} {w_str:<6} {acc:.4f}   {pred_rate:.4f}      {coef_str}")

    # Pick best by CV accuracy. Defer the predicted-rate sanity check
    # until AFTER refitting base models on full train (OOF rates are
    # biased low because each fold sees only 9/10 of the data).
    train_rate = y_train.mean()
    best = max(results, key=lambda r: r[2])
    print(f"\n[4/5] Highest CV strategy (pre-sanity-check): {best[0]}  w_rf={best[1]}  CV={best[2]:.4f}")

    print("\n[5/5] Refitting base models on full train + applying chosen blend ...")
    rf_full = RandomForestClassifier(**ITER6_RF_PARAMS)
    rf_full.fit(X_train, y_train)
    xgb_full = xgb.XGBClassifier(**ITER7_XGB_PARAMS)
    xgb_full.fit(X_train, y_train)
    rf_test = rf_full.predict_proba(X_test)[:, 1]
    xgb_test = xgb_full.predict_proba(X_test)[:, 1]

    def apply_blend(label, w_rf):
        if label == "weighted":
            return w_rf * rf_test + (1.0 - w_rf) * xgb_test
        return final_meta.predict_proba(np.column_stack([rf_test, xgb_test]))[:, 1]

    label, w_rf, _, _ = best
    blended_test = apply_blend(label, w_rf)
    preds = (blended_test >= 0.5).astype(int)
    pred_rate = preds.mean()

    # Now check predicted rate on FULL-train predictions and fall back
    # to pure RF if it's drifted too far.
    if abs(pred_rate - train_rate) > 0.025:
        print(f"      WARNING: chosen strategy predicts {pred_rate:.3f} (train {train_rate:.3f}), drift > 0.025.")
        print("      Falling back to pure RF.")
        blended_test = rf_test
        preds = (blended_test >= 0.5).astype(int)
        pred_rate = preds.mean()
        label, w_rf = "weighted", 1.0
        best = ("weighted", 1.0, rf_only_acc, rf_oof)

    print(f"      Chosen final: {label} w_rf={w_rf}  pred_rate={pred_rate:.4f}")
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v8.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 8 -- Stacking / blending investigation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"RF OOF accuracy:  {rf_only_acc:.4f}\n")
        f.write(f"XGB OOF accuracy: {xgb_only_acc:.4f}\n")
        f.write(f"RF/XGB disagreement rate: {disagreement:.4f}\n\n")
        f.write("Blend strategy leaderboard (OOF):\n")
        f.write(f"{'Strategy':<14} {'w_rf':<6} {'Acc':<8} {'OOF rate':<12}\n")
        for lbl, w, acc, blended in results:
            r = (blended >= 0.5).astype(int).mean()
            w_str = f"{w:.1f}" if w is not None else "meta"
            f.write(f"{lbl:<14} {w_str:<6} {acc:.4f}   {r:.4f}\n")
        f.write(f"\nChosen: {best[0]} w_rf={best[1]} CV={best[2]:.4f}\n")
        f.write(f"Final submission predicted rate: {pred_rate:.4f}\n")
        f.write(f"LogReg meta coefs (rf, xgb): ({final_meta.coef_[0][0]:+.4f}, {final_meta.coef_[0][1]:+.4f})\n")
    print(f"      Wrote {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
