"""
Iter 7 finish: the Optuna phase already converged in the previous run
(CV 0.8530); only the final 10-fold CV / refit / submission failed due
to a Windows paging-file issue with nested parallelism.

This script just runs the tail end with the known-best params and
n_jobs=1 for the outer CV loop, avoiding the nested-pool blowout.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

# Best params from the iter-7 Optuna run (which completed before the
# paging-file error during the final CV step). Optuna CV: 0.8530.
BEST_PARAMS = {
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


def main() -> None:
    print("[1/4] Loading features (FamilySurvival + ticket-group) ...")
    X_train, y_train, X_test, test_ids = build_features(
        include_family_survival=True, include_ticket_group=True
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print("\n[2/4] 10-fold CV with iter-7 best params (n_jobs=1 outer to avoid paging issue) ...")
    model = xgb.XGBClassifier(**BEST_PARAMS)
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv10, scoring="accuracy", n_jobs=1)
    print(f"      Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    for i, s in enumerate(scores, 1):
        print(f"        fold {i:2d}: {s:.4f}")

    print("\n[3/4] Refit on full training set ...")
    model.fit(X_train, y_train)
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print("      Top 15 features (XGBoost gain):")
    for name, val in importances.head(15).items():
        print(f"        {val:.4f}  {name}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v7.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 7 -- Optuna-tuned regularized XGBoost\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optuna best (repeated 5-fold x2): 0.8530\n")
        f.write(f"Final 10-fold CV mean:             {scores.mean():.4f}\n")
        f.write(f"Final 10-fold CV std:              {scores.std():.4f}\n\n")
        f.write("Best params:\n")
        for k, v in BEST_PARAMS.items():
            if k in ("eval_metric", "random_state", "n_jobs", "verbosity"):
                continue
            f.write(f"  {k}: {v}\n")
        f.write("\nTop 15 feature importances (XGBoost gain):\n")
        for name, val in importances.head(15).items():
            f.write(f"  {val:.4f}  {name}\n")
    print(f"      Wrote {report_path}")

    print("\n[4/4] Writing submission ...")
    preds = model.predict(X_test).astype(int)
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")
    pred_rate = preds.mean()
    print(f"      Predicted survival rate: {pred_rate:.3f}  (train rate: {y_train.mean():.3f})")
    if abs(pred_rate - y_train.mean()) > 0.02:
        print(f"      WARNING: predicted rate drifted >0.02 from train rate")
    print("\nDone.")


if __name__ == "__main__":
    main()
