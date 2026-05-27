"""
Iteration 7: Optuna-tuned heavily-regularized XGBoost on iter-6 features.

Why XGBoost now?
  RF (bagging) and XGBoost (boosting) have orthogonal inductive biases.
  RF averages many independent deep trees -- it reduces VARIANCE.
  XGBoost builds many shallow trees sequentially, each correcting the
  previous trees' errors -- it reduces BIAS.
  On the same features, the two models will make DIFFERENT mistakes.
  This iteration tests how XGBoost alone performs; iter 8 will combine
  iter 6 (RF) and iter 7 (XGB) via stacking.

Critical lesson from iter 2:
  XGBoost overfits viciously on 891 rows without aggressive
  regularization. Iter 2's ensemble lost 3 LB points because XGB/LGB
  members fit training noise with high confidence. This iteration
  constrains Optuna to a regularization-friendly search space:
    max_depth 2-6           (shallow trees)
    learning_rate 0.01-0.08 (slow learning)
    min_child_weight 1-10   (require >= N samples per leaf)
    reg_lambda 0-5          (L2 weight penalty)
    reg_alpha 0-2           (L1 sparsity penalty)
    gamma 0-1               (min loss reduction to split)
    subsample 0.6-1.0
    colsample_bytree 0.5-1.0

This is "polite XGBoost" -- it can't be deep, fast, or unregularized.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
N_TRIALS = 60


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return scores.mean()


def main() -> None:
    print("[1/5] Loading features (FamilySurvival + ticket-group) ...")
    X_train, y_train, X_test, test_ids = build_features(
        include_family_survival=True, include_ticket_group=True
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print(f"\n[2/5] Optuna tuning XGBoost (regularized search space) -- {N_TRIALS} trials ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: objective(t, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    print(f"      Best CV (repeated 5-fold x2): {study.best_value:.4f}")
    print(f"      Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"        {k}: {v:.4f}")
        else:
            print(f"        {k}: {v}")

    print("\n[3/5] Final 10-fold CV with best params ...")
    best_params = {
        **study.best_params,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    # XGBoost already uses n_jobs=-1 internally; running outer CV with
    # n_jobs=-1 too spawns nested process pools and blows out Windows'
    # paging file. Keep outer loop serial.
    model = xgb.XGBClassifier(**best_params)
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    final_scores = cross_val_score(model, X_train, y_train, cv=cv10, scoring="accuracy", n_jobs=1)
    print(f"      Mean: {final_scores.mean():.4f}  Std: {final_scores.std():.4f}")
    for i, s in enumerate(final_scores, 1):
        print(f"        fold {i:2d}: {s:.4f}")

    print("\n[4/5] Refit on full training set ...")
    model.fit(X_train, y_train)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print("      Top 15 features (gain-based):")
    for name, val in importances.head(15).items():
        print(f"        {val:.4f}  {name}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v7.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 7 -- Optuna-tuned regularized XGBoost\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optuna best (repeated 5-fold x2): {study.best_value:.4f}\n")
        f.write(f"Final 10-fold CV mean:             {final_scores.mean():.4f}\n")
        f.write(f"Final 10-fold CV std:              {final_scores.std():.4f}\n\n")
        f.write("Best params:\n")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\nTop 15 feature importances (gain):\n")
        for name, val in importances.head(15).items():
            f.write(f"  {val:.4f}  {name}\n")
    print(f"      Wrote {report_path}")

    print("\n[5/5] Writing submission ...")
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
