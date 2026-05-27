"""
Iteration 5: Optuna tuning WITHOUT class_weight (clean A/B vs iter 3).

Iter 4 found CV 0.857 / LB 0.780 -- a 1.2pt LB drop vs iter 3 (CV 0.842
/ LB 0.792). The culprit was identified as `class_weight=balanced_subsample`,
which inflates CV by weighting positives up but pushes predicted
survival rate from 0.37 to 0.41, hurting LB.

This iteration removes class_weight from the search space entirely
(forced to None). If the OTHER tuning changes from iter 4
(min_samples_leaf, max_features, max_samples, etc.) were real
improvements, we should see:
  - CV close to iter 4's 0.857
  - LB above iter 3's 0.792
  - Predicted survival rate back near 0.37

If LB doesn't beat iter 3, then iter 3's defaults were already near
the ceiling for plain RF on these features, and the entire CV gain
in iter 4 was the class-weight cheat.

This is a deliberately conservative iteration: same model family, same
features, just better regularization knobs (hopefully).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42
N_TRIALS = 60


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]
        ),
        "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        # class_weight FORCED to None -- the lesson from iter 4 is that
        # this lever inflates CV without helping LB on a dataset with
        # known and shared train/test class balance.
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return scores.mean()


def main() -> None:
    print("[1/5] Loading features (with FamilySurvival) ...")
    X_train, y_train, X_test, test_ids = build_features(include_family_survival=True)
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print(f"\n[2/5] Optuna tuning RF (class_weight=None forced) -- {N_TRIALS} trials ...")
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
        print(f"        {k}: {v}")

    print("\n[3/5] Final 10-fold CV with best params ...")
    best_params = {
        **study.best_params,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**best_params)
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    final_scores = cross_val_score(model, X_train, y_train, cv=cv10, scoring="accuracy", n_jobs=-1)
    print(f"      Mean: {final_scores.mean():.4f}  Std: {final_scores.std():.4f}")
    for i, s in enumerate(final_scores, 1):
        print(f"        fold {i:2d}: {s:.4f}")

    print("\n[4/5] Refit on full training set ...")
    model.fit(X_train, y_train)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v5.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 5 -- Optuna-tuned RF, class_weight forced to None\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optuna best (repeated 5-fold x2): {study.best_value:.4f}\n")
        f.write(f"Final 10-fold CV mean:             {final_scores.mean():.4f}\n")
        f.write(f"Final 10-fold CV std:              {final_scores.std():.4f}\n\n")
        f.write("Best params (class_weight forced to None):\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop 15 feature importances:\n")
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
        print(f"      WARNING: predicted rate drifted >0.02 from train rate -- possible class-weight artifact")
    print("\nDone.")


if __name__ == "__main__":
    main()
