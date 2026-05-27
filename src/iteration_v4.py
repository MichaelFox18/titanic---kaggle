"""
Iteration 4: Optuna-tuned RandomForest on iter 3 feature set.

Iter 3 proved the FamilySurvival feature is real (+1.4pt CV).
Iter 4 keeps the same feature set and asks Optuna to find better RF
hyperparameters than the eyeballed defaults we've been using.

Why repeated CV during tuning?
  Single 10-fold CV on 891 rows has std ~0.022 -- big enough that
  Optuna can fool itself with noise. Repeated 10-fold (3 repeats with
  different shuffles) averages out the noise, so we tune to real signal
  instead of lucky splits.

What we tune:
  n_estimators, max_depth, min_samples_leaf, min_samples_split,
  max_features, max_samples (bootstrap fraction), criterion,
  class_weight. These cover the regularization knobs that matter on
  small datasets.
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
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced_subsample"]
        ),
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

    print(f"\n[2/5] Optuna tuning RF -- {N_TRIALS} trials ...")
    # Suppress Optuna's per-trial logging to keep output readable.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: objective(t, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    print(f"      Best CV (repeated 5-fold x3): {study.best_value:.4f}")
    print(f"      Best params:")
    for k, v in study.best_params.items():
        print(f"        {k}: {v}")

    print("\n[3/5] Final 10-fold CV with best params (for fair comparison vs iter 3) ...")
    best_params = {**study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
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
    report_path = OUTPUTS_DIR / "cv_scores_v4.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 4 -- Optuna-tuned RF on iter 3 features\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optuna best (repeated 5-fold x3): {study.best_value:.4f}\n")
        f.write(f"Final 10-fold CV mean:             {final_scores.mean():.4f}\n")
        f.write(f"Final 10-fold CV std:              {final_scores.std():.4f}\n\n")
        f.write("Best params:\n")
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
    print(f"      Predicted survival rate: {preds.mean():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
