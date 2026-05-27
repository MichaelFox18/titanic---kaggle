"""
Iteration 3: back to single Random Forest, plus FamilySurvival feature.

Why this approach?
- Iter 2's ensemble dropped LB by 3 points despite higher CV. The
  boosting members (GB / XGB / LGB) were overfitting noise on the 891-row
  train set; their high-confidence wrong votes dragged the soft average
  away from RF's regularized predictions.
- RF alone scored 0.7775 on LB. Keep that as the workhorse.
- Add the FamilySurvival feature -- the single highest-impact known
  Titanic trick. For each passenger, look at whether OTHER members of
  their family (Surname+Fare) or ticket group survived/died in train.
- Compare CV with and without the feature to confirm it helps before
  trusting the submission.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main() -> None:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    print("[1/4] Loading features WITHOUT FamilySurvival (baseline check) ...")
    X_tr_base, y, X_te_base, ids = build_features(include_family_survival=False)
    s_base = cross_val_score(make_rf(), X_tr_base, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      RF baseline CV mean: {s_base.mean():.4f}  (std {s_base.std():.4f})")

    print("\n[2/4] Loading features WITH FamilySurvival ...")
    X_tr, y, X_te, ids = build_features(include_family_survival=True)
    print(f"      Train: {X_tr.shape}  Test: {X_te.shape}")
    s = cross_val_score(make_rf(), X_tr, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      Per-fold accuracy:")
    for i, val in enumerate(s, 1):
        print(f"        fold {i:2d}: {val:.4f}")
    print(f"      Mean:  {s.mean():.4f}")
    print(f"      Std:   {s.std():.4f}")
    print(f"      Delta from baseline: {s.mean() - s_base.mean():+.4f}")

    print("\n[3/4] Refit on full training set ...")
    model = make_rf()
    model.fit(X_tr, y)

    importances = pd.Series(
        model.feature_importances_, index=X_tr.columns
    ).sort_values(ascending=False)
    print("      Top 15 features:")
    for name, val in importances.head(15).items():
        print(f"        {val:.4f}  {name}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_path = OUTPUTS_DIR / "cv_scores_v3.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 3 -- RF + FamilySurvival feature\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Baseline (no FS) CV mean: {s_base.mean():.4f}\n")
        f.write(f"With FS         CV mean: {s.mean():.4f}\n")
        f.write(f"Delta:                    {s.mean() - s_base.mean():+.4f}\n\n")
        f.write("Per-fold (with FS):\n")
        for i, val in enumerate(s, 1):
            f.write(f"  fold {i:2d}: {val:.4f}\n")
        f.write("\nTop 15 feature importances:\n")
        for name, val in importances.head(15).items():
            f.write(f"  {val:.4f}  {name}\n")
    print(f"      Wrote {report_path}")

    print("\n[4/4] Writing submission ...")
    preds = model.predict(X_te).astype(int)
    submission = pd.DataFrame({"PassengerId": ids, "Survived": preds})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}  ({len(submission)} rows)")
    print(f"      Predicted survival rate: {preds.mean():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
