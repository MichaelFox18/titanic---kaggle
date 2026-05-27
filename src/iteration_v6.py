"""
Iteration 6: iter-5 tuned RF + TicketGroupSize + FarePerTicketPerson.

Iter 5 hit LB 0.799 (CV 0.851) with a heavily-regularized RF tuned by
Optuna. Iter 6 keeps that exact model and isolates the contribution of
two new features:

  TicketGroupSize       -- how many passengers share a ticket. Catches
                           travel companions who aren't blood relatives
                           and so don't show up in SibSp/Parch.

  FarePerTicketPerson   -- Fare / TicketGroupSize. Honest per-head
                           ticket cost; the existing FarePerPerson
                           divides by FamilySize, which misattributes
                           cost when unrelated people share a ticket.

A/B design: run the SAME tuned model twice, once without the new
features (= iter 5 reproduction) and once with them, on the same CV
splits. Any CV delta is attributable to the features alone.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline import DATA_DIR, PROJECT_ROOT, build_features


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

# Iter 5's Optuna-discovered best params (class_weight forced to None).
ITER5_PARAMS = {
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


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**ITER5_PARAMS)


def main() -> None:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    print("[1/4] A/B baseline: features WITHOUT ticket-group additions (reproduce iter 5) ...")
    X_tr_base, y, X_te_base, ids = build_features(
        include_family_survival=True, include_ticket_group=False
    )
    s_base = cross_val_score(make_rf(), X_tr_base, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      Shape: {X_tr_base.shape}")
    print(f"      CV mean: {s_base.mean():.4f}  (std {s_base.std():.4f})")

    print("\n[2/4] A/B test: features WITH ticket-group additions ...")
    X_tr, y, X_te, ids = build_features(
        include_family_survival=True, include_ticket_group=True
    )
    print(f"      Shape: {X_tr.shape}")
    s = cross_val_score(make_rf(), X_tr, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      CV mean: {s.mean():.4f}  (std {s.std():.4f})")
    print(f"      Per-fold:")
    for i, val in enumerate(s, 1):
        print(f"        fold {i:2d}: {val:.4f}")
    print(f"      Delta from baseline (iter 5): {s.mean() - s_base.mean():+.4f}")

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
    report_path = OUTPUTS_DIR / "cv_scores_v6.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Iteration 6 -- iter 5 tuned RF + ticket-group features\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Baseline (iter 5 features) CV: {s_base.mean():.4f}\n")
        f.write(f"With ticket-group features:    {s.mean():.4f}\n")
        f.write(f"Delta:                          {s.mean() - s_base.mean():+.4f}\n\n")
        f.write("Per-fold (with ticket-group):\n")
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
    pred_rate = preds.mean()
    print(f"      Predicted survival rate: {pred_rate:.3f}  (train rate: {y.mean():.3f})")
    if abs(pred_rate - y.mean()) > 0.02:
        print(f"      WARNING: predicted rate drifted >0.02 from train rate")
    print("\nDone.")


if __name__ == "__main__":
    main()
