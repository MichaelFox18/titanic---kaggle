"""The 10-iteration scoreboard, hard-coded from README.md's final table.

Screens 1 (Iteration Journey) and 4 (RF vs XGBoost) read from here so they
don't have to parse outputs/cv_scores_*.txt. Numbers are rounded to 3 decimals
to match the README scoreboard and the project's display convention.

Iter 10 regressed (threshold tuning was a CV mirage), so iter 9 is the best.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Iteration:
    n: int
    approach: str
    cv: float
    lb: float
    verdict: str
    good: bool  # did this iteration improve on the prior best LB?

    @property
    def gap(self) -> float:
        return round(self.cv - self.lb, 3)


# n, approach, cv, lb, verdict, good
ITERATIONS = [
    Iteration(1, "Random Forest baseline", 0.827, 0.778, "Starting point", True),
    Iteration(2, "Soft-voting ensemble (RF+GB+XGB+LGB)", 0.835, 0.749, "Overfit — boosting members memorize noise", False),
    Iteration(3, "RF + FamilySurvival feature", 0.842, 0.792, "Big lift — the single best feature", True),
    Iteration(4, "Optuna RF + class_weight=balanced_subsample", 0.857, 0.780, "CV mirage — over-predicts survivors", False),
    Iteration(5, "Optuna RF (no class_weight)", 0.851, 0.799, "Clean tuning win", True),
    Iteration(6, "Add TicketGroupSize / FarePerTicketPerson", 0.853, 0.806, "Crossed 0.80", True),
    Iteration(7, "Tuned XGBoost solo", 0.845, 0.782, "XGB underperforms RF on small data", False),
    Iteration(8, "0.7 RF + 0.3 XGB blend", 0.856, 0.792, "Stacking contagion — XGB drags it down", False),
    Iteration(9, "5-seed RF averaging", 0.854, 0.809, "Best result — variance reduction", True),
    Iteration(10, "5-seed RF + threshold tuning (t=0.47)", 0.855, 0.794, "Threshold overfit — calibration mirage", False),
]

GAP_DANGER = 0.06  # CLAUDE.md lesson 3 — CV-LB gap above this means overfit
TRAIN_RATE = 0.384
HONEST_CEILING = 0.82

# Reference points for the "where does our score land" context (README intro).
SCORE_REFERENCES = [
    ("All women survive (no ML)", 0.766),
    ("Typical AI baseline", 0.78),
    ("Strong honest ML", 0.81),
    ("Honest-ML ceiling", 0.82),
    ("Ground-truth lookup (cheating)", 1.00),
]

BEST = max(ITERATIONS, key=lambda it: it.lb)  # iter 9


def as_records() -> list[dict]:
    """List-of-dicts view, convenient for building a pandas DataFrame."""
    return [
        {
            "iter": it.n,
            "approach": it.approach,
            "cv": it.cv,
            "lb": it.lb,
            "gap": it.gap,
            "verdict": it.verdict,
            "good": it.good,
        }
        for it in ITERATIONS
    ]
