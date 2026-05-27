# Titanic Survival Prediction — Claude Code Session Guide

Project context, layout, and conventions for any future Claude Code session working on this repo. For the full project narrative, results, and concepts, see [README.md](README.md).

---

## What this is

Kaggle Titanic survival prediction. Binary classification: predict which of 418 unlabeled passengers survived, using 891 labeled training rows. Evaluation = classification accuracy on the held-out test set.

**Best result achieved:** public LB **0.80861** (iter 9, 5-seed RF averaging). Iter 10 (threshold-tuned variant) submission file is ready but its LB outcome is recorded in `README.md`.

---

## Project layout

```
.
├── CLAUDE.md                    this file (Claude session guide)
├── README.md                    project narrative + concepts + iteration log
├── requirements.txt             Python deps
├── .venv/                       local virtualenv (gitignored)
├── data/
│   ├── train.csv                891 labeled rows
│   ├── test.csv                 418 unlabeled rows
│   └── submission.csv           latest predictions (overwritten each iteration)
├── notebooks/
│   └── 01_eda.ipynb             exploratory data analysis
├── src/
│   ├── pipeline.py              feature engineering + preprocessing (single source of truth)
│   ├── baseline_rf.py           iter 1 baseline (RF, default features)
│   ├── iteration_v2.py          iter 2: RF+GB+XGB+LGB soft-voting ensemble
│   ├── iteration_v3.py          iter 3: RF + FamilySurvival feature
│   ├── iteration_v4.py          iter 4: Optuna-tuned RF (incl. class_weight)
│   ├── iteration_v5.py          iter 5: Optuna-tuned RF (no class_weight)
│   ├── iteration_v6.py          iter 6: + ticket-group features (best single RF)
│   ├── iteration_v7.py          iter 7: tuned XGBoost
│   ├── iteration_v7_finish.py   iter 7 helper (rerun after a paging-file crash)
│   ├── iteration_v8.py          iter 8: RF + XGB blend / stacking investigation
│   ├── iteration_v9.py          iter 9: 5-seed RF averaging (best LB)
│   └── iteration_v10.py         iter 10: 5-seed RF + OOF threshold tuning
└── outputs/
    ├── cv_scores.txt            iter 1 CV report
    └── cv_scores_v2..v10.txt    one per iteration
```

`src/pipeline.py` is the **single source of truth** for features. Every iteration imports `build_features(...)` from it. Two flags toggle the late-added features:
- `include_family_survival=True` → adds the FamilySurvival train-label-aware grouping feature (introduced in iter 3)
- `include_ticket_group=True` → adds TicketGroupSize + FarePerTicketPerson (introduced in iter 6)

---

## How to run

From the project root on Windows PowerShell:

```powershell
# one-time setup
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run any iteration (each writes data/submission.csv + outputs/cv_scores_*.txt)
py src\iteration_v9.py        # best LB submission as of writing
py src\iteration_v10.py       # threshold-tuned variant

# launch EDA notebook
jupyter notebook notebooks\01_eda.ipynb
```

Each iteration script is self-contained — it imports `build_features` from `pipeline.py`, fits, prints a CV report, and writes `data/submission.csv` (overwriting the previous iteration's submission).

---

## Dataset facts that matter

- **Size:** 891 train + 418 test = small. Anything that adds model capacity (deep trees, big ensembles, complex meta-learners) tends to overfit.
- **Class balance:** 38.4% survival rate. Train and test share this rate; **don't** mess with class_weight or balanced sampling unless you have a specific reason — it inflates CV without helping LB. See iter 4.
- **Missing data:**
  - Age: 177/891 train missing (~20%), 86/418 test missing
  - Cabin: 687/891 train missing (~77%) — "having a cabin" is a class proxy
  - Embarked: 2 train missing
  - Fare: 1 test missing
- **Public LB ceiling for honest ML:** ~0.82. Scores above 0.83 typically come from looking up actual passenger fates (the passenger list is public historical record). 0.79–0.82 is the realistic strong-ML band.

---

## Hard-won learnings (project-specific — read before iterating)

These are *non-obvious* lessons from this project. If a future session ignores them, it will probably waste submissions.

1. **RF beats XGBoost on this dataset.** On 891 rows, even carefully regularized XGBoost underperforms a tuned Random Forest by ~2pt LB (iter 7: 0.782 vs iter 6: 0.806). Don't be tempted to retry boosting models — we already did with iter 2 ensemble, iter 7 solo XGB, and iter 8 stacked blend. All three regressed LB. RF's variance-reduction-via-bagging is structurally better than boosting's bias-reduction on small tabular data.

2. **Don't use `class_weight='balanced_subsample'`.** Train and test share the same class balance (38%). Forcing the model to behave as if classes are balanced over-predicts positives. Iter 4 lost 1.2pt LB this way despite gaining 1.5pt CV. **Predicted survival rate** is the canary: if it drifts >0.02 from train rate (0.384), suspect a CV mirage.

3. **The CV–LB gap is the diagnostic.** Normal is ~0.05. >0.06 means the model is fitting CV-protocol-specific quirks that don't generalize:
   - Iter 1 RF: gap 0.049 ✓
   - Iter 6 best: gap 0.047 ✓
   - Iter 9 best LB: gap 0.045 ✓
   - Iter 2 ensemble: gap 0.086 ✗
   - Iter 4 class_weight: gap 0.077 ✗
   - Iter 8 blend: gap 0.064 ✗

4. **Multi-seed RF averaging works even though CV barely moves.** Iter 9 got +0.23pt CV but +0.24pt LB by averaging 5 RFs with different `random_state`. The CV–LB tracking is the signal it's real. Don't dismiss small-CV-gain changes if their mechanism (variance reduction) is theoretically sound.

5. **Threshold tuning is a legitimate calibration step, not a trick.** The 0.5 default consistently under-predicted by ~2 percentage points across iterations (iter 6 rate 0.361, iter 9 rate 0.359 vs train 0.384). Iter 10's OOF-validated threshold of 0.47 brought test predicted rate to 0.383 — essentially perfect calibration. Always sanity-check the predicted rate.

6. **Windows joblib + nested process pools = hangs.** Setting `n_jobs=-1` on the outer `cross_val_score` AND on the inner sklearn estimator spawns pools-of-pools that exhaust Windows' worker pool / paging file. Symptom: process alive, 0 CPU work for minutes. **Fix:** outer CV loop uses `n_jobs=1`; let the model parallelize internally. See `iteration_v7.py` / `iteration_v9.py` / `iteration_v10.py` notes.

---

## Feature engineering reference (what's in `pipeline.py`)

Always-on features:
- **Title** — extracted from Name regex (Mr, Mrs, Miss, Master, Rare). High signal; Master = young boys = high survival.
- **FamilySize / IsAlone** — SibSp + Parch + 1; binary alone flag.
- **AgeBand / FareBand** — binned versions, useful for non-linear cutoffs.
- **Deck** — first letter of Cabin (or "N" if missing). 'T' (1 person) → 'Rare'.
- **FarePerPerson** — Fare ÷ FamilySize.
- **TitlePclass** — interaction encoding (e.g. `Title_Mrs × Pclass_1`).
- **Age imputation by (Title, Pclass) group median.** Never global median — see CONCEPTS in README.

Opt-in features (set `include_*=True`):
- **FamilySurvival** (iter 3+) — for each passenger, look up survival outcomes of OTHER members of their family/ticket group using *train labels only*. Iter 3's +1.4pt LB lift.
- **TicketGroupSize / FarePerTicketPerson** (iter 6+) — count of people sharing a ticket; honest per-head fare. Iter 6's +0.7pt LB lift.

Drop-on-output (never reach the model): Name, Ticket, Cabin, PassengerId.

---

## Submission format

```
PassengerId,Survived
892,0
893,1
...
```

All 418 test rows, `Survived` ∈ {0, 1} as integers (not 0.0/1.0). Iteration scripts handle this with `preds.astype(int)`.
