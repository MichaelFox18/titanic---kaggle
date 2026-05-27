# Titanic Survival Prediction

A learning-focused project for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).
The goal is to predict which passengers survived the 1912 sinking using
the passenger manifest (class, age, sex, family, cabin, fare, port).

This repo is built to *teach* the data science alongside producing a
submission. Code is heavily commented; concepts are explained as they
appear; the EDA notebook walks through what each chart is telling us.

See [`CLAUDE.md`](CLAUDE.md) for the full strategy doc and
[`docs/CONCEPTS.md`](docs/CONCEPTS.md) for an ML glossary that grows
with the project.

## Project layout

```
.
├── CLAUDE.md                strategy / methodology
├── README.md                this file
├── requirements.txt         Python deps
├── .venv/                   local virtualenv (gitignored)
├── data/
│   ├── train.csv            891 labeled rows
│   ├── test.csv             418 unlabeled rows
│   └── submission.csv       generated predictions
├── docs/
│   └── CONCEPTS.md          ML glossary (stratified k-fold, leakage, etc.)
├── notebooks/
│   └── 01_eda.ipynb         exploratory data analysis with charts
├── src/
│   ├── pipeline.py          feature engineering + preprocessing
│   └── baseline_rf.py       Random Forest baseline + CV + submission
└── outputs/
    └── cv_scores.txt        per-fold CV scores + feature importances
```

## How to run

From the project root, on Windows PowerShell:

```powershell
# one-time setup
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run the baseline (writes data/submission.csv + outputs/cv_scores.txt)
python src\baseline_rf.py

# launch the EDA notebook
jupyter notebook notebooks\01_eda.ipynb
```

Or without activating the venv:

```powershell
.\.venv\Scripts\python.exe src\baseline_rf.py
```

## What the baseline does

1. Loads `train.csv` + `test.csv` and engineers features (Title, FamilySize,
   IsAlone, AgeBand, FareBand, Deck, FarePerPerson, Title×Pclass).
2. Imputes missing Age values **by (Title, Pclass) group median** — never
   a global median (see `docs/CONCEPTS.md` for why).
3. Trains a Random Forest with 10-fold Stratified Cross-Validation.
4. Prints per-fold accuracy, top 20 feature importances, and writes:
   - `data/submission.csv` — Kaggle-format predictions
   - `outputs/cv_scores.txt` — the CV report

Expected CV accuracy: **~0.82 – 0.83**. Next iteration (XGBoost, then an
ensemble) should push toward 0.84+.

## Roadmap

- [x] Baseline RF + clean feature pipeline
- [ ] EDA notebook with charts and commentary
- [ ] XGBoost with Optuna hyperparameter tuning
- [ ] LightGBM
- [ ] Voting / stacking ensemble
- [ ] Submission to Kaggle leaderboard
