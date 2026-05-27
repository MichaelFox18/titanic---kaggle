# Titanic Survival Prediction — Claude Code Session Guide

## Goal
Achieve maximum accuracy on the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). The task is binary classification: predict which passengers survived the Titanic disaster. Evaluation metric is **classification accuracy** (% of correct predictions on the test set).

---

## Project Context

### Files
- `train.csv` — 891 rows, labeled (has `Survived` column)
- `test.csv` — 418 rows, unlabeled (submit predictions for these)
- `submission.csv` — output file to submit to Kaggle (format: `PassengerId`, `Survived`)

### Columns
| Column | Description |
|---|---|
| PassengerId | Unique ID |
| Survived | **Target**: 0 = No, 1 = Yes |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | Full name (contains title like Mr., Mrs., Miss., etc.) |
| Sex | male / female |
| Age | Age in years (177 missing in train, 86 in test) |
| SibSp | # siblings/spouses aboard |
| Parch | # parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare (1 missing in test) |
| Cabin | Cabin number (687 missing in train — treat as its own signal) |
| Embarked | Port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton (2 missing in train) |

### Known Data Issues to Handle
- `Age`: 177/891 missing in train, 86/418 in test — impute carefully (by title/Pclass group, not global median)
- `Cabin`: 687/891 missing — extract deck letter; binary "has cabin" feature is valuable
- `Embarked`: 2 missing in train — fill with mode (S)
- `Fare`: 1 missing in test — fill with median of same Pclass

---

## Strategy for Maximum Accuracy

### 1. Feature Engineering (highest ROI)
These features are known to significantly boost performance on this dataset:

- **Title** — extract from Name (Mr, Mrs, Miss, Master, Rare). `Master` = boys, strong survival signal.
- **FamilySize** — `SibSp + Parch + 1`. Solo travelers and very large families had lower survival.
- **IsAlone** — binary flag when FamilySize == 1
- **AgeBand** — bin Age into groups after imputing (child/teen/adult/senior)
- **FareBand** — bin Fare into quartiles
- **Deck** — extract first letter of Cabin; `NoCabin` as its own category
- **TicketPrefix** — some ticket prefixes correlate with survival
- **FarePerPerson** — Fare divided by FamilySize (shared tickets skew fare)
- **Title + Pclass interaction** — encode combinations

### 2. Age Imputation
Do NOT use global median. Use median Age grouped by `Title` and `Pclass`. This substantially reduces imputation error.

### 3. Models to Try (in order)
1. **Random Forest** — strong baseline, interpretable
2. **Gradient Boosting (XGBoost / LightGBM)** — typically best single model here
3. **Voting Ensemble** — combine RF + XGB + LightGBM + SVM with soft/hard voting
4. **Stacking** — use out-of-fold predictions as meta-features for a second-level model

### 4. Validation Strategy
- Use **Stratified K-Fold (k=10)** to preserve class balance (38.4% survived)
- Track both CV score and public LB score — they can diverge
- Target CV accuracy: **0.83+**; public LB: **0.78–0.80+**

### 5. Hyperparameter Tuning
Use `GridSearchCV` or `Optuna` on:
- n_estimators, max_depth, min_samples_leaf, max_features (RF)
- learning_rate, n_estimators, max_depth, subsample, colsample_bytree (XGB)

---

## Recommended Implementation Plan

```
Step 1: EDA
  - Survival rates by Sex, Pclass, Age group, FamilySize
  - Correlation heatmap
  - Missing value audit

Step 2: Preprocessing pipeline
  - Impute Age (by Title+Pclass group median)
  - Impute Fare (Pclass median), Embarked (mode)
  - Extract all engineered features
  - Encode categoricals (LabelEncoder or OHE)
  - Drop: Name, Ticket, Cabin (after extracting Deck/Title/FarePerPerson)

Step 3: Baseline model
  - Random Forest with default params + stratified 10-fold CV
  - Print feature importances

Step 4: Tune best model(s)
  - GridSearch or Optuna on RF + XGBoost
  - Evaluate on held-out fold

Step 5: Ensemble
  - VotingClassifier (RF + XGB + LGB)
  - Optional: StackingClassifier

Step 6: Generate submission
  - Predict on test.csv
  - Save as submission.csv with columns: PassengerId, Survived
```

---

## Environment Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna matplotlib seaborn
```

---

## Submission Format

```
PassengerId,Survived
892,0
893,1
894,0
...
```
Must contain all 418 test rows. `Survived` must be 0 or 1 (integer, not float).

---

## Known Accuracy Ceilings

| Approach | Typical CV Accuracy |
|---|---|
| Logistic Regression (baseline) | ~0.80 |
| Random Forest (tuned) | ~0.83 |
| XGBoost (tuned) | ~0.83–0.84 |
| Ensemble (RF + XGB + LGB) | ~0.84–0.85 |
| Stacking | ~0.84–0.86 |

Public LB scores are typically 1–3 points lower than CV due to dataset size and noise.

---

## Tips for This Specific Dataset

- Women in 1st/2nd class had near-perfect survival — the model needs to capture this interaction strongly
- `Master` (young boys) survived at high rates — title extraction is critical
- Large families (5+) had low survival — FamilySize nonlinearity matters
- Cabin deck A/B/C correlates with 1st class; use it even with heavy missingness
- Avoid overfitting — the training set is only 891 rows, regularize aggressively
- The test set has a known "correct" answer ceiling of ~82% for fully manual labeling; top LB scores above 0.82 often exploit leaked/external data
