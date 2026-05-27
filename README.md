# Titanic Survival Prediction — A Learning-Focused Kaggle Project

Predicting which passengers survived the 1912 Titanic disaster, using the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) data. This repo is organized for **learning** as much as for placing on the leaderboard — every iteration is documented, every method explained, every failed experiment kept as a lesson.

**Best result:** public leaderboard accuracy **0.80861** (iteration 9), with iteration 10's calibration-tuned variant submitted as the final attempt. For context, scores break down roughly:

| Strategy | Public LB |
|---|---|
| All women survive (no ML) | 0.76555 |
| Typical AI baseline | 0.77–0.79 |
| Strong ML + careful tuning | 0.80–0.82 |
| Honest-ML ceiling | ~0.82 |
| Ground-truth lookup (cheating) | 0.83–1.00 |

So a final LB of 0.80–0.81 lands solidly in the "strong honest ML" band, near the realistic ceiling for an automated pipeline that doesn't peek at the public passenger fate list.

---

## Table of contents

1. [Final scoreboard](#final-scoreboard)
2. [Quick start](#quick-start)
3. [Project layout](#project-layout)
4. [The methods that worked](#the-methods-that-worked) — the four techniques that did most of the work
5. [The iteration journey](#the-iteration-journey) — all 10 attempts with lessons
6. [Hard-won lessons](#hard-won-lessons)
7. [Concepts reference](#concepts-reference) — glossary of every ML idea used here
8. [What I'd do differently](#what-id-do-differently)

---

## Final scoreboard

| Iter | Approach | CV | Public LB | CV–LB gap | Verdict |
|---|---|---|---|---|---|
| 1 | Random Forest baseline | 0.827 | 0.778 | 0.049 | Starting point |
| 2 | Soft-voting ensemble (RF+GB+XGB+LGB) | 0.835 | 0.749 | 0.086 | ❌ Overfit |
| 3 | RF + FamilySurvival feature | 0.842 | 0.792 | 0.050 | ✅ Big lift |
| 4 | Optuna RF + `class_weight=balanced_subsample` | 0.857 | 0.780 | 0.077 | ❌ CV mirage |
| 5 | Optuna RF (no class_weight) | 0.851 | 0.799 | 0.052 | ✅ Clean tuning |
| 6 | Add TicketGroupSize / FarePerTicketPerson | 0.853 | **0.806** | **0.047** | ✅ Crossed 0.80 |
| 7 | Tuned XGBoost solo | 0.845 | 0.782 | 0.063 | ❌ XGB underperforms |
| 8 | 0.7 RF + 0.3 XGB blend | 0.856 | 0.792 | 0.064 | ❌ Stacking contagion |
| 9 | 5-seed RF averaging | 0.854 | **0.809** | **0.045** | ✅ New best |
| 10 | 5-seed RF + threshold tuning (t=0.47) | 0.855 | _pending_ | — | ? |

CV = mean 10-fold stratified cross-validation accuracy on the 891 training rows.
LB = Kaggle's accuracy score on the 418-row test set after submitting `data/submission.csv`.

---

## Quick start

```powershell
# one-time setup (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# reproduce the best LB submission
py src\iteration_v9.py

# or the calibration-tuned variant
py src\iteration_v10.py

# explore the data
jupyter notebook notebooks\01_eda.ipynb
```

Each iteration script overwrites `data/submission.csv`. Run the iteration whose submission you want before uploading to Kaggle.

---

## Project layout

```
.
├── CLAUDE.md                    Claude Code session guide (project facts, lessons)
├── README.md                    this file — the master narrative + reference
├── requirements.txt
├── data/                        train.csv, test.csv, submission.csv
├── notebooks/01_eda.ipynb       exploratory data analysis
├── src/
│   ├── pipeline.py              shared feature engineering (used by every iteration)
│   ├── baseline_rf.py           iter 1
│   └── iteration_v2..v10.py     one file per iteration
└── outputs/cv_scores_*.txt      per-iteration CV reports + feature importances
```

`pipeline.py` is the single source of truth for feature engineering. Each iteration imports its `build_features(...)` function and adds modeling on top. Two opt-in flags toggle late-added features (`include_family_survival`, `include_ticket_group`).

---

## The methods that worked

Out of dozens of things tried across 10 iterations, four techniques accounted for almost all the LB improvement from 0.778 → 0.809.

### 1. Feature engineering with domain reasoning

The raw columns (Name, Ticket, Cabin) are useless to a model as strings. But hidden inside them are **socially-meaningful** signals that a tree model can use:

| Engineered feature | What it captures |
|---|---|
| **Title** (Mr/Mrs/Miss/Master/Rare) | Marital status + age bracket + social class, all in one |
| **FamilySize** = SibSp + Parch + 1 | Solo travelers and very large families had lower survival; non-linear |
| **TicketGroupSize** | Catches travel companions who weren't blood relatives |
| **FarePerPerson** (÷ FamilySize) | Per-head cost using family group |
| **FarePerTicketPerson** (÷ TicketGroupSize) | Per-head cost using actual booking group (more honest) |
| **AgeBand / FareBand** | Discrete buckets for non-linear cutoffs (e.g. "children under 12") |
| **Deck** (first letter of Cabin, "N" if missing) | Class proxy — first-class records were better kept |
| **TitlePclass** | Interaction: 1st-class Mrs survived ~97%, 3rd-class Mrs ~50% |
| **FamilySurvival** | Train-label-aware: did other members of this family survive? |

The biggest single feature gain came from **FamilySurvival** (iter 3, +1.4pt LB). It encodes a real-world fact: families on the Titanic largely lived or died together. For each passenger, the feature looks at survival outcomes of their family/ticket-group members *in the training labels only* (test labels stay NaN by construction — no leakage). When the model meets a test passenger whose mother is in the training set and survived, it now has direct evidence pointing toward survival.

Lesson: **features motivated by domain reasoning punch above their CV weight on the leaderboard.** A feature that's hand-engineered to encode something structurally true about the world (per-head fare, family bond) generalizes from train to test more reliably than a feature found by data fishing.

### 2. Imputation by group, not globally

Age is 20% missing. The naïve fix is to fill missing values with the global median (28). The better fix is to fill with the median **within (Title, Pclass) groups**:

- A 1st-class Mrs. is much older on average than a 3rd-class Miss
- Mrs. implies married → older; 1st-class implies wealthier → older
- Global median biases the older group downward and younger group upward

Same logic for the single missing Fare in test (use Pclass median, since fare scales with class) and missing Embarked (mode = "S").

### 3. Random Forest is the right model family here

We tested Random Forest, Gradient Boosting (sklearn), XGBoost, and LightGBM. Every iteration that strayed from a single tuned Random Forest regressed on the leaderboard:

- Iter 2 (4-way ensemble): -3pt LB
- Iter 7 (XGBoost solo): -2.4pt LB
- Iter 8 (RF + XGB blend): -1.4pt LB

This is **unusual** — XGBoost usually wins on tabular data. The probable cause: Titanic has only 891 training rows. Boosting builds shallow trees sequentially that each correct the previous trees' residual errors — but on too few samples, those residuals are noise. RF averages many independent bagged deep trees, which is structurally a better fit when there isn't enough data to find fine-grained signal beyond noise.

Tuned RF hyperparameters that mattered (from Optuna search in iter 5):
- `max_depth=4` — kept shallow; deeper trees overfit
- `min_samples_leaf=4` — leaves need ≥4 samples
- `max_features=0.5` — each split sees 50% of features (vs default sqrt ≈ 14%)
- `max_samples=0.64` — each tree trained on 64% of rows (extra bagging)
- `criterion='entropy'` — vs default gini (marginal but real here)
- `n_estimators=400` — diminishing returns above this

### 4. Multi-seed averaging + threshold calibration

Even a tuned RF has residual randomness from its own internal bagging. Iter 9 trained the same RF 5 times with different `random_state` values and averaged the predicted probabilities. CV gain was tiny (+0.23pt), but **LB gain matched it almost exactly (+0.24pt)** — a signal that the change was structural, not a CV-protocol artifact.

Iter 10 added one more layer of calibration: **threshold tuning**. The standard 0.5 cutoff for binary classification assumes equal cost of false positives and false negatives at the natural class balance. But across every iteration, the model's predicted survival rate was consistently ~0.36 vs the true train rate of 0.384 — a 2-point under-prediction. Lowering the threshold to 0.47 (chosen by OOF accuracy + a predicted-rate sanity check) brings the test predicted rate to 0.383, essentially perfect calibration.

These two techniques together are *low-variance* moves — they don't add model capacity, so they can't overfit in the traditional sense. The only risk is that the calibration value (threshold) is picked from a noisy CV signal, which we guard against with a sanity check on the predicted rate.

---

## The iteration journey

Each iteration was a deliberate experiment with a hypothesis. The pattern that emerged: small gains from any single change, careful A/B isolation of effects, and a brutal reminder that *higher CV does not mean higher LB*.

### Iteration 1 — Baseline (LB 0.778)

A single Random Forest with sensible defaults (500 trees, depth 6) on a well-engineered feature set. Predicted survival rate 0.366 (under by 0.018).

**What worked:** Feature engineering already covered Title, FamilySize, Deck, FarePerPerson, TitlePclass interaction.

**What didn't:** Stuck in the "typical AI baseline" band. Single RF with default thresholds wasn't enough.

### Iteration 2 — Ensemble experiment (LB 0.749, -2.9pt) ❌

Soft-voting blend of RF + GradientBoosting + XGBoost + LightGBM. CV went up (+0.8pt) but LB collapsed.

**Lesson:** On 891 rows, the boosting members overfit training noise. Their high-confidence wrong predictions dragged the soft-average ensemble away from RF's safer outputs. **Model variety doesn't help when the variety adds capacity rather than regularization on a small dataset.** First big CV–LB gap warning (0.086 vs the normal 0.05).

### Iteration 3 — FamilySurvival feature (LB 0.792, +1.4pt) ✅

Added a single feature: for each passenger, look up whether OTHER members of their family (Surname+Fare) or ticket group survived/died in the training labels. 0.5 default for passengers with no group info.

**Why no leakage:** the feature only reads training labels. For test rows, the feature value comes from looking at family members who happen to be in the training set. Test labels stay unknown by construction.

**The result:** CV gain (+1.46pt) and LB gain (+1.44pt) almost exactly matched — the cleanest "real signal" iteration yet. FamilySurvival landed as the 4th most important feature on first attempt.

### Iteration 4 — Optuna RF tuning, with `class_weight` (LB 0.780, -1.2pt) ❌

100 Optuna trials searching 8 hyperparameters. Found CV 0.857 (+1.6pt) using `class_weight='balanced_subsample'`. Predicted survival rate jumped to 0.407 (over by 0.023).

**Lesson:** `class_weight='balanced_subsample'` weights positive examples up in each bootstrap, training the model as if class balance were 50%. But train and test share ~38% positive rate. The model now over-predicts survivors, costing real points on every spurious "survived" call. **The predicted survival rate is a smoke detector** — if it drifts >0.02 from train rate, suspect a CV mirage.

### Iteration 5 — Optuna without `class_weight` (LB 0.799, +0.7pt) ✅

Rerun iter 4 with `class_weight` forced to `None`. Tested the hypothesis that the *other* tuning changes were honest gains and only `class_weight` was the LB killer.

**Best params discovered:** max_depth 4 (down from 6), max_features 0.5, max_samples 0.64, criterion entropy. Optuna compensated for the loss of `class_weight` by picking even more aggressive regularization — confirming RF on this dataset wants to be kept simple.

**The clean A/B:** CV dropped from 0.857 → 0.851 (the `class_weight` was worth ~0.6pt CV), but LB rose from 0.780 → 0.799 (a 2pt LB swing in the same change). Most of iter 4's CV gain was the cheat; the regularization improvements were real.

### Iteration 6 — Add ticket-group features (LB 0.806, +0.7pt) ✅ **crossed 0.80**

Added two domain-motivated features:
- **TicketGroupSize**: count of passengers sharing a ticket (different from FamilySize — catches non-family companions like servants, friends, business associates)
- **FarePerTicketPerson**: Fare ÷ TicketGroupSize (more honest than the old FarePerPerson ÷ FamilySize when unrelated people share a ticket)

CV gain was small (+0.22pt) but LB gain was 3× larger (+0.72pt). Both new features landed in the top 10 importance ranking; FarePerTicketPerson edged out the older FarePerPerson at #5. CV–LB gap dropped to 0.047 — the cleanest of any iteration so far.

### Iteration 7 — Tuned XGBoost solo (LB 0.782, -2.4pt) ❌

Optuna with a regularization-friendly search space (max_depth 2–6, lr 0.01–0.08, high reg_alpha/lambda). Best XGB CV 0.845, but LB dropped to 0.782.

**Lesson:** XGBoost is meaningfully weaker than tuned RF on this dataset, even when carefully constrained. Feature importance comparison: XGB heavily weights raw Title × Sex × Pclass features and barely uses the engineered FarePerTicketPerson. RF made better use of the engineered features. CV–LB gap was 0.063 — XGB generalizes *worse* than RF on this small dataset.

### Iteration 8 — RF + XGB blend (LB 0.792, -1.4pt) ❌

Even though XGB is weaker overall, blending could still help if the two models make uncorrelated errors. Disagreement analysis on OOF:
- RF and XGB disagreed on 5.05% of training rows
- When they disagreed, RF was right 57.8% and XGB right 42.2% — almost coin flip

This *looked* like a favorable signal for ensembling. We swept weights 0.5 to 1.0 and the meta-learner; 0.7 RF + 0.3 XGB won at CV 0.856.

**LB regressed by 1.4pt.** The disagreement diagnostic lied: it was right about OOF, but on the test set, XGB's predictions on the disagreement rows must have been wrong far more often. The 30% XGB weight cost more on LB than it gained on CV. Pattern matched iter 4's CV mirage exactly.

**Generalizable rule:** *every* non-RF addition on this dataset has regressed LB (iter 2 ensemble, iter 4 class_weight, iter 7 XGB, iter 8 blend). RF is genuinely the right model family for Titanic's 891-row size.

### Iteration 9 — 5-seed RF averaging (LB 0.809, +0.3pt vs iter 6) ✅ **new best**

Train the iter-6 tuned RF five times with seeds [42, 0, 1, 7, 13]. Average the predicted probabilities. Threshold at 0.5.

CV gain was tiny (+0.23pt) — RF's internal bagging already does most of the variance reduction. But CV and LB moved together almost exactly (+0.24pt LB), the cleanest CV–LB tracking yet. **Lesson:** when a CV gain is small but its *mechanism* has clear statistical theory (variance reduction across independent fits), the LB gain typically shows up. Contrast with iters 4 and 8 where the CV gains came from optimization tricks that overfit the CV protocol itself.

### Iteration 10 — Threshold tuning (LB: _pending_)

Same 5-seed RF as iter 9, but with the prediction threshold optimized via OOF.

Looking back at iters 1, 3, 5, 6, 9, every single one predicted a test survival rate of ~0.36, ~2 percentage points below the true train rate of 0.384. Consistent miscalibration. Iter 10 sweeps thresholds from 0.40 to 0.55 on the OOF probabilities and picks 0.47 (CV-best), which brings the final test predicted rate to 0.3828 — within 0.1% of train rate.

OOF accuracy gain was small (+0.11pt, at the noise floor), but the calibration argument is structural: the model was systematically under-confident on positive predictions, and threshold 0.47 corrects exactly that bias. The change flips ~10 borderline test predictions from 0 to 1.

---

## Hard-won lessons

These are the meta-lessons that emerged from the 10-iteration arc.

1. **CV is a guess; LB is the answer.** Track the gap between them. A CV–LB gap of ~0.05 is normal for Titanic. >0.06 is a red flag. Iters 2, 4, 7, 8 all blew this out.

2. **The predicted survival rate is a calibration smoke detector.** Train rate is 0.384. Models that predict ~0.36 are under by a healthy amount and safe. Models that predict 0.40+ are usually gaming class-imbalance tricks. Iter 4 was flagged in advance by this check; iter 10 used it as a sanity guard.

3. **Higher CV does not mean higher LB on small datasets.** Iters 2, 4, 8 each had CV ≥ the eventual winner but LB worse. The cause every time: the CV gain came from fitting noise that didn't generalize. Iter 9 was the opposite pattern — tiny CV gain, but the mechanism (averaging independent fits) was sound.

4. **The "different model bias = stacking gains" intuition fails on small noisy data.** RF and XGB had different feature reliance patterns and disagreed in ways that *looked* good for blending. They blended badly anyway. Boosting models on 891 rows just overfit harder.

5. **Most of the gain came from features, not models.** From 0.778 (baseline) to 0.806 (best single RF), the path was: tuned hyperparameters (+2.1pt total via iters 4-5) + two feature additions (FamilySurvival +1.4pt, ticket-group +0.7pt). The remaining +0.3pt came from variance reduction and (likely) calibration.

6. **A/B isolation matters.** Iter 3's `+with FS` vs `without FS` comparison, iter 5 reproducing iter 4 minus class_weight, iter 6 reproducing iter 5 plus ticket features — every clean test let us attribute gains to the specific change rather than guessing.

7. **Documentation is a multiplier.** Iter 8's failure made sense because we'd already documented why iter 4 failed. Iter 10's threshold tuning was motivated by noticing the predicted-rate pattern that documentation surfaced across iters 1, 3, 5, 6, 9. Without the running log, those signals would have been invisible.

8. **Windows + sklearn + joblib has a nested-parallelism trap.** Outer `cross_val_score(n_jobs=-1)` combined with inner estimator `n_jobs=-1` spawns pools-of-pools that exhaust Windows' worker resources. Symptom: process alive, 0 CPU work for minutes. Fix: serialize the outer loop.

---

## Concepts reference

A glossary for the data science / ML ideas this project uses. Each entry: the *what*, the *why*, and where it shows up in this codebase.

### Stratified K-Fold Cross-Validation

**What.** Split the training data into K equal chunks ("folds"). Train on K-1 of them, score on the held-out one. Rotate which fold is held out until each has been scored. Report the mean (and standard deviation) of the K scores.

**Stratified** means each fold preserves the overall class distribution. Titanic is 38.4% survivors; without stratification, random folds could swing 30%–47%, making scores noisy and incomparable.

**Why.** A single train/test split gives one noisy estimate. K-fold averages K of them — much more stable. On 891-row datasets, a single 80/20 split has only ~180 test rows, too few to trust.

**Where.** Every iteration uses `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.

### Imputation by Group vs. Globally

**What.** When a column has missing values, fill with the median *within a meaningful subgroup* — for Age, that's (Title, Pclass). A 1st-class Mrs is on average older than a 3rd-class Miss. Filling both with the global median (28) biases your data systematically.

**Where.** `src/pipeline.py`, lines using `groupby(...).transform(lambda s: s.fillna(s.median()))`.

### Target Leakage

**What.** Information that wouldn't be available at prediction time sneaking into the training data. CV looks great, production fails.

**The danger.** A feature that's a near-duplicate of the label. E.g., if `submission.csv` from a past run accidentally got merged into the features.

**The mild case here.** We fit Age imputation on train+test combined. That's technically using test-row Ages to compute group medians. It's defensible because: (a) Age is a feature, not the label; (b) the medians don't depend on `Survived`. The fatal version would be using `Survived` itself to impute or compute features.

**The strict version.** FamilySurvival uses *only* train labels. Test rows have NaN labels, so when we compute "did any group member survive?", test rows can never contribute. Clean.

### Random Forest

**What.** An ensemble of decision trees, each trained on (a) a random bootstrap sample of rows and (b) a random subset of features at every split point. Final prediction = majority vote (classification) or mean (regression) across all trees.

**Why.** A single decision tree is high-variance — small changes in training data produce big changes in the tree. RF averages out that variance. The randomness in row/feature sampling forces trees to *decorrelate*, which is what makes averaging actually help (averaging identical trees gives you... the same tree).

**Tuned hyperparameters used here:**
- `n_estimators=400` — more trees = more stable vote; diminishing returns past a few hundred
- `max_depth=4` — caps depth to prevent memorization on 891 rows
- `min_samples_leaf=4` — leaves need ≥4 samples (prevents single-passenger leaves)
- `max_features=0.5` — at each split, consider 50% of features (decorrelates trees)
- `max_samples=0.64` — each tree trained on 64% of rows (extra bagging)
- `criterion='entropy'` — vs default gini; entropy slightly favors deeper splits when classes are mixed

### XGBoost vs Random Forest (boosting vs bagging)

**Random Forest** = bagging: train many independent trees on random subsets, average their predictions. Reduces **variance**. Each tree can be deep; the averaging is what regularizes.

**XGBoost** = boosting: train trees *sequentially*, each new tree fitting the residual errors of the previous trees. Reduces **bias**. Each tree must be shallow because deep trees on residuals overfit hard.

**The Titanic finding.** Boosting usually wins on tabular data, but it needs enough samples for sequential residual fitting to find real signal. On 891 rows, the residuals are mostly noise after a few trees, so boosting just amplifies overfitting. RF's bagging is structurally a better fit for small tabular data.

### Feature Importance (Gini / impurity-based)

**What.** A score per feature summing to 1.0, representing the share of total impurity reduction the feature contributed across all splits in all trees.

**Useful for.** Sanity-checking that engineered features are doing work. If `FamilySurvival` is in your top 5, the engineering paid off.

**Be skeptical of.** Impurity-based importance is biased toward high-cardinality features (many unique values), even when they're not actually informative. **Permutation importance** is more honest but more expensive. For development sanity checks, Gini is fine.

### One-Hot Encoding

**What.** Convert a categorical column with K levels into K binary columns. `Embarked` with values {C, Q, S} becomes `Embarked_C`, `Embarked_Q`, `Embarked_S`, with exactly one set to 1 per row.

**Why.** Most models need numeric input. Label-encoding (C=0, Q=1, S=2) implies a fictitious ordering. One-hot has none. Tree models can sometimes handle label-encoded categoricals fine in practice, but one-hot makes feature importances readable (you see `Embarked_C` not opaque integer 0).

### Why Tree Models Don't Need Feature Scaling

Linear models (logistic regression, SVM, neural nets) need features on similar scales — otherwise a column ranging 0–500 dominates one ranging 0–1 in any distance calculation. **Tree-based models don't care about scale at all** — a tree splits on `Age > 30` or `Age > 3000`; the split point adapts. This pipeline has no `StandardScaler` because every model used was tree-based.

### Why We Don't Drop NaN Rows

Tempting: `df.dropna()`. But 20% of training Ages are missing — dropping them shrinks training data by 1/5, almost always a worse trade-off than careful imputation. The exception: when missingness itself is informative. For Cabin (77% missing), we keep "NoCabin" as its own Deck category because the *fact* of missingness correlates strongly with class and survival.

### Optuna Hyperparameter Tuning

**What.** A library that uses Tree-structured Parzen Estimator (TPE) sampling to search a hyperparameter space more efficiently than grid search or random search. Each trial gets a parameter combination from the sampler, evaluates it (here: via CV accuracy), and the result informs the next sample.

**How we used it.** Iters 4, 5, 7 each used 60–100 Optuna trials with repeated stratified K-fold inside each trial to average out single-shuffle noise. The search spaces were deliberately constrained to regularization-favoring values (max_depth 2–6 for XGBoost, modest learning rates) to guard against overfitting tricks.

**Watch out for.** Optuna will happily find params that game the CV protocol if you let it. Iter 4's `class_weight='balanced_subsample'` discovery was a clean example: technically optimal by CV, materially bad on LB. Solution: hold out a class-balance or predicted-rate sanity check separate from CV.

### Predicted Survival Rate as a Calibration Check

**What.** Compute the fraction of test predictions that are class 1 (survived). Compare to the train fraction (0.384). Any drift indicates miscalibration.

**Why this works.** The train and test sets on Titanic share the same survival rate. A well-calibrated model should predict positives at roughly the same rate. If it doesn't, the model is biased — either over-predicting (false positives accumulate) or under-predicting (false negatives accumulate).

**How we used it.**
- As a *guard* against CV mirages (iter 4 was flagged in advance).
- As a *signal* for threshold tuning (iter 10's threshold 0.47 was chosen partly because it brought predicted rate from 0.36 → 0.38).

### Threshold Tuning

**What.** The default binary classification threshold (0.5) assumes equal cost of false positives and false negatives at equal class priors. Neither necessarily holds. You can pick a different threshold by CV.

**The risk.** Threshold is one more dimension to overfit CV on. Iter 4 is the cautionary tale — a CV-optimal `class_weight` cost real points on LB.

**The guardrail.** Validate the chosen threshold both by (a) OOF accuracy AND (b) the resulting predicted survival rate matching train rate. Iter 10 chose 0.47 because the OOF accuracy peaked there *and* the resulting test predicted rate landed within 0.1% of train rate (0.3828 vs 0.3838).

### Out-Of-Fold (OOF) Predictions

**What.** For each row in the training set, the prediction made by a model that was trained WITHOUT that row (its CV holdout fold). The result is a per-row prediction that's honest about generalization.

**Why useful.** OOF predictions can serve as training data for a *meta-learner* (stacking) or as the basis for threshold tuning (iter 10). They're what CV would say if you could ask it "what would you predict for THIS specific row?".

**Caveat.** OOF predictions are systematically biased — each fold's model has 10% less training data than the final full-data model. Predicted survival rates from OOF are typically ~0.02–0.04 lower than the final model's. Iter 10's sanity check applies to the *final* (full-data) test predictions, not OOF.

### Multi-Seed Model Averaging

**What.** Train the same model architecture multiple times with different `random_state` values. Average the predicted probabilities. Threshold the average.

**Why.** Even when a model architecture has its own randomness (RF's bootstrap sampling, neural net initialization), repeating with different seeds explores slightly different regions of parameter space. Averaging cancels the seed-specific noise.

**When it helps most.** High-variance models — single trees, small forests, deep nets. A well-regularized RF already has lots of internal averaging (400 trees × bootstrap sampling × feature subsetting), so multi-seed averaging on top adds only a small gain.

**Iter 9 result.** +0.23pt CV, +0.24pt LB with 5 seeds. The mechanism is sound enough that even small gains translate cleanly to LB.

---

## What I'd do differently

A short list of choices I'd revisit on a future Titanic attempt, given what we know now.

1. **Skip the ensemble / stacking experiments entirely.** Iters 2, 7, and 8 collectively burned 3 of 10 submissions to learn that boosting models hurt LB on this dataset. The right answer would have been to commit to tuned RF after iter 3 and spend those submissions on feature engineering instead.

2. **Try threshold tuning earlier.** Every iteration from 1 onward under-predicted survival rate by ~2 percentage points — a clean miscalibration. If iter 10's threshold-tuning approach had been applied at iter 6, the iter-6 LB might have been ~0.812 instead of 0.806.

3. **Investigate one more feature angle.** TicketGroupSize captured ticket-sharing groups, but ticket-prefix patterns (e.g., "PC", "STON") encode booking-class information that might be more granular than Pclass alone. Worth a careful A/B.

4. **Drop the `class_weight` search space from the start.** It's the single biggest CV-mirage trap for this dataset.

5. **Always check predicted survival rate as a hard guard.** It surfaced iter 4's failure before the LB confirmed it, and could have surfaced iter 8's failure too (predicted rate 0.371 was suspicious in retrospect — too healthy for a blend with a model that itself predicts 0.347).

---

## Acknowledgements & references

- [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) — the dataset and the leaderboard
- [scikit-learn](https://scikit-learn.org/) — RandomForest, model_selection, ensembles
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting (even though it lost here)
- [Optuna](https://optuna.org/) — TPE hyperparameter search

This project was developed as a learning exercise. Code is heavily commented; every iteration script reproduces its result from `data/train.csv` + `data/test.csv` with no manual steps.
