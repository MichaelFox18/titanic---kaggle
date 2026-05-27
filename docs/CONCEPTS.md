# Concepts Glossary

A growing reference for the data science / ML ideas this project uses.
Each entry explains the *what*, the *why*, and where it appears in this
codebase. Entries get added as new techniques come in.

---

## Stratified K-Fold Cross-Validation

**What.** Split the training data into K equal chunks ("folds"). Train
on K-1 of them, score on the held-out 1. Rotate which fold is held out
until each fold has been scored once. Report the mean of the K scores
(and the standard deviation, which measures consistency).

**Stratified** means each fold is sampled to preserve the overall class
distribution. The Titanic data has 38.4% survivors; without stratification,
a random fold could have anywhere from 30% to 47% survivors, making
its score noisy and incomparable to other folds.

**Why.** A single train/test split gives ONE noisy estimate of model
quality — you might get lucky or unlucky with which rows ended up
where. K-fold gives K estimates, and the mean is much more stable.
For small datasets like Titanic (891 rows), this is essential — a
single 80/20 split has only ~180 rows in the test fold, which is too
few to trust.

**Where.** `src/baseline_rf.py`, via `StratifiedKFold(n_splits=10, ...)`
and `cross_val_score`.

---

## Imputation by Group vs. Globally

**What.** When a column has missing values, you have to put *something*
there before a model can use the row. The naïve approach is to fill
with the column's mean or median. The better approach is to fill with
the median *within a meaningful subgroup* — e.g. (Title, Pclass) for
Age in this dataset.

**Why.** The naïve median assumes all rows are interchangeable. They're
not. A 1st-class Mrs. is on average much older than a 3rd-class Miss,
because Mrs. implies married (older) and 1st-class passengers were
wealthier (older). Filling both with the global median (28) systematically
biases your data downward for the older group and upward for the
younger group.

This is also the reason we fill `Fare` (1 missing value in test) with
the median of the same `Pclass` rather than the overall median —
fares scaled with class.

**Where.** `src/pipeline.py`, the lines using `groupby(...).transform(...)`.

---

## Target Leakage

**What.** When information that would NOT be available at prediction
time sneaks into the training data, the model looks great in
cross-validation but fails in production.

**Concrete example.** If we imputed Age using the median of the FULL
combined dataset including the test set's known ages, the model has
"seen" test data implicitly. CV would be optimistic. The safe version:
fit imputation values only on the training fold during each CV round.
(We slightly cheat here — Age is imputed on train+test combined — but
since Age is purely a feature, not the label, and the imputation uses
no label information, it's acceptable. The fatal version would be using
`Survived` to impute or compute features.)

A worse leakage: a feature that's a near-duplicate of the label. E.g.
if `submission.csv` from a past run accidentally got merged into the
features.

**Why it matters.** Leakage is the #1 reason model performance collapses
between dev and production. Always ask: "If I were predicting on a
single new row, would this feature value exist at prediction time?"

**Where.** Not in this project (we're careful), but a thing to watch
for. The pipeline split (train vs. test) explicitly remembers `n_train`
to avoid leaking label info back.

---

## Random Forest

**What.** An ensemble of many decision trees, each trained on (a) a
random bootstrap sample of the rows and (b) a random subset of the
features at every split point. Final prediction = majority vote
(classification) or mean (regression) across all trees.

**Why.** A single decision tree is high-variance — small changes in
the training data cause big changes in the tree. RF averages out that
variance. The randomness in row/feature sampling forces trees to
decorrelate, which is what makes the averaging actually help (averaging
identical trees gives you... the same tree).

**Hyperparameters we used and why:**
- `n_estimators=500` — more trees = more stable vote. Diminishing
  returns past a few hundred; cost is linear in time.
- `max_depth=6` — caps tree depth. Without this, each tree can
  memorize the training set. Shallower trees underfit individually
  but the forest still captures the signal.
- `min_samples_leaf=2` — a leaf node must contain at least 2
  training samples. Prevents leaves that fit a single passenger.
- `max_features='sqrt'` — at each split, consider only sqrt(n_features)
  random features. This is what decorrelates trees most strongly.

**Where.** `src/baseline_rf.py`.

---

## Feature Importance (Gini / impurity-based)

**What.** A score per feature, summing to 1.0, representing how much
total "impurity reduction" the feature contributed across all splits
in all trees. Higher = the model leaned on this feature more.

**Why useful.** Tells you which features mattered. If `Sex_female`
dominates and your hand-engineered `TitlePclass_Mrs_1` is near zero,
maybe the interaction isn't adding much over `Sex` alone.

**Why to be skeptical.** Impurity-based importance is biased toward
features with high cardinality (many unique values), even if they're
not actually informative. Permutation importance is more honest but
more expensive. For a baseline sanity check, the Gini version is fine.

**Where.** `src/baseline_rf.py` prints the top 20.

---

## One-Hot Encoding

**What.** Convert a categorical column with K levels into K binary
columns. E.g. `Embarked` with values {C, Q, S} becomes three columns
`Embarked_C`, `Embarked_Q`, `Embarked_S`, where exactly one is 1 per
row.

**Why.** Most models need numeric input. We could label-encode
(C=0, Q=1, S=2), but that implies an order (S > Q > C) which is
fictitious. One-hot has no such implied ordering.

**Tree models specifically** can handle label-encoded categoricals
OK in practice, but one-hot makes feature importances readable and
avoids any chance of the model treating the integer encoding as
ordinal.

**Where.** `src/pipeline.py`, via `pd.get_dummies(...)`.

---

## Why Tree Models Don't Need Feature Scaling

Linear models (logistic regression, SVM, neural nets) need features
on similar scales — otherwise a feature ranging 0–500 dominates one
ranging 0–1 in distance calculations. **Tree-based models don't care
about scale at all**: a tree splits on `Age > 30` or `Age > 3000` —
the split point adapts; absolute magnitudes don't matter. That's why
this pipeline has no `StandardScaler` step.

---

## Why We Don't Drop NaN Rows

Tempting shortcut: `df.dropna()`. But the training set has only 891
rows and ~20% have missing Age. Dropping them would shrink training
data by 20% — usually a worse trade-off than carefully imputing.

The exception is when missingness is itself informative: we keep
"NoCabin" as its own Deck category rather than discarding cabin-less
passengers, because the FACT of cabin missingness correlates strongly
with class and survival.
