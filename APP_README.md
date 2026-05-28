# Titanic ML Concepts — Interactive Learning App

A locally-hosted **Streamlit** app that turns this repo's modeling work into an
interactive teaching tool. Every screen runs the project's *real* code — it
imports `build_features` from `src/pipeline.py` and trains the same models from
the iteration scripts — so the visualisations reflect the actual Titanic
pipeline, not a toy reimplementation.

This file is both **documentation** (for a human) and a **build spec** (for a
Claude Code session). If you're a Claude Code instance: read
[`README.md`](README.md) and [`CLAUDE.md`](CLAUDE.md) first, then
[`src/pipeline.py`](src/pipeline.py) to learn the exact `build_features`
signature, then build the app described below. A ready-to-paste starter prompt
is at the bottom.

---

## What it teaches

Seven screens, each mapping a hard-to-grasp concept to a control you can move
and a chart that responds:

| Screen | Concept | The "aha" |
|---|---|---|
| 1. Iteration journey | CV vs LB, the CV mirage | Higher CV ≠ higher LB; watch the gap |
| 2. Decision tree explorer | Overfitting, impurity | Crank depth, watch train acc soar while CV collapses |
| 3. Forest builder | Bagging, variance reduction | Add trees, watch CV variance shrink and stabilise |
| 4. RF vs XGBoost | Bagging vs boosting on small data | Boosting loses here — see why |
| 5. Feature lab | Feature engineering, importance | Toggle `FamilySurvival` / ticket features, watch CV + importances move |
| 6. Calibration studio | Threshold tuning, predicted-rate canary | Slide the threshold, watch predicted rate hit 0.384 |
| 7. Build-a-passenger | Model inference, feature attribution | Set Age/Sex/Pclass, get a live survival probability |

---

## Where it lives in the repo

The app sits in a new top-level `app/` folder so it doesn't clutter `src/`.
It reaches back into `src/` for the pipeline — no code is duplicated.

```
.
├── CLAUDE.md
├── README.md
├── APP_README.md                 ← this file
├── requirements.txt              ← add the app deps (see Setup)
├── data/                         ← reused as-is (train.csv, test.csv)
├── src/
│   └── pipeline.py               ← imported by the app, never modified
└── app/                          ← NEW
    ├── streamlit_app.py          home page + navigation + intro
    ├── app_utils.py              shared: path setup, data loading, cached training
    ├── iteration_data.py         the 10-iteration scoreboard (from README)
    └── pages/                    Streamlit auto-discovers these as nav tabs
        ├── 1_Iteration_Journey.py
        ├── 2_Decision_Tree.py
        ├── 3_Forest_Builder.py
        ├── 4_RF_vs_XGBoost.py
        ├── 5_Feature_Lab.py
        ├── 6_Calibration_Studio.py
        └── 7_Build_a_Passenger.py
```

Streamlit's multipage convention auto-loads everything in `pages/` as sidebar
navigation, sorted by the numeric prefix. The main script (`streamlit_app.py`)
is the landing page.

---

## Setup

Add these to `requirements.txt` (sklearn / xgboost / optuna are already there
from the modeling work):

```
streamlit>=1.36
plotly>=5.22
matplotlib>=3.8        # only for sklearn's plot_tree in screen 2
```

Then, from the project root on Windows PowerShell (matching the repo's existing
convention):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

It opens at `http://localhost:8501`. Edits hot-reload on save.

---

## Architecture notes (read before building)

**Reuse the pipeline, never reimplement it.** The whole point is that the app
runs real project code. `app_utils.py` adds `src/` to the path and imports
`build_features`. Inspect `pipeline.py` for the true signature before wiring
this up — the stub below assumes it takes train/test DataFrames plus the two
documented flags, but **verify and adapt**.

**Cache aggressively or the app feels sluggish.** Training an RF on every
slider nudge is wasteful. Use:
- `@st.cache_data` for CSV loading and feature building (returns DataFrames).
- `@st.cache_resource` for fitted model objects.
- Key caches on the inputs that should trigger a retrain (feature flags,
  `max_depth`, `n_estimators`, seed). Streamlit does this automatically from
  the function arguments, so pass those as plain args, not via globals.

**Carry forward the Windows joblib gotcha** (CLAUDE.md lesson 6). Nested
process pools hang on Windows. Inside the app, set `n_jobs=1` on any
`cross_val_score` / outer loop and let the estimator parallelise internally
(`RandomForestClassifier(n_jobs=-1)`). Do **not** set `n_jobs=-1` on both.

**Keep the modeling honest to the project's findings.** Use the same
`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` the iterations
use, so CV numbers shown in the app line up with the ones in `README.md` and
`outputs/cv_scores_*.txt`.

---

## Shared utilities — `app/app_utils.py`

Build this first; every page depends on it. Full reference implementation
(adapt the `build_features` call to the real signature):

```python
"""Shared helpers for the learning app. Imports the project's real pipeline."""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# --- make src/pipeline.py importable from inside app/ -----------------------
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from pipeline import build_features  # noqa: E402  <-- verify signature!

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
TRAIN_RATE = 0.384  # project-wide reference survival rate


@st.cache_data
def load_raw():
    train = pd.read_csv(os.path.join(_DATA, "train.csv"))
    test = pd.read_csv(os.path.join(_DATA, "test.csv"))
    return train, test


@st.cache_data
def get_features(include_family_survival: bool, include_ticket_group: bool):
    """Return (X_train, y_train, X_test) built by the REAL pipeline.

    NOTE: adapt this call to pipeline.build_features' actual signature.
    The stub assumes it accepts the raw frames + the two documented flags
    and returns engineered train/test feature frames + the target.
    """
    train, test = load_raw()
    X_train, y_train, X_test = build_features(
        train, test,
        include_family_survival=include_family_survival,
        include_ticket_group=include_ticket_group,
    )
    return X_train, y_train, X_test


@st.cache_resource
def fit_rf(_X, _y, **params):
    """Fit a Random Forest. Underscored args aren't hashed by Streamlit,
    so we also pass the hyperparameters as kwargs to key the cache."""
    model = RandomForestClassifier(n_jobs=-1, random_state=42, **params)
    model.fit(_X, _y)
    return model


def cv_accuracy(model, X, y):
    """10-fold stratified CV accuracy. n_jobs=1 on the OUTER loop (Windows
    joblib safety — see CLAUDE.md lesson 6)."""
    scores = cross_val_score(model, X, y, cv=CV, scoring="accuracy", n_jobs=1)
    return scores.mean(), scores.std()
```

Pages then do `from app_utils import get_features, fit_rf, cv_accuracy, ...`.

---

## Screen specs

Each spec lists: the **controls**, what it **computes**, what it **renders**, the
**teaching caption** to show, and the **data source**. Captions are short, plain
sentences — the goal is intuition, not a textbook.

### 1 — Iteration Journey  (`pages/1_Iteration_Journey.py`)
- **Controls:** a selectbox to highlight one iteration; a toggle to show/hide the
  CV–LB gap band.
- **Computes:** nothing live — reads the scoreboard from `iteration_data.py`.
- **Renders:** a Plotly line chart with two series (CV and LB) across iterations
  1–10, plus a shaded ribbon for the gap. Colour the gap red where it exceeds
  0.06 (the danger threshold from CLAUDE.md lesson 3). Below the chart, a card
  for the selected iteration showing its approach, CV, LB, gap, and the
  one-line verdict.
- **Caption:** "CV is a guess; LB is the answer. When the gap blows past ~0.06,
  the model is fitting quirks of the cross-validation that don't survive
  contact with the test set. Iterations 2, 4, and 8 all did this."
- **Source:** the scoreboard table in `README.md`, encoded in `iteration_data.py`.

### 2 — Decision Tree Explorer  (`pages/2_Decision_Tree.py`)
- **Controls:** sliders for `max_depth` (1–20) and `min_samples_leaf` (1–30);
  feature-flag toggles.
- **Computes:** fits a single `DecisionTreeClassifier(criterion="entropy")` at the
  chosen settings; computes **train accuracy** and **10-fold CV accuracy**.
- **Renders:** (a) the tree itself via `sklearn.tree.plot_tree` into a matplotlib
  figure (cap displayed depth at ~3–4 for legibility even if trained deeper, or
  warn when too deep to draw); (b) a Plotly line chart sweeping `max_depth` 1→20
  with two curves — train accuracy climbing toward 1.0, CV accuracy peaking then
  falling. Mark the current depth.
- **Caption:** "Watch the two curves diverge. Past a certain depth the tree keeps
  getting better on data it has already seen (train) while getting worse on data
  it hasn't (CV). That divergence is overfitting, and it's exactly why the
  project capped `max_depth=4`."
- **Source:** live training on the real features.

### 3 — Forest Builder  (`pages/3_Forest_Builder.py`)
- **Controls:** slider for `n_estimators` (1–500); slider for `max_features`
  (0.1–1.0); a "show individual tree votes" toggle.
- **Computes:** fits RFs across a sweep of `n_estimators`; for each, records CV
  mean and std. Optionally fits several single trees on different bootstrap
  samples to show their disagreement.
- **Renders:** a Plotly chart of CV accuracy vs number of trees, with the std as
  an error band that visibly *narrows* as trees are added — this is variance
  reduction made visual. If the votes toggle is on, show a small panel of
  per-tree predicted probabilities for one example passenger, scattered, with
  the average marked.
- **Caption:** "Each tree is a noisy guess. Averaging many independent guesses
  cancels the noise — the band tightens as you add trees. This is why a forest
  beats a single tree, and why `n_estimators=400` was plenty: the band has
  basically stopped shrinking."
- **Source:** live training on the real features.

### 4 — RF vs XGBoost  (`pages/4_RF_vs_XGBoost.py`)
- **Controls:** a shared `max_depth` slider; feature-flag toggles.
- **Computes:** fits a tuned-ish RF and an XGBoost at matched depth; reports CV
  for both. (Keep XGB regularisation-friendly per iter 7.)
- **Renders:** side-by-side metric cards (CV for each), plus a static annotated
  bar comparing the *documented LB* outcomes — iter 6 RF 0.806 vs iter 7 XGB
  0.782 vs iter 8 blend 0.792. A short note explains the small-data reason.
- **Caption:** "Boosting usually wins on tabular data — but it needs enough rows
  for its sequential error-correction to find real signal. On 891 rows the
  residuals turn to noise fast, so boosting overfits and RF's averaging wins.
  Every non-RF experiment in this project regressed the leaderboard."
- **Source:** live CV training; LB numbers hard-coded from `README.md`.

### 5 — Feature Lab  (`pages/5_Feature_Lab.py`)
- **Controls:** the two real pipeline flags — `include_family_survival`,
  `include_ticket_group` — as toggles. (These are the project's actual opt-in
  features, so this screen literally reproduces iters 3 and 6.)
- **Computes:** rebuilds features with the chosen flags, fits the project RF,
  reports CV and the model's **predicted survival rate** on the test set.
- **Renders:** a horizontal Plotly bar chart of `feature_importances_` (top ~15),
  re-sorting live as flags change; a metric card showing CV delta vs the
  no-extras baseline; the predicted-rate readout with a marker at 0.384.
- **Caption:** "Turn on `FamilySurvival` and watch it jump into the top ranks —
  that single feature was the project's biggest single lift (+1.4pt LB). Good
  features come from domain reasoning, not data fishing. Note: impurity-based
  importance is biased toward high-cardinality features, so read it as a sanity
  check, not gospel."
- **Source:** live training; uses the pipeline's real `include_*` flags.

### 6 — Calibration Studio  (`pages/6_Calibration_Studio.py`)
- **Controls:** a threshold slider (0.40–0.55, step 0.01); feature-flag toggles.
- **Computes:** fits the project RF (ideally the 5-seed average from iter 9),
  gets test-set probabilities, then at the chosen threshold computes the
  resulting **predicted survival rate** and the **OOF accuracy**.
- **Renders:** a gauge or number showing predicted rate vs the 0.384 target
  line; an OOF-accuracy-vs-threshold curve with the current threshold marked and
  the 0.47 project choice annotated.
- **Caption:** "The default 0.5 cutoff made the model under-predict survivors by
  ~2 points across every iteration. Lowering the threshold to 0.47 nudged the
  predicted rate to 0.383 — almost perfect. The predicted rate is a calibration
  smoke detector: drift more than ~0.02 from 0.384 and something's off."
- **Source:** live training and OOF prediction.

### 7 — Build-a-Passenger  (`pages/7_Build_a_Passenger.py`)
- **Controls:** input widgets for a hypothetical passenger — Sex, Pclass, Age,
  SibSp, Parch, Fare, Embarked; feature-flag toggles.
- **Computes:** assembles a one-row frame, runs it through `build_features`
  (alongside the training data so group-based imputation and FamilySurvival
  behave), and predicts with the fitted RF.
- **Renders:** a big survival probability number with a coloured bar; a short
  list of which engineered feature values that passenger ended up with
  (Title, FamilySize, Deck, FarePerPerson…) so the user connects inputs to the
  features the model actually sees.
- **Caption:** "Change Sex from male to female and watch the probability leap —
  that's the single strongest signal in the data. This is the model from
  iteration 9 making a live prediction on a passenger you invented."
- **Source:** live inference. **Build this screen last** — single-row inference
  through a pipeline written for batch train/test frames is the fiddliest part;
  you may need a small wrapper that appends the synthetic row to the test frame,
  runs the pipeline, then pulls that row's features back out.

---

## Iteration data — `app/iteration_data.py`

Hard-code the scoreboard so screens 1 and 4 don't need to parse output files.
Straight from the `README.md` final scoreboard:

```python
ITERATIONS = [
    # iter, approach, cv, lb, gap, verdict
    (1,  "Random Forest baseline",                 0.827, 0.778, 0.049, "Starting point"),
    (2,  "Soft-voting ensemble (RF+GB+XGB+LGB)",   0.835, 0.749, 0.086, "Overfit"),
    (3,  "RF + FamilySurvival feature",            0.842, 0.792, 0.050, "Big lift"),
    (4,  "Optuna RF + class_weight",               0.857, 0.780, 0.077, "CV mirage"),
    (5,  "Optuna RF (no class_weight)",            0.851, 0.799, 0.052, "Clean tuning"),
    (6,  "Add ticket-group features",              0.853, 0.806, 0.047, "Crossed 0.80"),
    (7,  "Tuned XGBoost solo",                      0.845, 0.782, 0.063, "XGB underperforms"),
    (8,  "0.7 RF + 0.3 XGB blend",                 0.856, 0.792, 0.064, "Stacking contagion"),
    (9,  "5-seed RF averaging",                     0.854, 0.809, 0.045, "New best"),
    (10, "5-seed RF + threshold tuning (t=0.47)",  0.855, None,  None,  "Pending"),
]

GAP_DANGER = 0.06   # CLAUDE.md lesson 3
TRAIN_RATE = 0.384
HONEST_CEILING = 0.82
```

---

## Build order (recommended)

1. `app_utils.py` — verify the `build_features` signature against `pipeline.py`
   and get `get_features` / `fit_rf` / `cv_accuracy` working in isolation.
2. `iteration_data.py` — trivial, unblocks screens 1 and 4.
3. `streamlit_app.py` — landing page with project intro + nav.
4. Screen 1 (no live training — fastest win, confirms Streamlit + Plotly work).
5. Screen 2, then 3 (core overfitting / variance-reduction story).
6. Screens 5 and 6 (use the real flags + calibration).
7. Screen 4 (mostly static once CV training exists).
8. Screen 7 last (single-row inference is the trickiest plumbing).

---

## Design conventions

- Plain language in every caption — explain the intuition, name the project
  iteration it came from, keep it to two or three sentences.
- Plotly for interactive charts, matplotlib only for the tree drawing.
- Sentence case on headings; avoid heavy formatting.
- Round every displayed number (`f"{x:.3f}"` for accuracies, the LB convention).
- Don't modify anything in `src/` or `data/`. The app is strictly a consumer of
  the existing pipeline.

---

## What to skip (scope guard)

- No retraining of Optuna searches in-app — they're slow; use the documented
  best params as fixed defaults.
- No writing to `data/submission.csv` — this is a learning app, not a submission
  pipeline.
- No auth, no database, no deployment config — local `streamlit run` only.
  (Streamlit Cloud deploy is a fine stretch goal once it works locally.)

---

## Starter prompt for Claude Code

Paste this into a Claude Code session opened at the repo root:

> Read `APP_README.md`, `README.md`, and `CLAUDE.md`, then open
> `src/pipeline.py` and tell me the exact signature and return shape of
> `build_features` before writing any code. Once I confirm, build the Streamlit
> learning app exactly as specified in `APP_README.md`: create the `app/`
> folder, `app_utils.py`, `iteration_data.py`, `streamlit_app.py`, and the seven
> pages in `pages/`. Reuse the real pipeline — do not reimplement feature
> engineering. Follow the Windows joblib safety rule (`n_jobs=1` on outer CV
> loops). Add the three app dependencies to `requirements.txt`. Build in the
> recommended order, and after each screen, run `streamlit run
> app\streamlit_app.py` so I can check it before moving on.
