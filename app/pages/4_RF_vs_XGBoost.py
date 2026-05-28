"""Screen 4 — RF vs XGBoost.

Trains a Random Forest and a regularisation-friendly XGBoost at the SAME depth
on the same features, reports CV for both, and shows the documented leaderboard
outcomes (iter 6 RF vs iter 7 XGB vs iter 8 blend).

Teaches: bagging vs boosting, and why boosting loses on this small dataset.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from app_utils import CV, cv_accuracy_rf, get_features, rf_param_items
from iteration_data import ITERATIONS

st.title("4 · Random Forest vs XGBoost")

st.markdown(
    "Two ways to combine trees. A **Random Forest** trains many deep trees "
    "independently and averages them (reduces *variance*). **XGBoost** trains "
    "shallow trees in sequence, each fixing the last one's mistakes (reduces "
    "*bias*). Boosting usually wins on tabular data — but watch what happens on "
    "this tiny 891-row dataset."
)

# --- controls ---------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    max_depth = st.slider("Shared tree depth (max_depth)", 2, 10, 6,
                          help="Both models use this depth, for a fair comparison.")
with c2:
    ifs = st.toggle("FamilySurvival", value=True)
with c3:
    itg = st.toggle("Ticket-group", value=True)


@st.cache_data(show_spinner="Training XGBoost…")
def cv_xgb(ifs: bool, itg: bool, max_depth: int):
    X, y, _, _ = get_features(ifs, itg)
    model = xgb.XGBClassifier(
        n_estimators=450, learning_rate=0.0303, max_depth=max_depth,
        min_child_weight=2, subsample=0.9035, colsample_bytree=0.6438,
        gamma=0.2273, reg_alpha=1.2444, reg_lambda=1.5685,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    # Outer CV serial (Windows joblib safety); XGB parallelises internally.
    scores = cross_val_score(model, X, y, cv=CV, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std())


rf_cv, rf_std = cv_accuracy_rf(ifs, itg, rf_param_items(max_depth=max_depth))
xgb_cv, xgb_std = cv_xgb(ifs, itg, max_depth)

# --- live CV comparison -----------------------------------------------------
st.subheader(f"Live cross-validation at depth {max_depth}")
m1, m2, m3 = st.columns(3)
m1.metric("Random Forest CV", f"{rf_cv:.3f}",
          help=f"10-fold cross-validation accuracy for the bagging model (±{rf_std:.3f} across folds). "
               f"Many deep trees trained independently on random data slices, then averaged.")
m2.metric("XGBoost CV", f"{xgb_cv:.3f}",
          help=f"10-fold cross-validation accuracy for the boosting model (±{xgb_std:.3f} across folds). "
               f"Shallow trees trained in sequence, each correcting the last one's errors. Regularised per iter 7.")
m3.metric("RF advantage", f"{rf_cv - xgb_cv:+.3f}",
          delta="RF ahead" if rf_cv >= xgb_cv else "XGB ahead",
          delta_color="normal" if rf_cv >= xgb_cv else "inverse",
          help="RF CV minus XGBoost CV at this depth. Positive = the forest is winning. On this small dataset RF "
               "tends to stay ahead — and the leaderboard gap below was even wider than CV suggested.")

# --- documented LB outcomes -------------------------------------------------
st.subheader("What actually happened on the leaderboard")
rf6 = ITERATIONS[5].lb   # iter 6 — tuned RF
xgb7 = ITERATIONS[6].lb  # iter 7 — XGB solo
blend8 = ITERATIONS[7].lb  # iter 8 — RF+XGB blend
best9 = ITERATIONS[8].lb  # iter 9 — 5-seed RF (best)

labels = ["iter 6\nRF", "iter 7\nXGB solo", "iter 8\nRF+XGB blend", "iter 9\n5-seed RF"]
vals = [rf6, xgb7, blend8, best9]
colors = ["#4C78A8", "#E45756", "#E45756", "#54A24B"]
fig = go.Figure(go.Bar(x=labels, y=vals, marker=dict(color=colors),
                       text=[f"{v:.3f}" for v in vals], textposition="outside"))
fig.update_layout(height=360, yaxis=dict(title="Public LB", range=[0.76, 0.82]),
                  margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Boosting usually wins on tabular data — but it needs enough rows for its "
    "sequential error-correction to find real signal. On 891 rows the residuals "
    "turn to noise fast, so XGBoost overfits (iter 7: 0.782) and even a blend "
    "with RF drags the score down (iter 8: 0.792). Every non-RF experiment in "
    "this project regressed the leaderboard — RF's averaging is structurally the "
    "better fit for small tabular data."
)
