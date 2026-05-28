"""Screen 5 — Feature Lab.

Toggles the project's two real opt-in features (FamilySurvival, ticket-group)
and shows how CV accuracy, the predicted survival rate, and the feature
importance ranking respond. This screen literally reproduces iterations 3 and 6.

Teaches: feature engineering, feature importance, and that good features come
from domain reasoning.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import (
    TRAIN_RATE,
    cv_accuracy_rf,
    fit_rf,
    get_features,
    predicted_rate,
    rf_param_items,
)

st.title("5 · The feature lab")

st.markdown(
    "These are the project's two **real** opt-in features, wired straight to "
    "`build_features`. Toggle them and watch the model's accuracy and its "
    "feature-importance ranking shift. With both on, you're looking at exactly "
    "the iteration-6 model."
)

# --- controls ---------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    ifs = st.toggle("FamilySurvival", value=True,
                    help="For each passenger, did other members of their family/ticket group survive in the training labels? (iter 3)")
with c2:
    itg = st.toggle("Ticket-group features", value=True,
                    help="TicketGroupSize + FarePerTicketPerson — honest per-head fare. (iter 6)")

params = rf_param_items()

cv_mean, cv_std = cv_accuracy_rf(ifs, itg, params)
cv_base, _ = cv_accuracy_rf(False, False, params)
model = fit_rf(ifs, itg, params)
X_train, y_train, X_test, _ = get_features(ifs, itg)
rate = predicted_rate(model, X_test)

# --- metrics ----------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("CV accuracy", f"{cv_mean:.3f}", delta=f"{cv_mean - cv_base:+.3f} vs no extras",
          help="10-fold cross-validation accuracy on the training passengers. The delta compares against the "
               "baseline with BOTH opt-in features off, so you can see exactly what toggling them buys you.")
m2.metric("Features in model", f"{X_train.shape[1]}",
          help="Total columns the model sees after one-hot encoding. Categorical features like Title or Deck "
               "expand into several 0/1 columns, so this is larger than the handful of raw inputs.")
m3.metric("Predicted survival rate", f"{rate:.3f}",
          delta=f"{rate - TRAIN_RATE:+.3f} vs train", delta_color="off",
          help="Fraction of the 418 test passengers predicted to survive. The true training rate is 0.384; "
               "healthy models sit a touch below. A big jump above would hint the model is over-predicting survivors.")

# --- importance bar chart ---------------------------------------------------
importances = (
    pd.Series(model.feature_importances_, index=X_train.columns)
    .sort_values(ascending=True)
    .tail(15)
)
# highlight the opt-in features so they're easy to spot
special = {"FamilySurvival", "TicketGroupSize", "FarePerTicketPerson"}
colors = ["#E45756" if name in special else "#4C78A8" for name in importances.index]

fig = go.Figure(go.Bar(
    x=importances.values, y=importances.index, orientation="h",
    marker=dict(color=colors),
))
fig.update_layout(
    height=460, xaxis=dict(title="feature importance (impurity-based)"),
    margin=dict(l=10, r=10, t=30, b=10),
    title="Top 15 features (opt-in features highlighted)",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Turn on FamilySurvival and watch it jump into the top ranks — that single "
    "feature was the project's biggest lift (+1.4pt LB at iter 3). Good features "
    "come from domain reasoning (families shared a fate), not data fishing. One "
    "caveat: impurity-based importance is biased toward high-cardinality "
    "features, so read it as a sanity check, not gospel."
)

if not ifs and not itg:
    st.info("Both features off — this is the iteration-1/2 baseline feature set.")
elif ifs and itg:
    st.success("Both features on — this is the iteration-6 model (best single RF, LB 0.806).")
