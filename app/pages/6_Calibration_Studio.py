"""Screen 6 — Calibration Studio.

Fits the project's 5-seed RF (iter 9), computes out-of-fold probabilities, and
lets you slide the decision threshold. Shows the predicted survival rate vs the
0.384 target and the OOF-accuracy-vs-threshold curve.

Teaches: threshold tuning, the predicted-rate canary — AND the project's hard
lesson that "perfect" aggregate calibration (iter 10, t=0.47) still *regressed*
on the leaderboard. Matching the global rate is not the same as getting more
individual passengers right.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from app_utils import BEST_RF_PARAMS, CV, SEEDS, TRAIN_RATE, get_features

st.title("6 · The calibration studio")

st.markdown(
    "After the model outputs a probability of survival, we still have to pick a "
    "**cut-off**: above it → predict survived, below → died. The default is 0.5. "
    "But across every iteration this model under-predicted survivors (~0.36 vs "
    "the true 0.384). Iteration 10 tried lowering the cut-off to fix that — slide "
    "it and see what happened."
)

# --- controls ---------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    threshold = st.slider("Decision threshold", 0.40, 0.55, 0.50, 0.01)
with c2:
    ifs = st.toggle("FamilySurvival", value=True)
with c3:
    itg = st.toggle("Ticket-group", value=True)


@st.cache_data(show_spinner="Training the 5-seed forest + out-of-fold probabilities (one-time, ~30s)…")
def calibration_data(ifs: bool, itg: bool):
    X, y, X_test, _ = get_features(ifs, itg)
    yv = y.values
    # OOF probabilities, averaged across the 5 seeds (matches iter 9/10).
    oof = np.zeros(len(yv))
    for seed in SEEDS:
        fold = np.zeros(len(yv))
        for tr, va in CV.split(X, y):
            m = RandomForestClassifier(**BEST_RF_PARAMS, random_state=seed, n_jobs=-1)
            m.fit(X.iloc[tr], y.iloc[tr])
            fold[va] = m.predict_proba(X.iloc[va])[:, 1]
        oof += fold
    oof /= len(SEEDS)
    # Test probabilities: refit each seed on full train, average.
    test_probs = np.zeros(len(X_test))
    for seed in SEEDS:
        m = RandomForestClassifier(**BEST_RF_PARAMS, random_state=seed, n_jobs=-1)
        m.fit(X, y)
        test_probs += m.predict_proba(X_test)[:, 1]
    test_probs /= len(SEEDS)
    return oof, yv, test_probs


oof, yv, test_probs = calibration_data(ifs, itg)

# --- metrics at the chosen threshold ---------------------------------------
oof_acc = float(((oof >= threshold).astype(int) == yv).mean())
test_rate = float((test_probs >= threshold).mean())

m1, m2, m3 = st.columns(3)
m1.metric("OOF accuracy", f"{oof_acc:.3f}",
          help="Out-of-fold (OOF) accuracy. Every training passenger gets a survival probability from a model "
               "trained on the OTHER 9 folds — i.e. a model that never saw them — so it's an honest estimate of "
               "real-world accuracy. Here we apply YOUR threshold to those probabilities: above it = predict "
               "survived, below = died, then count how many we got right. Move the slider and this changes because "
               "you're re-drawing the survived/died line, not retraining.")
m2.metric("Predicted survival rate", f"{test_rate:.3f}",
          delta=f"{test_rate - TRAIN_RATE:+.3f} vs train", delta_color="off",
          help="The fraction of the 418 test passengers this threshold labels as survived. The true training rate "
               "is 0.384. If this drifts far above, the model is over-predicting survivors (usually bad); the 0.5 "
               "default tended to under-predict at ~0.36.")
m3.metric("Threshold", f"{threshold:.2f}",
          help="The probability cut-off. A passenger with P(survived) at or above this is predicted to survive. "
               "0.5 is the neutral default; lower it and the model calls more passengers survivors.")

# --- OOF accuracy vs threshold curve ---------------------------------------
ts = np.round(np.arange(0.40, 0.551, 0.01), 2)
accs = [float(((oof >= t).astype(int) == yv).mean()) for t in ts]
rates = [float((test_probs >= t).mean()) for t in ts]
best_t = float(ts[int(np.argmax(accs))])

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=accs, mode="lines+markers", name="OOF accuracy",
                         line=dict(color="#4C78A8", width=3)))
fig.add_vline(x=threshold, line=dict(color="#B279A2", width=2, dash="dash"),
              annotation_text="you are here", annotation_position="top left")
fig.add_vline(x=0.50, line=dict(color="#54A24B", width=1, dash="dot"),
              annotation_text="0.50 default (iter 9 → LB 0.809)", annotation_position="bottom right")
fig.add_vline(x=0.47, line=dict(color="#E45756", width=1, dash="dot"),
              annotation_text="0.47 (iter 10 → LB 0.794)", annotation_position="top right")
fig.update_layout(
    height=380, xaxis=dict(title="decision threshold"),
    yaxis=dict(title="OOF accuracy"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --- predicted-rate gauge vs target ----------------------------------------
st.subheader("Predicted survival rate vs the 0.384 target")
figr = go.Figure()
figr.add_trace(go.Bar(x=[test_rate], y=["predicted"], orientation="h",
                      marker=dict(color="#4C78A8"), name="predicted rate",
                      text=[f"{test_rate:.3f}"], textposition="outside"))
figr.add_vline(x=TRAIN_RATE, line=dict(color="#E45756", width=2),
               annotation_text="train rate 0.384", annotation_position="top")
figr.update_layout(height=160, xaxis=dict(range=[0.30, 0.45], title="fraction predicted to survive"),
                   margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
st.plotly_chart(figr, use_container_width=True)

st.caption(
    "The default 0.50 cut-off made the model under-predict survivors by ~2 points. "
    "Lowering it toward 0.47 (red line) drives the predicted rate onto the 0.384 "
    "target — apparently perfect calibration. But on the real leaderboard that "
    "0.47 model scored **0.794**, *below* the 0.50 model's **0.809**. The lesson: "
    "matching the aggregate survival rate is not the same as getting more "
    "individual passengers right. A threshold picked to maximise CV accuracy is "
    "just one more knob that can overfit the cross-validation."
)

if abs(threshold - 0.47) < 1e-9:
    st.warning("This is iteration 10's threshold. It looked perfectly calibrated — and regressed the leaderboard by 1.4 points.")
elif abs(threshold - 0.50) < 1e-9:
    st.success("This is the default — and iteration 9's choice. It was the project's best submission (LB 0.809).")
