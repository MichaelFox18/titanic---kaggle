"""Titanic ML Concepts — interactive learning app (landing page).

Run from the project root:
    py -m streamlit run app/streamlit_app.py

Streamlit auto-discovers the files in app/pages/ and lists them in the sidebar,
sorted by their numeric prefix. This file is the home page.
"""

from __future__ import annotations

import streamlit as st

from iteration_data import BEST, HONEST_CEILING, ITERATIONS, SCORE_REFERENCES

st.set_page_config(
    page_title="Titanic ML Concepts",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Titanic ML concepts — an interactive walkthrough")

st.markdown(
    """
This app turns a real Kaggle Titanic modeling project into a hands-on teaching
tool. Every screen runs the project's **actual code** — it imports
`build_features` from `src/pipeline.py` and trains the same Random Forests the
iteration scripts use — so what you see reflects the real pipeline, not a toy
reimplementation.

The project ran **10 iterations** trying to predict which passengers survived.
The story isn't a straight climb: about half the experiments *regressed* the
leaderboard, and those failures taught more than the wins. This app lets you
move the controls behind each lesson and watch the result respond.
"""
)

# --- headline numbers -------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Best public LB", f"{BEST.lb:.3f}", help=f"Iteration {BEST.n}: {BEST.approach}")
col2.metric("Honest-ML ceiling", f"{HONEST_CEILING:.2f}", help="Scores above ~0.83 usually come from looking up real passenger fates, not modeling.")
col3.metric("Iterations run", len(ITERATIONS), help="Each is a documented experiment with a hypothesis.")

st.divider()

# --- what the screens teach -------------------------------------------------
st.subheader("What you can explore")

st.markdown(
    """
Use the sidebar to move between screens. Each maps one hard-to-grasp idea to a
control you can move and a chart that responds.

| Screen | Concept | The "aha" |
|---|---|---|
| **1. Iteration journey** | CV vs LB, the CV mirage | Higher CV ≠ higher LB — watch the gap |
| **2. Decision tree explorer** | Overfitting, tree depth | Crank depth, watch train accuracy soar while CV collapses |
| **3. Forest builder** | Bagging, variance reduction | Add trees, watch the CV spread shrink and stabilise |
| **4. RF vs XGBoost** | Bagging vs boosting on small data | Boosting loses here — see why |
| **5. Feature lab** | Feature engineering, importance | Toggle the real pipeline features, watch CV + importances move |
| **6. Calibration studio** | Threshold tuning, the predicted-rate canary | Slide the threshold — and see why "perfect calibration" still lost |
| **7. Build a passenger** | Model inference, feature attribution | Invent a passenger, get a live survival probability |
"""
)

st.divider()

# --- orientation ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("The two numbers that matter")
    st.markdown(
        """
- **CV** = cross-validation accuracy, measured locally on the 891 labeled
  training rows. It's our *guess* at how good the model is.
- **LB** = the Kaggle public leaderboard accuracy on 418 hidden test rows.
  It's the *answer* — but we only see it after submitting.

A healthy model has **CV ≈ LB**. When CV climbs but LB doesn't, the model is
fitting quirks of the cross-validation that don't survive contact with real
unseen data. Screen 1 makes this gap visible.
"""
    )

with right:
    st.subheader("Where this score lands")
    for label, score in SCORE_REFERENCES:
        st.markdown(f"- **{score:.3f}** — {label}")
    st.caption(
        "Our best honest-ML result (0.809) sits near the realistic ceiling. "
        "The perfect scores on the public leaderboard come from looking up the "
        "published passenger survivor list, not from a better model."
    )

st.divider()
st.caption(
    "Built on the real project pipeline. Nothing here modifies src/ or data/ — "
    "the app is strictly a consumer of the existing code. See README.md for the "
    "full project narrative and CLAUDE.md for the engineering notes."
)
