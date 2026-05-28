"""Screen 2 — Decision Tree Explorer.

Fits a single DecisionTreeClassifier (entropy, like the project's RF) at a
chosen depth and shows (a) the tree itself and (b) the train-vs-CV accuracy
divergence as depth grows. Teaches: overfitting, and why the project capped
max_depth=4.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from app_utils import CV, RANDOM_STATE, get_features

st.title("2 · The decision tree explorer")

st.markdown(
    "A decision tree asks yes/no questions until it reaches a verdict. Let it "
    "ask **too many** and it stops learning patterns and starts memorising "
    "individual passengers. Crank the depth and watch the two accuracy curves "
    "pull apart — that gap *is* overfitting."
)

# --- controls ---------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    max_depth = st.slider("Tree depth (max_depth)", 1, 20, 4,
                          help="How many questions deep the tree can go.")
with c2:
    min_samples_leaf = st.slider("Min samples per leaf", 1, 30, 1,
                                 help="A leaf must hold at least this many passengers. Higher = simpler tree.")

f1, f2 = st.columns(2)
with f1:
    ifs = st.toggle("Include FamilySurvival feature", value=True)
with f2:
    itg = st.toggle("Include ticket-group features", value=True)


@st.cache_data(show_spinner="Fitting the tree…")
def fit_tree(ifs: bool, itg: bool, max_depth: int, min_samples_leaf: int):
    X, y, _, _ = get_features(ifs, itg)
    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    cv_mean = float(cross_val_score(clf, X, y, cv=CV, scoring="accuracy", n_jobs=1).mean())
    return clf, train_acc, cv_mean, list(X.columns)


@st.cache_data(show_spinner="Sweeping depth 1→20…")
def depth_sweep(ifs: bool, itg: bool, min_samples_leaf: int):
    X, y, _, _ = get_features(ifs, itg)
    out = []
    for d in range(1, 21):
        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=d,
            min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE,
        )
        clf.fit(X, y)
        train_acc = float(clf.score(X, y))
        cv_mean = float(cross_val_score(clf, X, y, cv=CV, scoring="accuracy", n_jobs=1).mean())
        out.append((d, train_acc, cv_mean))
    return out


clf, train_acc, cv_mean, feat_names = fit_tree(ifs, itg, max_depth, min_samples_leaf)
sweep = depth_sweep(ifs, itg, min_samples_leaf)

# --- metrics ----------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Train accuracy", f"{train_acc:.3f}",
          help="How often the tree is right on the SAME 891 passengers it was trained on. "
               "A deep tree can memorise them and push this toward 1.0 — impressive-looking but meaningless, "
               "because it has seen the answers.")
m2.metric("CV accuracy", f"{cv_mean:.3f}",
          help="10-fold cross-validation: the data is split into 10 parts; the tree trains on 9 and is tested "
               "on the 10th it never saw, rotated until every part has been the test set once. The average is "
               "an honest estimate of accuracy on NEW passengers — what actually matters.")
overfit = train_acc - cv_mean
m3.metric("Overfit gap (train − CV)", f"{overfit:.3f}",
          delta=f"{'high' if overfit > 0.10 else 'ok'}",
          delta_color="inverse",
          help="Train accuracy minus CV accuracy. A small gap means the tree learned general patterns; a large "
               "gap means it memorised the training passengers instead of learning rules that transfer. The wider "
               "this gap, the worse the overfitting.")

# --- divergence chart -------------------------------------------------------
depths = [r[0] for r in sweep]
train_curve = [r[1] for r in sweep]
cv_curve = [r[2] for r in sweep]
best_cv_depth = max(sweep, key=lambda r: r[2])[0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=depths, y=train_curve, name="Train accuracy",
                         mode="lines+markers", line=dict(color="#E45756", width=3)))
fig.add_trace(go.Scatter(x=depths, y=cv_curve, name="CV accuracy",
                         mode="lines+markers", line=dict(color="#4C78A8", width=3)))
fig.add_vline(x=max_depth, line=dict(color="#B279A2", width=2, dash="dash"),
              annotation_text="you are here", annotation_position="top")
fig.add_vline(x=best_cv_depth, line=dict(color="#54A24B", width=1, dash="dot"),
              annotation_text=f"best CV (depth {best_cv_depth})", annotation_position="bottom")
fig.update_layout(
    height=380, xaxis=dict(title="max_depth", dtick=1),
    yaxis=dict(title="Accuracy", range=[0.7, 1.02]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Watch the curves diverge. Past a certain depth the tree keeps getting "
    "better on data it has already seen (red) while getting worse on data it "
    "hasn't (blue). That divergence is overfitting — and it's exactly why the "
    "project capped max_depth=4, near where the blue CV curve peaks."
)

# --- the tree itself --------------------------------------------------------
st.subheader("The tree (top levels shown)")
display_depth = min(max_depth, 3)
if max_depth > 3:
    st.caption(f"Trained at depth {max_depth}; showing the top {display_depth} levels for legibility.")

fig2, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    clf, max_depth=display_depth, feature_names=feat_names,
    class_names=["died", "survived"], filled=True, rounded=True,
    fontsize=8, impurity=False, proportion=True, ax=ax,
)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

# --- full tree as a text outline -------------------------------------------
st.subheader("The whole tree, broadly")
st.caption(
    "The diagram above gets crowded fast, so here is the entire tree as a "
    "text outline — every split and what it asks, top to bottom."
)
outline_depth = min(max_depth, 10)
txt = export_text(
    clf, feature_names=list(feat_names), class_names=["died", "survived"],
    max_depth=outline_depth,
)
st.code(txt, language="text")
st.caption(
    "Read it like nested questions. Each `|--- feature <= value` is one yes/no test; "
    "deeper indentation = further down the tree. A leaf marked `survived`/`died` is the "
    "verdict once you reach it. One-hot columns like `Sex_female <= 0.50` just mean "
    "\"is this passenger NOT female?\" (0 = no, 1 = yes)."
    + (f" Showing the top {outline_depth} levels of a depth-{max_depth} tree." if max_depth > outline_depth else "")
)
