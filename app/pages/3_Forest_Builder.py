"""Screen 3 — Forest Builder.

Shows variance reduction: as you add trees, the forest's accuracy stabilises
and the spread across random seeds narrows. Uses a "first-k-trees" trick —
train a few full forests once, then evaluate growing prefixes of their trees —
so the curve comes from a handful of fits, not a full sweep.

Teaches: bagging / variance reduction, and why n_estimators=400 was plenty.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app_utils import get_features, load_raw

st.title("3 · The forest builder")

st.markdown(
    "One decision tree is a noisy guess. A **forest** trains many trees on "
    "different random slices of the data and averages their votes — and "
    "averaging many independent noisy guesses cancels the noise. Add trees and "
    "watch the accuracy stabilise while the seed-to-seed spread shrinks."
)

N_MAX = 200
KS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 75, 100, 140, N_MAX]
SEEDS = [42, 0, 1, 7, 13]

# --- controls ---------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    n_trees = st.slider("Number of trees (n_estimators)", 1, N_MAX, 50,
                        help="How many trees the forest averages over. Each is trained on a different random "
                             "bootstrap sample of the passengers, so they make different mistakes. More trees = "
                             "a steadier average — but with diminishing returns once the noise is mostly cancelled.")
with c2:
    max_features = st.select_slider(
        "max_features (fraction considered per split)",
        options=[0.1, 0.3, 0.5, 1.0], value=0.5,
        help="At each split, a tree only considers this fraction of the features (chosen at random). Lower values "
             "force trees to rely on different features, so they DECORRELATE — and averaging decorrelated trees "
             "cancels more noise. At 1.0 every tree sees all features and they look too alike for averaging to help much.",
    )

f1, f2, f3 = st.columns(3)
with f1:
    ifs = st.toggle("FamilySurvival", value=True)
with f2:
    itg = st.toggle("Ticket-group features", value=True)
with f3:
    show_votes = st.toggle("Show individual tree votes", value=True)


def _fmt_age(v) -> str:
    try:
        return str(int(v)) if not np.isnan(v) else "unknown"
    except (TypeError, ValueError):
        return "unknown"


@st.cache_data(show_spinner="Growing forests…")
def forest_growth(ifs: bool, itg: bool, max_features: float):
    """Train SEEDS forests of N_MAX trees on a stratified holdout, then for
    each k in KS measure accuracy using only the first k trees. Returns the
    growth curve (mean + min/max across seeds) and the per-tree vote trace for
    the single most-disputed holdout passenger."""
    X, y, _, _ = get_features(ifs, itg)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    y_te_arr = y_te.values

    per_seed_acc = []  # (seed, len(KS))
    proba_stacks = []  # each: (N_MAX, n_holdout)
    for s in SEEDS:
        rf = RandomForestClassifier(
            n_estimators=N_MAX, max_features=max_features, max_depth=4,
            min_samples_leaf=4, min_samples_split=3, criterion="entropy",
            random_state=s, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        tree_probas = np.array([est.predict_proba(X_te)[:, 1] for est in rf.estimators_])
        proba_stacks.append(tree_probas)
        cumavg = np.cumsum(tree_probas, axis=0) / np.arange(1, N_MAX + 1)[:, None]
        accs = [float(((cumavg[k - 1] >= 0.5).astype(int) == y_te_arr).mean()) for k in KS]
        per_seed_acc.append(accs)

    per_seed = np.array(per_seed_acc)
    mean = per_seed.mean(axis=0).tolist()
    lo = per_seed.min(axis=0).tolist()
    hi = per_seed.max(axis=0).tolist()

    # Pick the holdout passenger the trees disagree on most (highest std of
    # per-tree probability), from the first seed's forest.
    stack0 = proba_stacks[0]
    disagree = stack0.std(axis=0)
    loc = int(disagree.argmax())
    votes = stack0[:, loc].tolist()          # length N_MAX
    truth = int(y_te_arr[loc])
    raw_train, _ = load_raw()
    raw_row = raw_train.iloc[int(X_te.index[loc])]
    desc = f"{raw_row['Sex']}, class {int(raw_row['Pclass'])}, age {_fmt_age(raw_row['Age'])}"
    return KS, mean, lo, hi, votes, truth, desc


ks, mean, lo, hi, votes, truth, desc = forest_growth(ifs, itg, max_features)

# --- growth curve -----------------------------------------------------------
fig = go.Figure()
# min-max band across seeds
fig.add_trace(go.Scatter(x=ks + ks[::-1], y=hi + lo[::-1], fill="toself",
                         fillcolor="rgba(76,120,168,0.18)", line=dict(color="rgba(0,0,0,0)"),
                         name="seed-to-seed spread", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ks, y=mean, mode="lines+markers", name="mean accuracy",
                         line=dict(color="#4C78A8", width=3)))
fig.add_vline(x=n_trees, line=dict(color="#B279A2", width=2, dash="dash"),
              annotation_text=f"{n_trees} trees", annotation_position="top")
fig.update_layout(
    height=380, xaxis=dict(title="number of trees", range=[0, N_MAX + 5]),
    yaxis=dict(title="holdout accuracy"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Each tree is a noisy guess. Averaging many independent guesses cancels the "
    "noise — the shaded band (the spread across 5 random seeds) tightens as you "
    "add trees. This is why a forest beats a single tree, and why n_estimators=400 "
    "was plenty: the curve has flattened and the band has basically stopped "
    "shrinking long before then."
)

# --- votes panel ------------------------------------------------------------
if show_votes:
    st.subheader("How the first trees vote on one tricky passenger")
    st.caption(f"A held-out passenger the trees disagree on most — {desc}. "
               f"True outcome: **{'survived' if truth else 'died'}**.")

    k = n_trees
    v = votes[:k]
    running = (np.cumsum(v) / np.arange(1, k + 1)).tolist()
    xs = list(range(1, k + 1))

    figv = go.Figure()
    figv.add_trace(go.Scatter(x=xs, y=v, mode="markers", name="each tree's vote",
                              marker=dict(size=6, color="#BAB0AC")))
    figv.add_trace(go.Scatter(x=xs, y=running, mode="lines", name="running average",
                              line=dict(color="#4C78A8", width=3)))
    figv.add_hline(y=0.5, line=dict(color="#E45756", width=1, dash="dot"),
                   annotation_text="0.5 decision line", annotation_position="right")
    figv.update_layout(
        height=320, xaxis=dict(title="tree number"),
        yaxis=dict(title="P(survived)", range=[-0.05, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(figv, use_container_width=True)
    final = running[-1] if running else 0.5
    st.caption(
        f"Individual trees scatter all over (grey). Their running average (blue) "
        f"settles down as trees accumulate — here landing at {final:.2f}, "
        f"{'above' if final >= 0.5 else 'below'} the line, predicting "
        f"**{'survived' if final >= 0.5 else 'died'}**. The forest is calmer than any single tree."
    )
