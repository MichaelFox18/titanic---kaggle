"""Screen 1 — Iteration Journey.

Reads the hard-coded scoreboard from iteration_data.py (no live training) and
plots CV vs LB across all 10 iterations, with the CV-LB gap shaded and the
danger threshold (0.06) flagged. Teaches: higher CV does not mean higher LB.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from iteration_data import GAP_DANGER, ITERATIONS

st.title("1 · The iteration journey")

st.markdown(
    "Ten experiments, each with a hypothesis. The line you'd *expect* — CV and "
    "LB rising together — only holds for the honest iterations. When the gap "
    "between them blows past **0.06**, the model is fitting quirks of the "
    "cross-validation that don't survive contact with the test set."
)

# --- controls ---------------------------------------------------------------
c1, c2 = st.columns([3, 2])
with c1:
    labels = [f"Iter {it.n} — {it.approach}" for it in ITERATIONS]
    selected_idx = st.selectbox(
        "Highlight an iteration",
        options=list(range(len(ITERATIONS))),
        format_func=lambda i: labels[i],
        index=8,  # iter 9, the best
    )
with c2:
    show_gap = st.toggle("Show the CV–LB gap band", value=True)

selected = ITERATIONS[selected_idx]

# --- chart ------------------------------------------------------------------
xs = [it.n for it in ITERATIONS]
cv = [it.cv for it in ITERATIONS]
lb = [it.lb for it in ITERATIONS]
danger_x = [it.n for it in ITERATIONS if it.gap > GAP_DANGER]
danger_y = [it.lb for it in ITERATIONS if it.gap > GAP_DANGER]

fig = go.Figure()

# CV line (drawn first so the gap fill targets it)
fig.add_trace(go.Scatter(
    x=xs, y=cv, name="CV (local estimate)",
    mode="lines+markers", line=dict(color="#4C78A8", width=3),
    marker=dict(size=8),
))

# LB line with optional fill up to the CV line above it
fig.add_trace(go.Scatter(
    x=xs, y=lb, name="LB (Kaggle test score)",
    mode="lines+markers", line=dict(color="#54A24B", width=3),
    marker=dict(size=8),
    fill="tonexty" if show_gap else None,
    fillcolor="rgba(150,150,150,0.18)",
))

# danger markers: iterations whose gap exceeds 0.06
if danger_x:
    fig.add_trace(go.Scatter(
        x=danger_x, y=danger_y, name=f"CV–LB gap > {GAP_DANGER:g} (overfit)",
        mode="markers",
        marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#E45756", width=3), symbol="circle"),
        hoverinfo="skip",
    ))

# highlight selected iteration with a vertical line
fig.add_vline(x=selected.n, line=dict(color="#B279A2", width=2, dash="dash"))

fig.update_layout(
    height=460,
    xaxis=dict(title="Iteration", dtick=1),
    yaxis=dict(title="Accuracy", range=[0.73, 0.87]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# --- selected iteration card ------------------------------------------------
st.subheader(f"Iteration {selected.n} — {selected.approach}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("CV", f"{selected.cv:.3f}",
          help="Cross-validation accuracy, measured locally on the 891 labeled training passengers. Our private "
               "guess at quality before submitting anything.")
m2.metric("Public LB", f"{selected.lb:.3f}",
          help="The Kaggle public leaderboard score: accuracy on 418 hidden test passengers. The real answer — "
               "but you only see it after submitting, and the project had a 10-submission budget.")
gap_flag = "  ⚠️" if selected.gap > GAP_DANGER else ""
m3.metric("CV–LB gap", f"{selected.gap:.3f}{gap_flag}",
          help=f"CV minus LB. A healthy model has a small, stable gap (~0.05 here). Above {GAP_DANGER:g} (⚠️) the "
               f"model was fitting cross-validation quirks that didn't transfer to the real test set.")
m4.metric("Outcome", "improved" if selected.good else "regressed",
          help="Did this iteration beat the best leaderboard score that came before it? Half of them did not — "
               "and the failures taught the most.")

if selected.gap > GAP_DANGER:
    st.error(f"**{selected.verdict}.** The gap of {selected.gap:.3f} exceeds the {GAP_DANGER:g} danger line — this iteration's CV gain did not survive to the leaderboard.")
elif selected.good:
    st.success(f"**{selected.verdict}.** CV and LB moved together — a real gain.")
else:
    st.warning(f"**{selected.verdict}.**")

st.caption(
    "CV is a guess; LB is the answer. When the gap blows past ~0.06, the model "
    "is fitting cross-validation quirks that don't generalize. Iterations 2, 4, "
    "8, and 10 all did this — and three of them had *higher* CV than the "
    "iteration that actually won (iter 9)."
)
