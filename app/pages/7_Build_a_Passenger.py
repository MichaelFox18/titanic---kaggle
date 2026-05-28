"""Screen 7 — Build a Passenger.

Invent a passenger, run them through the REAL pipeline (via build_features'
new extra_test_rows hook), and get a live survival probability from the
project's RF. Shows which engineered feature values the passenger ended up with.

Teaches: model inference + feature attribution — connecting raw inputs to the
features the model actually sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from app_utils import BEST_RF_PARAMS, RANDOM_STATE, build_features

st.title("7 · Build a passenger")

st.markdown(
    "Invent a passenger and watch the iteration-9 model decide their fate. "
    "Your inputs run through the **real** `build_features` pipeline — the same "
    "title extraction, group-median age imputation, and one-hot encoding the "
    "project uses — then the trained Random Forest predicts."
)

# --- inputs -----------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    sex = st.radio("Sex", ["female", "male"], index=0)
    pclass = st.radio("Passenger class", [1, 2, 3], index=2,
                      format_func=lambda p: f"{p} — {['1st','2nd','3rd'][p-1]} class")
with c2:
    age_unknown = st.checkbox("Age unknown (let the pipeline impute it)", value=False)
    age = st.slider("Age", 0, 80, 28, disabled=age_unknown)
    embarked = st.selectbox("Port embarked", ["S", "C", "Q"],
                            format_func=lambda e: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[e])
with c3:
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 8, 0)
    fare = st.slider("Fare paid (£)", 0.0, 300.0, 15.0, 0.5)

c4, c5, c6 = st.columns(3)
with c4:
    deck = st.selectbox("Cabin deck", ["none", "A", "B", "C", "D", "E", "F", "G"],
                        help="The deck letter from the cabin number. Most 3rd-class passengers had no recorded "
                             "cabin ('none'), which is itself a strong class signal the model uses.")
with c5:
    title_choice = st.selectbox(
        "Title", ["auto (from sex & age)", "Mr", "Mrs", "Miss", "Master", "Rare"],
        help="Title is the single strongest feature in the data — it bundles sex, rough age, and social role. "
             "'Master' meant a young boy (prioritised in 'women and children first'); 'Mr' an adult man (low "
             "survival). In 'auto' mode it's guessed from sex and age; pick a fixed value to control it directly.",
    )
with c6:
    ifs = st.toggle("FamilySurvival", value=True)

itg = st.toggle("Ticket-group features", value=True)


def derive_title(sex: str, age_val: float) -> str:
    """Rough title from sex + age, mirroring how Name encodes social role."""
    a = 30.0 if (age_val is None or (isinstance(age_val, float) and np.isnan(age_val))) else age_val
    if sex == "male":
        return "Master" if a < 13 else "Mr"
    return "Miss" if a < 18 else "Mrs"


if title_choice.startswith("auto"):
    title = derive_title(sex, np.nan if age_unknown else age)
else:
    title = title_choice

st.info(
    f"**Title used: {title}.** This is the model's strongest single feature. Heads-up on a quirk: in **auto** "
    "mode a male flips from *Master* (young boy) to *Mr* (adult man) at age 13 — and the model gives boys a far "
    "higher survival chance ('women and children first'), so the verdict can jump sharply right at that one-year "
    "boundary. That cliff is the **title changing**, not the single year of age. Pick a fixed Title above to hold "
    "it steady while you sweep age."
)


@st.cache_data(show_spinner="Running your passenger through the pipeline…")
def predict_passenger(ifs: bool, itg: bool, row_items: tuple):
    row = dict(row_items)
    extra = pd.DataFrame([row])
    # Train on the SAME build that includes the synthetic row, so the one-hot
    # columns line up exactly between train and the synthetic test row.
    X_train, y_train, X_test, _ = build_features(
        include_family_survival=ifs, include_ticket_group=itg, extra_test_rows=extra,
    )
    model = RandomForestClassifier(**BEST_RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    synth = X_test.iloc[[-1]]
    proba = float(model.predict_proba(synth)[0, 1])
    return proba, synth.iloc[0].to_dict()


row = {
    "PassengerId": 99999,
    "Pclass": pclass,
    "Name": f"Synthetic, {title}. Passenger",
    "Sex": sex,
    "Age": np.nan if age_unknown else float(age),
    "SibSp": int(sibsp),
    "Parch": int(parch),
    "Ticket": "SYNTH0000",
    "Fare": float(fare),
    "Cabin": np.nan if deck == "none" else f"{deck}50",
    "Embarked": embarked,
}
proba, feats = predict_passenger(ifs, itg, tuple(sorted(row.items(), key=lambda kv: kv[0])))

# --- verdict ----------------------------------------------------------------
survived = proba >= 0.5
st.subheader("Verdict")
color = "#54A24B" if survived else "#E45756"
st.markdown(
    f"<div style='font-size:3rem;font-weight:700;color:{color}'>"
    f"{proba*100:.1f}% — {'likely survived' if survived else 'likely died'}</div>",
    unsafe_allow_html=True,
)
fig = go.Figure(go.Bar(x=[proba], y=[""], orientation="h", marker=dict(color=color)))
fig.add_vline(x=0.5, line=dict(color="#333", width=2, dash="dot"))
fig.update_layout(height=90, xaxis=dict(range=[0, 1], title="P(survived)"),
                  margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- what the model actually saw -------------------------------------------
st.subheader("The features your passenger turned into")

def active_onehot(prefix: str) -> str:
    for k, v in feats.items():
        if k.startswith(prefix) and v == 1:
            return k[len(prefix):]
    return "—"

rows = [
    ("Title", active_onehot("Title_")),
    ("FamilySize", f"{feats.get('FamilySize', '—'):.0f}" if "FamilySize" in feats else "—"),
    ("IsAlone", "yes" if feats.get("IsAlone") == 1 else "no"),
    ("Deck", active_onehot("Deck_")),
    ("FarePerPerson", f"{feats.get('FarePerPerson', float('nan')):.2f}"),
]
if itg and "FarePerTicketPerson" in feats:
    rows.append(("FarePerTicketPerson", f"{feats['FarePerTicketPerson']:.2f}"))
if ifs and "FamilySurvival" in feats:
    fs = feats["FamilySurvival"]
    fs_label = {1.0: "1.0 (family survived)", 0.0: "0.0 (family died)", 0.5: "0.5 (no group info)"}.get(fs, f"{fs}")
    rows.append(("FamilySurvival", fs_label))

st.table(pd.DataFrame(rows, columns=["engineered feature", "value"]))

st.caption(
    "Change Sex from male to female and watch the probability leap — that's the "
    "single strongest signal in the data. Notice how your raw inputs became the "
    "features the model actually consumes (Title, FamilySize, Deck…). A synthetic "
    "passenger has no real family in the data, so FamilySurvival defaults to 0.5."
)
