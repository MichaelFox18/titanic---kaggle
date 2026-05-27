"""
Titanic feature engineering + preprocessing pipeline.

LEARNING NOTES
==============
This module does TWO jobs that machine-learning code keeps separate from
the model itself:

  1. FEATURE ENGINEERING — inventing new columns from the raw ones that
     carry more signal than what's there. E.g. "Title" (Mr/Mrs/Miss/Master)
     extracted from Name is a stronger predictor of survival than Name
     itself, because Name is unique per passenger (zero generalization
     value) but Title groups people by social role and age cohort.

  2. PREPROCESSING — putting columns into a shape a model can actually
     consume. Models can't eat "C85" (the Cabin string) or NaN (missing
     Age) directly. We have to encode strings as numbers and decide what
     to do about every missing value.

Why split this out from the model code? Because every model we try
(Random Forest now, XGBoost later, an ensemble after that) needs the
SAME features. If we baked feature engineering into baseline_rf.py, we'd
copy-paste it three times and they'd drift apart.

There's a critical rule we follow here: the train and test sets must be
processed with the same transformations. The cleanest way to guarantee
that is to CONCATENATE them, do all feature work, then split them apart
again. Otherwise you risk training on one schema and predicting on
another (e.g. a new Deck letter shows up in test that train never saw,
breaking one-hot encoding).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths -- resolved relative to this file so the script works no
# matter what directory you call it from. PROJECT_ROOT is two levels up
# (src/pipeline.py -> src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ----------------------------------------------------------------------------
# Title extraction
# ----------------------------------------------------------------------------
#
# Names look like: "Braund, Mr. Owen Harris" or
#                  "Cumings, Mrs. John Bradley (Florence Briggs Thayer)".
# The token between the comma and the first period is the title. We grab
# it with a regex.
#
# WHY THIS MATTERS: Title is a powerful proxy for (Sex, Age, Social class)
# all bundled into one feature. "Master" specifically means a young boy
# (titled gentlemen above ~12 became "Mr."), so it captures the "women
# and children first" lifeboat policy in a way the raw Age column can't
# when 20% of ages are missing.
TITLE_REGEX = re.compile(r",\s*([^.]+)\.")

# Many titles appear only a few times (Don, Lady, the Countess, Capt,
# Jonkheer...). Models hate rare categories -- they generally can't
# learn from <5 examples. We collapse them into a single "Rare" bucket.
# Mlle/Ms get mapped to Miss, Mme to Mrs (French equivalents).
TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def extract_title(name: str) -> str:
    """Return the cleaned title from a Name string, or 'Rare' if unusual."""
    match = TITLE_REGEX.search(name)
    if not match:
        return "Rare"
    title = match.group(1).strip()
    title = TITLE_MAP.get(title, title)
    return title if title in COMMON_TITLES else "Rare"


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def build_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load train + test, engineer features on the combined set, then split.

    Returns
    -------
    X_train : DataFrame of model-ready features for the 891 labeled rows
    y_train : Series of 0/1 survival labels
    X_test  : DataFrame of model-ready features for the 418 unlabeled rows
    test_ids: PassengerId values for X_test (needed to build submission.csv)
    """
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Pull out what we need to remember before merging.
    # y is the label -- only train has it. We keep test_ids because the
    # submission file requires PassengerId alongside our prediction.
    y_train = train["Survived"].astype(int)
    test_ids = test["PassengerId"].copy()
    n_train = len(train)  # we'll split back at this index after processing

    # Concat for joint processing. drop Survived so the schemas match.
    # ignore_index=True gives us a clean 0..N-1 index.
    combined = pd.concat(
        [train.drop(columns=["Survived"]), test],
        ignore_index=True,
        sort=False,
    )

    # --- Title ---
    # Extract title from Name. We'll use this both as a feature AND as a
    # grouping variable for Age imputation below.
    combined["Title"] = combined["Name"].apply(extract_title)

    # --- Family features ---
    # FamilySize: how many people in this passenger's traveling party,
    # INCLUDING themselves. SibSp = siblings + spouses on board, Parch =
    # parents + children on board. Why does this matter? Lone travelers
    # had to make their own decisions; small families could move together
    # to a lifeboat; very large families (5+) had a hard time staying
    # together and survived poorly. So the relationship between family
    # size and survival is NON-LINEAR -- a tree-based model captures
    # this nicely without us having to spell it out.
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

    # --- Embarked ---
    # Only 2 rows are missing this. Fill with the mode (most common
    # value). Both missing passengers were 1st-class women who paid
    # similar fares to Southampton boarders, so this is a safe default.
    combined["Embarked"] = combined["Embarked"].fillna(
        combined["Embarked"].mode()[0]
    )

    # --- Fare ---
    # Exactly 1 missing Fare in test. We fill with the median for that
    # passenger's Pclass -- fares varied wildly by class so a global
    # median would be wrong. This is a tiny instance of the same logic
    # we use for Age below.
    combined["Fare"] = combined.groupby("Pclass")["Fare"].transform(
        lambda s: s.fillna(s.median())
    )

    # --- Age imputation by (Title, Pclass) ---
    # THIS IS THE BIG ONE. 263 of 1309 rows (20%) are missing Age.
    # If we filled with the global median (28), we'd say a 1st-class
    # Mrs. is the same age as a 3rd-class Miss -- they're not. Mrs.
    # implies married, which historically skewed older. Pclass correlates
    # with age too (1st-class passengers were wealthier and older on
    # average).
    #
    # The transform() call below means: for each (Title, Pclass) group,
    # compute that group's median Age, and use it to fill missing values
    # WITHIN that group. The fillback handles the edge case where a
    # whole group is empty (rare here, but defensive).
    combined["Age"] = combined.groupby(["Title", "Pclass"])["Age"].transform(
        lambda s: s.fillna(s.median())
    )
    # Belt-and-suspenders: if any Age is still NaN (e.g. group had no
    # known ages at all), fall back to the global median. In this
    # dataset this catches zero rows but it's a cheap safety net.
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())

    # --- Deck from Cabin ---
    # Cabin is 77% missing, so dropping it loses a lot of rows. But the
    # FACT of having a cabin (vs. not) is itself a strong signal: cabin
    # records were better-kept for first-class passengers, who survived
    # at much higher rates. So "NoCabin" becomes its own category and
    # the deck letter (A, B, C...) becomes a feature where we have it.
    combined["Deck"] = combined["Cabin"].fillna("N").str[0]
    # 'T' deck has only one passenger -- collapse into 'Rare' to avoid
    # a one-hot column that fires on a single row.
    combined.loc[combined["Deck"] == "T", "Deck"] = "Rare"

    # --- FarePerPerson ---
    # Tickets were often shared by families: a single Fare value covers
    # the whole group, not the individual. Dividing by FamilySize gives
    # us a per-person fare, which is a more honest measure of how much
    # money each person was paying for their accommodations.
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]

    # --- AgeBand and FareBand ---
    # Binning continuous variables into discrete buckets can help in
    # two ways: (a) it gives the model a clean signal for non-linear
    # cutoffs (e.g. "children under 12 survived"), (b) it reduces
    # sensitivity to outliers. We use pd.qcut for FareBand because Fare
    # is heavily right-skewed (one passenger paid 512!) -- qcut bins by
    # quantile so each bucket has equal population. We use pd.cut for
    # Age with hand-picked breakpoints aligned to life stages.
    combined["AgeBand"] = pd.cut(
        combined["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Mid", "Senior"],
    ).astype(str)
    # duplicates='drop' handles the case where quantile boundaries
    # coincide (e.g. lots of $7.75 tickets).
    combined["FareBand"] = pd.qcut(
        combined["Fare"], q=4, labels=False, duplicates="drop"
    )

    # --- Title x Pclass interaction ---
    # Sometimes the combined effect of two features is stronger than
    # either alone. A 1st-class Mrs. survived at ~97%; a 3rd-class Mrs.
    # at ~50%. Encoding the pair as a single categorical lets the model
    # learn this interaction directly. (Tree models can discover
    # interactions on their own, but giving them a direct feature
    # usually helps a little.)
    combined["TitlePclass"] = (
        combined["Title"] + "_" + combined["Pclass"].astype(str)
    )

    # --- Drop raw columns we've fully extracted from ---
    # Name was useful only for Title. Ticket has ~700 unique values --
    # too sparse to be useful without much more work. Cabin -> Deck.
    # PassengerId is just an index. We keep it OUT of features but
    # remembered test_ids above for the submission.
    combined = combined.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

    # --- One-hot encode categoricals ---
    # Tree models like Random Forest CAN handle ordinal-encoded
    # categoricals, but one-hot is the safest universal format and
    # makes feature importances readable ("Title_Master" vs. an
    # opaque integer). drop_first=False keeps all levels because
    # tree models don't suffer from collinearity the way linear
    # models do.
    cat_cols = ["Sex", "Embarked", "Title", "Deck", "AgeBand", "TitlePclass"]
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    # Pandas' get_dummies emits boolean dtype in modern versions; some
    # estimators warn about that. Cast to int8 for consistency and
    # small memory footprint.
    bool_cols = combined.select_dtypes(include="bool").columns
    combined[bool_cols] = combined[bool_cols].astype("int8")

    # Split back into train / test using the index we remembered.
    X_train = combined.iloc[:n_train].reset_index(drop=True)
    X_test = combined.iloc[n_train:].reset_index(drop=True)

    return X_train, y_train, X_test, test_ids


if __name__ == "__main__":
    # Smoke test: run the pipeline and print a summary so a human can
    # eyeball that nothing exploded. Useful during development.
    X_tr, y_tr, X_te, ids = build_features()
    print(f"X_train shape : {X_tr.shape}")
    print(f"y_train shape : {y_tr.shape}  (survival rate: {y_tr.mean():.3f})")
    print(f"X_test shape  : {X_te.shape}")
    print(f"test_ids len  : {len(ids)}")
    print(f"NaNs in X_train: {X_tr.isna().sum().sum()}")
    print(f"NaNs in X_test : {X_te.isna().sum().sum()}")
    print(f"\nFirst 10 columns: {list(X_tr.columns[:10])}")
