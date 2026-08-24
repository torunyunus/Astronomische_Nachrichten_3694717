# Maintainer: Yunis Torun


from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
CLASS_ORDER = ["GALAXY", "STAR", "QSO"]
LABEL_MAP = {"GALAXY": 0, "STAR": 1, "QSO": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "GaussianNB",
    "Random Forest",
    "LightGBM",
]


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    df["class"] = df["class"].astype(str).str.upper().str.strip()
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    for c in ["u", "g", "r", "i", "z", "redshift"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["u-g"] = df["u"] - df["g"]
    df["g-r"] = df["g"] - df["r"]
    df["r-i"] = df["r"] - df["i"]
    df["i-z"] = df["i"] - df["z"]
    df["u-r"] = df["u"] - df["r"]
    df["g-i"] = df["g"] - df["i"]
    df["r-z"] = df["r"] - df["z"]
    df["target"] = df["class"].map(LABEL_MAP).astype(int)

    mags = ["u", "g", "r", "i", "z"]
    colors = ["u-g", "g-r", "r-i", "i-z", "u-r", "g-i", "r-z"]
    feature_sets = {
        "F1 Redshift": ["redshift"],
        "F2 Magnitudes": mags,
        "F3 Colors": colors,
        "F4 Mag+Colors": mags + colors,
        "F5 Mag+Redshift": mags + ["redshift"],
        "F6 Colors+Redshift": colors + ["redshift"],
        "F7 Full": mags + colors + ["redshift"],
    }

    all_features = list(dict.fromkeys(sum(feature_sets.values(), [])))
    finite = np.isfinite(df[all_features].to_numpy(dtype=float)).all(axis=1)
    return df.loc[finite].reset_index(drop=True), feature_sets


def make_model(model_name: str):
    if model_name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ))),
        ])

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "GaussianNB":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ])

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "LightGBM":
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=1.00,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")
