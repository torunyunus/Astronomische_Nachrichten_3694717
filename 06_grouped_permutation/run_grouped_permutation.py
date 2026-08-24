# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from analysis_common import RANDOM_STATE, prepare_dataframe


def grouped_permutation(model, X: pd.DataFrame, y: np.ndarray, groups: dict, n_perm=100, seed=42):
    rng = np.random.default_rng(seed)
    baseline = f1_score(y, model.predict(X.to_numpy(dtype=float)), average="macro")
    rows = []
    for group_name, cols in groups.items():
        drops = []
        for p in range(n_perm):
            order = rng.permutation(len(X))
            Xp = X.copy()
            Xp.loc[:, cols] = X.iloc[order][cols].to_numpy()
            score = f1_score(y, model.predict(Xp.to_numpy(dtype=float)), average="macro")
            drops.append(baseline - score)
        a = np.asarray(drops)
        rows.append({
            "group": group_name,
            "baseline_macro_f1": baseline,
            "mean_delta_macro_f1": a.mean(),
            "sd_delta_macro_f1": a.std(ddof=1),
            "q025_delta_macro_f1": np.quantile(a, 0.025),
            "q975_delta_macro_f1": np.quantile(a, 0.975),
            "positive_fraction": np.mean(a > 0),
            "n_permutations": n_perm,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--f4-model", required=True)
    ap.add_argument("--f7-model", required=True)
    ap.add_argument("--out", default="grouped_permutation_output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, fs = prepare_dataframe(pd.read_csv(args.data))
    _, test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    test = test.reset_index(drop=True)
    y = test["target"].to_numpy()

    mags = ["u", "g", "r", "i", "z"]
    colors = ["u-g", "g-r", "r-i", "i-z", "u-r", "g-i", "r-z"]

    f4 = joblib.load(args.f4_model)
    f7 = joblib.load(args.f7_model)

    r4 = grouped_permutation(
        f4,
        test[fs["F4 Mag+Colors"]],
        y,
        {
            "Magnitude group": mags,
            "Color group": colors,
            "Full photometric block": mags + colors,
        },
        n_perm=100,
        seed=RANDOM_STATE,
    )
    r4.insert(0, "feature_configuration", "F4")
    r4.insert(1, "model", "Random Forest")

    r7 = grouped_permutation(
        f7,
        test[fs["F7 Full"]],
        y,
        {
            "Magnitude group": mags,
            "Color group": colors,
            "Full photometric block": mags + colors,
            "Redshift": ["redshift"],
        },
        n_perm=100,
        seed=RANDOM_STATE,
    )
    r7.insert(0, "feature_configuration", "F7")
    r7.insert(1, "model", "LightGBM")

    pd.concat([r4, r7], ignore_index=True).to_csv(
        out / "grouped_permutation_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
