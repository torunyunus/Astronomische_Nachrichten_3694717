# Maintainer: Y.Torun, ytorun@cumhuriyet.edu.tr

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from analysis_common import RANDOM_STATE, prepare_dataframe, make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="spatial_validation_output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, feature_sets = prepare_dataframe(pd.read_csv(args.data))
    dev, _ = train_test_split(
        df,
        test_size=0.20,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )
    dev = dev.reset_index(drop=True)

    if "ra" not in dev.columns:
        raise ValueError("Input CSV must contain RA for spatial validation.")

    order = np.argsort(dev["ra"].to_numpy(dtype=float), kind="mergesort")
    region = np.empty(len(dev), dtype=int)
    for rid, idx in enumerate(np.array_split(order, 5), start=1):
        region[idx] = rid
    dev["ra_region"] = region

    selected = {
        "F4 Mag+Colors": "Random Forest",
        "F7 Full": "LightGBM",
    }
    rows = []

    for fs_name, model_name in selected.items():
        cols = feature_sets[fs_name]
        for rid in range(1, 6):
            tr = dev["ra_region"] != rid
            te = dev["ra_region"] == rid

            X_train = dev.loc[tr, cols].to_numpy(dtype=float)
            y_train = dev.loc[tr, "target"].to_numpy()
            X_test = dev.loc[te, cols].to_numpy(dtype=float)
            y_test = dev.loc[te, "target"].to_numpy()

            model = make_model(model_name)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            recalls = recall_score(
                y_test,
                pred,
                labels=[0, 1, 2],
                average=None,
                zero_division=0,
            )
            support = np.bincount(y_test, minlength=3)

            rows.append({
                "feature_configuration": fs_name,
                "model": model_name,
                "ra_region": rid,
                "n": int(te.sum()),
                "ra_min": float(dev.loc[te, "ra"].min()),
                "ra_max": float(dev.loc[te, "ra"].max()),
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "mcc": matthews_corrcoef(y_test, pred),
                "auc_macro_ovr": roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                ),
                "recall_galaxy": recalls[0],
                "recall_star": recalls[1],
                "recall_qso": recalls[2],
                "support_galaxy": int(support[0]),
                "support_star": int(support[1]),
                "support_qso": int(support[2]),
            })

    results = pd.DataFrame(rows)
    results.to_csv(out / "spatial_holdout_results.csv", index=False)

    summary = (
        results.groupby(["feature_configuration", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            mcc_mean=("mcc", "mean"),
            auc_macro_ovr_mean=("auc_macro_ovr", "mean"),
            recall_galaxy_mean=("recall_galaxy", "mean"),
            recall_star_mean=("recall_star", "mean"),
            recall_qso_mean=("recall_qso", "mean"),
        )
    )
    summary.to_csv(out / "spatial_holdout_summary.csv", index=False)

    cols = [
        c
        for c in ["objid", "specobjid", "class", "ra", "ra_region"]
        if c in dev.columns
    ]
    dev[cols].to_csv(out / "spatial_region_assignment_development.csv", index=False)


if __name__ == "__main__":
    main()
