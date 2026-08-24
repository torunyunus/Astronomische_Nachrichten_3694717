# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
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
            model = make_model(model_name)
            model.fit(dev.loc[tr, cols].to_numpy(dtype=float), dev.loc[tr, "target"].to_numpy())
            pred = model.predict(dev.loc[te, cols].to_numpy(dtype=float))
            yt = dev.loc[te, "target"].to_numpy()
            rows.append({
                "feature_configuration": fs_name,
                "model": model_name,
                "ra_region": rid,
                "n": int(te.sum()),
                "ra_min": float(dev.loc[te, "ra"].min()),
                "ra_max": float(dev.loc[te, "ra"].max()),
                "accuracy": accuracy_score(yt, pred),
                "macro_f1": f1_score(yt, pred, average="macro"),
                "mcc": matthews_corrcoef(yt, pred),
            })

    results = pd.DataFrame(rows)
    results.to_csv(out / "spatial_holdout_results.csv", index=False)
    summary = (
        results.groupby(["feature_configuration", "model"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
        )
    )
    summary.to_csv(out / "spatial_holdout_summary.csv", index=False)

    cols = [c for c in ["objid", "specobjid", "class", "ra", "ra_region"] if c in dev.columns]
    dev[cols].to_csv(out / "spatial_region_assignment_development.csv", index=False)


if __name__ == "__main__":
    main()
