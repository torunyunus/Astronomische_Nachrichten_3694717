# Maintainer: Yunis Torun

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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

from analysis_common import RANDOM_STATE, LABEL_MAP, prepare_dataframe


def equal_count_bins(values, n_bins=5):
    order = np.argsort(np.asarray(values), kind="mergesort")
    out = np.empty(len(order), dtype=int)
    for bid, idx in enumerate(np.array_split(order, n_bins), start=1):
        out[idx] = bid
    return out


def bootstrap_multiclass(y, pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_by_class = {c: np.flatnonzero(y == c) for c in classes}
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(idx_by_class[c], size=len(idx_by_class[c]), replace=True)
            for c in classes
        ])
        yt, yp = y[idx], pred[idx]
        rec = recall_score(
            yt, yp, labels=[0, 1, 2], average=None, zero_division=0
        )
        rows.append([
            accuracy_score(yt, yp),
            f1_score(yt, yp, average="macro", zero_division=0),
            matthews_corrcoef(yt, yp),
            rec[0], rec[1], rec[2],
        ])
    a = np.asarray(rows)
    return np.quantile(a, 0.025, axis=0), np.quantile(a, 0.975, axis=0)


def binary_recall_ci(correct, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = rng.choice(correct, size=len(correct), replace=True).mean()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def write_table_6(out: Path, overall: pd.DataFrame) -> None:
    cols = [
        "feature_configuration", "model", "r_bin", "r_min", "r_max",
        "galaxy_n", "star_n", "qso_n",
        "macro_f1", "macro_f1_ci95_low", "macro_f1_ci95_high",
        "recall_galaxy", "recall_galaxy_ci95_low", "recall_galaxy_ci95_high",
        "recall_star", "recall_star_ci95_low", "recall_star_ci95_high",
        "recall_qso", "recall_qso_ci95_low", "recall_qso_ci95_high",
    ]
    overall[cols].to_csv(out / "Table_6_magnitude_robustness.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--f4-model", required=True, help="Fixed F4 model from validation stage")
    ap.add_argument("--f7-model", required=True, help="Fixed F7 model from validation stage")
    ap.add_argument("--out", default="magnitude_robustness_output")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, fs = prepare_dataframe(pd.read_csv(args.data))
    _, test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    test = test.reset_index(drop=True)
    test["r_bin_id"] = equal_count_bins(test["r"].to_numpy(), 5)

    selected = {
        "F4": ("F4 Mag+Colors", "Random Forest", joblib.load(args.f4_model)),
        "F7": ("F7 Full", "LightGBM", joblib.load(args.f7_model)),
    }
    overall_rows = []
    class_rows = []

    for tag, (fs_name, model_name, model) in selected.items():
        pred_all = model.predict(test[fs[fs_name]].to_numpy(dtype=float))
        proba_all = model.predict_proba(test[fs[fs_name]].to_numpy(dtype=float))

        for bid in range(1, 6):
            m = test["r_bin_id"].eq(bid).to_numpy()
            d = test.loc[m]
            y = d["target"].to_numpy()
            pred = pred_all[m]
            proba = proba_all[m]
            rec = recall_score(
                y, pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            lo, hi = bootstrap_multiclass(
                y,
                pred,
                n_boot=args.bootstrap,
                seed=RANDOM_STATE + bid + (100 if tag == "F7" else 0),
            )
            counts = d["class"].value_counts()
            overall_rows.append({
                "feature_configuration": tag,
                "model": model_name,
                "r_bin_id": bid,
                "r_bin": f"Q{bid}",
                "n": len(d),
                "r_min": d["r"].min(),
                "r_max": d["r"].max(),
                "r_mean": d["r"].mean(),
                "r_median": d["r"].median(),
                "galaxy_n": int(counts.get("GALAXY", 0)),
                "star_n": int(counts.get("STAR", 0)),
                "qso_n": int(counts.get("QSO", 0)),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y, pred),
                "auc_macro_ovr": roc_auc_score(
                    y, proba, multi_class="ovr", average="macro"
                ),
                "recall_galaxy": rec[0],
                "recall_star": rec[1],
                "recall_qso": rec[2],
                "accuracy_ci95_low": lo[0],
                "accuracy_ci95_high": hi[0],
                "macro_f1_ci95_low": lo[1],
                "macro_f1_ci95_high": hi[1],
                "mcc_ci95_low": lo[2],
                "mcc_ci95_high": hi[2],
                "recall_galaxy_ci95_low": lo[3],
                "recall_galaxy_ci95_high": hi[3],
                "recall_star_ci95_low": lo[4],
                "recall_star_ci95_high": hi[4],
                "recall_qso_ci95_low": lo[5],
                "recall_qso_ci95_high": hi[5],
            })

        for class_name, class_id in LABEL_MAP.items():
            dc = test[test["target"].eq(class_id)].copy().reset_index()
            dc["class_r_bin_id"] = equal_count_bins(dc["r"].to_numpy(), 5)
            for bid in range(1, 6):
                m = dc["class_r_bin_id"].eq(bid).to_numpy()
                idx = dc.loc[m, "index"].to_numpy(dtype=int)
                correct = (pred_all[idx] == class_id).astype(float)
                lo, hi = binary_recall_ci(
                    correct,
                    n_boot=args.bootstrap,
                    seed=RANDOM_STATE + class_id * 10 + bid + (100 if tag == "F7" else 0),
                )
                d = dc.loc[m]
                class_rows.append({
                    "feature_configuration": tag,
                    "model": model_name,
                    "true_class": class_name,
                    "class_r_bin_id": bid,
                    "class_r_bin": f"Q{bid}",
                    "n": len(d),
                    "r_min": d["r"].min(),
                    "r_max": d["r"].max(),
                    "r_mean": d["r"].mean(),
                    "r_median": d["r"].median(),
                    "recall": correct.mean(),
                    "recall_ci95_low": lo,
                    "recall_ci95_high": hi,
                })

    overall = pd.DataFrame(overall_rows)
    class_conditional = pd.DataFrame(class_rows)
    overall.to_csv(out / "magnitude_bin_results.csv", index=False)
    class_conditional.to_csv(out / "class_conditional_recall_results.csv", index=False)
    write_table_6(out, overall)

    bins = (
        test.groupby("r_bin_id", as_index=False)
        .agg(
            n=("r", "size"),
            r_min=("r", "min"),
            r_max=("r", "max"),
            r_mean=("r", "mean"),
            r_median=("r", "median"),
        )
    )
    bins["r_bin"] = "Q" + bins["r_bin_id"].astype(str)
    bins.to_csv(out / "magnitude_bin_definition.csv", index=False)


if __name__ == "__main__":
    main()
