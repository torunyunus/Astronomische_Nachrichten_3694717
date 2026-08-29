# Maintainer: Y.Torun, ytorun@cumhuriyet.edu.tr

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from analysis_common import RANDOM_STATE, MODEL_ORDER, prepare_dataframe, make_model


def stratified_bootstrap_ci(y_true, y_pred, proba=None, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_idx = {c: np.flatnonzero(y_true == c) for c in classes}
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(class_idx[c], size=len(class_idx[c]), replace=True)
            for c in classes
        ])
        yt, yp = y_true[idx], y_pred[idx]
        row = [
            accuracy_score(yt, yp),
            f1_score(yt, yp, average="macro"),
            matthews_corrcoef(yt, yp),
        ]
        if proba is not None:
            row.append(roc_auc_score(yt, proba[idx], multi_class="ovr", average="macro"))
        rows.append(row)
    a = np.asarray(rows)
    return np.quantile(a, [0.025, 0.975], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to randomized 150k CSV")
    ap.add_argument("--out", default="validation_output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, feature_sets = prepare_dataframe(pd.read_csv(args.data))
    dev, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )
    dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)

    split_map = pd.concat([
        dev.assign(split="development"),
        test.assign(split="final_test"),
    ], ignore_index=True)
    id_cols = [c for c in ["objid", "specobjid", "class", "split"] if c in split_map.columns]
    split_map[id_cols].to_csv(out / "split_assignment.csv", index=False)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cv_rows = []

    for fs_name, cols in feature_sets.items():
        X = dev[cols].to_numpy(dtype=float)
        y = dev["target"].to_numpy()
        for model_name in MODEL_ORDER:
            fold_scores = []
            for fold_id, (tr, va) in enumerate(cv.split(X, y), start=1):
                model = make_model(model_name)
                model.fit(X[tr], y[tr])
                pred = model.predict(X[va])
                score = f1_score(y[va], pred, average="macro")
                fold_scores.append(score)
                cv_rows.append({
                    "feature_configuration": fs_name,
                    "model": model_name,
                    "fold": fold_id,
                    "macro_f1": score,
                })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out / "repeated_cv_fold_scores.csv", index=False)
    summary = (
        cv_df.groupby(["feature_configuration", "model"], as_index=False)
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"))
    )
    summary.to_csv(out / "repeated_cv_summary.csv", index=False)

    selected = {
        "F4 Mag+Colors": "Random Forest",
        "F7 Full": "LightGBM",
    }
    final_rows = []

    for fs_name, model_name in selected.items():
        cols = feature_sets[fs_name]
        model = make_model(model_name)
        model.fit(dev[cols].to_numpy(dtype=float), dev["target"].to_numpy())
        X_test = test[cols].to_numpy(dtype=float)
        y_test = test["target"].to_numpy()
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        mcc = matthews_corrcoef(y_test, pred)
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        ci = stratified_bootstrap_ci(y_test, pred, proba, n_boot=2000, seed=RANDOM_STATE)

        final_rows.append({
            "feature_configuration": fs_name,
            "model": model_name,
            "accuracy": acc,
            "accuracy_ci95_low": ci[0, 0],
            "accuracy_ci95_high": ci[1, 0],
            "macro_f1": mf1,
            "macro_f1_ci95_low": ci[0, 1],
            "macro_f1_ci95_high": ci[1, 1],
            "mcc": mcc,
            "mcc_ci95_low": ci[0, 2],
            "mcc_ci95_high": ci[1, 2],
            "auc_macro_ovr": auc,
            "auc_ci95_low": ci[0, 3],
            "auc_ci95_high": ci[1, 3],
        })

        tag = "F4" if fs_name.startswith("F4") else "F7"
        pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(
            out / f"{tag}_final_test_confusion_matrix.csv", index=False, header=False
        )
        joblib.dump(model, out / f"{tag}_selected_model.joblib")
        np.savez_compressed(
            out / f"{tag}_final_test_predictions.npz",
            y_true=y_test,
            y_pred=pred,
            proba=proba,
        )

    pd.DataFrame(final_rows).to_csv(out / "untouched_final_test_results_F4_F7.csv", index=False)


if __name__ == "__main__":
    main()
