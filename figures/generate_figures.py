# Maintainer: Yunis Torun

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def _save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1():
    """FIGURE 1: repeated-development-validation macro-F1 heatmap."""
    path = ROOT / "tables" / "Table_3_repeated_CV_macroF1_mean_sd.csv"
    df = pd.read_csv(path)
    label_col = df.columns[0]
    model_cols = list(df.columns[1:])
    means = np.array([
        [float(str(v).split("±")[0].strip()) for v in row]
        for row in df[model_cols].to_numpy()
    ])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    im = ax.imshow(means, aspect="auto", vmin=0.55, vmax=1.0)
    ax.set_xticks(np.arange(len(model_cols)))
    ax.set_xticklabels(model_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df[label_col])
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Feature configuration")
    ax.set_title("Repeated stratified validation: mean macro-F1 (5-fold × 3 repeats)")

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means[i, j]:.4f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean macro-F1")
    _save(fig, "Figure_1_repeated_CV_macroF1")


def figure_2():
    """FIGURE 2: final-test confusion matrices for the fixed F4/F7 models."""
    files = [
        (
            ROOT / "tables" / "F4_final_test_confusion_matrix.csv",
            "(a) F4 – Random Forest",
        ),
        (
            ROOT / "tables" / "F7_final_test_confusion_matrix.csv",
            "(b) F7 – LightGBM",
        ),
    ]
    classes = ["Galaxy", "Star", "QSO"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))

    last_im = None
    for ax, (path, title) in zip(axes, files):
        cm = pd.read_csv(path, index_col=0).to_numpy(dtype=int)
        row_total = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, row_total, out=np.zeros_like(cm, dtype=float), where=row_total > 0)
        last_im = ax.imshow(cm, aspect="equal")
        accuracy = np.trace(cm) / cm.sum()
        ax.set_title(f"{title}\nOverall Accuracy = {100 * accuracy:.2f}%")
        ax.set_xticks(range(3))
        ax.set_xticklabels(classes)
        ax.set_yticks(range(3))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        threshold = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                text = f"{cm[i, j]:,}\n({100 * frac[i, j]:.2f}%)"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=9,
                )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.88)
        cbar.set_label("Count")
    _save(fig, "Figure_2_confusion_matrices")


def figure_3():
    """FIGURE 3: magnitude robustness from manuscript Table 6."""
    path = ROOT / "tables" / "Table_6_magnitude_robustness.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))

    specs = [
        ("F4", "Random Forest", "F4 redshift-free"),
        ("F7", "LightGBM", "F7 full"),
    ]
    for config, model, label in specs:
        d = df[
            (df["feature_configuration"] == config)
            & (df["model"] == model)
        ].copy()
        d["r_bin_id"] = d["r_bin"].str.extract(r"(\d+)").astype(int)
        d = d.sort_values("r_bin_id")

        axes[0].errorbar(
            d["r_bin_id"],
            d["macro_f1"],
            yerr=np.vstack([
                d["macro_f1"] - d["macro_f1_ci95_low"],
                d["macro_f1_ci95_high"] - d["macro_f1"],
            ]),
            marker="o",
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            d["r_bin_id"],
            d["recall_star"],
            yerr=np.vstack([
                d["recall_star"] - d["recall_star_ci95_low"],
                d["recall_star_ci95_high"] - d["recall_star"],
            ]),
            marker="o",
            capsize=3,
            label=label,
        )

    counts = df[
        (df["feature_configuration"] == "F4")
        & (df["model"] == "Random Forest")
    ].copy()
    counts["r_bin_id"] = counts["r_bin"].str.extract(r"(\d+)").astype(int)
    counts = counts.sort_values("r_bin_id")

    for ax in axes:
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_xlabel("r-band magnitude bin")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.70, 1.01)
    axes[0].set_title("(a)")
    axes[1].set_ylabel("STAR recall")
    axes[1].set_ylim(0.50, 1.01)
    axes[1].set_title("(b)")

    count_text = ";  ".join(
        f"Q{int(r.r_bin_id)} = {int(r.galaxy_n)}/{int(r.star_n)}/{int(r.qso_n)}"
        for r in counts.itertuples()
    )
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.035, f"Class counts (G/S/Q): {count_text}", ha="center", fontsize=8)
    _save(fig, "Figure_3_magnitude_robustness")


def figure_4():
    """FIGURE 4: reliability curves for raw and calibrated probabilities."""
    path = ROOT / "07_probability_calibration" / "reliability_curve_data.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))

    for ax, config, title in zip(
        axes,
        ["F4", "F7"],
        ["(a) F4: Random Forest", "(b) F7: LightGBM"],
    ):
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
        for method in ["Raw", "Sigmoid", "Isotonic"]:
            d = df[
                (df["feature_configuration"] == config)
                & (df["method"] == method)
                & (df["n"] > 0)
            ].sort_values("bin_id")
            yerr = np.vstack([
                d["empirical_accuracy"] - d["accuracy_ci95_low"],
                d["accuracy_ci95_high"] - d["empirical_accuracy"],
            ])
            ax.errorbar(
                d["mean_confidence"],
                d["empirical_accuracy"],
                yerr=yerr,
                marker="o",
                capsize=2.5,
                label=method,
            )
        ax.set_xlim(0.3, 1.01)
        ax.set_ylim(0.3, 1.01)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(title)
        ax.legend(frameon=False)

    _save(fig, "Figure_4_reliability_curves")


def figure_5():
    """FIGURE 5: grouped permutation importance from manuscript Table 8."""
    path = ROOT / "tables" / "Table_8_grouped_permutation_importance.csv"
    df = pd.read_csv(path)

    config_col = "Feature configuration"
    group_col = "Feature group"
    value_col = "ΔMacro-F1 mean"
    sd_col = "SD"

    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = [f"{r[config_col]} - {r[group_col]}" for _, r in df.iterrows()]
    y = np.arange(len(df))
    ax.barh(y, df[value_col], xerr=df[sd_col], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in macro-F1 after grouped permutation")
    _save(fig, "Figure_5_grouped_permutation_importance")


def supplementary_figure_s1():
    path = ROOT / "04_magnitude_robustness" / "class_conditional_recall_results.csv"
    df = pd.read_csv(path)
    stars = df[df["true_class"] == "STAR"].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for config, model in [("F4", "Random Forest"), ("F7", "LightGBM")]:
        d = stars[
            (stars["feature_configuration"] == config)
            & (stars["model"] == model)
        ].sort_values("class_r_bin_id")
        ax.errorbar(
            d["class_r_bin_id"],
            d["recall"],
            yerr=np.vstack([
                d["recall"] - d["recall_ci95_low"],
                d["recall_ci95_high"] - d["recall"],
            ]),
            marker="o",
            capsize=3,
            label=f"{config} {model}",
        )
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("STAR class-conditional r-band magnitude bin")
    ax.set_ylabel("STAR recall")
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False)
    _save(fig, "Supplementary_Figure_S1_STAR_class_conditional_recall")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    supplementary_figure_s1()


if __name__ == "__main__":
    main()
