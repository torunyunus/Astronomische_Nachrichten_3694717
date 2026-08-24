# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def figure_1():
    path = ROOT / "tables" / "Table_3_repeated_CV_macroF1_mean_sd.csv"
    df = pd.read_csv(path)
    label_col = df.columns[0]
    model_cols = list(df.columns[1:])
    means = np.array([
        [float(str(v).split("±")[0].strip()) for v in row]
        for row in df[model_cols].to_numpy()
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    for j, model in enumerate(model_cols):
        ax.plot(x, means[:, j], marker="o", label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col], rotation=30, ha="right")
    ax.set_ylabel("Development macro-F1")
    ax.set_xlabel("Feature configuration")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_1_repeated_CV_macroF1.png", dpi=300)
    fig.savefig(OUT / "Figure_1_repeated_CV_macroF1.pdf")
    plt.close(fig)


def figure_3():
    path = ROOT / "04_magnitude_robustness" / "magnitude_bin_results.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))

    for config, model in [("F4", "Random Forest"), ("F7", "LightGBM")]:
        d = df[(df["feature_configuration"] == config) & (df["model"] == model)].sort_values("r_bin_id")
        label = f"{config} {model}"
        axes[0].errorbar(
            d["r_bin_id"], d["macro_f1"],
            yerr=np.vstack([
                d["macro_f1"] - d["macro_f1_ci95_low"],
                d["macro_f1_ci95_high"] - d["macro_f1"],
            ]),
            marker="o", capsize=3, label=label,
        )
        axes[1].errorbar(
            d["r_bin_id"], d["recall_star"],
            yerr=np.vstack([
                d["recall_star"] - d["recall_star_ci95_low"],
                d["recall_star_ci95_high"] - d["recall_star"],
            ]),
            marker="o", capsize=3, label=label,
        )

    for ax in axes:
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_xlabel("r-band magnitude bin")
        ax.set_ylim(0.45, 1.02)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("STAR recall")
    axes[0].set_title("(a)")
    axes[1].set_title("(b)")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_magnitude_robustness.png", dpi=300)
    fig.savefig(OUT / "Figure_3_magnitude_robustness.pdf")
    plt.close(fig)


def figure_4():
    path = ROOT / "07_probability_calibration" / "reliability_curve_data.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))

    for ax, config, title in zip(axes, ["F4", "F7"], ["(a) F4 Random Forest", "(b) F7 LightGBM"]):
        for method in ["Raw", "Sigmoid", "Isotonic"]:
            d = df[(df["feature_configuration"] == config) & (df["method"] == method)].copy()
            d = d[d["n"] > 0].sort_values("bin_id")
            yerr = np.vstack([
                d["empirical_accuracy"] - d["accuracy_ci95_low"],
                d["accuracy_ci95_high"] - d["empirical_accuracy"],
            ])
            ax.errorbar(
                d["mean_confidence"], d["empirical_accuracy"],
                yerr=yerr, marker="o", capsize=2.5, label=method,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0.3, 1.01)
        ax.set_ylim(0.3, 1.01)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(title)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_reliability_curves.png", dpi=300)
    fig.savefig(OUT / "Figure_4_reliability_curves.pdf")
    plt.close(fig)


def figure_5():
    path = ROOT / "tables" / "Table_7_grouped_permutation_importance.csv"
    df = pd.read_csv(path)

    value_col = next(c for c in df.columns if "mean" in c.lower() and "macro" in c.lower())
    sd_col = next(c for c in df.columns if "sd" in c.lower() and "macro" in c.lower())
    config_col = next(c for c in df.columns if "feature" in c.lower() or "configuration" in c.lower())
    group_col = next(c for c in df.columns if c.lower() == "group")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = [f"{r[config_col]}: {r[group_col]}" for _, r in df.iterrows()]
    y = np.arange(len(df))
    ax.barh(y, df[value_col], xerr=df[sd_col], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in macro-F1 after grouped permutation")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_5_grouped_permutation_importance.png", dpi=300)
    fig.savefig(OUT / "Figure_5_grouped_permutation_importance.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_3()
    figure_4()
    figure_5()


if __name__ == "__main__":
    main()
