from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot publication-ready F1 robustness curves.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness/f1_robustness.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.results)
    baseline = frame[frame.perturbation == "baseline"][["model", "f1_macro"]]
    panels = [
        ("gaussian_noise", "Gaussian noise sigma", [0.0, 0.05, 0.10, 0.20]),
        ("motion_blur", "Motion blur kernel", [0.0, 5.0, 9.0, 13.0]),
        ("brightness", "Brightness change", [-0.40, -0.20, 0.0, 0.20, 0.40]),
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for axis, (kind, label, ticks) in zip(axes, panels):
        for model in frame.model.unique():
            clean_f1 = float(baseline[baseline.model == model].f1_macro.iloc[0])
            subset = frame[(frame.model == model) & (frame.perturbation == kind)].copy()
            x = subset.value.to_numpy()
            if kind == "brightness":
                x = x - 1.0
            x = list(x) + [0.0]
            y = list(subset.f1_macro) + [clean_f1]
            ordered = sorted(zip(x, y))
            axis.plot([p[0] for p in ordered], [p[1] for p in ordered], marker="o", linewidth=2, label=model)
        axis.set_title(kind.replace("_", " ").title())
        axis.set_xlabel(label)
        axis.set_xticks(ticks)
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle("Robustness on the fixed test split", y=0.985, fontsize=14, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

