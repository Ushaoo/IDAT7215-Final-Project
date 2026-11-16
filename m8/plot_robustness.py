# -*- coding: utf-8 -*-
"""plot_robustness.py — PPT page-11 "robustness curve" draft figure (M8).

Reads data/robustness_results_all_models.csv (M2) and produces a 1x3
small-multiple line chart: Gaussian noise / motion blur / brightness, each
plotting Macro-F1 vs perturbation level for all four models.

Style matches plot_data_card.py and the group model palette.

Usage:
    python plot_robustness.py [--csv data/robustness_results_all_models.csv]
                              [--out figures/p11_robustness.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

MODEL_ORDER = ("VGG16", "MobileNetV2", "ResNet50", "MobileNetV3-Small")
MODEL_COLORS = {
    "VGG16": "#2a78d6",
    "MobileNetV2": "#eb6834",
    "ResNet50": "#1baf7a",
    "MobileNetV3-Small": "#eda100",
}

# (perturbation type, panel title, [(x label, value-string-or-None)]).
# value None = the shared "baseline" (clean) point.
PANELS = [
    ("gaussian_noise", "Gaussian noise (σ)",
     [("0", None), ("0.05", "0.05"), ("0.1", "0.1"), ("0.2", "0.2")]),
    ("motion_blur", "Motion blur (kernel)",
     [("0", None), ("5", "5.0"), ("9", "9.0"), ("13", "13.0")]),
    ("brightness", "Brightness change",
     [("-40%", "0.6"), ("-20%", "0.8"), ("0%", None), ("+20%", "1.2"), ("+40%", "1.4")]),
]


def load_robustness(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        sys.exit("robustness CSV is empty")
    baseline = {}
    table = {}
    for row in rows:
        model = row["model"].strip()
        pert = row["perturbation"].strip()
        val = row["value"].strip()
        f1 = float(row["f1_macro"])
        if pert == "baseline":
            baseline[model] = f1
        else:
            table[(model, pert, val)] = f1
    missing = [m for m in MODEL_ORDER if m not in baseline]
    if missing:
        sys.exit("robustness CSV missing baseline for: %s" % missing)
    return baseline, table


def style_axes(ax, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)


def line_panel(ax, title, xlabel, labels, baseline, table, pert_type):
    xs = list(range(len(labels)))
    for model in MODEL_ORDER:
        ys = []
        for _, val in labels:
            ys.append(baseline[model] if val is None else table[(model, pert_type, val)])
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                color=MODEL_COLORS[model], label=model, zorder=3)
    style_axes(ax, "Macro-F1")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in labels])
    ax.set_ylim(0.0, 1.06)
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the page-11 robustness figure.")
    parser.add_argument("--csv", type=Path,
                        default=here / "data" / "robustness_results_all_models.csv")
    parser.add_argument("--out", type=Path, default=here / "figures" / "p11_robustness.png")
    args = parser.parse_args()

    baseline, table = load_robustness(args.csv)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    for ax, (pert_type, title, labels) in zip(axes, PANELS):
        line_panel(ax, title, "Perturbation level", labels, baseline, table, pert_type)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.text(0.5, -0.01, "Perturbation level", ha="center", color=INK_2, fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=10, labelcolor=INK_2, handlelength=1.4, handleheight=1.4,
               bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("Clean accuracy is not enough: Macro-F1 under corruptions",
                 color=INK, fontsize=13.5, fontweight="bold")
    fig.text(0.5, -0.215, "40 test images · one split · one training seed · one noise realization",
             ha="center", color=MUTED, fontsize=9)

    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
