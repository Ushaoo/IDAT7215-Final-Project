# -*- coding: utf-8 -*-
"""plot_efficiency.py — PPT page-10 "efficiency comparison" draft figure (M8).

Reads data/efficiency_cpu.csv (M2's unified CPU benchmark) and produces a
1x3 small-multiple bar chart: parameters / CPU latency / checkpoint size.

Style follows plot_data_card.py (same SURFACE/INK/MUTED/GRID tokens) and the
group model palette agreed in blind_review/README.md.

Usage:
    python plot_efficiency.py [--csv data/efficiency_cpu.csv]
                              [--out figures/p10_efficiency.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- design tokens (shared with plot_data_card.py) ----
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


def load_efficiency(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        sys.exit("efficiency CSV is empty")
    by_model = {r["model"].strip(): r for r in rows}
    missing = [m for m in MODEL_ORDER if m not in by_model]
    if missing:
        sys.exit("efficiency CSV missing models: %s" % missing)
    return by_model


def style_axes(ax, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)


def bar_panel(ax, title, values, fmt, ylabel):
    names = list(MODEL_ORDER)
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    ax.bar(names, values, width=0.55, color=colors,
           edgecolor=SURFACE, linewidth=2, zorder=3)
    for name, value in zip(names, values):
        ax.text(name, value, fmt % value, ha="center", va="bottom",
                color=INK, fontsize=10.5, fontweight="bold")
    style_axes(ax, ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the page-10 efficiency figure.")
    parser.add_argument("--csv", type=Path, default=here / "data" / "efficiency_cpu.csv")
    parser.add_argument("--out", type=Path, default=here / "figures" / "p10_efficiency.png")
    args = parser.parse_args()

    data = load_efficiency(args.csv)

    params_m = [int(data[m]["total_parameters"]) / 1e6 for m in MODEL_ORDER]
    latency_ms = [float(data[m]["cpu_mean_ms"]) for m in MODEL_ORDER]
    size_mb = [float(data[m]["full_checkpoint_mb_decimal"]) for m in MODEL_ORDER]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 4.4),
                                        constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)

    bar_panel(ax1, "Parameters (M)", params_m, "%.2f", "Parameters (M)")
    bar_panel(ax2, "CPU latency (ms/image)", latency_ms, "%.1f", "Mean ms / image")
    bar_panel(ax3, "Checkpoint size (MB)", size_mb, "%.1f", "Checkpoint (MB)")

    fig.suptitle("Model efficiency on a unified CPU benchmark (batch size 1, FP32)",
                 color=INK, fontsize=13.5, fontweight="bold")
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
