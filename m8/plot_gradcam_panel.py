# -*- coding: utf-8 -*-
"""plot_gradcam_panel.py — PPT page-12 Grad-CAM panel grid (M8).

Places 8 blind-review heatmaps (2 per model: one judged "covers the defect",
one judged "does not") into a 4x2 grid. Panels are copied verbatim from
M5's blind bundle, so nothing is re-rendered and the blind-review verdicts
stay attached to exactly the images the reviewers saw.

Selections come from m5_results/results/per_image_votes.csv (majority vote of
M1/M5/M8); localization rates come from blind_review_stats.csv.

Usage:
    python m8/plot_gradcam_panel.py [--panel-dir m8/panels] [--results-dir m8/results]
                                 [--out figures/p12_gradcam_grid.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import FancyBboxPatch

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

# Chosen in step 3.2 (strongest blind-review agreement available per model).
SELECTION = {
    "VGG16":              {"success": "4F9AEAB48B04", "failure": "076BC3EBDE74"},
    "MobileNetV2":        {"success": "0F497B417C70", "failure": "3F58DD1BF702"},
    "ResNet50":           {"success": "187B63565DEC", "failure": "F9C5BC7DE2B2"},
    "MobileNetV3-Small":  {"success": "025F28C81816", "failure": "1E7A13C47459"},
}


def read_votes(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {r["blind_id"]: r for r in csv.DictReader(stream)}


def read_rates(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return {r["model"]: r for r in rows}


def votes_str(row):
    vals = [row["votes_%s" % k] for k in ("M1", "M5", "M8")]
    n_yes = sum(1 for v in vals if v == "1")
    return "%s → %d/3 rated 'covered'" % ("/".join(vals), n_yes)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the page-12 Grad-CAM grid.")
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--panel-dir", type=Path, default=here / "panels")
    parser.add_argument("--out", type=Path, default=here / "figures" / "p12_gradcam_grid.png")
    args = parser.parse_args()

    votes = read_votes(args.results_dir / "per_image_votes.csv")
    rates = read_rates(args.results_dir / "blind_review_stats.csv")

    fig, axes = plt.subplots(4, 2, figsize=(13.2, 9.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    col_titles = ("Localized — heatmap covers the defect", "Missed — heatmap does not cover it")
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color=INK, fontsize=13, fontweight="bold", pad=28)
        axes[0, col].annotate(
            "majority of M1 / M5 / M8", xy=(0.5, 1.075), xycoords="axes fraction",
            ha="center", va="bottom", color=MUTED, fontsize=9.5)

    for i, model in enumerate(MODEL_ORDER):
        rate = rates[model]["localization_rate"]
        pct = "%.0f%%" % (100.0 * float(rate)) if rate else "n/a"
        label = "%s\nlocalization %s" % (model, pct)
        axes[i, 0].annotate(
            label, xy=(-0.035, 0.5), xycoords="axes fraction", ha="right", va="center",
            color=MODEL_COLORS[model], fontsize=12, fontweight="bold", linespacing=1.5)
        for col, kind in enumerate(("success", "failure")):
            ax = axes[i, col]
            ax.set_facecolor(SURFACE)
            blind_id = SELECTION[model][kind]
            path = args.panel_dir / ("%s.png" % blind_id)
            if not path.is_file():
                sys.exit("missing heatmap: %s" % path)
            ax.imshow(mpimg.imread(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            row = votes[blind_id]
            ax.annotate(
                "%s  ·  blind %s" % (row["image_id"], blind_id),
                xy=(0.0, -0.035), xycoords="axes fraction", ha="left", va="top",
                color=INK_2, fontsize=8.5)
            ax.annotate(
                votes_str(row), xy=(0.0, -0.115), xycoords="axes fraction",
                ha="left", va="top", color=MUTED, fontsize=8.5)

    fig.suptitle("Grad-CAM localization: a bigger model is not a better explanation",
                 color=INK, fontsize=15, fontweight="bold")
    fig.text(0.5, -0.028,
             "Left / middle / right of each panel = input · Grad-CAM · overlay (M5's rendering, unmodified) · "
             "8 of 80 blind heatmaps · localization rates from the full 3-reviewer blind review",
             ha="center", color=MUTED, fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
