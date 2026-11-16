"""plot_data_card.py
Generate the PPT page-5 "data card" figures for the IDAT7215 Topic 4 project.

Reads the group's fixed split (data/split.csv, supplied by M2) and produces:
    figures/p05_data_composition.png   overall + per-split class composition
    figures/p05_split_ratio.png        train/val/test ratio donut
    figures/p05_samples.png            sample image montage (4 per class)

The montage downloads only the 8 needed images from the pinned AMdataset
commit and verifies each file's SHA-256 against split.csv (the same
integrity rule used by m4/fetch_data.py).

Usage:
    python plot_data_card.py [--split PATH] [--out-dir PATH]

All numbers are computed from split.csv - nothing is hard-coded, so the
figures regenerate automatically if M2 updates the split.
"""
import argparse
import csv
import hashlib
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- design tokens (validated reference palette, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis text
GRID = "#e1e0d9"         # hairline grid
C_NONDEFECT = "#2a78d6"  # categorical slot 1 (blue)
C_DEFECT = "#eb6834"     # categorical slot 2 (orange)
RAMP = {"train": "#184f95", "val": "#3987e5", "test": "#86b6ef"}  # blue ramp, dark = large

# Pinned commit used by M4's fetch_data.py (same byte-identical source).
SOURCE = "https://raw.githubusercontent.com/mahsan2/AMdataset/52fe75dec70c1e8046493b4e3256f8f752a4d39d/dataset/"
SPLIT_ORDER = ("train", "val", "test")
SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}


def load_split(path):
    """Read data/split.csv and validate its schema."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "path", "label", "class_name", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            sys.exit("split.csv must contain columns: %s" % sorted(required))
        rows = list(reader)
    if not rows:
        sys.exit("split.csv is empty")
    return rows


def split_stats(rows):
    """Return {split: {class_name: count}} plus per-class totals."""
    per_split = {name: Counter() for name in SPLIT_ORDER}
    per_class = Counter()
    for row in rows:
        per_split[row["split"]][row["class_name"]] += 1
        per_class[row["class_name"]] += 1
    return per_split, per_class


def style_axes(ax, ylabel):
    """Recessive axes: no top/right spines, hairline grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)


def figure_composition(per_split, per_class, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    # Panel A: overall Dataset One composition (two bars, categories named on x).
    names = ("Non-defect", "Defect")
    values = [per_class["Non-defect"], per_class["Defect"]]
    colors = [C_NONDEFECT, C_DEFECT]
    ax1.bar(names, values, width=0.45, color=colors,
            edgecolor=SURFACE, linewidth=2, zorder=3)
    for name, value in zip(names, values):
        ax1.text(name, value + 3, str(value), ha="center", va="bottom",
                 color=INK, fontsize=12, fontweight="bold")
    style_axes(ax1, "Images")
    ax1.set_ylim(0, 160)
    ax1.set_title("Dataset One composition", color=INK, fontsize=13, fontweight="bold")

    # Panel B: per-split stacked bars (Defect bottom, Non-defect top).
    bottoms = [0, 0, 0]
    labels = [SPLIT_NAMES[s] for s in SPLIT_ORDER]
    series = (("Non-defect", C_NONDEFECT), ("Defect", C_DEFECT))
    for class_name, color in series:
        heights = [per_split[s][class_name] for s in SPLIT_ORDER]
        ax2.bar(labels, heights, width=0.45, bottom=bottoms, color=color,
                edgecolor=SURFACE, linewidth=2, label=class_name, zorder=3)
        for label, height, bottom in zip(labels, heights, bottoms):
            if height > 0:
                ax2.text(label, bottom + height / 2, str(height),
                         ha="center", va="center", color="#ffffff",
                         fontsize=11, fontweight="bold")
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    total = sum(per_class.values())
    ax2.text(0.5, 1.12, "n = %d" % total, transform=ax2.transAxes,
             ha="center", color=INK_2, fontsize=10)
    style_axes(ax2, "Images")
    ax2.set_ylim(0, 210)
    ax2.set_title("Fixed split by class", color=INK, fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=False, fontsize=10,
               labelcolor=INK_2, handlelength=1.2, handleheight=1.2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return total


def figure_ratio(per_split, total, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    sizes = [per_split[s]["Non-defect"] + per_split[s]["Defect"] for s in SPLIT_ORDER]
    wedges, _, autotexts = ax.pie(
        sizes, startangle=90, counterclock=False,
        colors=[RAMP[s] for s in SPLIT_ORDER],
        wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2),
        autopct=lambda p: "%.1f%%" % p, pctdistance=0.79,
    )
    # Label color by segment luminance: dark fills get white, the light step gets ink.
    for autotext, split_name in zip(autotexts, SPLIT_ORDER):
        autotext.set_color("#ffffff" if split_name != "test" else INK)
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    ax.text(0, 0, "n = %d" % total, ha="center", va="center",
            color=INK, fontsize=15, fontweight="bold")
    ax.legend(
        ["%s  %d" % (SPLIT_NAMES[s], sizes[i]) for i, s in enumerate(SPLIT_ORDER)],
        loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False,
        fontsize=10, labelcolor=INK_2, handlelength=1.2, handleheight=1.2,
    )
    ax.set_title("Fixed split ratio", color=INK, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def download_verified(relative, expected_sha, cache_dir):
    """Fetch one image and refuse to use it if the bytes don't match split.csv."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / relative.replace("/", "__")
    if dest.exists() and sha256_bytes(dest.read_bytes()) == expected_sha:
        return dest
    url = SOURCE + urllib.parse.quote(relative)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if sha256_bytes(data) != expected_sha:
        sys.exit("SHA-256 mismatch for %s - refusing to plot it" % relative)
    dest.write_bytes(data)
    return dest


def pick_samples(rows, per_class=4):
    """4 evenly spaced train images per class (deterministic, byte-verified)."""
    by_class = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_class[row["class_name"]].append(row)
    samples = []
    for class_name in ("Non-defect", "Defect"):
        group = sorted(by_class[class_name], key=lambda r: r["path"])
        n = len(group)
        indexes = sorted({0, n // 3, 2 * n // 3, n - 1})
        samples.extend((class_name, group[i]) for i in indexes)
    return samples


def figure_samples(samples, cache_dir, out_path):
    try:
        prepared = []
        for class_name, row in samples:
            path = download_verified(row["path"], row["sha256"].lower(), cache_dir)
            prepared.append((class_name, path, row["path"].split("/")[-1]))
    except Exception as exc:  # network / checksum problems: skip montage, keep the rest
        print("WARNING: sample montage skipped (%s)" % exc)
        return False

    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(11.2, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 5, width_ratios=(0.045, 1, 1, 1, 1))
    rows = {"Non-defect": prepared[:4], "Defect": prepared[4:]}
    for row_index, (class_name, items) in enumerate(rows.items()):
        label_ax = fig.add_subplot(gs[row_index, 0])
        label_ax.text(0.5, 0.5, class_name, rotation=90, ha="center", va="center",
                      color=INK, fontsize=12, fontweight="bold")
        label_ax.axis("off")
        for col_index, (_, path, filename) in enumerate(items):
            ax = fig.add_subplot(gs[row_index, col_index + 1])
            ax.imshow(mpimg.imread(str(path)))
            ax.set_title(filename, fontsize=7, color=MUTED, pad=4)
            ax.axis("off")
    fig.suptitle("Sample images from the fixed split (train)", color=INK,
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the page-5 data card figures.")
    parser.add_argument("--split", type=Path, default=here.parent / "data" / "split.csv")
    parser.add_argument("--out-dir", type=Path, default=here / "figures")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = load_split(args.split)
    per_split, per_class = split_stats(rows)
    total = sum(per_class.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.out_dir / "_sample_cache"

    print("== data card stats (from %s) ==" % args.split)
    print("total images: %d" % total)
    for class_name in ("Non-defect", "Defect"):
        print("  %-11s total=%3d  train=%3d val=%2d test=%2d" % (
            class_name, per_class[class_name],
            per_split["train"][class_name], per_split["val"][class_name],
            per_split["test"][class_name]))
    for split_name in SPLIT_ORDER:
        n = per_split[split_name]["Non-defect"] + per_split[split_name]["Defect"]
        print("  %-11s %3d (%.1f%%)" % (split_name, n, 100.0 * n / total))

    figure_composition(per_split, per_class, args.out_dir / "p05_data_composition.png")
    figure_ratio(per_split, total, args.out_dir / "p05_split_ratio.png")
    ok = figure_samples(pick_samples(rows), cache_dir, args.out_dir / "p05_samples.png")
    print("== outputs ==")
    for name in ("p05_data_composition.png", "p05_split_ratio.png", "p05_samples.png"):
        path = args.out_dir / name
        if path.exists():
            print("  %s  (%d KB)" % (path, path.stat().st_size // 1024))
        elif not ok:
            print("  %s  NOT generated" % path)


if __name__ == "__main__":
    main()
