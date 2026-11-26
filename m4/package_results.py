"""Package measured metrics and slide figures from the two completed M4 runs."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ARCHITECTURES = {
    "resnet50": "ResNet50",
    "mobilenet_v3_small": "MobileNetV3-Small",
}


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_run(folder, architecture):
    result = json.loads((folder / "metrics.json").read_text())
    if result["architecture"] != architecture:
        raise ValueError(f"Wrong run architecture: {folder}")
    for filename, key in (("best_full.pt", "full_checkpoint_sha256"), ("best_head.pt", "head_checkpoint_sha256")):
        if file_sha256(folder / filename) != result[key]:
            raise ValueError(f"Checkpoint hash mismatch: {folder / filename}")
    return result


def plot_confusion(runs, path):
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), layout="constrained")
    maximum = max(value for run in runs for row in run["test"]["confusion_matrix"] for value in row)
    for axis, (label, run) in zip(axes, zip(ARCHITECTURES.values(), runs)):
        matrix = run["test"]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            for column in range(2):
                value = matrix[row][column]
                axis.text(column, row, value, ha="center", va="center", fontsize=16,
                          color="white" if value > maximum * 0.6 else "#162235")
        axis.set_title(label, fontsize=13)
        axis.set_xticks([0, 1], run["classes"])
        axis.set_yticks([0, 1], run["classes"])
        axis.set(xlabel="Predicted class", ylabel="True class")
    figure.colorbar(image, ax=axes, label="Images", shrink=0.82)
    figure.suptitle(f"Confusion matrices on the fixed test split (n={runs[0]['test']['count']})")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_efficiency(runs, path):
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), layout="constrained")
    names = list(ARCHITECTURES.values())
    panels = [
        ("Total parameters (millions)", [run["total_parameters"] / 1e6 for run in runs]),
        ("Mean forward latency (ms)", [run["latency"]["mean_ms"] for run in runs]),
    ]
    for axis, (title, values) in zip(axes, panels):
        bars = axis.barh(names, values, color=["#234e70", "#74a5c2"], height=0.5)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4)
        axis.set_xlim(0, max(values) * 1.28)
        axis.set_title(title, fontsize=12)
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Efficiency on {runs[0]['device'].upper()} · FP32 · batch size 1")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def package(source, output):
    runs = [read_run(source / architecture, architecture) for architecture in ARCHITECTURES]
    for field in ("split_sha256", "device", "platform", "classes", "image_size", "seed", "batch_size", "epochs", "optimizer", "learning_rate"):
        if runs[0][field] != runs[1][field]:
            raise ValueError(f"Runs differ in {field}")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for (architecture, name), run in zip(ARCHITECTURES.items(), runs):
        folder = output / architecture
        folder.mkdir()
        for filename in ("best_head.pt", "metrics.json", "history.csv", "test_predictions.csv"):
            shutil.copyfile(source / architecture / filename, folder / filename)
        test = run["test"]
        timing = run["latency"]
        rows.append((
            name, run["selected_epoch"], test["count"], test["accuracy"], test["f1_defect"],
            test["mcc"], run["total_parameters"], run["trainable_parameters"],
            run["full_checkpoint_bytes"], timing["mean_ms"], timing["median_ms"], timing["p95_ms"],
        ))
    fields = (
        "model", "selected_epoch", "test_n", "accuracy", "f1_defect", "mcc", "total_parameters",
        "trainable_parameters", "full_checkpoint_bytes", "mean_ms", "median_ms", "p95_ms",
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    plot_confusion(runs, output / "confusion_matrices.png")
    plot_efficiency(runs, output / "efficiency.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("m4_output"))
    parser.add_argument("--output", type=Path, default=Path("m4/results"))
    args = parser.parse_args()
    package(args.source, args.output)


if __name__ == "__main__":
    main()
