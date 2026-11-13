"""Package measured metrics and slide figures from the two completed M3 runs."""
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
    "vgg16": "VGG16",
    "mobilenet_v2": "MobileNetV2",
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


def plot_history(source, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for axis, (architecture, name) in zip(axes, ARCHITECTURES.items()):
        with (source / architecture / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        axis.plot(epochs, [float(row["train_loss"]) for row in rows], label="Training loss")
        axis.plot(epochs, [float(row["val_loss"]) for row in rows], label="Validation loss")
        axis.set(title=name, xlabel="Epoch", ylabel="Cross-entropy loss")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_materials(runs, output):
    names = list(ARCHITECTURES.values())
    table = ["| Model | Accuracy | Defect F1 | MCC | Parameters | Full model MB | Mean latency ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in zip(names, runs):
        table.append(f"| {name} | {run['test']['accuracy']:.1%} | {run['test']['f1_defect']:.4f} | "
                     f"{run['test']['mcc']:.4f} | {run['total_parameters']:,} | "
                     f"{run['full_checkpoint_bytes']/1e6:.2f} | {run['latency']['mean_ms']:.2f} |")
    paragraphs = []
    for name, run in zip(names, runs):
        (tn, fp), (fn, tp) = run["test"]["confusion_matrix"]
        paragraphs.append(f"{name}: {tn+tp}/{run['test']['count']} correct; "
                          f"{fn} missed defects; {fp} false alarms; selected epoch {run['selected_epoch']}.")
    text = "\n".join([
        "# M3 result-page materials (generated from measured results)", "",
        "## Page 1: How we trained the two models", "",
        "- VGG16 and MobileNetV2 use official Torchvision ImageNet V1 weights.",
        "- Frozen feature extractors; 4x4 average pooling, flatten, 64-unit ReLU, dropout 0.5, two logits.",
        "- Adam, learning rate 0.001, batch size 8, 30 epochs, seed 42.",
        "- Group fixed split: 190 train, 41 validation, 40 test. Defect is positive (label 1).",
        "- Best validation defect F1 selects the checkpoint; validation loss breaks ties.",
        "- The full saved model is reloaded before final test evaluation.", "",
        "## Page 2: Results on the fixed test split", "", *table, "", *paragraphs, "",
        "Figures: confusion_matrices.png and training_curves.png.", "",
        "## Page 3: Accuracy, size and latency", "",
        "Figure: efficiency.png. Use the measured table above to describe the observed trade-off.",
        f"Latency device: {runs[0]['device']}; FP32, batch size 1, 10 warm-ups and 60 synchronized forwards; excludes file loading and preprocessing.",
        "Cross-group speed comparisons must use the same hardware and timing boundaries.",
        "Limitations: only 40 test images, one seed and one fixed split; no claim of general deployment reliability.",
        "Robustness and Grad-CAM are evaluated separately by the designated teammates.",
        "These are the group's measured results, not the original paper's reported scores.", "",
        "## Short spoken explanation", "",
        "I trained VGG16 and MobileNetV2 using the same fixed data split as the other group models. "
        "Both started from ImageNet weights. I froze the feature extractors and trained a small classification head for thirty epochs. "
        "I used the validation set to select the best checkpoint, then reloaded it before evaluating the test images.",
        *paragraphs,
        "The table also shows the total parameters, saved model size and single-image forward time. "
        "These measurements help us compare classification performance with computational cost. "
        "However, the test set contains only forty images, so the results apply to this particular split. "
        "Our teammates will test noise, blur and brightness changes, and inspect the trained models using Grad-CAM.", "",
    ])
    (output / "PPT结果素材与讲稿.md").write_text(text, encoding="utf-8")
    (output / "交接说明.md").write_text(
        "# 给数据评估与 Grad-CAM 同学\n\n"
        "每个模型文件夹包含 best_full.pt（独立完整模型）、best_head.pt（分类头）、config.json、metrics.json、history.csv 和 test_predictions.csv。\n\n"
        "在交付包根目录使用：\n\n```python\n"
        "from PIL import Image\nimport torch\nfrom m3.models import load_checkpoint, image_transform\n"
        "model, metadata = load_checkpoint('交付结果/vgg16/best_full.pt', device='cpu')\n"
        "with Image.open('example.jpg') as image:\n"
        "    batch = image_transform()(image.convert('RGB')).unsqueeze(0)\n"
        "with torch.inference_mode():\n    p_defect = model(batch).softmax(1)[:, 1]\n```\n\n"
        "MobileNetV2 改成 mobilenet_v2 文件夹。完整权重无需联网下载骨干；分类头文件则需要对应官方 ImageNet V1 权重。\n\n"
        "Grad-CAM 时不要用 inference_mode，设 batch.requires_grad_(True)。VGG16 可选 model.backbone[28]，MobileNetV2 可选 model.backbone[-1] 作为起点。\n\n"
        "类别顺序固定为 [Non-defect, Defect]。预处理沿用 M4：RGB，直接缩放到224×224，ImageNet均值/标准差。扰动应在归一化前施加。\n",
        encoding="utf-8",
    )


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
        for filename in ("best_full.pt", "best_head.pt", "config.json", "metrics.json", "history.csv", "test_predictions.csv"):
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
    plot_history(source, output / "training_curves.png")
    write_materials(runs, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("m3_output"))
    parser.add_argument("--output", type=Path, default=Path("m3/results"))
    args = parser.parse_args()
    package(args.source, args.output)


if __name__ == "__main__":
    main()
