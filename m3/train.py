import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from m3.data import SplitImages, read_split, sha256
from m3.models import CLASSES, IMAGE_SIZE, WEIGHTS, DefectClassifier, load_checkpoint


def metrics(labels, predictions):
    # Matrix orientation: true labels are rows, predictions are columns.
    matrix = torch.bincount(2 * labels + predictions, minlength=4).reshape(2, 2).tolist()
    (tn, fp), (fn, tp) = matrix
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1_denominator = 2 * tp + fp + fn
    return {
        "accuracy": (tn + tp) / len(labels),
        "f1_defect": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "confusion_matrix": matrix,
        "count": len(labels),
    }


@torch.inference_mode()
def score(model, loader, device, loss_fn):
    model.eval()
    scores, labels, paths = [], [], []
    for images, targets, relative in loader:
        scores.append(model(images.to(device)).cpu())
        labels.append(targets)
        paths.extend(relative)
    logits, truth = torch.cat(scores), torch.cat(labels)
    predicted = logits.argmax(1)
    return (
        {**metrics(truth, predicted), "loss": loss_fn(logits, truth).item()},
        list(zip(paths, truth.tolist(), predicted.tolist(), logits.softmax(1)[:, 1].tolist())),
    )


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(model, samples, device, warmup=10, repeats=60):
    model.eval()
    inputs = [sample.unsqueeze(0).to(device) for sample in samples]
    for index in range(warmup):
        model(inputs[index % len(inputs)])
    sync(device)
    elapsed = []
    for index in range(repeats):
        start = time.perf_counter_ns()
        model(inputs[index % len(inputs)])
        sync(device)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "mean_ms": statistics.mean(elapsed),
        "median_ms": statistics.median(elapsed),
        "p95_ms": sorted(elapsed)[math.ceil(0.95 * repeats) - 1],
        "warmup": warmup,
        "repeats": repeats,
        "input_count": len(inputs),
        "batch_size": 1,
        "precision": "float32",
        "scope": "model forward, prepared inputs on device, synchronized, excludes loading and transforms",
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def save_checkpoint(path, model, metadata, full=False):
    state = model.state_dict() if full else model.head.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save({
        **metadata,
        "format": "m3-full-v1" if full else "m3-head-v1",
        "state_dict": {name: value.detach().cpu() for name, value in state.items()},
    }, temporary)
    temporary.replace(path)


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run with GPU access or select --device cpu.")
    rows = read_split(args.split, args.data_root)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    datasets = {name: SplitImages(samples, training=name == "train") for name, samples in rows.items()}
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            generator=torch.Generator().manual_seed(args.seed),
        )
        for name, dataset in datasets.items()
    }
    metadata = {
        "architecture": args.arch,
        "classes": list(CLASSES),
        "base_weights": WEIGHTS[args.arch].name,
        "base_weights_url": WEIGHTS[args.arch].url,
        "split_sha256": sha256(args.split),
        "image_size": IMAGE_SIZE,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "platform": platform.platform(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "checkpoint_selection": "highest validation defect F1, then lowest validation loss, then earliest epoch",
        "split_counts": {name: len(items) for name, items in rows.items()},
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{args.arch} on {device}: {metadata['split_counts']}", flush=True)
    model = DefectClassifier(args.arch, pretrained=True, weights_dir=args.weights_dir).to(device)
    optimizer = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}[args.optimizer](
        model.head.parameters(), lr=args.lr,
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best = (-math.inf, -math.inf)
    best_path = output / "best_head.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, labels, _ in loaders["train"]:
            with torch.no_grad():
                features = model.features(images.to(device))
            optimizer.zero_grad()
            loss = loss_fn(model.head(features), labels.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        validation, _ = score(model, loaders["val"], device, loss_fn)
        key = (validation["f1_defect"], -validation["loss"])
        if key > best:
            best = key
            save_checkpoint(best_path, model, {**metadata, "selected_epoch": epoch})
            # Give M2/M5 a reloadable full model while the formal run continues.
            save_checkpoint(output / "best_full.pt", model,
                            {**metadata, "selected_epoch": epoch}, full=True)
        history.append((epoch, total / len(rows["train"]), validation["loss"], validation["accuracy"], validation["f1_defect"], validation["mcc"]))
        write_csv(output / "history.csv", ("epoch", "train_loss", "val_loss", "val_accuracy", "val_f1_defect", "val_mcc"), history)
        print(f"epoch {epoch:02d}/{args.epochs} train loss={history[-1][1]:.4f} val F1={key[0]:.4f}", flush=True)

    best_head = torch.load(best_path, map_location="cpu", weights_only=True)
    model.head.load_state_dict(best_head["state_dict"])
    full_path = output / "best_full.pt"
    save_checkpoint(full_path, model, {**metadata, "selected_epoch": best_head["selected_epoch"]}, full=True)
    # Evaluate the actual reloadable artifact rather than the last training epoch.
    model, selected = load_checkpoint(full_path, device)
    validation, _ = score(model, loaders["val"], device, loss_fn)
    test, predictions = score(model, loaders["test"], device, loss_fn)
    write_csv(output / "test_predictions.csv", ("path", "label", "prediction", "p_defect"), predictions)
    inputs = [datasets["test"][index][0] for index in range(min(len(datasets["test"]), 12))]
    result = {
        **selected,
        "validation": validation,
        "test": test,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "full_checkpoint_bytes": full_path.stat().st_size,
        "full_checkpoint_sha256": sha256(full_path),
        "head_checkpoint_bytes": best_path.stat().st_size,
        "head_checkpoint_sha256": sha256(best_path),
        "latency": benchmark(model, inputs, device),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the M3 defect classifier on the provided fixed split.")
    parser.add_argument("--arch", choices=WEIGHTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="AMdataset/dataset")
    parser.add_argument("--split", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--weights-dir", type=Path, help="Optional directory of official Torchvision weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "rmsprop"), default="adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, batch size, threads and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
