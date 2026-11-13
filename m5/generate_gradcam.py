import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from m5.data import read_split, select_rows
from m5.gradcam import GradCAM
from m5.modeling import (
    CLASSES,
    IMAGE_SIZE,
    default_target_layer,
    image_transform,
    load_checkpoint,
    resolve_target_layer,
)


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/checkpoint.pt")
    name, checkpoint = value.split("=", 1)
    if not name.strip() or not checkpoint.strip():
        raise argparse.ArgumentTypeError("Model name and checkpoint path are required")
    return name.strip(), Path(checkpoint)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "blind" / "images").mkdir(parents=True)
    (path / "private" / "images").mkdir(parents=True)
    return path


def slug(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "model"


def blind_identifier(seed, model_name, relative_path):
    payload = f"{seed}|{model_name}|{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12].upper()


def render_panel(original, heatmap, destination, title=None):
    image = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image)
    color = plt.get_cmap("turbo")(heatmap)[..., :3]
    color_uint8 = np.uint8(np.clip(color * 255, 0, 255))
    overlay = np.uint8(np.clip(0.55 * array + 0.45 * color_uint8, 0, 255))

    figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for axis, content, label in zip(
        axes,
        (array, color_uint8, overlay),
        ("Original", "Grad-CAM", "Overlay"),
    ):
        axis.imshow(content)
        axis.set_title(label)
        axis.axis("off")
    if title:
        figure.suptitle(title, fontsize=10)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def predict_all(model, rows, transform, device):
    predictions = {}
    for row in rows:
        with Image.open(row["path"]) as source:
            batch = transform(source.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            predictions[row["relative_path"]] = int(model(batch).argmax(dim=1).item())
    return predictions


def target_class(mode, prediction, truth):
    if mode == "predicted":
        return prediction
    if mode == "defect":
        return 1
    if mode == "ground-truth":
        return truth
    raise ValueError(mode)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and blinded review material.")
    parser.add_argument("--model", action="append", required=True, type=parse_model, metavar="NAME=CHECKPOINT")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-mode", choices=("predicted", "defect", "ground-truth"), default="predicted")
    parser.add_argument("--include-non-defect", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--target-layer", action="append", default=[], metavar="MODEL=MODULE.PATH")
    parser.add_argument("--rater", action="append", default=["M1", "M5", "M8"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = prepare_output(args.output)
    device = choose_device(args.device)
    transform = image_transform()
    rows = read_split(
        args.split_csv,
        args.data_root,
        split="test",
        verify_hashes=not args.skip_hash_check,
    )

    layer_overrides = dict(parse_model(value) for value in args.target_layer)
    loaded = {}
    all_predictions = {}
    for model_name, checkpoint in args.model:
        if model_name in loaded:
            raise ValueError(f"Duplicate model name: {model_name}")
        model, metadata = load_checkpoint(checkpoint, device=device, weights_dir=args.weights_dir)
        loaded[model_name] = (model, metadata, checkpoint)
        all_predictions[model_name] = predict_all(model, rows, transform, device)

    selected = select_rows(
        rows,
        all_predictions,
        count=args.samples,
        seed=args.seed,
        defect_only=not args.include_non_defect,
    )
    manifest_rows = []
    for model_name, (model, metadata, checkpoint) in loaded.items():
        override = layer_overrides.get(model_name)
        layer = resolve_target_layer(model, str(override)) if override else default_target_layer(
            model, metadata["architecture"]
        )
        with GradCAM(model, layer) as gradcam:
            for row in selected:
                relative = row["relative_path"]
                with Image.open(row["path"]) as source:
                    original = source.convert("RGB")
                    batch = transform(original).unsqueeze(0).to(device)
                prediction = all_predictions[model_name][relative]
                cam_target = target_class(args.target_mode, prediction, row["label"])
                maps, logits, _ = gradcam.generate(batch, cam_target)
                probabilities = logits.softmax(dim=1)[0]
                heatmap = maps[0, 0].cpu().numpy()
                blind_id = blind_identifier(args.seed, model_name, relative)
                blind_path = output / "blind" / "images" / f"{blind_id}.png"
                private_path = output / "private" / "images" / f"{slug(model_name)}_{blind_id}.png"
                render_panel(original, heatmap, blind_path)
                render_panel(
                    original,
                    heatmap,
                    private_path,
                    title=(
                        f"{model_name} | true={CLASSES[row['label']]} | "
                        f"pred={CLASSES[prediction]} | target={CLASSES[cam_target]}"
                    ),
                )
                manifest_rows.append(
                    {
                        "blind_id": blind_id,
                        "model": model_name,
                        "architecture": metadata["architecture"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_format": metadata["format"],
                        "relative_path": relative,
                        "true_label": row["label"],
                        "true_class": CLASSES[row["label"]],
                        "prediction": prediction,
                        "predicted_class": CLASSES[prediction],
                        "correct": prediction == row["label"],
                        "predicted_confidence": f"{float(probabilities[prediction]):.8f}",
                        "target_class": CLASSES[cam_target],
                        "blind_image": blind_path.relative_to(output).as_posix(),
                        "private_image": private_path.relative_to(output).as_posix(),
                    }
                )

    manifest_path = output / "private" / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    unique_raters = list(dict.fromkeys(args.rater))
    for rater in unique_raters:
        review_path = output / "blind" / f"review_{slug(rater)}.csv"
        review_rows = manifest_rows.copy()
        rater_seed = int(
            hashlib.sha256(f"{args.seed}|{rater}".encode("utf-8")).hexdigest()[:16], 16
        )
        random.Random(rater_seed).shuffle(review_rows)
        with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("blind_id", "image_path", "covers_defect", "comments"),
            )
            writer.writeheader()
            for row in review_rows:
                writer.writerow(
                    {
                        "blind_id": row["blind_id"],
                        "image_path": row["blind_image"],
                        "covers_defect": "",
                        "comments": "",
                    }
                )

    summary = {
        "device": str(device),
        "seed": args.seed,
        "target_mode": args.target_mode,
        "defect_only": not args.include_non_defect,
        "requested_samples": args.samples,
        "selected_unique_images": len(selected),
        "blind_items": len(manifest_rows),
        "models": list(loaded),
        "raters": unique_raters,
        "instructions": "Keep private/manifest.csv hidden until every independent review is complete.",
    }
    (output / "private" / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
