from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from am_m2.evaluation import evaluate_model
from am_m2.perturbations import make_perturbation


CONDITIONS = [
    ("baseline", 0.0),
    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.10),
    ("gaussian_noise", 0.20),
    ("motion_blur", 5.0),
    ("motion_blur", 9.0),
    ("motion_blur", 13.0),
    ("brightness", 0.60),
    ("brightness", 0.80),
    ("brightness", 1.20),
    ("brightness", 1.40),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M4 ResNet50 and MobileNetV3-Small robustness.")
    parser.add_argument("--m4-repo", required=True, type=Path, help="Repository root containing m4/models.py")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m4_robustness"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.m4_repo.resolve()))
    from m4.models import load_checkpoint

    specs = [
        ("ResNet50", "resnet50", args.m4_repo / "m4/results/resnet50/best_head.pt"),
        ("MobileNetV3-Small", "mobilenet_v3_small", args.m4_repo / "m4/results/mobilenet_v3_small/best_head.pt"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_predictions = []
    for model_name, architecture, checkpoint in specs:
        model, metadata = load_checkpoint(checkpoint, device="cpu")
        baseline_f1 = None
        for kind, value in CONDITIONS:
            metrics, predictions = evaluate_model(
                model=model,
                data_root=args.data_root,
                split_csv=args.split_csv,
                split="test",
                perturbation=make_perturbation(kind, value, seed=args.seed),
                batch_size=args.batch_size,
                workers=args.workers,
                device=args.device,
                image_size=224,
            )
            if kind == "baseline":
                baseline_f1 = float(metrics["f1_macro"])
            row = {
                "model": model_name,
                "architecture": architecture,
                "perturbation": kind,
                "value": value,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_defect": metrics["f1_defect"],
                "mcc": metrics["mcc"],
                "f1_drop": baseline_f1 - float(metrics["f1_macro"]),
                "f1_drop_percent": 0.0 if baseline_f1 == 0 else (baseline_f1 - float(metrics["f1_macro"])) / baseline_f1 * 100,
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "parameter_count": metrics["parameter_count"],
                "inference_ms_per_image": metrics["inference_ms_per_image"],
                "n": metrics["n"],
                "checkpoint": checkpoint.relative_to(args.m4_repo).as_posix(),
                "checkpoint_format": metadata["format"],
            }
            rows.append(row)
            for prediction in predictions:
                all_predictions.append(
                    {"model": model_name, "perturbation": kind, "value": value, **prediction}
                )
            print(f"{model_name}: {kind}={value:g}, F1={metrics['f1_macro']:.4f}")

    results_path = args.output_dir / "robustness_results.csv"
    with results_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_predictions[0].keys())
        writer.writeheader()
        writer.writerows(all_predictions)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "corruption_order": "resize to 224x224, corrupt RGB image, then ImageNet normalize",
                "noise_seed": args.seed,
                "split": "test",
                "test_n": 40,
                "conditions": CONDITIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
