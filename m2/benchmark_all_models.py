from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

from am_m2.data import SplitImageDataset
from am_m2.evaluation import imagenet_transform
from am_m2.models import parameter_count


def parse_args():
    parser = argparse.ArgumentParser(description="Fair batch-1 CPU latency benchmark for all four group models.")
    parser.add_argument("--m3-repo", required=True, type=Path)
    parser.add_argument("--m4-repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_robustness/efficiency_cpu.csv"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--input-count", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.m3_repo.resolve()))
    from m3.models import load_checkpoint as load_m3

    sys.path.insert(0, str(args.m4_repo.resolve()))
    from m4.models import load_checkpoint as load_m4

    specs = [
        ("VGG16", load_m3, args.m3_repo / "m3/results/vgg16/best_head.pt", 59_000_977),
        ("MobileNetV2", load_m3, args.m3_repo / "m3/results/mobilenet_v2/best_head.pt", 9_457_483),
        ("ResNet50", load_m4, args.m4_repo / "m4/results/resnet50/best_head.pt", 94_868_555),
        ("MobileNetV3-Small", load_m4, args.m4_repo / "m4/results/mobilenet_v3_small/best_head.pt", 3_978_891),
    ]
    dataset = SplitImageDataset(
        data_root=args.data_root,
        split_csv=args.split_csv,
        split="test",
        transform=imagenet_transform(224),
    )
    tensors = [dataset[index][0].unsqueeze(0) for index in range(min(args.input_count, len(dataset)))]
    rows = []
    for model_name, loader, checkpoint, full_checkpoint_bytes in specs:
        model, _ = loader(checkpoint, device="cpu")
        with torch.inference_mode():
            for index in range(args.warmup):
                model(tensors[index % len(tensors)])
            times = []
            for index in range(args.repeats):
                started = time.perf_counter()
                model(tensors[index % len(tensors)])
                times.append((time.perf_counter() - started) * 1000)
        total, trainable = parameter_count(model)
        ordered = sorted(times)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        rows.append(
            {
                "model": model_name,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "full_checkpoint_mb_decimal": full_checkpoint_bytes / 1_000_000,
                "cpu_mean_ms": statistics.mean(times),
                "cpu_median_ms": statistics.median(times),
                "cpu_p95_ms": p95,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "batch_size": 1,
                "precision": "float32",
            }
        )
        print(f"{model_name}: {statistics.mean(times):.2f} ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
