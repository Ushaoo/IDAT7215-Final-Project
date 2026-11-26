"""Stress-test seed 42 on held-out groups and generate out-of-fold Grad-CAM panels."""

import argparse
import csv
import gzip
import re
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from m2.am_m2.perturbations import make_perturbation
from m3.models import DefectClassifier as M3, WEIGHTS as M3_WEIGHTS
from m4.models import DefectClassifier as M4, image_transform
from m5.gradcam import GradCAM
from grouped.protocol import ARCHS, ARMS
from grouped.train import metrics

CONDITIONS = [
    ("clean", 0),
    ("gaussian_noise", 0.2),
    ("motion_blur", 13),
    ("brightness", 0.6),
    ("brightness", 1.4),
]


def save_rows(path, rows):
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def main(weights, cache, out, device):
    torch.set_num_threads(4)
    rgb = np.load(cache / "images.npy", mmap_mode="r")
    rows = list(csv.DictReader((out / "split.csv").open()))
    scores = []
    latency = []
    transform = image_transform(resize=False)
    with gzip.open(out / "stress_predictions.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "fold", "arm", "condition", "level", "path", "label", "p_defect"]
        )
        for name in ARCHS:
            constructor = M3 if name in M3_WEIGHTS else M4
            model = (
                constructor(name, pretrained=True, weights_dir=weights)
                .eval()
                .to(device)
            )
            fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
            for fold in range(4):
                ids = [i for i, r in enumerate(rows) if r[f"fold{fold}"] == "test"]
                labels = torch.tensor([int(rows[i]["label"]) for i in ids])
                heads = {}
                for arm in ARMS:
                    import copy

                    head = copy.deepcopy(model.head)
                    state = torch.load(
                        out / name / f"{arm}_fold{fold}_head.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    head.load_state_dict(state)
                    heads[arm] = head.eval()
                for condition, level in CONDITIONS:
                    perturb = make_perturbation(condition, level)
                    prob = {a: [] for a in ARMS}
                    with torch.inference_mode():
                        for start in range(0, len(ids), 16):
                            batch = []
                            for i in ids[start : start + 16]:
                                image = Image.fromarray(rgb[i])
                                batch.append(
                                    transform(perturb(image) if perturb else image)
                                )
                            fmap = model.backbone(torch.stack(batch).to(device))
                            legacy = fmap[:, :, :4, :4].mean(dim=(-2, -1))
                            glob = fmap.mean(dim=(-2, -1))
                            for arm in ARMS:
                                x = (
                                    glob
                                    if arm in ("global_pool", "balanced_global")
                                    else legacy
                                )
                                prob[arm].append(heads[arm](x).softmax(1)[:, 1].cpu())
                    for arm, parts in prob.items():
                        ps = torch.cat(parts)
                        scores.append(
                            dict(
                                model=name,
                                fold=fold,
                                arm=arm,
                                condition=condition,
                                level=level,
                                **metrics(labels, (ps > 0.5).long()),
                            )
                        )
                        writer.writerows(
                            (
                                name,
                                fold,
                                arm,
                                condition,
                                level,
                                rows[i]["path"],
                                rows[i]["label"],
                                p,
                            )
                            for i, p in zip(ids, ps.tolist())
                        )
                model.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
                model.head = heads["balanced_global"]
                layer = model.backbone[28] if name == "vgg16" else model.backbone[-1]
                with GradCAM(model, layer) as cam:
                    for label in (0, 1):
                        cases = sorted(
                            (i for i in ids if rows[i]["label"] == str(label)),
                            key=lambda i: int(
                                re.search(r"_(\d+)\.jpg$", rows[i]["path"])[1]
                            ),
                        )
                        i = cases[len(cases) // 2]
                        image = Image.fromarray(rgb[i])
                        heat, logits, _ = cam.generate(
                            transform(image)[None].to(device), 1
                        )
                        ax = axes[label, fold]
                        ax.imshow(image)
                        ax.imshow(
                            heat[0, 0].cpu(), alpha=0.42, cmap="inferno", vmin=0, vmax=1
                        )
                        ax.set_title(
                            f"{Path(rows[i]['path']).stem}\ny={label}, p(defect)={logits.softmax(1)[0, 1].item():.2f}",
                            fontsize=8,
                        )
                        ax.axis("off")
            fig.suptitle(f"{name}: held-out groups, defect-class Grad-CAM (seed 42)")
            fig.tight_layout()
            fig.savefig(out / name / "gradcam.png", dpi=125)
            plt.close(fig)
            model.cpu().eval()
            x = transform(Image.fromarray(rgb[0]))[None]
            elapsed = []
            with torch.inference_mode():
                for it in range(70):
                    begin = perf_counter()
                    model(x)
                    if it >= 10:
                        elapsed.append(1000 * (perf_counter() - begin))
            latency.append(
                dict(
                    model=name,
                    parameters=sum(p.numel() for p in model.parameters()),
                    cpu_threads=4,
                    mean_ms=np.mean(elapsed),
                    median_ms=np.median(elapsed),
                    p95_ms=np.percentile(elapsed, 95),
                )
            )
            print("diagnostics:", name, flush=True)
    save_rows(out / "stress_metrics.csv", scores)
    save_rows(out / "efficiency.csv", latency)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "cache", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    main(a.weights, a.cache, a.output, a.device)
