"""Generate paired, blinded Grad-CAM panels from fold-specific trained heads."""

import argparse
import csv
import hashlib
import random
import secrets
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import torch
from torch import nn

from grouped.protocol import ARCHS
from m3.models import DefectClassifier as M3, WEIGHTS as M3_WEIGHTS
from m4.models import DefectClassifier as M4, image_transform
from m5.generate_gradcam import render_panel
from m5.gradcam import GradCAM

ARMS = ("expanded", "global_pool")
REVIEWERS = ("A", "B", "C")
SHEET_FIELDS = ("blind_id", "image_path", "covers_defect", "comments")
SCOPES = 4
REVIEW_INSTRUCTIONS = """IDAT7215 — Grad-CAM anonymous review

Open gallery.html and ratings.csv. The panel order is the same in both.
Assess each panel independently. First look for a visible printing defect
in the original image, then look at the heatmap and overlay.

yes: The strongest highlighted area clearly overlaps a visible defect.
no: The strongest area is on the background or an unrelated feature, or
    there is no distinct hotspot despite a clear visible defect.
unsure: A visible defect or the overlap cannot be judged confidently.

Enter yes, no or unsure in covers_defect. Comments are optional. Preserve
blind_id and the CSV format (UTF-8). Save a copy while working. Return only
the finished ratings.csv to the coordinator, using the assigned reviewer letter
in the filename (for example ratings_A.csv). Review without discussing answers
or consulting the project's labelled examples. Look at the full original
panel when the highlight is ambiguous.

打開 gallery.html 和 ratings.csv，兩個檔案的案例順序相同。先看原圖，確認
可見缺陷的位置，再看熱圖與疊圖。主要高亮區清楚覆蓋缺陷填 yes；落在背景、
無關部位，或缺陷清楚但沒有明顯高亮，填 no；看不清缺陷或無法確定重疊則填
unsure。請獨立評分，保留 ID，評完只交 UTF-8 CSV。可分批完成並儲存。

Original images are from Justin Ruan, 3D-Printer Defected Dataset v1
(https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset).
Resized images, heatmaps and overlays are distributed under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).
"""


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_cases(rows, per_fold, seed):
    """Draw one case per consecutive filename band, before inspecting predictions."""
    rng = random.Random(seed)
    selected = {}
    for fold in range(SCOPES):
        candidates = sorted(
            (
                i
                for i, row in enumerate(rows)
                if row["source"] == "kaggle"
                and row["label"] == "1"
                and row[f"fold{fold}"] == "test"
            ),
            key=lambda i: int(Path(rows[i]["path"]).stem.rsplit("_", 1)[1]),
        )
        if len(candidates) < per_fold:
            raise ValueError(f"Fold {fold} has only {len(candidates)} defect images")
        selected[fold] = [
            rng.choice(list(band)) for band in np.array_split(candidates, per_fold)
        ]
    return selected


def package_reviewer(root, reviewer, blinded):
    ordered = blinded.copy()
    # Separate independent permutations. The seed is never distributed.
    secrets.SystemRandom().shuffle(ordered)
    dest = root / f"reviewer_{reviewer}.zip"
    sheet = root / "_ratings.csv"
    write_csv(sheet, ordered, SHEET_FIELDS)
    gallery = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Anonymous Grad-CAM review</title>",
        "<style>body{font:1rem system-ui;max-width:980px;margin:auto;padding:1rem}"
        "article{border-top:1px solid #999;padding-top:1rem;margin:1.5rem 0}"
        "img{display:block;width:100%;height:auto}nav{display:flex;gap:.6rem;flex-wrap:wrap}"
        "</style><h1>Anonymous Grad-CAM review</h1>",
        "<p>Open ratings.csv alongside this gallery. Read instructions.txt for the scoring rules.</p>",
        "<nav>"
        + " ".join(f'<a href="#panel-{i}">{i}</a>' for i in range(1, len(ordered) + 1))
        + "</nav>",
    ]
    with ZipFile(dest, "w", compression=ZIP_DEFLATED) as archive:
        for index, row in enumerate(ordered, 1):
            image = row["image_path"]
            label = row["blind_id"]
            gallery.append(
                f'<article id="panel-{index}"><h2>{index}. {label}</h2><a href="{image}">'
                f'<img loading="lazy" src="{image}" alt="Original image, heatmap and overlay for {label}">'
                "</a></article>"
            )
            archive.write(root / image, image)
        archive.writestr("gallery.html", "\n".join(gallery) + "</html>")
        archive.writestr("instructions.txt", REVIEW_INSTRUCTIONS)
        archive.write(sheet, "ratings.csv")
    sheet.unlink()


def generate(results, cache, weights, output, per_fold, seed, device):
    output.mkdir(parents=True, exist_ok=False)
    image_dir = output / "images"
    image_dir.mkdir()
    torch.set_num_threads(4)
    with (results / "split.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    pixels = np.load(cache / "images.npy", mmap_mode="r")
    if pixels.shape != (len(rows), 224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError("Image cache must match the 224 × 224 manifest")
    samples = sample_cases(rows, per_fold, seed)
    transform = image_transform(resize=False)
    manifest = []
    used_ids = set()
    for arch in ARCHS:
        constructor = M3 if arch in M3_WEIGHTS else M4
        model = (
            constructor(arch, pretrained=True, weights_dir=weights).eval().to(device)
        )
        layer = model.backbone[28] if arch == "vgg16" else model.backbone[-1]
        original_pool = model.pool
        for arm in ARMS:
            model.pool = (
                original_pool
                if arm == "expanded"
                else nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            )
            for fold, indices in samples.items():
                path = results / arch / f"{arm}_fold{fold}_head.pt"
                model.head.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                model.zero_grad(set_to_none=True)
                with GradCAM(model, layer) as cam:
                    for i in indices:
                        original = Image.fromarray(np.array(pixels[i]))
                        x = transform(original).unsqueeze(0).to(device)
                        heat, logits, _ = cam.generate(x, 1)
                        blind_id = secrets.token_hex(8)
                        while blind_id in used_ids:
                            blind_id = secrets.token_hex(8)
                        used_ids.add(blind_id)
                        path = f"images/{blind_id}.png"
                        render_panel(original, heat[0, 0].cpu().numpy(), output / path)
                        manifest.append(
                            dict(
                                blind_id=blind_id,
                                image_path=path,
                                model=arch,
                                arm=arm,
                                fold=fold,
                                source_path=rows[i]["path"],
                                label=rows[i]["label"],
                                seed=42,
                                defect_probability=float(logits.softmax(1)[0, 1]),
                                panel_sha256=hashlib.sha256(
                                    (output / path).read_bytes()
                                ).hexdigest(),
                            )
                        )
        print(f"{arch}: panels generated", flush=True)
    assert all(
        n == per_fold
        for n in Counter((r["model"], r["arm"], r["fold"]) for r in manifest).values()
    )
    key_dir = output / "coordinator_only"
    key_dir.mkdir(mode=0o700)
    write_csv(
        key_dir / "blind_key.csv",
        sorted(manifest, key=lambda r: r["blind_id"]),
        tuple(manifest[0]),
    )
    sheet_rows = [
        dict(
            blind_id=r["blind_id"],
            image_path=r["image_path"],
            covers_defect="",
            comments="",
        )
        for r in manifest
    ]
    for reviewer in REVIEWERS:
        package_reviewer(output, reviewer, sheet_rows)
    print(f"{len(manifest)} panels per reviewer · key in coordinator_only", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "cache", "weights", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--per-fold", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=26)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.per_fold < 1:
        parser.error("--per-fold must be positive")
    generate(
        args.results,
        args.cache,
        args.weights,
        args.output,
        args.per_fold,
        args.sample_seed,
        args.device,
    )
