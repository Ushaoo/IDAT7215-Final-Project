"""Score Grad-CAM overlap with regions marked by one Codex visual reviewer."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from grouped.blind_review.aggregate import read_csv, write_csv
from grouped.blind_review.generate import ARMS, ARCHS
from m3.models import DefectClassifier as M3, WEIGHTS as M3_WEIGHTS
from m4.models import DefectClassifier as M4, image_transform
from m5.gradcam import GradCAM

THRESHOLDS = (0.40, 0.50, 0.60)
HOT_CUTOFF = 0.80


def make_mask(boxes):
    mask = np.zeros((224, 224), dtype=bool)
    for box in boxes.split("|") if boxes else []:
        coordinates = [
            round(224 * float(component) / 100) for component in box.split(":")
        ]
        left, top, right, bottom = coordinates
        if not (0 <= left < right <= 224 and 0 <= top < bottom <= 224):
            raise ValueError(f"Invalid region: {box}")
        mask[top:bottom, left:right] = True
    return mask


def score(results, cache, weights, kit_key, case_key, annotations, output):
    torch.set_num_threads(4)
    manifest = read_csv(results / "split.csv")
    kaggle = {r["path"]: i for i, r in enumerate(manifest) if r["source"] == "kaggle"}
    indices = {r["case_id"]: r["source_path"] for r in read_csv(case_key)}
    regions = {
        indices[r["case_id"]]: make_mask(r["boxes"]) for r in read_csv(annotations)
    }
    panels = read_csv(kit_key)
    assert regions.keys() == {r["source_path"] for r in panels}
    pixels = np.load(cache / "images.npy", mmap_mode="r")
    assert len(manifest) == len(pixels)
    transform = image_transform(resize=False)
    ratings = []
    for arch in ARCHS:
        constructor = M3 if arch in M3_WEIGHTS else M4
        model = constructor(arch, pretrained=True, weights_dir=weights).eval()
        original_pool = model.pool
        layer = model.backbone[28] if arch == "vgg16" else model.backbone[-1]
        for arm in ARMS:
            model.pool = (
                original_pool
                if arm == "expanded"
                else nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            )
            for fold in range(4):
                checkpoint = results / arch / f"{arm}_fold{fold}_head.pt"
                model.head.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                )
                subset = (
                    r
                    for r in panels
                    if (r["model"], r["arm"], int(r["fold"])) == (arch, arm, fold)
                )
                with GradCAM(model, layer) as cam:
                    for row in subset:
                        source = row["source_path"]
                        idx = kaggle[source]
                        assert manifest[idx][f"fold{fold}"] == "test"
                        assert manifest[idx]["label"] == "1"
                        x = transform(Image.fromarray(pixels[idx])).unsqueeze(0)
                        heat, logits, _ = cam.generate(x, 1)
                        probability = float(logits.softmax(1)[0, 1])
                        if abs(probability - float(row["defect_probability"])) > 0.0001:
                            raise ValueError(
                                f"Panel replay mismatch for {arch} fold {fold}"
                            )
                        hot = heat[0, 0].numpy() >= HOT_CUTOFF
                        mask = regions[source]
                        overlap = (
                            int(np.count_nonzero(hot & mask)) / int(hot.sum())
                            if mask.any() and hot.any()
                            else (0.0 if mask.any() else None)
                        )
                        ratings.append(
                            dict(
                                blind_id=row["blind_id"],
                                model=arch,
                                arm=arm,
                                fold=fold,
                                source_path=source,
                                hotspot_overlap=overlap,
                                covers_defect=(
                                    "unsure"
                                    if overlap is None
                                    else ("yes" if overlap >= 0.50 else "no")
                                ),
                            )
                        )
        print(arch, "scored", flush=True)
    assert len(ratings) == len(panels) == 160
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "private_panel_ratings.csv", ratings)
    summarize(output, ratings)


def summarize(output, ratings):
    groups = defaultdict(dict)
    for row in ratings:
        key = row["model"]
        source, arm = row["source_path"], row["arm"]
        groups[key].setdefault(source, {})[arm] = row["hotspot_overlap"]
    summary = []
    for model, cases in sorted(groups.items()):
        assert len(cases) == 20 and all(
            set(pair) == set(ARMS) for pair in cases.values()
        )
        good = [
            list(pair[arm] for arm in ARMS)
            for pair in cases.values()
            if all(v is not None for v in pair.values())
        ]
        for threshold in THRESHOLDS:
            pairs = [(left >= threshold, right >= threshold) for left, right in good]
            summary.append(
                dict(
                    reviewer="Codex (single AI reviewer)",
                    model=model,
                    overlap_threshold=threshold,
                    hotspot_threshold=HOT_CUTOFF,
                    total_cases=len(cases),
                    assessable=len(good),
                    unsure=len(cases) - len(good),
                    control_yes=sum(left for left, _ in pairs),
                    global_yes=sum(right for _, right in pairs),
                    both_yes=pairs.count((True, True)),
                    control_only=pairs.count((True, False)),
                    global_only=pairs.count((False, True)),
                    both_no=pairs.count((False, False)),
                )
            )
    write_csv(output / "codex_localization_summary.csv", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "results",
        "cache",
        "weights",
        "kit_key",
        "case_key",
        "annotations",
        "output",
    ):
        parser.add_argument(name, type=Path)
    score(**vars(parser.parse_args()))
