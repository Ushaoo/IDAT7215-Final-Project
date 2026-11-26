from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def canonical_class(folder_name: str) -> tuple[str, int] | None:
    name = _normalise_name(folder_name)
    non_defect_aliases = {"nondefect", "nondefective", "nodefect", "normal", "good", "ok", "negative"}
    defect_aliases = {"defect", "defective", "fault", "faulty", "bad", "positive"}
    if name in non_defect_aliases or name.startswith(("nondefect", "good", "normal")):
        return "Non-defect", 0
    if name in defect_aliases or name.startswith(("defect", "bad", "fault")):
        return "Defect", 1
    return None


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def collect_samples(data_root: Path) -> list[dict]:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    samples: list[dict] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_info = None
        for parent in [path.parent, *path.parents]:
            if parent == data_root.parent:
                break
            label_info = canonical_class(parent.name)
            if label_info:
                break
        if label_info is None:
            continue
        class_name, label = label_info
        samples.append(
            {
                "path": path.relative_to(data_root).as_posix(),
                "class_name": class_name,
                "label": label,
                "sha256": sha256_file(path),
            }
        )
    if not samples:
        raise ValueError(
            "No labelled images found. Expected class folders named Defect/Non-defect "
            "(common aliases such as defective, normal and good are also accepted)."
        )
    if {s["label"] for s in samples} != {0, 1}:
        raise ValueError("Both Defect and Non-defect classes are required")
    return samples


def _target_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(x) for x in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def assign_splits(
    samples: list[dict], ratios: tuple[float, float, float], seed: int
) -> list[dict]:
    if any(r <= 0 for r in ratios):
        raise ValueError("all split ratios must be positive")
    ratio_sum = sum(ratios)
    ratios = tuple(r / ratio_sum for r in ratios)
    rng = random.Random(seed)
    split_names = ["train", "val", "test"]
    result: list[dict] = []

    by_label: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        by_label[int(sample["label"])].append(sample)

    for label, labelled_samples in sorted(by_label.items()):
        groups: dict[str, list[dict]] = defaultdict(list)
        for sample in labelled_samples:
            groups[sample["sha256"]].append(sample)
        group_list = list(groups.values())
        rng.shuffle(group_list)
        group_list.sort(key=len, reverse=True)
        targets = _target_counts(len(labelled_samples), ratios)
        assigned_counts = [0, 0, 0]
        for group in group_list:
            deficits = [targets[i] - assigned_counts[i] for i in range(3)]
            split_index = max(range(3), key=lambda i: (deficits[i], -assigned_counts[i], -i))
            for sample in group:
                result.append({**sample, "split": split_names[split_index]})
            assigned_counts[split_index] += len(group)

    return sorted(result, key=lambda row: (row["split"], row["label"], row["path"]))


def write_split_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "path", "label", "class_name", "sha256"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def build_data_card(rows: list[dict], data_root: Path, seed: int, ratios: tuple[float, float, float]) -> dict:
    counts = Counter((row["split"], row["class_name"]) for row in rows)
    hashes = Counter(row["sha256"] for row in rows)
    widths, heights = [], []
    unreadable = []
    for row in rows:
        try:
            with Image.open(data_root / row["path"]) as image:
                widths.append(image.width)
                heights.append(image.height)
        except Exception as exc:  # surfaced in card rather than silently dropped
            unreadable.append({"path": row["path"], "error": str(exc)})
    split_table = {
        split: {
            class_name: counts[(split, class_name)]
            for class_name in ("Defect", "Non-defect")
        }
        for split in ("train", "val", "test")
    }
    for split in split_table:
        split_table[split]["total"] = sum(split_table[split].values())
    return {
        "task": "binary classification: Defect (1) vs Non-defect (0)",
        "data_root": str(data_root.resolve()),
        "total_images": len(rows),
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
        "split_counts": split_table,
        "split_seed": seed,
        "requested_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "duplicate_files": sum(count - 1 for count in hashes.values() if count > 1),
        "duplicate_groups": sum(count > 1 for count in hashes.values()),
        "duplicate_policy": "identical SHA-256 files are kept in the same split",
        "image_width_range": [min(widths), max(widths)] if widths else None,
        "image_height_range": [min(heights), max(heights)] if heights else None,
        "unreadable_images": unreadable,
        "notes": [
            "The public repository contains only a portion of the paper dataset.",
            "Compare models on this identical split; do not claim exact replication of paper numbers.",
            "Augmented images must not cross split boundaries with their originals.",
        ],
    }


def write_data_card(card: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    table = card["split_counts"]
    lines = [
        "# AM 缺陷二分类数据卡片",
        "",
        f"- 任务：{card['task']}",
        f"- 实际扫描图片：{card['total_images']} 张",
        f"- 固定随机种子：{card['split_seed']}",
        f"- 重复文件：{card['duplicate_files']} 个（同 SHA-256 文件强制留在同一划分）",
        f"- 尺寸范围：宽 {card['image_width_range']}，高 {card['image_height_range']}",
        "",
        "| 划分 | Defect | Non-defect | 合计 |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        row = table[split]
        lines.append(f"| {split} | {row['Defect']} | {row['Non-defect']} | {row['total']} |")
    lines += [
        "",
        "## 使用边界",
        "",
        "公开仓库只含论文数据的一部分。本项目只比较相同公开子集、相同划分上的模型行为，"
        "不宣称与论文绝对数值完全一致。标签映射固定为 Defect=1、Non-defect=0。",
    ]
    if card["unreadable_images"]:
        lines += ["", f"> 警告：发现 {len(card['unreadable_images'])} 张无法读取的图片，详见 JSON。"]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class SplitImageDataset(Dataset):
    data_root: Path
    split_csv: Path
    split: str = "test"
    transform: Callable | None = None
    perturbation: Callable | None = None

    def __post_init__(self):
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [row for row in rows if row["split"] == self.split]
        if not self.rows:
            raise ValueError(f"no rows for split '{self.split}' in {self.split_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.data_root / row["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.perturbation is not None:
                image = self.perturbation(image)
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["label"]), row["path"]

