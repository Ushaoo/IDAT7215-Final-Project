import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from m4.models import CLASSES, image_transform, resize_image


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_split(csv_path, root):
    root = root.resolve()
    by_split = {name: [] for name in ("train", "val", "test")}
    seen_paths = set()
    digest_splits = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "path", "label", "class_name", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected CSV columns: {sorted(required)}")
        for row in reader:
            relative = row["path"]
            split = row["split"]
            label = int(row["label"])
            if split not in by_split or label not in (0, 1) or CLASSES[label] != row["class_name"]:
                raise ValueError(f"Invalid split or class mapping: {row}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path in seen_paths:
                raise ValueError(f"Duplicate or out-of-root image path: {relative}")
            seen_paths.add(path)
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing. Set --data-root to AMdataset/dataset.")
            digest = sha256(path)
            if digest != row["sha256"].lower():
                raise ValueError(f"SHA-256 mismatch: {relative}")
            if digest_splits.setdefault(digest, split) != split:
                raise ValueError(f"Same image bytes appear across splits: {relative}")
            by_split[split].append((path, label, relative))
    for name, rows in by_split.items():
        if set(Counter(row[1] for row in rows)) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
    return by_split


class SplitImages(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.images = [None] * len(rows)
        self.transform = image_transform(training, resize=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label, relative = self.rows[index]
        image = self.images[index]
        if image is None:
            with Image.open(path) as source:
                image = resize_image(source.convert("RGB"))
            self.images[index] = image
        return self.transform(image), label, relative
