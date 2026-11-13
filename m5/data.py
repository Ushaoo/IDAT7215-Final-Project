import csv
import hashlib
from pathlib import Path


REQUIRED_COLUMNS = {"split", "path", "label", "class_name", "sha256"}
CLASSES = ("Non-defect", "Defect")


def read_split(split_csv, data_root, split="test", verify_hashes=True):
    data_root = Path(data_root).resolve()
    rows = []
    with Path(split_csv).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Split CSV must contain {sorted(REQUIRED_COLUMNS)}")
        for row in reader:
            if row["split"] != split:
                continue
            label = int(row["label"])
            if label not in (0, 1) or row["class_name"] != CLASSES[label]:
                raise ValueError(f"Invalid label mapping: {row}")
            image_path = (data_root / row["path"]).resolve()
            if not image_path.is_relative_to(data_root):
                raise ValueError(f"Image path escapes data root: {row['path']}")
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if verify_hashes:
                with image_path.open("rb") as image_stream:
                    digest = hashlib.file_digest(image_stream, "sha256").hexdigest()
                if digest != row["sha256"].lower():
                    raise ValueError(f"SHA-256 mismatch: {row['path']}")
            rows.append(
                {
                    "path": image_path,
                    "relative_path": row["path"],
                    "label": label,
                    "class_name": row["class_name"],
                }
            )
    if not rows:
        raise ValueError(f"No rows found for split={split}")
    return rows


def select_rows(rows, predictions, count, seed, defect_only=True):
    """Prefer misclassified cases, then deterministically fill a balanced sample."""
    import random

    eligible = [row for row in rows if not defect_only or row["label"] == 1]
    if count <= 0:
        raise ValueError("count must be positive")
    by_key = {}
    for row in eligible:
        relative = row["relative_path"]
        any_error = any(model_predictions[relative] != row["label"] for model_predictions in predictions.values())
        by_key.setdefault((any_error, row["label"]), []).append(row)

    rng = random.Random(seed)
    for bucket in by_key.values():
        rng.shuffle(bucket)

    chosen = []
    labels = (1,) if defect_only else (1, 0)
    for error_status in (True, False):
        while len(chosen) < count:
            added = False
            for label in labels:
                bucket = by_key.get((error_status, label), [])
                if bucket:
                    chosen.append(bucket.pop())
                    added = True
                    if len(chosen) == count:
                        break
            if not added:
                break
    return chosen
