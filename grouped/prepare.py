"""Verify the two archives and prepare a deterministic RGB resize cache."""

import argparse
import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

ARCHIVES = {
    "kaggle": "kaggle_source.zip",
    "smis": "smis_source.zip",
}
PAIR = [
    ("scratch", "bed_not_stick"),
    ("scratch_2", "leg_broken"),
    ("scratch_3", "no_bottom"),
    ("scratch_4", "no_support"),
]
LABELS = {
    "goodcylinder": 0,
    "badcylinder": 1,
    "Goodaug": 0,
    "Badaug": 1,
    "defected": 1,
    "no_defected": 0,
}


def split_role(row, fold):
    if row["source"] == "smis":
        return "train"
    if row["group"] in PAIR[fold]:
        return "test"
    if row["group"] in PAIR[(fold + 1) % 4]:
        return "val"
    return "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = {
        r["path"]: r["sha256"] for r in csv.DictReader(Path("data/split.csv").open())
    }
    rows = []
    pixels = np.lib.format.open_memmap(
        args.cache / "images.npy", mode="w+", dtype="uint8", shape=(4845, 224, 224, 3)
    )
    for source, filename in ARCHIVES.items():
        with ZipFile(args.data / filename) as z:
            for name in sorted(z.namelist()):
                if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                path = name.split("/", 1)[1] if source == "smis" else name
                parent = Path(path).parent.name
                label = LABELS[parent]
                data = z.read(name)
                digest = hashlib.sha256(data).hexdigest()
                if path.startswith("dataset/"):
                    assert digest == raw[path.removeprefix("dataset/")]
                group = (
                    parent
                    if source == "smis"
                    else re.sub(r"_\d+$", "", Path(path).stem)
                )
                if source == "kaggle":
                    assert group in {pair[label] for pair in PAIR}
                item = dict(
                    source=source,
                    path=path,
                    group=group,
                    label=label,
                    augmented=int(path.startswith("Augmented_data/")),
                    sha256=digest,
                )
                for fold in range(4):
                    item[f"fold{fold}"] = split_role(item, fold)
                with Image.open(io.BytesIO(data)) as im:
                    pixels[len(rows)] = np.asarray(
                        im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    )
                rows.append(item)
    assert len(rows) == len(pixels)
    assert Counter((r["source"], r["augmented"]) for r in rows) == {
        ("smis", 0): 271,
        ("smis", 1): 3017,
        ("kaggle", 0): 1557,
    }
    for fold in range(4):
        sets = [
            {r["sha256"] for r in rows if r[f"fold{fold}"] == role}
            for role in ("train", "val", "test")
        ]
        assert all(
            not sets[i].intersection(sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))
        )
    pixels.flush()
    with (args.output / "split.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print("Verified 4845 files and four disjoint folds.", flush=True)


if __name__ == "__main__":
    main()
