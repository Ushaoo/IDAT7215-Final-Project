"""Fetch only the public AMdataset images listed by the group's split."""
import argparse
import csv
import hashlib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/mahsan2/AMdataset/52fe75dec70c1e8046493b4e3256f8f752a4d39d/dataset/"


def fetch(row, root):
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Out-of-root path: {row['path']}")
    expected = row["sha256"].lower()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return
        raise ValueError(f"Existing file has the wrong SHA-256: {path}")
    with urllib.request.urlopen(SOURCE + urllib.parse.quote(row["path"]), timeout=45) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Downloaded file has the wrong SHA-256: {row['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    with args.split.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    root = args.data_root.resolve()
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("The split contains duplicate paths")
    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, _ in enumerate(pool.map(lambda row: fetch(row, root), rows), 1):
            if index % 50 == 0 or index == len(rows):
                print(f"Verified {index}/{len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
