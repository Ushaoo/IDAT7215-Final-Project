"""Download and verify only the fixed-split images needed by M5."""

import argparse
import csv
import hashlib
import os
import ssl
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi


SOURCE = "https://raw.githubusercontent.com/mahsan2/AMdataset/52fe75dec70c1e8046493b4e3256f8f752a4d39d/dataset/"


def fetch(row, root):
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes data root: {row['path']}")
    expected = row["sha256"].lower()
    if path.is_file():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                return
        raise ValueError(f"Existing image has the wrong SHA-256: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            url = SOURCE + urllib.parse.quote(row["path"], safe="/")
            request = urllib.request.Request(url, headers={"User-Agent": "IDAT7215-M5"})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, context=context, timeout=90) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded image has the wrong SHA-256: {row['path']}")
            os.replace(temporary, path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    with args.split_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if args.subset != "all":
        rows = [row for row in rows if row["split"] == args.subset]
    if not rows:
        raise ValueError(f"No rows found for subset={args.subset}")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("The selected split contains duplicate paths")

    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/checking {len(rows)} {args.subset} images", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(lambda row: fetch(row, root), rows), start=1):
            if index % 10 == 0 or index == len(rows):
                print(f"Verified {index}/{len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
