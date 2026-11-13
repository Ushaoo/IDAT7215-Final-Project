"""Adapted from M4: resumable, SHA256-verified download of the fixed manifest."""
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
        raise ValueError("Image path outside data root")
    expected = row["sha256"].lower()
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            url = SOURCE + urllib.parse.quote(row["path"], safe="/")
            request = urllib.request.Request(url, headers={"User-Agent": "IDAT7215-M3"})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, context=context, timeout=90) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Image SHA256 mismatch: {row['path']}")
            os.replace(temporary, path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    with args.split.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths in manifest")
    root = args.data_root.resolve()
    print(f"Downloading/checking {len(rows)} images (about 849 MB total).", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(lambda row: fetch(row, root), rows), 1):
            if index % 10 == 0 or index == len(rows):
                print(f"Verified {index}/{len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
