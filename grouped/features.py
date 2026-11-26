"""Cache original pooling and global pooling from exactly the same ImageNet feature map."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch

from m3.models import DefectClassifier as M3, WEIGHTS as M3_WEIGHTS
from m4.models import DefectClassifier as M4, WEIGHTS as M4_WEIGHTS

from grouped.protocol import ARCHS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(5)
    images = np.load(args.cache / "images.npy", mmap_mode="r")
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    for name in ARCHS:
        constructor, weights = (
            (M3, M3_WEIGHTS) if name in M3_WEIGHTS else (M4, M4_WEIGHTS)
        )
        weightfile = args.weights / Path(weights[name].url).name
        with weightfile.open("rb") as src:
            digest = hashlib.file_digest(src, "sha256").hexdigest()
        assert digest.startswith(weightfile.stem.split("-")[-1])
        model = (
            constructor(name, pretrained=True, weights_dir=args.weights)
            .to(args.device)
            .eval()
        )
        results = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                data = (
                    torch.from_numpy(images[start : start + 16].copy())
                    .permute(0, 3, 1, 2)
                    .to(args.device, torch.float32)
                )
                fmap = model.backbone((data / 255 - mean) / std)
                assert fmap.shape[-2:] == (7, 7)
                results.append(
                    torch.stack(
                        (model.pool(fmap), fmap.mean(dim=(-2, -1))), dim=1
                    ).cpu()
                )
        path = args.cache / (name + ".pt")
        temp = path.with_suffix(".partial")
        torch.save(torch.cat(results), temp)
        temp.replace(path)
        print(name, len(images), "feature pairs cached", flush=True)


if __name__ == "__main__":
    main()
