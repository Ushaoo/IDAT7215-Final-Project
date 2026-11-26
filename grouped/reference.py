"""Study Three reference-style split and matched Kaggle-only grouped controls."""

import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
import torch
from grouped.protocol import ARCHS, SEEDS
from grouped.train import train_one, metrics


def holdout(indices, fraction):
    # sklearn ShuffleSplit: ceil(n*fraction) test rows from the leading permutation.
    order = np.random.RandomState(42).permutation(len(indices))
    count = int(np.ceil(len(indices) * fraction))
    return indices[order[count:]], indices[order[:count]]


def run(cache, out):
    rows = list(csv.DictReader((out / "split.csv").open()))
    ids = np.array([i for i, r in enumerate(rows) if r["source"] == "kaggle"])
    train, temp = holdout(ids, 0.30)
    val, test = holdout(temp, 0.66)
    roles = [("random", 0, train, val, test)]
    for f in range(4):
        parts = [
            [i for i in ids if rows[i][f"fold{f}"] == s]
            for s in ("train", "val", "test")
        ]
        roles.append(("grouped", f, *parts))
    with (out / "reference_split.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scheme", "fold", "path", "role"])
        for scheme, fold, *parts in roles:
            for role, indices in zip(("train", "val", "test"), parts):
                writer.writerows((scheme, fold, rows[i]["path"], role) for i in indices)
    y = torch.tensor([int(r["label"]) for r in rows])
    w = torch.ones(len(rows))
    summary = []
    with gzip.open(out / "reference_predictions.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "scheme", "fold", "seed", "path", "label", "p_defect"]
        )
        for arch in ARCHS:
            x = torch.load(cache / (arch + ".pt"), weights_only=True)[:, 0].contiguous()
            for scheme, fold, tr, va, te in roles:
                tr, va, te = map(torch.as_tensor, (tr, va, te))
                for seed in SEEDS:
                    prob, _, epoch, score = train_one(
                        x, y, w, tr, va, te, len(tr), seed
                    )
                    summary.append(
                        dict(
                            model=arch,
                            scheme=scheme,
                            fold=fold,
                            seed=seed,
                            n_train=len(tr),
                            n_val=len(va),
                            n_test=len(te),
                            epoch=epoch,
                            val_macro_f1=score,
                            **metrics(y[te], (prob > 0.5).long()),
                        )
                    )
                    writer.writerows(
                        (arch, scheme, fold, seed, rows[i]["path"], int(y[i]), p)
                        for i, p in zip(te.tolist(), prob.tolist())
                    )
            print("reference:", arch, flush=True)
    with (out / "reference_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(1)
    run(a.cache, a.output)
