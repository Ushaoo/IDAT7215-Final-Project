"""Paired frozen-backbone ablations, selected on validation groups."""

import argparse
import copy
import csv
import gzip
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from grouped.protocol import ARMS, SEEDS


def metrics(y, p):
    tn, fp, fn, tp = torch.bincount(2 * y + p, minlength=4).tolist()
    errors = fp + fn
    f1 = 2 * tp / max(1, 2 * tp + errors)
    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=(tp + tn) / len(y),
        macro_f1=(f1 + 2 * tn / max(1, 2 * tn + errors)) / 2,
        defect_f1=f1,
        recall=tp / max(tp + fn, 1),
        fpr=fp / max(fp + tn, 1),
    )


def shuffled_epoch(count, target_size, gen):
    parts = [
        torch.randperm(count, generator=gen)
        for _ in range((target_size + count - 1) // count)
    ]
    return torch.cat(parts)[:target_size]


def train_one(x, y, weights, train, val, test, budget, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    head = nn.Sequential(
        nn.Linear(x.shape[-1], 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2)
    )
    for layer in (head[0], head[3]):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    opt = torch.optim.Adam(head.parameters(), lr=0.001)
    best_key = (-1.0, float("-inf"))
    for epoch in range(1, 31):
        head.train()
        for order in shuffled_epoch(len(train), budget, gen).split(64):
            batch = train[order]
            opt.zero_grad()
            logits = head(x[batch])
            losses = nn.functional.cross_entropy(logits, y[batch], reduction="none")
            (losses * weights[batch]).mean().backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            logits = head(x[val])
            key = (
                metrics(y[val], logits.argmax(1))["macro_f1"],
                -nn.functional.cross_entropy(logits, y[val]).item(),
            )
        if key > best_key:
            best_key = key
            selected = epoch
            state = copy.deepcopy(head.state_dict())
    head.load_state_dict(state)
    with torch.no_grad():
        prob = head(x[test]).softmax(1)[:, 1]
    return prob, state, selected, best_key[0]


def run(arch, cache, output):
    with (output / "split.csv").open() as f:
        rows = list(csv.DictReader(f))
    features = torch.load(cache / (arch + ".pt"), weights_only=True)
    assert features.size(0) == len(rows)
    y = torch.tensor([int(r["label"]) for r in rows])
    summary = []
    model_dir = output / arch
    model_dir.mkdir(exist_ok=True)
    with gzip.open(model_dir / "out_of_fold.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "seed", "arm", "path", "label", "p_defect"])
        for fold in range(4):
            roles = [r[f"fold{fold}"] for r in rows]
            train_all = [i for i, s in enumerate(roles) if s == "train"]
            val = torch.tensor([i for i, s in enumerate(roles) if s == "val"])
            test = torch.tensor([i for i, s in enumerate(roles) if s == "test"])
            for arm in ARMS:
                train = torch.tensor(
                    [
                        i
                        for i in train_all
                        if arm != "raw" or rows[i]["augmented"] == "0"
                    ]
                )
                weights = torch.ones(len(rows))
                if arm == "balanced_global":
                    # Equal domain mass, equal class mass per domain, equal groups per class.
                    # Pool SMIS raw and augmented images of a class into a single group.
                    def stratum(r):
                        group = r["group"] if r["source"] == "kaggle" else "lab"
                        return r["source"], r["label"], group

                    counts = Counter(stratum(rows[i]) for i in train_all)
                    groups = Counter((source, label) for source, label, group in counts)
                    for i in train_all:
                        key = stratum(rows[i])
                        weights[i] = len(train) / (4 * groups[key[:2]] * counts[key])
                x = features[
                    :, int(arm in ("global_pool", "balanced_global"))
                ].contiguous()
                for seed in SEEDS:
                    prob, state, epoch, score = train_one(
                        x, y, weights, train, val, test, len(train_all), seed
                    )
                    stats = metrics(y[test], (prob > 0.5).long())
                    summary.append(
                        dict(
                            model=arch,
                            fold=fold,
                            seed=seed,
                            arm=arm,
                            n_train=len(train),
                            n_val=len(val),
                            n_test=len(test),
                            epoch=epoch,
                            val_macro_f1=score,
                            **stats,
                        )
                    )
                    writer.writerows(
                        (fold, seed, arm, rows[i]["path"], int(y[i]), p)
                        for i, p in zip(test.tolist(), prob.tolist())
                    )
                    if seed == 42:
                        torch.save(state, model_dir / f"{arm}_fold{fold}_head.pt")
            print(arch, "fold", fold, "finished", flush=True)
    with (model_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("architectures", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for arch in args.architectures:
        run(arch, args.cache, args.output)
