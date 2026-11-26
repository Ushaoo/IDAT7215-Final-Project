"""Validate out-of-fold predictions independently and regenerate the report figure."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from grouped.protocol import ARMS, SEEDS, ARCHS


def main(root):
    manifest = pd.read_csv(root / "split.csv")
    validated = []
    for arch in ARCHS:
        predictions = pd.read_csv(root / arch / "out_of_fold.csv.gz")
        runs = pd.read_csv(root / arch / "metrics.csv").set_index(
            ["fold", "seed", "arm"]
        )
        assert len(runs) == 4 * len(SEEDS) * len(ARMS) and runs.index.is_unique
        assert len(predictions) == 1557 * len(ARMS) * len(SEEDS)
        for (fold, seed, arm), frame in predictions.groupby(["fold", "seed", "arm"]):
            run = runs.loc[fold, seed, arm]
            expected = (
                manifest[manifest[f"fold{fold}"] == "test"]
                .set_index("path")
                .label.sort_index()
            )
            actual = frame.set_index("path").label.sort_index()
            pd.testing.assert_series_equal(expected, actual)
            assert frame.p_defect.between(0, 1).all()
            pred = frame.p_defect.gt(0.5)
            tn, fp, fn, tp = confusion_matrix(frame.label, pred, labels=[0, 1]).ravel()
            assert np.array_equal(
                [tn, fp, fn, tp], run[["tn", "fp", "fn", "tp"]].values
            )
            score = f1_score(frame.label, pred, average="macro")
            assert np.isclose(score, run.macro_f1)
            validated.append(
                dict(
                    model=arch,
                    fold=fold,
                    seed=seed,
                    arm=arm,
                    macro_f1=score,
                    recall=tp / (tp + fn),
                    fpr=fp / (fp + tn),
                    accuracy=(tp + tn) / len(frame),
                )
            )
    data = pd.DataFrame(validated)
    summary = (
        data.groupby(["model", "arm"])[["macro_f1", "recall", "fpr", "accuracy"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(root / "summary.csv", index=False)
    folds = data.groupby(["model", "arm", "fold"]).macro_f1.mean().unstack("fold")
    folds.to_csv(root / "fold_summary.csv")
    table = summary.pivot(index="model", columns="arm", values="macro_f1").loc[
        list(ARCHS), list(ARMS)
    ]
    ax = table.rename(
        columns=dict(
            raw="Raw only",
            expanded="With augmented images",
            global_pool="Global average",
            balanced_global="Global + balanced loss",
        )
    ).plot.bar(figsize=(10, 5), rot=0)
    ax.set(
        xlabel="", ylabel="Macro-F1 (mean of four folds and three seeds)", ylim=(0, 1)
    )
    ax.legend(ncol=2, loc="upper right")
    ax.figure.tight_layout()
    ax.figure.savefig(root / "ablation.png", dpi=180)
    plt.close(ax.figure)
    print(table.round(4).to_string())
    print(folds.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
